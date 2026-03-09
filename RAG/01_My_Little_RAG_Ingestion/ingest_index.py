import os
import json
import hashlib
import time
import uuid
import math
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import re
import threading

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, TextIndexParams, Filter, FieldCondition, MatchValue, FilterSelector, SparseVectorParams, SparseVector

# Configuration
# folder and file storage
INPUT_DIR = "input"
CHUNKS_DIR = "chunks"
METADATA_FILE = "metadata/file_metadata.json"

# Provider
PROVIDER = os.getenv("PROVIDER", "openai").lower()  # "openai" or "ollama"

# Embedding
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "CPP_snowflake-embed-l-v2.0-GGUF")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1024))  # Embedding dimension

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Qdrant
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Chunking
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 600))  # Maximum tokens per chunk
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", 200))  # Minimum tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))    # Overlap between chunks

# Setup logging
class Logger:
    def __init__(self):
        self.run_id = str(uuid.uuid4())[:8]
        self.start_time = time.time()
        self.errors = []
        self.warnings = []

    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{self.run_id}] [{level}] {message}"
        print(log_msg)

        # Capture errors and warnings
        if level == "ERROR":
            self.errors.append(log_msg)
        elif level == "WARNING":
            self.warnings.append(log_msg)

    def print_summary(self):
        """Print a summary of all errors and warnings at the end of the run"""
        total_time = time.time() - self.start_time
        print(f"\n{'='*50}")
        print(f"Run completed in {total_time:.2f} seconds")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*50}")

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        else:
            print("\n✅ No errors detected")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        else:
            print("\n✅ No warnings detected")

        print(f"{'='*50}")

logger = Logger()

# Sparse vector configuration
SPARSE_INDEX_SPACE = int(os.getenv("SPARSE_INDEX_SPACE", 100000))  # Same as previous hash space

# Semaphore to limit concurrent embedding requests to prevent server overload
# With larger ubatch-size, we can process more requests safely, but still want some control
embedding_semaphore = threading.Semaphore(4)  # Allow up to 4 concurrent embedding requests


def stable_term_index(term: str, mod: int = SPARSE_INDEX_SPACE) -> int:
    """Stable, cross-process deterministic hash -> index."""
    # md5 returns hex; convert to int
    h = hashlib.md5(term.encode('utf-8')).hexdigest()
    return int(h, 16) % mod

def generate_sparse_vector(text: str) -> SparseVector:
    """Generate a sparse vector representation of text with improved term weighting"""
    if not text or not text.strip():
        return SparseVector(indices=[], values=[])
    
    # Enhanced tokenization to handle German characters and special symbols including legal notation like §98
    # This pattern captures:
    # 1. § followed by optional whitespace and numbers (e.g., "§ 98", "§98") as single tokens
    # 2. Words that may contain special symbols
    # 3. Numbers
    # 4. Other word-like tokens
    tokens = re.findall(r'(?:§\s*\d+|[a-zA-ZäöüÄÖÜß§]+\d*|\d+[a-zA-ZäöüÄÖÜß§]*|[a-zA-ZäöüÄÖÜß§]+)', text.lower())
    
    # Remove stopwords and short terms
    filtered_tokens = [token for token in tokens if len(token) >= 2]
    
    if not filtered_tokens:
        return SparseVector(indices=[], values=[])

    # Count term frequencies
    tf = Counter(filtered_tokens)

    # Calculate TF-IDF like weighting to give more importance to rare terms
    total_terms = len(filtered_tokens)
    index_values = {}

    # Create indices based on stable hash of terms (to maintain consistency)
    # Handle potential hash collisions by summing frequencies
    for term, freq in tf.items():
        # Create a stable hash-based index for the term
        term_hash = stable_term_index(term)  # Limit index space

        # Apply a log-based weight to reduce the impact of very frequent terms
        # This is a simplified TF-IDF approach
        log_weight = 1 + math.log(freq)  # Adding 1 to avoid log(0)

        # If index already exists, sum the weights (handle hash collisions)
        if term_hash in index_values:
            index_values[term_hash] += log_weight
        else:
            index_values[term_hash] = log_weight

    # Convert to lists and sort by indices for Qdrant sparse vector format
    if index_values:
        # Sort by indices to maintain consistent order
        sorted_items = sorted(index_values.items())
        indices, values = zip(*sorted_items)
        indices = list(indices)
        values = list(values)

        # Apply min-max normalization to ensure consistent value range
        min_val, max_val = min(values), max(values)
        if max_val > min_val:
            # Normalize to [0, 1] range
            values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            # If all values are the same, set them to 1.0
            values = [1.0 for _ in values]
    else:
        indices = []
        values = []

    return SparseVector(indices=indices, values=values)

def load_file_metadata() -> Dict[str, Any]:
    """Load file metadata from disk"""
    if not os.path.exists(METADATA_FILE):
        return {}

    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.log(f"Error loading metadata: {e}", "ERROR")
        return {}

def save_file_metadata(metadata: Dict[str, Any]):
    """Save metadata with atomic write"""
    try:
        os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
        temp_file = f"{METADATA_FILE}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        os.replace(temp_file, METADATA_FILE)
    except Exception as e:
        logger.log(f"Error saving metadata: {e}", "ERROR")

def mark_as_embedded(file_keys: List[str], collection_name: str):
    """Mark files as successfully embedded (commit phase).
    
    This is the commit point - files are only considered processed after this succeeds.
    
    Args:
        file_keys: List of file paths (keys) to mark as embedded
        collection_name: Collection name
    """
    metadata = load_file_metadata()
    updated = False
    
    for file_key in file_keys:
        if file_key in metadata and collection_name in metadata[file_key].get('collections', {}):
            metadata[file_key]['collections'][collection_name]['embedded'] = True
            updated = True
    
    if updated:
        save_file_metadata(metadata)
        logger.log(f"Marked {len(file_keys)} files as successfully embedded for collection {collection_name}", "INFO")

def get_file_hash(file_path: str) -> str:
    """Get SHA256 hash of file content"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.log(f"Error hashing file {file_path}: {e}", "ERROR")
        return ""

def has_file_changed(file_path: str, stored_metadata: Dict[str, Any], collection_name: str) -> bool:
    """Check if file has changed, was processed with a different collection, or has incomplete embedding"""
    file_key = str(file_path)
    if file_key not in stored_metadata:
        return True

    # Check if file was processed with a different collection
    if collection_name not in stored_metadata[file_key].get('collections', {}):
        return True

    # Check if embedding is incomplete (missing or false)
    collection_metadata = stored_metadata[file_key]['collections'][collection_name]
    embedded = collection_metadata.get('embedded', False)
    if not embedded:
        return True

    try:
        current_mtime = os.path.getmtime(file_path)
        current_hash = get_file_hash(file_path)
        stored_mtime = collection_metadata.get('mtime', 0)
        stored_hash = collection_metadata.get('hash', '')

        return (abs(current_mtime - stored_mtime) > 1 or current_hash != stored_hash)
    except Exception as e:
        logger.log(f"Error checking file changes {file_path}: {e}", "ERROR")
        return True

def remove_old_chunks(doc_id: int, chunks_dir: str = CHUNKS_DIR, collection_name: str = None) -> int:
    """Move old chunks for a specific document into a .removed folder (returns moved count).
    
    Also removes chunks from .pending directory if they exist.
    """
    if collection_name is None:
        collection_name = QDRANT_COLLECTION  # Maintain backward compatibility
    collection_chunks_dir = os.path.join(chunks_dir, collection_name)
    removed_dir = os.path.join(collection_chunks_dir, ".removed")
    pending_dir = os.path.join(collection_chunks_dir, ".pending")
    os.makedirs(removed_dir, exist_ok=True)

    pattern = f"{doc_id}_*.json"
    moved = 0
    errors = 0

    # Remove from main directory
    for chunk_file in Path(collection_chunks_dir).glob(pattern):
        if ".pending" in str(chunk_file) or ".removed" in str(chunk_file):
            continue
        try:
            dest = Path(removed_dir) / chunk_file.name
            # If a file with same name exists in removed, add a suffix
            if dest.exists():
                dest = dest.with_name(f"{chunk_file.stem}_{int(time.time())}{chunk_file.suffix}")
            chunk_file.rename(dest)
            moved += 1
        except Exception as e:
            logger.log(f"Could not move old chunk {chunk_file} to .removed: {e}", "WARNING")
            errors += 1
    
    # Remove from .pending directory (chunks that were never embedded)
    if os.path.exists(pending_dir):
        for chunk_file in Path(pending_dir).glob(pattern):
            try:
                chunk_file.unlink()
                moved += 1
            except Exception as e:
                logger.log(f"Could not remove pending chunk {chunk_file}: {e}", "WARNING")
                errors += 1

    if moved or errors:
        logger.log(f"Moved {moved} old chunks (errors: {errors}) for doc_id {doc_id} in collection {collection_name}", "INFO")

    return moved

def get_existing_doc_ids(collection_name: str = None) -> set:
    """Get all existing document IDs from chunks (excluding .pending directory)"""
    doc_ids = set()
    # Use collection-specific subfolder
    if collection_name is None:
        collection_name = QDRANT_COLLECTION  # Maintain backward compatibility
    collection_chunks_dir = os.path.join(CHUNKS_DIR, collection_name)
    for chunk_file in Path(collection_chunks_dir).glob("*.json"):
        # Skip .pending directory
        if ".pending" in str(chunk_file):
            continue
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk = json.load(f)
                if isinstance(chunk, dict) and 'doc_id' in chunk:
                    doc_ids.add(int(chunk['doc_id']))
        except Exception as e:
            logger.log(f"Could not read chunk file {chunk_file}: {e}", "WARNING")
    return doc_ids

def remove_yaml_front_matter(text: str) -> str:
    """Remove YAML front matter from the beginning of documents.

    The YAML front matter is enclosed between '---' lines at the beginning of the document.
    """
    # Pattern to match YAML front matter at the beginning of the document
    # It starts with --- on a line by itself, followed by any content,
    # and ends with --- on a line by itself
    pattern = r'^---\n.*?\n---(?:\n|$)'

    # Use re.DOTALL to match across multiple lines
    result = re.sub(pattern, '', text, count=1, flags=re.DOTALL)
    return result.strip()

def get_next_doc_id(collection_name: str = None) -> int:
    """Get the next available document ID"""
    existing_ids = get_existing_doc_ids(collection_name)
    return max(existing_ids, default=0) + 1

def get_collection_name_from_path(file_path: Path) -> str:
    """Get collection name from file path based on subfolder, with sanitization"""
    # Get the parent directory relative to INPUT_DIR
    relative_path = file_path.relative_to(INPUT_DIR)
    # If file is directly in input dir (no subfolder), use default collection
    if len(relative_path.parts) <= 1:
        return QDRANT_COLLECTION
    else:
        # Use the first part (immediate subfolder) as collection name
        raw_collection_name = relative_path.parts[0]
        # Sanitize collection name to comply with Qdrant requirements
        # Qdrant collection names should contain lowercase letters, numbers, hyphens, underscores
        sanitized = re.sub(r'[^a-z0-9_-]', '_', raw_collection_name.lower())
        # Ensure it doesn't start with underscore/dash and has appropriate length
        sanitized = sanitized.strip('_-')
        sanitized = sanitized[:255]  # Limit length
        if not sanitized:  # If it becomes empty after sanitization
            sanitized = "default_collection"
        return sanitized

def load_documents_incremental() -> tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """Load only new or changed documents based on collection"""
    # Check for force reindex environment variable
    force_reindex = os.getenv("FORCE_REINDEX", "false").lower() == "true"
    if force_reindex:
        logger.log("FORCE_REINDEX is enabled. All documents will be reprocessed.", "INFO")

    stored_metadata = load_file_metadata()
    current_metadata = {}

    # Group files by collection first
    files_by_collection = {}
    for path in Path(INPUT_DIR).rglob("*"):
        if path.suffix not in [".txt", ".md", ".html"]:
            continue

        file_key = str(path)

        # Determine collection name based on subfolder
        collection_name = get_collection_name_from_path(path)

        if collection_name not in files_by_collection:
            files_by_collection[collection_name] = []
        files_by_collection[collection_name].append((path, file_key))

    # Build comprehensive mapping of file paths to doc_ids for ALL collections first
    # Include both main directory and .pending directory to handle failed runs
    all_path_to_doc_id = {}
    for collection_name in files_by_collection.keys():
        collection_chunks_dir = os.path.join(CHUNKS_DIR, collection_name)
        pending_chunks_dir = os.path.join(collection_chunks_dir, ".pending")
        
        # Load from main directory
        if os.path.exists(collection_chunks_dir):
            for chunk_file in Path(collection_chunks_dir).glob("*.json"):
                if ".pending" in str(chunk_file):
                    continue
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        source_path = chunk['source']
                        # Store with collection context to avoid conflicts
                        key = f"{collection_name}:{source_path}"
                        all_path_to_doc_id[key] = {
                            'doc_id': chunk['doc_id'],
                            'collection': collection_name
                        }
                except Exception as e:
                    logger.log(f"Could not read chunk file {chunk_file}: {e}", "WARNING")
        
        # Load from .pending directory (to handle chunks that weren't successfully embedded)
        if os.path.exists(pending_chunks_dir):
            for chunk_file in Path(pending_chunks_dir).glob("*.json"):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk = json.load(f)
                        source_path = chunk['source']
                        # Store with collection context to avoid conflicts
                        key = f"{collection_name}:{source_path}"
                        # If the file is also in the main directory, prefer that (it was successfully embedded)
                        if key not in all_path_to_doc_id:
                            all_path_to_doc_id[key] = {
                                'doc_id': chunk['doc_id'],
                                'collection': collection_name
                            }
                except Exception as e:
                    logger.log(f"Could not read pending chunk file {chunk_file}: {e}", "WARNING")

    new_docs = []
    changed_doc_ids_by_collection: Dict[str, set] = defaultdict(set)

    # Process each collection separately
    for collection_name, file_list in files_by_collection.items():
        # Get doc ID counter for this collection
        doc_id_counter = get_next_doc_id(collection_name)

        for path, file_key in file_list:
            try:
                current_mtime = os.path.getmtime(path)
                current_hash = get_file_hash(path)
            except OSError as e:
                logger.log(f"Could not access file {path}: {e}", "WARNING")
                continue

            # Check if file changed or needs reprocessing for this collection
            # When FORCE_REINDEX is true, process all files regardless of changes
            file_has_changed = has_file_changed(path, stored_metadata, collection_name)
            should_process_file = force_reindex or file_has_changed

            # Update current metadata
            if file_key not in current_metadata:
                current_metadata[file_key] = {
                    'collections': {}
                }

            # Copy existing collections data if available
            if file_key in stored_metadata and 'collections' in stored_metadata[file_key]:
                current_metadata[file_key]['collections'] = stored_metadata[file_key]['collections'].copy()

            # Only set embedded: false for files that are actually being processed
            # Unchanged files should preserve their existing embedded status
            if should_process_file:
                current_metadata[file_key]['collections'][collection_name] = {
                    'mtime': current_mtime,
                    'hash': current_hash,
                    'embedded': False
                }
            else:
                # File unchanged - update mtime and hash but preserve embedded status
                if collection_name in current_metadata[file_key]['collections']:
                    current_metadata[file_key]['collections'][collection_name]['mtime'] = current_mtime
                    current_metadata[file_key]['collections'][collection_name]['hash'] = current_hash
                else:
                    # Collection not in metadata yet (shouldn't happen but handle gracefully)
                    current_metadata[file_key]['collections'][collection_name] = {
                        'mtime': current_mtime,
                        'hash': current_hash,
                        'embedded': False
                    }

            if should_process_file:
                logger.log(f"Processing {'new' if file_key not in stored_metadata or collection_name not in stored_metadata[file_key].get('collections', {}) else 'changed/re-collection'} file: {path} for collection: {collection_name}")

                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        if path.suffix == ".html":
                            text = BeautifulSoup(text, "html.parser").get_text()

                        # Remove YAML front matter from the document
                        text = remove_yaml_front_matter(text)
                except Exception as e:
                    logger.log(f"Could not read file {path}: {e}", "WARNING")
                    continue

                # CRITICAL FIX: Consistent document ID handling
                collection_key = f"{collection_name}:{file_key}"
                old_doc_id = None

                # Check if this file was previously indexed in THIS collection
                if collection_key in all_path_to_doc_id:
                    old_doc_id = all_path_to_doc_id[collection_key]['doc_id']
                    # Record for deletion BEFORE removing chunk files (avoid timing issue)
                    changed_doc_ids_by_collection[collection_name].add(int(old_doc_id))
                    # Remove old chunks
                    remove_old_chunks(old_doc_id, chunks_dir=CHUNKS_DIR, collection_name=collection_name)
                    # Reuse the same doc_id
                    doc_id = old_doc_id
                else:
                    # Assign new doc_id
                    doc_id = doc_id_counter
                    doc_id_counter += 1

                new_docs.append({
                    "text": text,
                    "source": file_key,
                    "doc_id": doc_id,
                    "collection_name": collection_name  # Add collection name to track where to store this doc
                })

    # Check for deleted files (only if not force reindexing)
    if not force_reindex:
        deleted_files = set(stored_metadata.keys()) - set(current_metadata.keys())
        for deleted_file in deleted_files:
            logger.log(f"File deleted: {deleted_file}")
            # Determine which collection this file belonged to
            deleted_file_path = Path(deleted_file)
            collection_name = get_collection_name_from_path(deleted_file_path)

            # Find and remove chunks for deleted files
            # Need to load doc_id from the specific collection's chunks (both main and .pending)
            path_to_doc_id = {}
            collection_chunks_dir = os.path.join(CHUNKS_DIR, collection_name)
            pending_chunks_dir = os.path.join(collection_chunks_dir, ".pending")
            
            # Check main directory
            if os.path.exists(collection_chunks_dir):
                for chunk_file in Path(collection_chunks_dir).glob("*.json"):
                    if ".pending" in str(chunk_file):
                        continue
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            chunk = json.load(f)
                            source_path = chunk['source']
                            if source_path == deleted_file:
                                path_to_doc_id[source_path] = chunk['doc_id']
                    except Exception as e:
                        logger.log(f"Could not read chunk file {chunk_file}: {e}", "WARNING")
            
            # Check .pending directory
            if os.path.exists(pending_chunks_dir):
                for chunk_file in Path(pending_chunks_dir).glob("*.json"):
                    try:
                        with open(chunk_file, 'r', encoding='utf-8') as f:
                            chunk = json.load(f)
                            source_path = chunk['source']
                            if source_path == deleted_file:
                                # Only add if not already in path_to_doc_id (prefer main directory)
                                if source_path not in path_to_doc_id:
                                    path_to_doc_id[source_path] = chunk['doc_id']
                    except Exception as e:
                        logger.log(f"Could not read pending chunk file {chunk_file}: {e}", "WARNING")

            if deleted_file in path_to_doc_id:
                doc_id_to_remove = path_to_doc_id[deleted_file]
                # Record for deletion BEFORE removing chunk files (avoid timing issue)
                changed_doc_ids_by_collection[collection_name].add(int(doc_id_to_remove))
                remove_old_chunks(doc_id_to_remove, chunks_dir=CHUNKS_DIR, collection_name=collection_name)
                logger.log(f"Marked doc_id {doc_id_to_remove} for removal from collection {collection_name}")

    # Save updated metadata
    save_file_metadata(current_metadata)

    # Convert sets to sorted lists for deterministic output
    changed_doc_ids_by_collection_final = {k: sorted(list(v)) for k, v in changed_doc_ids_by_collection.items()}

    return new_docs, changed_doc_ids_by_collection_final

def split_into_paragraphs(text: str) -> list[str]:
    """
    Splits text into paragraphs.
    A paragraph is separated by one or more blank lines.
    """
    paragraphs = re.split(r'\n\s*\n+', text)
    return [p.strip() for p in paragraphs if p.strip()]


def estimate_token_count(text: str) -> int:
    # Very rough but good enough for chunk sizing
    return len(text.split())


def split_sentences_respecting_bounds(text: str) -> List[str]:
    """
    Split text into sentences while respecting legal abbreviations and notation.
    This prevents splitting at abbreviations like 'Abs.', 'Nr.', etc. in legal texts.
    """
    # List of common German legal abbreviations that should not be treated as sentence endings
    abbreviations = [
        'Abs', 'Nr', 'S', 'Ziff', 'Bsp', 'u.a', 'ff', 'z.B', 'Art', 'EG',
        'u.U', 'd.h', 'i.S.v', 'z.T', 'usw', 'etc', 'vgl', 's.o', 's.u',
        'f', 'm', 'o.k', 'g.g.A', 'u.g', 'i.H.v', 'i.V.m', 'z.G', 'sog',
        '§§'  # Multiple paragraph symbols
    ]

    # Create a pattern that matches sentence endings but excludes legal abbreviations
    # This pattern looks for sentence ending punctuation followed by whitespace and capital letter
    # but excludes known abbreviations

    # First, protect abbreviations by replacing them with a placeholder
    protected_text = text
    placeholder_map = {}

    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR_{i}__"
        # Use word boundaries to match abbreviations followed by a period
        pattern = r'\b' + re.escape(abbr) + r'\.'
        protected_text = re.sub(pattern, f"{placeholder}.", protected_text)
        placeholder_map[placeholder] = abbr

    # Now split on sentence boundaries (., !, ?) followed by whitespace and capital letter
    # Also handle cases where sentences end with quotes or parentheses before the next capital
    sentence_pattern = r'[.!?]+[\'"»\])]*\s+(?=[A-ZÄÖÜ])'
    sentences = re.split(sentence_pattern, protected_text)

    # Restore the original abbreviations
    restored_sentences = []
    for sentence in sentences:
        restored_sentence = sentence
        for placeholder, abbr in placeholder_map.items():
            restored_sentence = restored_sentence.replace(f"{placeholder}.", f"{abbr}.")
        restored_sentences.append(restored_sentence.strip())

    # Filter out empty sentences
    return [s for s in restored_sentences if s.strip()]




def chunk_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split document into paragraph-level chunks with minimum size enforcement"""
    if not isinstance(doc, dict) or 'text' not in doc:
        logger.log(f"Invalid document format: {doc}", "WARNING")
        return []

    full_text = doc["text"]
    processed_chunks = []

    try:
        # Split text into paragraphs
        paragraphs = split_into_paragraphs(full_text)

        chunk_id = 0
        buffer = []
        buffer_token_count = 0

        for para in paragraphs:
            token_count = estimate_token_count(para)

            if token_count > MAX_CHUNK_SIZE:
                # Handle oversized paragraph by splitting it
                # First, flush current buffer if it has content
                if buffer:
                    processed_chunks.append({
                        "doc_id": doc["doc_id"],
                        "chunk_id": chunk_id,
                        "text": "\n\n".join(buffer),
                        "source": doc["source"],
                        "collection_name": doc.get("collection_name"),
                    })
                    chunk_id += 1
                    buffer = []
                    buffer_token_count = 0

                # Try sentence-aware splitting first for legal texts
                sentences = split_sentences_respecting_bounds(para)

                if len(sentences) > 1:
                    # We have multiple sentences, try to group them respecting MAX_CHUNK_SIZE
                    sentence_buffer = []
                    sentence_buffer_token_count = 0

                    for sentence in sentences:
                        sentence_token_count = estimate_token_count(sentence)

                        if sentence_token_count > MAX_CHUNK_SIZE:
                            # Individual sentence is too large, split by words as fallback
                            words = sentence.split()
                            sentence_chunks = [
                                " ".join(words[i:i+MAX_CHUNK_SIZE])
                                for i in range(0, len(words), MAX_CHUNK_SIZE)
                            ]

                            for chunk_text in sentence_chunks:
                                processed_chunks.append({
                                    "doc_id": doc["doc_id"],
                                    "chunk_id": chunk_id,
                                    "text": chunk_text,
                                    "source": doc["source"],
                                    "collection_name": doc.get("collection_name"),
                                })
                                chunk_id += 1
                        else:
                            # Check if adding this sentence would exceed MAX_CHUNK_SIZE
                            if sentence_buffer and (sentence_buffer_token_count + sentence_token_count) > MAX_CHUNK_SIZE:
                                # Emit current buffer as chunk
                                processed_chunks.append({
                                    "doc_id": doc["doc_id"],
                                    "chunk_id": chunk_id,
                                    "text": " ".join(sentence_buffer),
                                    "source": doc["source"],
                                    "collection_name": doc.get("collection_name"),
                                })
                                chunk_id += 1
                                # Start new buffer with current sentence
                                sentence_buffer = [sentence]
                                sentence_buffer_token_count = sentence_token_count
                            else:
                                # Add sentence to buffer
                                sentence_buffer.append(sentence)
                                sentence_buffer_token_count += sentence_token_count

                    # Emit remaining sentences in buffer if any
                    if sentence_buffer:
                        processed_chunks.append({
                            "doc_id": doc["doc_id"],
                            "chunk_id": chunk_id,
                            "text": " ".join(sentence_buffer),
                            "source": doc["source"],
                            "collection_name": doc.get("collection_name"),
                        })
                        chunk_id += 1
                else:
                    # Fallback to word-based splitting if no sentences were found
                    words = para.split()
                    para_chunks = [
                        " ".join(words[i:i+MAX_CHUNK_SIZE])
                        for i in range(0, len(words), MAX_CHUNK_SIZE)
                    ]

                    for chunk_text in para_chunks:
                        processed_chunks.append({
                            "doc_id": doc["doc_id"],
                            "chunk_id": chunk_id,
                            "text": chunk_text,
                            "source": doc["source"],
                            "collection_name": doc.get("collection_name"),
                        })
                        chunk_id += 1
            else:
                # Add paragraph to buffer
                buffer.append(para)
                buffer_token_count += token_count

                # If buffer has reached minimum size, emit as chunk
                if buffer_token_count >= MIN_CHUNK_SIZE:
                    processed_chunks.append({
                        "doc_id": doc["doc_id"],
                        "chunk_id": chunk_id,
                        "text": "\n\n".join(buffer),
                        "source": doc["source"],
                        "collection_name": doc.get("collection_name"),
                    })
                    chunk_id += 1
                    buffer = []
                    buffer_token_count = 0

        # Handle remaining content in buffer - emit as final chunk even if smaller than MIN_CHUNK_SIZE
        if buffer:
            processed_chunks.append({
                "doc_id": doc["doc_id"],
                "chunk_id": chunk_id,
                "text": "\n\n".join(buffer),
                "source": doc["source"],
                "collection_name": doc.get("collection_name"),
            })

        return processed_chunks
    except Exception as e:
        logger.log(f"Error chunking document {doc.get('source', 'unknown')}: {e}", "ERROR")
        return []

def get_embedding_function():
    """Get embeddings with retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if PROVIDER == "ollama":
                return OllamaEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    base_url=OLLAMA_BASE_URL,
                    # Ollama doesn't support chunk_size parameter
                )
            elif PROVIDER == "openai":
                return OpenAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_base=OPENAI_BASE_URL,
                    # Increased batch size for better performance with new server settings
                    chunk_size=50,  # Process up to 50 texts at a time for better performance
                )
            else:  # Default to OpenAI-compatible
                return OpenAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    openai_api_key=OPENAI_API_KEY,
                    openai_api_base=OPENAI_BASE_URL,
                    # Increased batch size for better performance with new server settings
                    chunk_size=50,  # Process up to 50 texts at a time for better performance
                )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = (attempt + 1) * 2
            logger.log(f"Embedding connection failed (attempt {attempt + 1}), retrying in {wait_time}s...", "WARNING")
            time.sleep(wait_time)

def save_chunks_to_disk(chunks: List[Dict[str, Any]], path: str = CHUNKS_DIR, collection_name: str = QDRANT_COLLECTION, pending: bool = True):
    """Save chunks to disk with error handling. If pending=True, save to .pending directory for two-phase commit."""
    saved_count = 0
    errors = 0

    # Use collection-specific subfolder
    if pending:
        # Save to .pending directory for two-phase commit
        collection_chunks_dir = os.path.join(path, collection_name, ".pending")
    else:
        # Save directly to main directory (for compatibility with old code)
        collection_chunks_dir = os.path.join(path, collection_name)
    os.makedirs(collection_chunks_dir, exist_ok=True)

    for chunk in chunks:
        if not isinstance(chunk, dict) or 'doc_id' not in chunk or 'chunk_id' not in chunk:
            logger.log(f"Invalid chunk format: {chunk}", "WARNING")
            errors += 1
            continue

        chunk_id = f"{chunk['doc_id']}_{chunk['chunk_id']}"
        chunk_file_path = f"{collection_chunks_dir}/{chunk_id}.json"
        temp_path = f"{chunk_file_path}.tmp"

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            os.replace(temp_path, chunk_file_path)
            saved_count += 1
        except Exception as e:
            logger.log(f"Could not save chunk {chunk_file_path}: {e}", "WARNING")
            errors += 1
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    logger.log(f"Saved {saved_count} chunks to {collection_chunks_dir} (errors: {errors})", "INFO")

def move_chunks_from_pending(doc_ids: List[int], path: str = CHUNKS_DIR, collection_name: str = QDRANT_COLLECTION) -> int:
    """Move chunks from .pending directory to main directory after successful embedding.
    
    Args:
        doc_ids: List of document IDs to move chunks for
        path: Base chunks directory
        collection_name: Collection name
    
    Returns:
        Number of chunks moved
    """
    moved_count = 0
    errors = 0
    
    pending_dir = os.path.join(path, collection_name, ".pending")
    main_dir = os.path.join(path, collection_name)
    
    if not os.path.exists(pending_dir):
        logger.log(f"Pending directory does not exist: {pending_dir}", "INFO")
        return 0
    
    os.makedirs(main_dir, exist_ok=True)
    
    for doc_id in doc_ids:
        pattern = f"{doc_id}_*.json"
        for chunk_file in Path(pending_dir).glob(pattern):
            dest = Path(main_dir) / chunk_file.name
            try:
                # Check if destination exists
                if dest.exists():
                    # Destination exists, remove it first
                    dest.unlink()
                # Move the file
                chunk_file.rename(dest)
                moved_count += 1
            except Exception as e:
                logger.log(f"Could not move chunk {chunk_file} to main directory: {e}", "WARNING")
                errors += 1
    
    logger.log(f"Moved {moved_count} chunks from .pending to {collection_name} (errors: {errors})", "INFO")
    return moved_count

def cleanup_pending_chunks(path: str = CHUNKS_DIR, collection_name: str = QDRANT_COLLECTION) -> int:
    """Clean up pending chunks for a collection. Called on startup to handle failed runs.
    
    Args:
        path: Base chunks directory
        collection_name: Collection name (or None to clean all collections)
    
    Returns:
        Number of chunks cleaned up
    """
    cleaned_count = 0
    
    if collection_name:
        collections = [collection_name]
    else:
        # Get all collection directories
        chunks_path = Path(path)
        if not chunks_path.exists():
            return 0
        collections = [d.name for d in chunks_path.iterdir() if d.is_dir() and d.name != ".removed"]
    
    for coll_name in collections:
        pending_dir = os.path.join(path, coll_name, ".pending")
        if os.path.exists(pending_dir):
            for chunk_file in Path(pending_dir).glob("*.json"):
                try:
                    chunk_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.log(f"Could not remove pending chunk {chunk_file}: {e}", "WARNING")
            
            # Try to remove the .pending directory if empty
            try:
                if not os.listdir(pending_dir):
                    os.rmdir(pending_dir)
            except Exception:
                pass
    
    if cleaned_count > 0:
        logger.log(f"Cleaned up {cleaned_count} pending chunks", "INFO")
    
    return cleaned_count

def load_cached_chunks(path: str = CHUNKS_DIR):
    # Use collection-specific subfolder
    collection_chunks_dir = os.path.join(path, QDRANT_COLLECTION)
    for file in Path(collection_chunks_dir).glob("*.json"):
        # Skip .pending directory
        if ".pending" in str(file):
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                yield json.load(f)
        except Exception as e:
            logger.log(f"Could not read chunk file {file}: {e}", "WARNING")

def update_qdrant_index(new_chunks: List[Dict[str, Any]], changed_doc_ids_by_collection: Dict[str, List[int]], embeddings):
    """Update Qdrant index incrementally for multiple collections.
    
    This function:
    1. Generates embeddings for chunks
    2. Upserts vectors to Qdrant
    3. On success, moves chunks from .pending to main directory
    4. Marks files as successfully embedded
    
    Returns:
        List of file keys (paths) that were successfully embedded
    """
    # Track files that were successfully embedded for marking
    successfully_embedded_files = set()
    
    # Increase timeout to handle larger processing times
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=300)

    # Group chunks by collection
    chunks_by_collection = {}
    for chunk in new_chunks:
        collection_name = chunk.get("collection_name", QDRANT_COLLECTION)  # Default to main collection if not specified
        if collection_name not in chunks_by_collection:
            chunks_by_collection[collection_name] = []
        chunks_by_collection[collection_name].append(chunk)

    # Process each collection separately
    for collection_name, collection_chunks in chunks_by_collection.items():
        # Check for force reindex environment variable
        force_reindex = os.getenv("FORCE_REINDEX", "false").lower() == "true"

        # Ensure collection exists
        collections = client.get_collections()
        collection_names = [collection.name for collection in collections.collections]

        if collection_name in collection_names:
            logger.log(f"Using existing Qdrant collection: {collection_name}")
            collection_info = client.get_collection(collection_name)
            # Check if vector configuration matches expected values
            dense_config = collection_info.config.params.vectors.get("dense")
            if not dense_config or dense_config.size != VECTOR_SIZE:
                logger.log(f"Vector configuration mismatch, recreating collection {collection_name}", "WARNING")
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.COSINE,
                            hnsw_config={
                                "m": 16,
                                "ef_construct": 100,
                                "full_scan_threshold": 10000,
                            },
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    },
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 100,
                        "full_scan_threshold": 10000,
                    },
                    optimizers_config={
                        "memmap_threshold": 20000,
                        "indexing_threshold": 20000,
                    }
                )
        else:
            logger.log(f"Creating new Qdrant collection: {collection_name}")
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.COSINE,
                            # Use hnsw config for better search performance
                            hnsw_config={
                                "m": 16,  # Number of edges per vertex (default 16, good for most cases)
                                "ef_construct": 100,  # Construction time parameter (default 100)
                                "full_scan_threshold": 10000,  # Use HNSW when collection size exceeds this (default 10000)
                            },
                            # Use quantization to reduce memory usage (optional)
                            quantization_config=None,  # Can set scalar quantization if needed
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    },
                    # Optimize for better performance
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 100,
                        "full_scan_threshold": 10000,
                    },
                    # Enable payload indexing for faster filtering
                    optimizers_config={
                        "memmap_threshold": 20000,      # Store index on disk if more than this many vectors
                        "indexing_threshold": 20000,    # Index vectors on disk after this many
                    }
                )

                # Create text index for BM25 search
                logger.log("Creating text index for sparse search...")
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="text",
                        field_schema=TextIndexParams(
                            type="text",
                            tokenizer="word",
                            min_token_len=2,
                            max_token_len=20,
                            lowercase=True,
                        )
                    )
                    logger.log("Text index created successfully")
                except Exception as e:
                    logger.log(f"Warning: Failed to create text index: {e}", "WARNING")
                    logger.log("Text search may not work properly", "WARNING")

                # Create payload indexes for better filtering performance
                logger.log("Creating payload indexes for doc_id and source...")
                try:
                    # Integer index for doc_id (faster filtering by document ID)
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="doc_id",
                        field_schema="integer"
                    )

                    # Keyword index for source (faster filtering by document source)
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source",
                        field_schema="keyword"
                    )

                    logger.log("Payload indexes created successfully")
                except Exception as e:
                    logger.log(f"Warning: Failed to create payload indexes: {e}", "WARNING")
            except Exception as e:
                # Handle the case where the collection was created by another process
                if "already exists" in str(e):
                    logger.log(f"Collection {collection_name} already exists, continuing...", "INFO")
                else:
                    logger.log(f"Failed to create collection {collection_name}: {e}", "ERROR")
                    continue  # Continue with other collections

        # Determine which doc ids to remove for THIS collection from the passed map
        collection_changed_doc_ids = changed_doc_ids_by_collection.get(collection_name, [])

        # Clear entire collection if force reindexing
        if force_reindex:
            logger.log(f"Force reindex enabled: Clearing all points from collection {collection_name}...")
            try:
                # Most efficient way to clear entire collection: delete and recreate
                # This avoids potential issues with filtering and ensures complete cleanup
                logger.log(f"Deleting collection {collection_name} for reindexing...")
                client.delete_collection(collection_name=collection_name)
                logger.log(f"Collection {collection_name} deleted successfully")

                # Recreate the collection with the same configuration
                logger.log(f"Recreating collection {collection_name}...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.COSINE,
                            # Use hnsw config for better search performance
                            hnsw_config={
                                "m": 16,  # Number of edges per vertex (default 16, good for most cases)
                                "ef_construct": 100,  # Construction time parameter (default 100)
                                "full_scan_threshold": 10000,  # Use HNSW when collection size exceeds this (default 10000)
                            },
                            # Use quantization to reduce memory usage (optional)
                            quantization_config=None,  # Can set scalar quantization if needed
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    },
                    # Optimize for better performance
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 100,
                        "full_scan_threshold": 10000,
                    },
                    # Enable payload indexing for faster filtering
                    optimizers_config={
                        "memmap_threshold": 20000,      # Store index on disk if more than this many vectors
                        "indexing_threshold": 20000,    # Index vectors on disk after this many
                    }
                )

                # Create text index for BM25 search
                logger.log("Creating text index for sparse search...")
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="text",
                        field_schema=TextIndexParams(
                            type="text",
                            tokenizer="word",
                            min_token_len=2,
                            max_token_len=20,
                            lowercase=True,
                        )
                    )
                    logger.log("Text index created successfully")
                except Exception as e:
                    logger.log(f"Warning: Failed to create text index: {e}", "WARNING")
                    logger.log("Text search may not work properly", "WARNING")

                # Create payload indexes for better filtering performance
                logger.log("Creating payload indexes for doc_id and source...")
                try:
                    # Integer index for doc_id (faster filtering by document ID)
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="doc_id",
                        field_schema="integer"
                    )

                    # Keyword index for source (faster filtering by document source)
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source",
                        field_schema="keyword"
                    )

                    logger.log("Payload indexes created successfully")
                except Exception as e:
                    logger.log(f"Warning: Failed to create payload indexes: {e}", "WARNING")

                logger.log(f"Collection {collection_name} recreated successfully for reindexing")
            except Exception as e:
                logger.log(f"Warning: Failed to clear collection {collection_name} by recreation: {e}", "WARNING")
        else:
            # Remove old points for changed documents (incremental update)
            if collection_changed_doc_ids:
                logger.log(f"Removing old vectors for {len(collection_changed_doc_ids)} changed documents from collection {collection_name}...")
                for doc_id in collection_changed_doc_ids:
                    try:
                        # Use FilterSelector to properly delete points by doc_id
                        # Ensure both stored and queried doc_id are integers
                        client.delete(
                            collection_name=collection_name,
                            points_selector=FilterSelector(
                                filter=Filter(
                                    must=[
                                        FieldCondition(
                                            key="doc_id",
                                            match=MatchValue(
                                                value=int(doc_id)  # Ensure this is an integer to match stored type
                                            )
                                        )
                                    ]
                                )
                            ),
                            wait=True
                        )

                        # VERIFY DELETION: Check if any points remain after deletion
                        try:
                            remaining_points = client.count(
                                collection_name=collection_name,
                                count_filter=Filter(
                                    must=[
                                        FieldCondition(
                                            key="doc_id",
                                            match=MatchValue(value=int(doc_id))
                                        )
                                    ]
                                )
                            ).count
                            if remaining_points > 0:
                                logger.log(f"Warning: {remaining_points} points still exist for doc_id {doc_id} after deletion in collection {collection_name}", "WARNING")
                            else:
                                logger.log(f"Verified deletion of all points for doc_id {doc_id} in collection {collection_name}")
                        except Exception as verify_e:
                            logger.log(f"Could not verify deletion for doc_id {doc_id} in collection {collection_name}: {verify_e}", "WARNING")

                    except Exception as e:
                        logger.log(f"Warning: Failed to delete old points for doc_id {doc_id} in collection {collection_name}: {e}", "WARNING")

        # Add new points to this collection
        if collection_chunks:
            # Optimize batch size based on available memory and performance with new server settings
            batch_size = int(os.getenv("QDRANT_BATCH_SIZE", 64))  # Increased for better performance
            logger.log(f"Processing {len(collection_chunks)} new chunks with batch size {batch_size} in collection {collection_name}")

            successful_points = 0
            failed_chunks = 0

            # Process chunks in batches for better performance
            for i in range(0, len(collection_chunks), batch_size):
                batch = collection_chunks[i:i + batch_size]
                batch_number = (i // batch_size) + 1
                total_batches = (len(collection_chunks) + batch_size - 1) // batch_size

                # Add retry mechanism for each batch
                max_retries = 3
                retry_count = 0

                while retry_count < max_retries:
                    try:
                        # Extract texts for batch embedding
                        texts = [chunk["text"] for chunk in batch]

                        # Skip extremely long chunks in batch
                        filtered_texts = []
                        filtered_chunks = []
                        for j, text in enumerate(texts):
                            if len(text) <= 10000:  # Limit text length
                                filtered_texts.append(text)
                                filtered_chunks.append(batch[j])
                            else:
                                logger.log(f"Skipping extremely long chunk ({len(text)} chars) in batch {batch_number}/{total_batches} in collection {collection_name}", "WARNING")
                                failed_chunks += 1

                        if not filtered_texts:
                            break  # All chunks in batch were too long

                        # Generate embeddings for the batch with semaphore to limit concurrent requests
                        with embedding_semaphore:
                            dense_vectors = embeddings.embed_documents(filtered_texts)

                        # Validate vector sizes
                        valid_chunks = []
                        valid_dense_vectors = []
                        for j, (chunk, vector) in enumerate(zip(filtered_chunks, dense_vectors)):
                            if len(vector) == VECTOR_SIZE:
                                valid_chunks.append(chunk)
                                valid_dense_vectors.append(vector)
                            else:
                                logger.log(f"Invalid vector size for chunk {chunk['chunk_id']} in batch {batch_number}/{total_batches} in collection {collection_name}", "WARNING")
                                failed_chunks += 1

                        if not valid_chunks:
                            break  # No valid chunks in batch

                        # Generate sparse vectors and create points
                        points = []
                        for chunk, dense_vector in zip(valid_chunks, valid_dense_vectors):
                            # Use paragraph for embeddings
                            sparse_vector = generate_sparse_vector(chunk["text"])

                            point = PointStruct(
                                id=str(uuid.uuid4()),
                                vector={
                                    "dense": dense_vector,
                                    "sparse": sparse_vector
                                },
                                payload={
                                    "text": chunk["text"],      # paragraph
                                    "doc_id": chunk["doc_id"],
                                    "chunk_id": chunk["chunk_id"],
                                    "source": chunk["source"],
                                    "text_length": len(chunk["text"])
                                }
                            )
                            points.append(point)

                        # Batch upsert points
                        client.upsert(
                            collection_name=collection_name,
                            points=points,
                            wait=True
                        )

                        successful_points += len(points)
                        logger.log(f"Successfully upserted batch {batch_number}/{total_batches} ({len(points)} points) in collection {collection_name}")
                        
                        # Track files that were successfully embedded for later marking
                        for chunk in valid_chunks:
                            successfully_embedded_files.add(chunk['source'])

                        # Add a small delay between batches to reduce server load
                        time.sleep(0.05)

                        break  # Success, exit retry loop

                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = retry_count * 5  # Exponential backoff
                            logger.log(f"Error processing batch {batch_number}/{total_batches} in collection {collection_name} (attempt {retry_count}): {e}. Retrying in {wait_time}s...", "WARNING")
                            time.sleep(wait_time)
                        else:
                            failed_chunks += len(batch)
                            logger.log(f"Failed to process batch {batch_number}/{total_batches} in collection {collection_name} after {max_retries} attempts: {e}", "ERROR")
                            break  # Exit retry loop on final failure

            logger.log(f"Completed upserting for collection {collection_name}. Successful points: {successful_points}, Failed chunks: {failed_chunks}")
            
            # Move chunks from .pending to main directory for successfully embedded files
            if successful_points > 0:
                successful_doc_ids = list(set([chunk['doc_id'] for chunk in collection_chunks if chunk['source'] in successfully_embedded_files]))
                if successful_doc_ids:
                    move_chunks_from_pending(successful_doc_ids, path=CHUNKS_DIR, collection_name=collection_name)
                    
                    # Mark files as successfully embedded (commit phase)
                    successfully_embedded_file_keys = [f for f in successfully_embedded_files]
                    mark_as_embedded(successfully_embedded_file_keys, collection_name)

def main():
    """Main ingestion process"""
    try:
        logger.log("Starting document ingestion")
        logger.log(f"Using collection: {QDRANT_COLLECTION}")
        
        # Clean up any pending chunks from failed previous runs
        logger.log("Cleaning up pending chunks from previous runs...")
        cleanup_pending_chunks()
        logger.log("Pending chunks cleanup completed")

        input_path = Path(INPUT_DIR)
        if not input_path.exists():
            logger.log(f"Input directory '{INPUT_DIR}' does not exist", "ERROR")
            return

        logger.log("Checking for new or changed documents...")
        new_docs, changed_doc_ids_by_collection = load_documents_incremental()

        if not new_docs and not changed_doc_ids_by_collection:
            logger.log("No changes detected. All documents are up to date!")
            return

        logger.log(f"Found {len(new_docs)} new/changed documents")

        # Chunk new/changed documents
        new_chunks = []
        for doc in tqdm(new_docs, desc="Chunking"):
            new_chunks.extend(chunk_document(doc))

        if new_chunks:
            logger.log(f"Saving {len(new_chunks)} new chunks to disk...")
            # Group chunks by collection for saving
            chunks_by_collection = {}
            for chunk in new_chunks:
                collection_name = chunk.get("collection_name", QDRANT_COLLECTION)  # Default to main collection if not specified
                if collection_name not in chunks_by_collection:
                    chunks_by_collection[collection_name] = []
                chunks_by_collection[collection_name].append(chunk)

            # Save chunks to their respective collection directories
            for collection_name, collection_chunks in chunks_by_collection.items():
                logger.log(f"Saving {len(collection_chunks)} chunks to collection {collection_name}...")
                save_chunks_to_disk(collection_chunks, collection_name=collection_name)

        logger.log("Loading embedding model...")
        embeddings = get_embedding_function()

        logger.log("Updating Qdrant index...")
        update_qdrant_index(new_chunks, changed_doc_ids_by_collection, embeddings)

        logger.log("Incremental indexing complete!")
    except Exception as e:
        logger.log(f"Fatal error: {e}", "ERROR")
        raise
    finally:
        # Print error/warning summary
        logger.print_summary()

if __name__ == "__main__":
    main()
