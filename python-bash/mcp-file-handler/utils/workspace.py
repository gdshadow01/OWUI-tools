import os
import shutil
from pathlib import Path
from typing import Optional

WORKSPACE_ROOT = os.environ.get("WORKSPACE_DIR", "/workspace")
UPLOADS_DIR = os.path.join(WORKSPACE_ROOT, "uploads")
OUTPUTS_DIR = os.path.join(WORKSPACE_ROOT, "outputs")


def ensure_directories():
    """Ensure workspace subdirectories exist."""
    for directory in [UPLOADS_DIR, OUTPUTS_DIR]:
        os.makedirs(directory, exist_ok=True)


def save_to_workspace(file_data: bytes, filename: str, subdir: str = "uploads") -> dict:
    """
    Save file data to workspace.

    Args:
        file_data: Raw file content as bytes
        filename: Name of the file
        subdir: Subdirectory ("uploads" or "outputs")

    Returns:
        dict: {"success": True, "path": "..."} or {"error": {...}}
    """
    try:
        ensure_directories()

        target_dir = UPLOADS_DIR if subdir == "uploads" else OUTPUTS_DIR
        safe_filename = Path(filename).name
        file_path = os.path.join(target_dir, safe_filename)

        with open(file_path, 'wb') as f:
            f.write(file_data)

        return {
            "success": True,
            "path": file_path,
            "filename": safe_filename,
            "size": len(file_data)
        }
    except Exception as e:
        return {"error": {"message": f"Failed to save file: {str(e)}"}}


def read_from_workspace(filename: str, subdir: str = "uploads") -> dict:
    """
    Read file content from workspace.

    Args:
        filename: Name of the file
        subdir: Subdirectory ("uploads" or "outputs")

    Returns:
        dict: {"content": "...", "size": ...} or {"error": {...}}
    """
    try:
        target_dir = UPLOADS_DIR if subdir == "uploads" else OUTPUTS_DIR
        file_path = os.path.join(target_dir, filename)

        if not os.path.exists(file_path):
            return {"error": {"message": f"File not found: {filename}"}}

        with open(file_path, 'rb') as f:
            content = f.read()

        return {
            "content": content,
            "path": file_path,
            "size": len(content)
        }
    except Exception as e:
        return {"error": {"message": f"Failed to read file: {str(e)}"}}


def list_workspace_files(subdir: str = None) -> dict:
    """
    List files in workspace directory.

    Args:
        subdir: Optional subdirectory ("uploads", "outputs", or None for all)

    Returns:
        dict: {"files": [...], "uploads": [...], "outputs": [...]}
    """
    try:
        result = {"files": [], "uploads": [], "outputs": []}

        if subdir:
            directories = [os.path.join(WORKSPACE_ROOT, subdir)]
        else:
            directories = [UPLOADS_DIR, OUTPUTS_DIR]

        for directory in directories:
            if not os.path.exists(directory):
                continue

            dir_name = os.path.basename(directory)
            for root, _, filenames in os.walk(directory):
                for name in filenames:
                    rel_path = os.path.relpath(os.path.join(root, name), WORKSPACE_ROOT)
                    size = os.path.getsize(os.path.join(root, name))
                    file_info = {
                        "path": rel_path,
                        "filename": name,
                        "size": size
                    }
                    result["files"].append(file_info)
                    result[dir_name].append(file_info)

        return result
    except Exception as e:
        return {"error": {"message": f"Failed to list files: {str(e)}"}}


def delete_workspace_file(filename: str, subdir: str = "uploads") -> dict:
    """
    Delete a file from workspace.

    Args:
        filename: Name of the file
        subdir: Subdirectory ("uploads" or "outputs")

    Returns:
        dict: {"success": True} or {"error": {...}}
    """
    try:
        target_dir = UPLOADS_DIR if subdir == "uploads" else OUTPUTS_DIR
        file_path = os.path.join(target_dir, filename)

        if not os.path.exists(file_path):
            return {"error": {"message": f"File not found: {filename}"}}

        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        return {"error": {"message": f"Failed to delete file: {str(e)}"}}


def move_to_outputs(filename: str) -> dict:
    """
    Move a file from uploads to outputs (for export).

    Args:
        filename: Name of the file in uploads directory

    Returns:
        dict: {"success": True, "new_path": "..."} or {"error": {...}}
    """
    try:
        ensure_directories()
        source_path = os.path.join(UPLOADS_DIR, filename)
        dest_path = os.path.join(OUTPUTS_DIR, filename)

        if not os.path.exists(source_path):
            return {"error": {"message": f"File not found in uploads: {filename}"}}

        shutil.move(source_path, dest_path)
        return {
            "success": True,
            "new_path": dest_path,
            "message": f"Moved {filename} to outputs"
        }
    except Exception as e:
        return {"error": {"message": f"Failed to move file: {str(e)}"}}
