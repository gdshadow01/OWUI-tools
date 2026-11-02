"""Stable per-lane Codex/ChatGPT session identities for the LiteLLM proxy.

Why this exists (2026-10-26): the ChatGPT/Codex backend partitions its prompt
cache by SESSION IDENTITY (outbound `session_id` header), not by prompt
content. LiteLLM (verified on the pinned v1.103.2, production image) stamps a
fresh uuid4 into that header for every request unless a caller-supplied id is
present, so identical prompts never reuse the backend cache (~10% hit rate
measured vs ~91% on the zai twins, which cache by prefix bytes). Resolution
order verified in-container on v1.103.2:
litellm_params.litellm_session_id -> litellm_params.session_id ->
litellm_params.metadata.session_id -> litellm_call_id/trace_id -> uuid4.
The proxy always builds litellm_params.metadata from the request `metadata`
dict, so setting data["metadata"]["session_id"] in async_pre_call_hook
determines the outbound header (end-to-end verified: header value ==
injected value on the /chat/completions -> Responses bridge).

Sharding is REQUIRED: concurrent requests sharing one session id serialize
server-side (upstream BerriAI/litellm#42012: 6 concurrent on one id = 61s
staircase vs 16s with distinct ids; OpenAI advises <=~15 req/min per key).

ID format: hermes-<model_group>-<shard>-<hash8>
  - model_group: client-facing group name (hook runs PRE-routing, so the
    deployment backing model is not yet chosen — activation must key off the
    group name, see CHATGPT_GROUPS)
  - shard: int(digest, 16) % CODEX_SESSION_SHARDS (default 4) — spreads
    concurrent traffic over N backend lanes
  - hash8: first 8 hex chars of sha256(group + first message content) —
    content-addressed: identical startup context -> identical id (cache
    reuse), changed system prompt -> new id automatically (no stale lane)

Scope: ONLY the groups in CHATGPT_GROUPS (= config.yml deployments whose
litellm_params.model starts with `chatgpt/`; test_codex_session_id.py
enforces the sync). Everything else passes through untouched. Fail-open:
any error returns data unchanged (= today's uuid4 behavior). Pure in-memory
hashing — no I/O, no prompt-content logging (spend logs already store
prompts; this module never prints message content, only the derived id).
"""

import hashlib
import os
from typing import Any, Literal, Optional

from litellm.integrations.custom_logger import CustomLogger

# chatgpt/-backed model groups (client-facing names) — keep in sync with
# litellm/config/config.yml deployments whose litellm_params.model starts
# with `chatgpt/` (test_codex_session_id.py enforces this).
CHATGPT_GROUPS = frozenset(
    {
        "chatgpt-gpt-6.1-sol",
        "chatgpt-gpt-6-luna",
        "chatgpt-gpt-6-astra",
    }
)

# Number of backend cache lanes per group. Concurrent requests on ONE id
# serialize server-side, so lanes spread load; identical content always maps
# to the same lane. Tunable at runtime via env (compose env_file litellm/.env)
# without a code change.
ENV_SHARD_COUNT = "CODEX_SESSION_SHARDS"
DEFAULT_SHARD_COUNT = 4
_HASH_CHARS = 8


def _safe_header_value(value: str) -> str:
    """Mirror LiteLLM's header sanitation: printable ASCII only."""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "_" for ch in value)
    return cleaned.strip() or "hermes-session"


def _first_message_text(messages: Any) -> str:
    """Best-effort stable text of the first message (system prompt lane).

    Handles content as str, None, or a list of parts (multimodal); degrades
    to "" on anything unexpected — never raises (fail-open is the caller's
    guarantee anyway, but hashing should not depend on it).
    """
    if not isinstance(messages, (list, tuple)) or not messages:
        return ""
    first = messages[0]
    if not isinstance(first, dict):
        return str(first)
    content = first.get("content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part.get("type", ""))))
            else:
                parts.append(str(part))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _shard_count() -> int:
    raw = os.environ.get(ENV_SHARD_COUNT, "")
    try:
        n = int(raw) if raw else DEFAULT_SHARD_COUNT
    except ValueError:
        return DEFAULT_SHARD_COUNT
    return n if n > 0 else DEFAULT_SHARD_COUNT


def compute_session_id(model_group: str, messages: Any) -> str:
    """Deterministic per-lane session id for a chatgpt-backed group."""
    digest = hashlib.sha256(
        f"{model_group}\x00{_first_message_text(messages)}".encode("utf-8", "replace")
    ).hexdigest()
    shard = int(digest, 16) % _shard_count()
    return _safe_header_value(f"hermes-{model_group}-{shard}-{digest[:_HASH_CHARS]}")


class CodexSessionIdHook(CustomLogger):
    """async_pre_call_hook: inject a stable sharded session id for chatgpt/
    backed groups so the Codex backend reuses its prompt cache across
    dispatches. All other groups: untouched. Any error: data unchanged."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[dict]:
        try:
            model = data.get("model")
            if not isinstance(model, str) or model not in CHATGPT_GROUPS:
                return data  # not a chatgpt-backed group: pass through untouched
            session_id = compute_session_id(model, data.get("messages"))
            if not isinstance(data.get("metadata"), dict):
                data["metadata"] = {}
            data["metadata"]["session_id"] = session_id
            # Defensive: some litellm versions carry a parallel litellm_metadata
            # dict; keep both in sync when present.
            litellm_metadata = data.get("litellm_metadata")
            if isinstance(litellm_metadata, dict):
                litellm_metadata["session_id"] = session_id
            print(f"[codex_session_id] {model} -> {session_id}")
        except Exception:
            pass  # fail open: request proceeds with litellm's default uuid4
        return data  # MUST return data, even if unchanged


# Module-level instance — this is what config.yml references
# (litellm_settings.callbacks: [..., "codex_session_id.codex_session_id_hook"])
codex_session_id_hook = CodexSessionIdHook()
