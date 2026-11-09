from typing import Literal, Optional

from litellm.integrations.custom_logger import CustomLogger

# NOTE: unlike strip_hermes.py we do not import UserAPIKeyAuth / DualCache
# from litellm.proxy.proxy_server — the hook only needs the parameter NAMES
# (litellm calls it with keywords), and keeping imports minimal makes the
# module unit-testable with a stubbed CustomLogger base class.


class StripThinkingHandler(CustomLogger):
    """
    Pop the OpenAI-style `thinking` parameter from request data before it
    reaches the zai provider transform.

    Why this exists (2026-09-05): opencode 1.17.9 unconditionally injects
    `thinking: {"type": "enabled", "clear_thinking": false}` into every
    request made through a provider whose id contains "zai"/"zhipuai"
    (@ai-sdk/openai-compatible path; client `reasoning: false` does NOT
    suppress it). litellm v1.93.0's zai transform forwards `thinking` as a
    top-level SDK kwarg whenever the DEPLOYMENT model name (e.g.
    zai/glm-5.3-flash) resolves in litellm.model_cost — get_supported_openai_params
    appends `thinking` iff supports_reasoning(deployment_name) — and the zai
    SDK then rejects it: `ZaiException: unexpected keyword argument
    'thinking'` -> HTTP 500 for the caller. `drop_params` cannot protect
    these groups because the param IS in the supported set.
    (refs litellm #31084 merged, #31085 open)

    Reasoning behavior is preserved: zai models reason natively, and the
    hermes-* virtuals pin `reasoning_effort` via extra_body in config.yml.

    Scope: ONLY the zai-backed model groups enumerated in ZAI_GROUPS — keep
    in sync with config.yml deployments whose litellm_params.model starts
    with `zai/` (test_strip_thinking.py enforces this). Everything else —
    deepseek-backed hermes-vision / hermes-fallback, openrouter-z-ai/* and
    openrouter-zai-* (openrouter transform), bare deepseek/groq/etc. — is
    untouched. Exact membership match, deliberately NOT substring matching:
    "openrouter-zai-glm-4.5-air-free" contains "zai-glm" and must NOT hit.
    """

    # zai-backed model groups (client-facing names) — keep in sync with
    # litellm/config/config.yml (`model: zai/...` deployments).
    ZAI_GROUPS = frozenset(
        {
            "zai-glm-5.3",
            "zai-glm-5.3-flash",
            "hermes-coding-plan",
            "hermes-coding-execute",
            "hermes-heartbeat-plan",
            "hermes-heartbeat-execute",
            "hermes-heartbeat-tasks",
        }
    )

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
        model = data.get("model")
        if not isinstance(model, str) or model not in self.ZAI_GROUPS:
            return data
        if "thinking" in data:
            data.pop("thinking", None)
            print("[strip_thinking] dropped thinking kwarg for", model)
        return data  # MUST return data, even if unchanged


# Module-level instance — this is what config.yaml references
strip_thinking_handler = StripThinkingHandler()
