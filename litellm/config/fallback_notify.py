import os
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
from litellm.integrations.custom_logger import CustomLogger

# Throttle state: unix timestamp of the last notification sent.
# One per direction so a recent failover ping cannot swallow a recovery
# ping (and vice versa).
LAST_FAILOVER = 0.0       # failover pings (primary -> fallback)
LAST_RECOVERY = 0.0       # recovery pings (back on primary)
THROTTLE_S = 600          # at most one gotify push per 10 minutes

# Serving state: model group -> unix ts of the failover that moved the
# group onto fallback. Set on EVERY fallback event (regardless of the
# notification throttle) and cleared on the first direct-primary success.
# In-memory only — a proxy restart while on fallback loses the flag.
_fallback_active: dict = {}

_warned = False           # warn about missing GOTIFY_* env only once

logger = logging.getLogger(__name__)

# --- Phase B: quota-"until" cooldown (Option Q1) -------------------------

# Statuses whose error body may carry a quota deadline: 429 is the classic
# quota case; 402/403 appear on some providers' paywall/permission errors.
# 408 (timeout) is deliberately absent — timeout text carries timestamps,
# not quota deadlines, and would be a false-positive vector.
QUOTA_STATUSES = frozenset((402, 403, 429))

# Deadlines further out than this are treated as parse garbage — no provider
# legitimately announces a 25-hour quota window, and a multi-day cooldown
# would wedge the primary behind the fallback semi-permanently.
MAX_QUOTA_COOLDOWN_S = 24 * 3600

# Providers whose quota text carries a **tz-naive wall clock in their own local
# zone** rather than UTC. Without this, a naive stamp is read as UTC and the
# deadline lands hours late (2026-10-12: Z.AI's "reset at 19:48:48" Beijing was
# read as 19:48:48 UTC → an 11 h cooldown instead of 3 h, wedging hermes-plan).
# Unlisted/unknown providers keep the UTC default: no silent behaviour change.
_PROVIDER_UTC_OFFSET_S = {
    "zai": 8 * 3600,  # Z.AI renders the reset as Beijing local time (UTC+8)
}

# Only the deadline's phrasing is provider-specific; the offset fills in a
# tz-naive stamp only, so an explicit "Z"/"+02:00" in the text always wins.
_PROVIDER_PREFIX_RE = re.compile(r"^([a-z0-9_]+)\s*/\s*", re.IGNORECASE)

# Quota-window phrasing: "Usage limit reached for 5 hour." The number is the
# LENGTH of the quota window, not a wait — reading it through the relative
# branch would manufacture a cooldown of exactly that length.
_QUOTA_WINDOW_PHRASE_RE = re.compile(
    r"\b(?:usage\s+)?(?:limit|quota)\s+"
    r"(?:reached|exceeded|hit|used\s+up)\s+for\s+\d+(?:\.\d+)?\s*"
    r"(?:seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)

# datetime.fromisoformat on py<3.11 cannot parse trailing "Z"; normalize.
_ZULU_RE = re.compile(r"Z\s*$")

# ISO-8601-ish deadline: date, optional time (with optional seconds frac),
# optional tz. Excludes bare times (a date must be present) so "17:00" in
# prose does not match.
_ISO_RE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}"                      # date
    r"(?:[T ]\d{2}:\d{2}"                        # optional HH:MM
    r"(?::\d{2}(?:\.\d+)?)?"                     # optional :SS(.fff)
    r"(?:Z|z|[+-]\d{2}:?\d{2})?)?)"              # optional tz
)
# Relative deadline: "<n> <unit>s", e.g. "retry after 90 seconds".
_REL_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
    re.IGNORECASE,
)
_REL_UNITS_S = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
}


def provider_zone_offset_s(kwargs):
    """UTC offset (seconds) for a deployment's tz-naive quota stamps.

    Z.AI's quota text carries a bare wall clock in Beijing local time; reading it
    as UTC put the 2026-10-12 deadline 8 h late. The offset is resolved from the
    router-stamped deployment metadata so it stays scoped per provider — an
    unknown provider gets 0 (UTC), i.e. the historical default.

    Candidates, most authoritative first: the resolved ``litellm_provider``, the
    deployment's own ``model``, then its registry ``key`` — the last two carry a
    ``<provider>/<model>`` prefix. Any surprise returns 0 rather than raising:
    the caller must never crash the proxy over a metadata shape.
    """
    try:
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        model_info = metadata.get("model_info") or {}
        candidates = (
            model_info.get("litellm_provider"),
            metadata.get("deployment"),
            (kwargs.get("litellm_params") or {}).get("model"),
            model_info.get("key"),
        )
    except AttributeError:
        return 0
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate:
            continue
        name = candidate.strip().lower()
        m = _PROVIDER_PREFIX_RE.match(name)
        if m:
            name = m.group(1)  # "zai/glm-5.3" -> "zai"
        if name in _PROVIDER_UTC_OFFSET_S:
            return _PROVIDER_UTC_OFFSET_S[name]
    return 0


def format_zone_offset(offset_s):
    """'+08:00' / '+00:00' — log-friendly rendering of a zone offset."""
    sign = "+" if offset_s >= 0 else "-"
    total = abs(int(offset_s))
    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"


def parse_quota_until(text, tz_offset_s=0):
    """Extract a quota deadline (unix ts) from error text.

    Tries, in order: ISO-8601, RFC-2822 (strict-ish), relative "<n> <unit>".
    Only FUTURE deadlines count — a past datetime in the text (e.g. a stale
    request timestamp) does not shadow a later valid candidate. Returns a
    unix timestamp, or None when no future deadline is found.

    ``tz_offset_s`` is the zone to assume for a tz-*naive* stamp: 0 (UTC) is the
    default so unlisted providers keep their historical behaviour, while
    providers whose text is local time pass their real offset (Z.AI = +8 h) —
    see ``_PROVIDER_UTC_OFFSET_S``. A stamp that carries its own tz/Z always wins.
    """
    if not isinstance(text, str) or not text:
        return None
    now = time.time()

    for m in _ISO_RE.finditer(text):
        raw = m.group(1).replace(" ", "T", 1) if " " in m.group(1) else m.group(1)
        candidate = _ZULU_RE.sub("+00:00", raw)
        try:
            dt = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(seconds=tz_offset_s)))
        if dt.timestamp() > now:
            return dt.timestamp()

    # RFC-2822: "Mon, 15 Sep 2026 23:30:00 GMT" — require a weekday prefix
    # to avoid false positives on bare "15 Sep 2026" prose.
    for m in re.finditer(r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),\s[^,\n]{3,80}", text):
        try:
            dt = parsedate_to_datetime(m.group(0))
        except (TypeError, ValueError):
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(seconds=tz_offset_s)))
        if dt.timestamp() > now:
            return dt.timestamp()

    # A quota-window phrase ("Usage limit reached for 5 hour") states the window
    # LENGTH, not a wait — drop it before the relative scan so a stale/absent
    # deadline cannot be turned into a cooldown of exactly that length.
    m = _REL_RE.search(_QUOTA_WINDOW_PHRASE_RE.sub(" ", text))
    if m:
        return now + float(m.group(1)) * _REL_UNITS_S[m.group(2).lower()]

    return None


def _quota_cooldown_from_failure(kwargs, now=None):
    """Apply a router cooldown when the failure carries a quota deadline.

    Called from the failure hooks with the litellm kwargs of the FAILED
    primary attempt (fallback machinery may still rescue the request
    afterwards — this hook fires per failed attempt regardless).

    Steps:
      1. extract the exception and its deployment id,
      2. check the status is quota-ish,
      3. parse the deadline from the error text,
      4. reach the live router (litellm.proxy.proxy_server.llm_router) and
         add the primary deployment to its cooldown cache until T.

    Any surprise degrades to a logged no-op: the callback must never crash
    the proxy. Returns True when a cooldown was actually registered.
    """
    if now is None:
        now = time.time()

    exc = kwargs.get("exception")
    if exc is None:
        return False

    deployment_id = None
    try:
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}
        model_info = metadata.get("model_info") or {}
        deployment_id = model_info.get("id")
    except AttributeError:
        deployment_id = None
    if not isinstance(deployment_id, str) or not deployment_id:
        return False

    status = getattr(exc, "status_code", None)
    try:
        status_int = int(status)
    except (TypeError, ValueError):
        return False
    if status_int not in QUOTA_STATUSES:
        return False

    zone_offset_s = provider_zone_offset_s(kwargs)
    until = parse_quota_until(str(exc), tz_offset_s=zone_offset_s)
    if until is None:
        return False

    cooldown_s = until - now
    if cooldown_s <= 0:
        return False  # deadline already passed — quota may have reset
    if cooldown_s > MAX_QUOTA_COOLDOWN_S:
        logger.info(
            "[fallback_notify] quota deadline %s on deployment %s exceeds "
            "24 h — ignored (garbage guard)",
            datetime.fromtimestamp(until, tz=timezone.utc).isoformat(),
            deployment_id,
        )
        return False

    router = _get_llm_router()
    if router is None:
        logger.warning(
            "[fallback_notify] quota deadline %s seen but no live router "
            "— cooldown skipped",
            datetime.fromtimestamp(until, tz=timezone.utc).isoformat(),
        )
        return False
    cooldown_cache = getattr(router, "cooldown_cache", None)
    if cooldown_cache is None or not hasattr(
        cooldown_cache, "add_deployment_to_cooldown"
    ):
        logger.warning(
            "[fallback_notify] router has no cooldown_cache — cooldown skipped"
        )
        return False

    try:
        cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment_id,
            original_exception=exc,
            exception_status=status_int,
            cooldown_time=cooldown_s,
        )
    except Exception:
        logger.exception(
            "[fallback_notify] add_deployment_to_cooldown failed for %s",
            deployment_id,
        )
        return False

    logger.info(
        "[fallback_notify] quota deadline parsed — deployment %s cooled down "
        "for %.0f s (until %s, zone %s)",
        deployment_id,
        cooldown_s,
        datetime.fromtimestamp(until, tz=timezone.utc).isoformat(),
        format_zone_offset(zone_offset_s),
    )
    return True


def _get_llm_router():
    """Live proxy router, or None outside the proxy (tests, SDK use)."""
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:  # ImportError plus anything the import triggers
        return None
    return llm_router


def _model_group(kwargs):
    """model_group from litellm kwargs; None when absent or malformed."""
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    group = metadata.get("model_group")
    return group if isinstance(group, str) and group else None


def _check_recovery(kwargs):
    """Decide whether a success event is a back-on-primary recovery.

    Returns (endpoint, payload) when a recovery ping must be sent, else
    None. The fallback-active flag is ALWAYS consumed for a direct-primary
    success — even when throttled or the gotify env is missing — because
    the serving state advances regardless of whether we could notify.
    """
    global LAST_RECOVERY, _warned

    group = _model_group(kwargs)
    if not group or group not in _fallback_active:
        return None
    if kwargs.get("fallback_depth", 0) >= 1:
        return None  # fallback-served traffic, not a recovery

    failed_at = _fallback_active.pop(group)

    # Throttle first: silently drop repeats inside the window
    if time.time() - LAST_RECOVERY < THROTTLE_S:
        return None
    # Consume the window BEFORE sending, so failed POSTs also count
    LAST_RECOVERY = time.time()

    gotify_url = os.environ.get("GOTIFY_URL")
    gotify_token = os.environ.get("GOTIFY_TOKEN")
    if not gotify_url or not gotify_token:
        if not _warned:
            _warned = True
            logger.warning(
                "[fallback_notify] GOTIFY_URL / GOTIFY_TOKEN not set; "
                "recovery notifications disabled"
            )
        return None

    minutes = max(0, int((time.time() - failed_at) // 60))
    msg = f"✅ '{group}' back on primary ({minutes}m on fallback)"
    payload = {
        "title": "LiteLLM recovery",
        "message": msg,
        "priority": 8,
    }
    return f"{gotify_url}/message?token={gotify_token}", payload


class FallbackNotifier(CustomLogger):
    """
    Push a gotify notification whenever ANY proxy-level fallback succeeds
    (e.g. plan-token exhaustion causing a failover to another deployment),
    and once more when the model group is back on its primary.

    Deliberately does NO model filtering — it fires on every fallback.

    Notifications are throttled to one per THROTTLE_S seconds PER
    DIRECTION (LAST_FAILOVER / LAST_RECOVERY); the window is consumed
    even if the POST fails, so a flaky gotify server cannot cause a
    notification storm.

    Additionally, on quota errors whose body carries a deadline
    ("... until 2026-09-15T23:30:00Z ..."), the failed primary
    deployment is put on router cooldown until that deadline (Option Q1)
    — the router then serves from the fallback without hammering the
    quota-dead primary, and the recovery ping above fires once the
    primary succeeds again after T. Deadlines >24 h are ignored.
    """

    async def log_success_fallback_event(
        self,
        original_model_group: str,
        kwargs: dict,
        original_exception: Exception,
    ) -> None:
        global LAST_FAILOVER, _warned

        # Serving state first: flag EVERY fallback event, even when the
        # notification below is throttled — this tracks serving state,
        # not notification state.
        _fallback_active[original_model_group] = time.time()

        # Throttle first: silently drop repeats inside the window
        if time.time() - LAST_FAILOVER < THROTTLE_S:
            return
        # Consume the window BEFORE sending, so failed POSTs also count
        LAST_FAILOVER = time.time()

        gotify_url = os.environ.get("GOTIFY_URL")
        gotify_token = os.environ.get("GOTIFY_TOKEN")
        if not gotify_url or not gotify_token:
            if not _warned:
                _warned = True
                logger.warning(
                    "[fallback_notify] GOTIFY_URL / GOTIFY_TOKEN not set; "
                    "fallback notifications disabled"
                )
            return

        msg = (
            f"⚠️ '{original_model_group}' failed "
            f"({type(original_exception).__name__}) — now serving on "
            f"'{kwargs.get('model')}'. Wrap up, no new heavy tasks."
        )

        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{gotify_url}/message?token={gotify_token}",
                    json={
                        "title": "LiteLLM fallback",
                        "message": msg,
                        "priority": 8,
                    },
                    timeout=10.0,
                )
        except Exception:
            logger.exception("[fallback_notify] failed to send gotify notification")

    # --- quota-"until" cooldown (Phase B, Option Q1) --------------------

    def _quota_cooldown(self, kwargs: dict) -> None:
        """Shared quota-cooldown step for the sync/async failure hooks."""
        try:
            _quota_cooldown_from_failure(kwargs)
        except Exception:
            # The callback must never take the proxy down with it.
            logger.exception("[fallback_notify] quota cooldown hook failed")

    def log_failure_event(
        self,
        kwargs: dict,
        response_obj,
        start_time,
        end_time,
    ) -> None:
        """Sync failure hook: observes quota errors on the primary."""
        self._quota_cooldown(kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj,
        start_time,
        end_time,
    ) -> None:
        """Async failure hook — the path the proxy actually invokes.

        Fires per FAILED attempt, before any fallback rescue, with
        kwargs["exception"] carrying the provider error and
        litellm_params.metadata.model_info.id naming the failed
        deployment (shape dumped live against litellm 1.93.0).
        """
        self._quota_cooldown(kwargs)

    def log_success_event(
        self,
        kwargs: dict,
        response_obj,
        start_time,
        end_time,
    ) -> None:
        """Sync success hook — fires on sync SDK calls (log_success_event)."""
        out = _check_recovery(kwargs)
        if out is None:
            return
        endpoint, payload = out
        try:
            with httpx.Client() as client:
                client.post(endpoint, json=payload, timeout=10.0)
        except Exception:
            logger.exception("[fallback_notify] failed to send gotify notification")

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj,
        start_time,
        end_time,
    ) -> None:
        """Async success hook — the path the proxy actually invokes."""
        out = _check_recovery(kwargs)
        if out is None:
            return
        endpoint, payload = out
        try:
            async with httpx.AsyncClient() as client:
                await client.post(endpoint, json=payload, timeout=10.0)
        except Exception:
            logger.exception("[fallback_notify] failed to send gotify notification")


# Module-level instance — this is what config.yaml references
fallback_notifier = FallbackNotifier()
