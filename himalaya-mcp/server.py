"""
Himalaya MCP Server — Streamable HTTP Transport (Draft-Only)
=============================================================
Read + draft email via the himalaya v2.0.0 CLI. NO send capability.
SMTP deliberately absent from config — structural enforcement.
"""

import asyncio
import base64
import json
import os
import sys
import tempfile
import email.message
import email.utils
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from attachments import ALLOWED_EXTS, classify_attachment, looks_like_markup

HIMALAYA_BIN = os.getenv("HIMALAYA_BIN", "himalaya")
HIMALAYA_CONFIG_DIR = os.getenv("HIMALAYA_CONFIG_DIR", "/config")
# Resolve the actual config FILE. Allow a direct override via
# HIMALAYA_CONFIG_FILE; otherwise expect config.toml inside the dir.
# himalaya's --config flag takes a FILE, not a directory — passing the
# directory causes "cannot read config file ... Is a directory".
HIMALAYA_CONFIG_FILE = os.getenv("HIMALAYA_CONFIG_FILE") or os.path.join(
    HIMALAYA_CONFIG_DIR, "config.toml"
)
# When set, forces a specific account for all tools that don't receive an
# explicit `account` argument. When empty (default), no `-a` flag is passed
# and himalaya resolves the account itself (the one with `default = true`).
DEFAULT_ACCOUNT = os.getenv("DEFAULT_ACCOUNT", "")
DRAFTS_FOLDER = os.getenv("DRAFTS_FOLDER", "Drafts")
TRASH_FOLDER = os.getenv("TRASH_FOLDER", "Trash")
ATTACHMENT_DL_TIMEOUT = int(os.getenv("ATTACHMENT_DL_TIMEOUT", "120"))
MAX_EXPORT_BYTES = int(os.getenv("ATTACHMENT_MAX_BYTES", str(20 * 1024 * 1024)))
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "9201"))

# Allowed Host header values for FastMCP's DNS-rebinding protection
# (Starlette TrustedHostMiddleware-equivalent). "*" disables protection and
# allows any host; otherwise a comma-separated list of allowed host values.
# Each entry may use a ":*" suffix as a port wildcard, e.g. "192.168.0.209:*".
_allowed_hosts_raw = os.getenv("ALLOWED_HOSTS", "*").strip()
if _allowed_hosts_raw == "*":
    _transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
else:
    _allowed_hosts = [h.strip() for h in _allowed_hosts_raw.split(",") if h.strip()]
    _transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=_allowed_hosts)


def _acc(account: Optional[str]) -> list:
    """Build the -a flag args for himalaya.

    Priority: explicit caller arg > DEFAULT_ACCOUNT env > omit (let
    himalaya pick the account marked `default = true` in config.toml).

    Passing a wrong account name (e.g. hardcoded "main" when the config
    defines "assistant") causes: 'cannot find configuration for account'.
    Omitting -a entirely lets himalaya resolve it correctly.
    """
    acc = account or DEFAULT_ACCOUNT
    return ["-a", acc] if acc else []


def _validate_config_path() -> None:
    """Fail fast on the classic misconfiguration: resolved path is a directory.
    A merely missing file is only a warning, so health_check and the MCP
    handshake still work before live IMAP creds are mounted."""
    if os.path.isdir(HIMALAYA_CONFIG_FILE):
        print(
            "FATAL: HIMALAYA_CONFIG_FILE "
            f"'{HIMALAYA_CONFIG_FILE}' is a directory, but himalaya "
            "--config expects a FILE. Point HIMALAYA_CONFIG_DIR at the dir "
            "containing config.toml, or set HIMALAYA_CONFIG_FILE to the "
            ".toml file directly.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not os.path.isfile(HIMALAYA_CONFIG_FILE):
        print(
            f"WARNING: config file not found at '{HIMALAYA_CONFIG_FILE}' "
            f"(dir '{HIMALAYA_CONFIG_DIR}'). IMAP tools will fail until a "
            "valid config.toml is mounted.",
            file=sys.stderr,
        )


_validate_config_path()

mcp = FastMCP("himalaya", json_response=True, transport_security=_transport_security)


async def _run(*args, stdin_data: Optional[str] = None, as_json: bool = True, timeout: float = 30) -> dict:
    """Run himalaya once; return parsed JSON (as_json=True) or raw stdout.

    v2 uses --json (not --output json) and has no --quiet. Global flags go
    before subcommand args. In --json mode v2 prints errors as a JSON object
    on STDOUT with rc=1 (stderr stays empty).
    """
    cmd = [HIMALAYA_BIN, "--config", HIMALAYA_CONFIG_FILE]
    if as_json:
        cmd.append("--json")
    cmd.extend(args)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(
                stdin_data.encode("utf-8") if stdin_data is not None else None),
            timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"error": f"himalaya command timed out after {timeout:g} seconds", "returncode": -1}
    if proc.returncode != 0:
        if as_json:
            try:
                parsed = json.loads(stdout.decode())
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and "error" in parsed:
                return {**parsed, "returncode": proc.returncode}
        return {"error": (stderr or stdout).decode().strip(), "returncode": proc.returncode}
    if not as_json:
        return {"raw": stdout.decode(errors="replace")}
    try:
        result = json.loads(stdout.decode())
    except json.JSONDecodeError:
        return {"raw": stdout.decode().strip()}
    # Backwards-compat unwrap for v1 bare-array responses: a dict with exactly
    # one key whose value is a list (e.g. {"mailboxes": [...]}) becomes the list.
    if isinstance(result, dict) and len(result) == 1:
        value = next(iter(result.values()))
        if isinstance(value, list):
            return value
    return result


async def _himalaya(*args, stdin_data: Optional[str] = None, timeout: float = 30) -> dict:
    """Run himalaya with --json and unwrap single-key list wrappers."""
    return await _run(*args, stdin_data=stdin_data, as_json=True, timeout=timeout)


async def _himalaya_raw(*args) -> dict:
    """Run himalaya WITHOUT --json, returning {"raw": stdout}.
    Used only by message_export (`message read --raw`). --raw and --json are
    mutually exclusive in v2.
    """
    return await _run(*args, as_json=False)


@mcp.tool()
async def folder_list(account: Optional[str] = None) -> str:
    """List all mailboxes/folders for the account."""
    result = await _himalaya("mailbox", "list", *_acc(account))
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def envelope_list(
    folder: str = "INBOX",
    query: str = "",
    page: int = 1,
    page_size: int = 20,
    account: Optional[str] = None,
) -> str:
    """List message envelopes with optional filter query.
    Query examples: 'not flag seen', 'from sender@example.com',
    'subject invoice', 'before 2026-01-01'.
    When a query is given this routes to himalaya `envelope search` (v2
    `envelope list` has no query positional)."""
    common = ["-m", folder, "-p", str(page), "-s", str(page_size)]
    if query:
        args = ["envelope", "search", *_acc(account), *common, "--", *query.split()]
    else:
        args = ["envelope", "list", *_acc(account), *common]
    result = await _himalaya(*args)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def message_read(
    id: str,
    peek: bool = True,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Read message body. Defaults to peek=true (does NOT mark as read —
    v2 read uses IMAP BODY.PEEK internally and never sets \\Seen).
    Set peek=false to add \\Seen afterwards."""
    result = await _himalaya("message", "read", *_acc(account), "-m", folder, "--", id)
    if "error" in result:
        return json.dumps(result, ensure_ascii=False, indent=2)
    if not peek:
        flagged = await _himalaya(
            "flag", "add", *_acc(account), "-m", folder, "-f", "seen", "--", id)
        if "error" in flagged:
            result["warn"] = f"flag add seen failed: {flagged['error']}"
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def message_export(
    id: str,
    peek: bool = True,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Export raw MIME (for ICS/calendar parsing). Defaults to peek=true.
    Uses `message read --raw` (RFC 5322 MIME on stdout). v2 read uses IMAP
    BODY.PEEK internally, so this is peek-safe by design — it never sets
    \\Seen and needs no race-condition workaround.
    Set peek=false to add \\Seen afterwards."""
    result = await _himalaya_raw(
        "message", "read", *_acc(account), "-m", folder, "--raw", "--", id)
    if "error" in result:
        output = result
    else:
        raw = result["raw"]
        output = {"id": id, "folder": folder, "raw_mime": raw, "mime_length": len(raw)}
        if not peek:
            flagged = await _himalaya(
                "flag", "add", *_acc(account), "-m", folder, "-f", "seen", "--", id)
            if "error" in flagged:
                output["warn"] = f"flag add seen failed: {flagged['error']}"
    return json.dumps(output, ensure_ascii=False, indent=2)


@mcp.tool()
async def attachment_list(
    id: str,
    folder: str = "INBOX",
    include_inline: bool = False,
    account: Optional[str] = None,
) -> str:
    """List attachments of a message with export-policy verdicts.

    Peek-safe: attachment list never sets \\Seen (verified on himalaya v2.0.0).
    Policy: only whitelist extensions (PDF, Office/ODF docs, pictures) are
    exportable; ICS/calendar is never exported as a file (see message_export).
    """
    args = ["attachment", "list", *_acc(account), "-m", folder]
    if include_inline:
        args.append("-i")
    args.extend(["--", id])
    result = await _himalaya(*args)
    if "error" in result:
        return json.dumps(result, ensure_ascii=False, indent=2)
    entries = result if isinstance(result, list) else []
    out = []
    for e in entries:
        exportable, reason = classify_attachment(
            e.get("filename", ""), e.get("mime", ""), e.get("size", 0), MAX_EXPORT_BYTES)
        item = dict(e)
        item["exportable"] = exportable
        item["block_reason"] = reason
        out.append(item)
    return json.dumps({"id": id, "folder": folder, "attachments": out}, ensure_ascii=False, indent=2)


@mcp.tool()
async def attachment_export(
    id: str,
    attachment_id: int,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Export one attachment as base64 JSON (size-capped, never truncated).

    Peek-safe: attachment download never sets \\Seen (verified on himalaya
    v2.0.0).
    """
    args = ["attachment", "list", *_acc(account), "-m", folder, "-i", "--", id]
    listing = await _himalaya(*args)
    if "error" in listing:
        return json.dumps(listing, ensure_ascii=False, indent=2)
    entries = listing if isinstance(listing, list) else []
    if attachment_id < 1 or attachment_id > len(entries):
        return json.dumps(
            {"error": f"no attachment with id {attachment_id} on message {id}", "stays_on_server": True},
            ensure_ascii=False, indent=2)
    entry = entries[attachment_id - 1]
    ok, reason = classify_attachment(
        entry.get("filename", ""), entry.get("mime", ""), entry.get("size", 0), MAX_EXPORT_BYTES)
    if not ok:
        return json.dumps(
            {"error": f"attachment not exported: {reason}", "attachment": entry, "stays_on_server": True},
            ensure_ascii=False, indent=2)
    with tempfile.TemporaryDirectory() as tmp:
        dl = await _himalaya(
            "attachment", "download", *_acc(account), "-m", folder,
            "-d", tmp, "--", id, str(attachment_id), timeout=ATTACHMENT_DL_TIMEOUT)
        if "error" in dl:
            return json.dumps(dl, ensure_ascii=False, indent=2)
        path = os.path.join(tmp, os.path.basename(entry["filename"]))
        if not os.path.isfile(path):
            return json.dumps({"error": "downloaded file not found", "attachment": entry}, ensure_ascii=False, indent=2)
        size = os.path.getsize(path)
        if size != entry.get("size"):
            return json.dumps(
                {
                    "error": f"size mismatch: listed {entry.get('size')} vs on-disk {size}",
                    "attachment": entry,
                    "stays_on_server": True,
                },
                ensure_ascii=False, indent=2)
        if size > MAX_EXPORT_BYTES:
            return json.dumps(
                {
                    "error": f"attachment not exported: size {size} exceeds cap {MAX_EXPORT_BYTES} bytes",
                    "stays_on_server": True,
                },
                ensure_ascii=False, indent=2)
        with open(path, "rb") as f:
            data = f.read()
    if looks_like_markup(data):
        return json.dumps(
            {
                "error": "attachment not exported: payload looks like markup (SVG/XML/HTML masquerade)",
                "attachment": entry,
                "stays_on_server": True,
            },
            ensure_ascii=False, indent=2)
    return json.dumps(
        {
            "id": id,
            "folder": folder,
            "attachment_id": attachment_id,
            "filename": entry.get("filename"),
            "mime": entry.get("mime"),
            "size": size,
            "encoding": "base64",
            "content_b64": base64.b64encode(data).decode("ascii"),
        },
        ensure_ascii=False, indent=2)


@mcp.tool()
async def template_write(account: Optional[str] = None) -> str:
    """Generate a blank MIME (RFC 5322) skeleton for composing."""
    result = await _himalaya("message", "compose", *_acc(account))
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def template_reply(
    id: str,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Generate a reply MIME skeleton pre-filled from the given message."""
    result = await _himalaya("message", "reply", *_acc(account), "-m", folder, "--", id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def template_forward(
    id: str,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Generate a forward MIME skeleton pre-filled from the given message."""
    result = await _himalaya("message", "forward", *_acc(account), "-m", folder, "--", id)
    return json.dumps(result, ensure_ascii=False, indent=2)


def _mml_to_mime(mml: str) -> bytes:
    """Compile an MML (Markdown Mail) string into raw MIME bytes.

    MML format: optional "Key: Value" header lines, then ONE blank line,
    then the body. The body is treated as plain text (utf-8) — no markdown
    detection or conversion is performed.
    """
    if "\n\n" in mml:
        headers_block, body = mml.split("\n\n", 1)
    else:
        headers_block, body = "", mml

    msg = email.message.EmailMessage()
    for line in headers_block.splitlines():
        line = line.strip()
        if not line:
            continue
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip()
            value = value.strip()
            if key:
                msg[key] = value

    msg.set_content(body)

    if "Message-ID" not in msg:
        msg["Message-ID"] = email.utils.make_msgid()
    if "Date" not in msg:
        msg["Date"] = email.utils.formatdate(localtime=True)

    return bytes(msg)


@mcp.tool()
async def template_save(
    mml: str,
    folder: Optional[str] = None,
    account: Optional[str] = None,
) -> str:
    """Save a message draft via IMAP APPEND. Does NOT send via SMTP.
    Input is the edited MIME from template_write/reply/forward; simple
    "Key: Value\n\n body" text is still compiled to MIME in Python.

    Implementation note: v2 `message add` takes raw MIME via STDIN, so we
    pipe it in (avoids argv size limits and `--` ordering issues). Input that
    already looks like MIME ("MIME-Version:" or "Content-Type:" present)
    passes through unchanged; other input is compiled with _mml_to_mime().
    Note: this does NOT send — `message add` is a pure IMAP APPEND. The v2
    --send flag exists but we never pass it (SMTP is not configured).
    """
    fld = folder or DRAFTS_FOLDER
    if "MIME-Version:" in mml or "Content-Type:" in mml:
        mime = mml
    else:
        mime = _mml_to_mime(mml).decode()
    result = await _himalaya("message", "add", *_acc(account), "-m", fld, stdin_data=mime)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def draft_delete(
    id: str,
    account: Optional[str] = None,
) -> str:
    """Delete a draft from the Drafts folder by moving it to the Trash folder
    (recoverable). Drafts folder only — cannot delete messages from other
    folders. Uses himalaya `message move` (v2 has no `message delete`)."""
    result = await _himalaya(
        "message", "move", *_acc(account), "-f", DRAFTS_FOLDER, "-t", TRASH_FOLDER, "--", id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def flag_set(
    id: str,
    flag: str,
    add: bool = True,
    folder: str = "INBOX",
    account: Optional[str] = None,
) -> str:
    """Add or remove a flag on a message.
    Common flags: 'seen' (\\Seen), 'flagged' (\\Flagged), 'answered' (\\Answered).
    Use add=false to remove the flag."""
    action = "add" if add else "remove"
    result = await _himalaya("flag", action, *_acc(account), "-m", folder, "-f", flag, "--", id)
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
async def health_check() -> str:
    """Check server health and confirm no send capability exists."""
    status = {
        "status": "ok",
        "server": "himalaya-mcp",
        "mode": "draft-only (no send capability)",
        "config": {
            "HIMALAYA_BIN": HIMALAYA_BIN,
            "HIMALAYA_CONFIG_DIR": HIMALAYA_CONFIG_DIR,
            "HIMALAYA_CONFIG_FILE": HIMALAYA_CONFIG_FILE,
            "DEFAULT_ACCOUNT": DEFAULT_ACCOUNT or "(himalaya default)",
            "DRAFTS_FOLDER": DRAFTS_FOLDER,
            "TRASH_FOLDER": TRASH_FOLDER,
            "ATTACHMENT_MAX_BYTES": MAX_EXPORT_BYTES,
            "ATTACHMENT_DL_TIMEOUT": ATTACHMENT_DL_TIMEOUT,
        },
        "tools_registered": [
            "folder_list", "envelope_list", "message_read",
            "message_export", "attachment_list", "attachment_export",
            "template_write", "template_reply", "template_forward",
            "template_save", "draft_delete", "flag_set", "health_check",
        ],
        "send_capability": False,
        "attachment_whitelist_exts": sorted(ALLOWED_EXTS),
    }
    return json.dumps(status, indent=2)


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=MCP_HOST, port=MCP_PORT)
