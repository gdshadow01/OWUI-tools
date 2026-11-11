"""Server-side attachment policy for the himalaya MCP server.

Whitelist semantics identical to the mail-checker wrapper: anything not
explicitly allowed stays on the mail server.
"""

from typing import Optional

ALLOWED_EXTS: frozenset = frozenset({
    'pdf',
    'doc', 'docx', 'docm', 'rtf', 'odt',
    'xls', 'xlsx', 'xlsm', 'ods',
    'ppt', 'pptx', 'pptm', 'odp',
    'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tif', 'tiff', 'heic', 'avif',
})

DENIED_MIME: frozenset = frozenset({
    'application/x-msdownload', 'application/x-msi', 'application/x-dos-bat',
    'application/x-sh', 'application/x-shellscript',
    'text/javascript', 'application/javascript',
    'text/ecmascript', 'application/ecmascript',
    'application/x-executable', 'application/x-elf', 'application/x-sharedlib',
    'application/java-archive',
    'application/zip', 'application/x-zip-compressed',
    'application/x-tar', 'application/gzip', 'application/x-gzip',
    'application/x-7z-compressed', 'application/vnd.rar', 'application/x-rar-compressed',
    'image/svg+xml', 'image/svg', 'text/html',
})

_EXT_GROUPS: dict = {
    'pdf': 'pdf',
    'doc': 'word', 'docx': 'word', 'docm': 'word', 'rtf': 'word',
    'xls': 'excel', 'xlsx': 'excel', 'xlsm': 'excel',
    'ppt': 'ppt', 'pptx': 'ppt', 'pptm': 'ppt',
    'odt': 'odf-word', 'ods': 'odf-sheet', 'odp': 'odf-slides',
    'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'gif': 'image',
    'webp': 'image', 'bmp': 'image', 'tif': 'image', 'tiff': 'image',
    'heic': 'image', 'avif': 'image',
}

_MIME_GROUPS: dict = {
    'application/pdf': 'pdf',
    'application/msword': 'word',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'word',
    'application/vnd.ms-word.document.macroEnabled.12': 'word',
    'application/rtf': 'word', 'text/rtf': 'word',
    'application/vnd.ms-excel': 'excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
    'application/vnd.ms-excel.sheet.macroEnabled.12': 'excel',
    'application/vnd.ms-powerpoint': 'ppt',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'ppt',
    'application/vnd.ms-powerpoint.presentation.macroEnabled.12': 'ppt',
    'application/vnd.oasis.opendocument.text': 'odf-word',
    'application/vnd.oasis.opendocument.spreadsheet': 'odf-sheet',
    'application/vnd.oasis.opendocument.presentation': 'odf-slides',
}


def _mime_group(mime: str) -> str:
    cleaned = mime.split(';')[0].strip().lower()
    if cleaned == '' or cleaned == 'application/octet-stream':
        return '*'
    return _MIME_GROUPS.get(cleaned, '?')


def _ext_of(filename: str) -> str:
    if '.' not in filename:
        return ''
    return filename.rsplit('.', 1)[1].lower()


def is_calendar(filename: str, mime: str) -> bool:
    ct = mime.split(';')[0]
    return 'calendar' in ct or 'ics' in ct or filename.lower().endswith('.ics')


def classify_attachment(filename: str, mime: str, size: int, cap: int) -> tuple[bool, Optional[str]]:
    """Return (allowed, refusal_reason). Deterministic refusal order."""
    ext = _ext_of(filename)
    clean = mime.split(';')[0].strip().lower()
    if is_calendar(filename, mime):
        return (False, 'calendar (ICS) content is never exported as a file; use message_export for the raw MIME')
    if ext not in ALLOWED_EXTS:
        return (False, f"extension '{ext}' not in whitelist")
    if clean in DENIED_MIME:
        return (False, f"declared content type '{clean}' is on the deny list")
    eg = _EXT_GROUPS.get(ext)
    mg = _mime_group(mime)
    if mg not in ('*', '?') and eg is not None and mg != eg:
        return (False, f"declared content type '{clean}' conflicts with extension '.{ext}'")
    if size > cap:
        return (False, f'size {size} exceeds cap {cap} bytes')
    return (True, None)


def looks_like_markup(payload: bytes) -> bool:
    head = payload[:1024]
    if head.startswith(b'\xef\xbb\xbf'):
        head = head[3:]
    head = head.lstrip(b' \t\r\n\v\f')
    if not head.startswith(b'<'):
        return False
    low = head.lower()
    return b'<svg' in low or b'<?xml' in low or b'<!doctype' in low
