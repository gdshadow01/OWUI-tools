"""Tests for the himalaya MCP server-side attachment policy."""

import pytest

import attachments


def _mime_for(ext: str) -> str:
    for mime, group in attachments._MIME_GROUPS.items():
        if attachments._EXT_GROUPS.get(ext) == group:
            return mime
    return f'image/{ext}'


@pytest.mark.parametrize('ext', sorted(attachments.ALLOWED_EXTS))
def test_whitelisted_ext_with_concrete_mime(ext):
    assert attachments.classify_attachment(f'f.{ext}', _mime_for(ext), 10, 1000) == (True, None)


@pytest.mark.parametrize('ext', sorted(attachments.ALLOWED_EXTS))
def test_whitelisted_ext_with_empty_mime(ext):
    assert attachments.classify_attachment(f'f.{ext}', '', 10, 1000) == (True, None)


@pytest.mark.parametrize('filename,mime', [
    ('setup.exe', 'application/octet-stream'),
    ('archive.zip', 'application/zip'),
    ('big.7z', 'application/x-7z-compressed'),
    ('script.bat', 'application/x-dos-bat'),
    ('photo.svg', 'image/svg+xml'),
])
def test_refusal_by_extension(filename, mime):
    allowed, reason = attachments.classify_attachment(filename, mime, 10, 1000)
    assert allowed is False
    assert reason.startswith('extension')


@pytest.mark.parametrize('filename,mime', [
    ('foto.png', 'image/svg+xml'),
    ('doc.pdf', 'application/zip'),
    ('x.docx', 'text/html'),
])
def test_denied_mime_with_whitelisted_ext(filename, mime):
    allowed, reason = attachments.classify_attachment(filename, mime, 10, 1000)
    assert allowed is False
    assert 'deny list' in reason


def test_group_conflict():
    allowed, reason = attachments.classify_attachment('a.png', 'application/pdf', 10, 1000)
    assert allowed is False
    assert 'conflicts with extension' in reason


@pytest.mark.parametrize('mime', ['application/msword', 'application/octet-stream', ''])
def test_lazy_sender_allowed(mime):
    assert attachments.classify_attachment('a.docx', mime, 10, 1000) == (True, None)


def test_no_extension():
    allowed, reason = attachments.classify_attachment('invoice', 'application/pdf', 10, 1000)
    assert allowed is False
    assert "extension '' not in whitelist" == reason


@pytest.mark.parametrize('filename,mime', [
    ('invite.ics', 'text/calendar'),
    ('invite.ics', 'text/calendar; method=REQUEST'),
    ('x.txt', 'application/ics'),
    ('x.ics', 'application/octet-stream'),
])
def test_calendar(filename, mime):
    assert attachments.is_calendar(filename, mime) is True
    allowed, reason = attachments.classify_attachment(filename, mime, 10, 1000)
    assert allowed is False
    assert 'calendar' in reason


def test_not_calendar():
    assert attachments.is_calendar('a.pdf', 'application/pdf') is False


def test_cap_boundary():
    assert attachments.classify_attachment('a.pdf', 'application/pdf', 1000, 1000) == (True, None)
    allowed, reason = attachments.classify_attachment('a.pdf', 'application/pdf', 1001, 1000)
    assert allowed is False
    assert reason == 'size 1001 exceeds cap 1000 bytes'


@pytest.mark.parametrize('payload,expected', [
    (b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', True),
    (b'<?xml version="1.0"?>', True),
    (b'\xef\xbb\xbf\n  <?xml version="1.0"?>', True),
    (b'<!DOCTYPE html><html>', True),
    (b'\x89PNG\r\n\x1a\n...', False),
    (b'plain text', False),
    (b'%PDF-1.7 ...', False),
])
def test_looks_like_markup(payload, expected):
    assert attachments.looks_like_markup(payload) is expected


def test_full_allowed_sanity():
    assert attachments.classify_attachment(
        'report.pdf', 'application/pdf', 325, 20 * 1024 * 1024
    ) == (True, None)
