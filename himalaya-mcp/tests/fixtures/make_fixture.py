#!/usr/bin/env python3
"""Build the attachment smoke-test fixture email.

Writes exactly ONE file: tests/fixtures/fixture.eml — a multipart/mixed
message with a text/plain body followed by SIX attachments (in order):

    1. report.pdf   application/pdf            real PDF header, >= 64 bytes
    2. bild.png     image/png                  PNG magic + filler
    3. invite.ics   text/calendar;method=REQUEST   VCALENDAR body
    4. setup.exe    application/octet-stream   MZ header
    5. grafik.png   image/png (declared)       SVG/XSS payload masquerade
    6. gross.pdf    application/pdf            over the smoke size cap

Populating the m2dir store (tests/fixtures/store/) is NOT this script's
job. The smoke harness does it itself, e.g.:

    docker run --rm \
        -v <abs-store>:/store \
        -v <abs-store>/.smoke-config.toml:/smoke-config.toml \
        -v <abs-fixtures>:/f \
        --entrypoint himalaya owui-himalaya-smoke:latest \
        --config /smoke-config.toml \
        m2dir messages save -m Inbox -- /f/fixture.eml

Host python3 stdlib only — no third-party imports (repo rule: Docker-only
tooling, clean host).
"""

import os

from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

FIXED_DATE = "Mon, 21 Sep 2026 09:00:00 +0000"
FIXED_MSGID = "<attachment-smoke-fixture@fixture.invalid>"
FIXED_BOUNDARY = "owui-smoke-fixture-boundary-1"

ICS_BODY = (
    "BEGIN:VCALENDAR\r\n"
    "VERSION:2.0\r\n"
    "PRODID:-//OWUI//Smoke Fixture//EN\r\n"
    "METHOD:REQUEST\r\n"
    "BEGIN:VEVENT\r\n"
    "UID:fixture-1@fixture.invalid\r\n"
    "DTSTAMP:20260921T090000Z\r\n"
    "DTSTART:20260921T100000Z\r\n"
    "DTEND:20260921T110000Z\r\n"
    "SUMMARY:Smoke fixture event\r\n"
    "END:VEVENT\r\n"
    "END:VCALENDAR\r\n"
)

SVG_PNG_PAYLOAD = (
    b'<?xml version="1.0"?>'
    b'<svg xmlns="http://www.w3.org/2000/svg">'
    b"<script>alert(1)</script></svg>"
)


def build_payloads() -> dict:
    """Return {filename: (declared_mime, exact_bytes)} for all 6 attachments."""
    pdf = b"%PDF-1.4\n%smoke-test-pdf\n"
    if len(pdf) < 64:
        pdf += b"%" * (64 - len(pdf))
    return {
        "report.pdf": ("application/pdf", pdf),
        "bild.png": ("image/png", b"\x89PNG\r\n\x1a\n" + b"P" * 40),
        "invite.ics": ("text/calendar; method=REQUEST", ICS_BODY.encode("utf-8")),
        "setup.exe": ("application/octet-stream", b"MZ" + b"E" * 40),
        "grafik.png": ("image/png", SVG_PNG_PAYLOAD),
        "gross.pdf": ("application/pdf", b"%PDF-1.7\n" + b"x" * 2000),
    }


def build_message():
    """Build the multipart/mixed fixture message (1 body part + 6 attachments)."""
    msg = MIMEMultipart("mixed", boundary=FIXED_BOUNDARY)
    msg["From"] = "smoke@fixture.invalid"
    msg["To"] = "agent@fixture.invalid"
    msg["Subject"] = "Attachment smoke fixture"
    msg["Date"] = FIXED_DATE
    msg["Message-ID"] = FIXED_MSGID

    msg.attach(MIMEText("smoke body", "plain", "us-ascii"))

    for filename, (mime, payload) in build_payloads().items():
        primary, subtype = mime.split(";")[0].strip().split("/", 1)
        extra_params = {}
        if ";" in mime:
            for param in mime.split(";")[1:]:
                k, _, v = param.strip().partition("=")
                if k:
                    extra_params[k] = v
        if primary == "text":
            part = MIMEBase(primary, subtype, **extra_params)
            part.set_payload(payload.decode("utf-8"), charset="utf-8")
        elif primary == "image":
            part = MIMEImage(payload, _subtype=subtype)
        else:
            part = MIMEApplication(payload, _subtype=subtype)
        part.add_header("Content-Disposition", "attachment", filename=filename)
        msg.attach(part)
    return msg


def main() -> None:
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture.eml")
    with open(out_path, "wb") as f:
        f.write(build_message().as_bytes())
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
