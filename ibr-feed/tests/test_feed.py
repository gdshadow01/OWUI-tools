import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fixed commit timestamps keep entry dates deterministic: each fixture commit
# gets explicit distinct GIT_AUTHOR_DATE/GIT_COMMITTER_DATE, so assertions on
# published/updated ordering never depend on wall-clock second precision.
BASE_DATE = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def commit(repo, message, seq=0):
    date = (BASE_DATE + timedelta(days=seq)).isoformat()
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True, capture_output=True, env=env)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    import app

    root = tmp_path / "fixture"
    (root / "Leitsätze").mkdir(parents=True)
    (root / "Entscheidungsgründe").mkdir()
    (tmp_path / "data").mkdir()
    subprocess.run(["git", "init", "-b", "main", str(root)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=root, check=True)
    (root / "Leitsätze" / "Canonical_01.md").write_text('---\ntitle: "VG Berlin, Urteil vom 01.01.2024"\nAZ: 1 A 1/24\n---\n\n# VG Berlin, Urteil vom 01.01.2024\n\n1. Erster Satz.\n2. Zweiter Satz.\n', encoding="utf-8")
    (root / "Leitsätze" / "LS-1.md").write_text("# OVG Hamburg, Beschluss vom 02.01.2024\n\nLegacy body.\n", encoding="utf-8")
    (root / "Leitsätze" / "Canonical_02.md").write_text('---\ntitle: "VG Köln, Urteil vom 03.01.2024"\n---\n\n# VG Köln, Urteil vom 03.01.2024\n\nDritter Satz.\n', encoding="utf-8")
    (root / "Leitsätze" / "LS-2.md").write_text("LG Bonn, Urteil vom 04.01.2024\n\nVierter Satz.\n", encoding="utf-8")
    (root / "Entscheidungsgründe" / "excluded.md").write_text("not included", encoding="utf-8")
    commit(root, "fixture", seq=1)
    monkeypatch.setattr(app, "REPO_DIR", root)
    monkeypatch.setattr(app, "FEED_FILE", tmp_path / "feed.atom")
    monkeypatch.setattr(app, "FEED_TOKEN", "a" * 48)
    monkeypatch.setattr(app, "PAT", "test")
    monkeypatch.setattr(app, "DATA_DIR", tmp_path / "data")
    return root


def test_byte_idempotence(repo):
    import app
    assert app.build_feed() == app.build_feed()


def test_append_preserves_existing_entries(repo):
    import app
    before = ET.fromstring(app.build_feed())
    (repo / "Leitsätze" / "LS-3.md").write_text("OVG Bremen, Urteil\n\nFünfter Satz.\n", encoding="utf-8")
    commit(repo, "append", seq=2)
    after = ET.fromstring(app.build_feed())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    before_entries = {e.findtext("a:id", namespaces=ns): ET.tostring(e) for e in before.findall("a:entry", ns)}
    after_entries = {e.findtext("a:id", namespaces=ns): ET.tostring(e) for e in after.findall("a:entry", ns)}
    assert len(after_entries) == len(before_entries) + 1
    assert all(after_entries[k] == v for k, v in before_entries.items())


def test_both_shapes_and_ordered_markdown(repo):
    import app
    xml = app.build_feed().decode()
    assert "VG Berlin, Urteil vom 01.01.2024" in xml
    assert "OVG Hamburg, Beschluss vom 02.01.2024" in xml
    assert any("<ol>" in (entry.find("a:content", {"a": "http://www.w3.org/2005/Atom"}).text or "") for entry in __import__("xml.etree.ElementTree", fromlist=["fromstring"]).fromstring(app.build_feed()).findall("{http://www.w3.org/2005/Atom}entry"))
    assert "excluded" not in xml


def test_token_gate(repo):
    import app
    app.FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    app.FEED_FILE.write_bytes(app.build_feed())
    with TestClient(app.app) as client:
        response = client.get("/" + "a" * 48 + "/leitsaetze.atom")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/atom+xml")
        assert client.get("/wrong/leitsaetze.atom").status_code == 404
        assert client.get("/leitsaetze.atom").status_code == 404


def test_modify_and_delete(repo):
    import app
    before = ET.fromstring(app.build_feed())
    path = repo / "Leitsätze" / "Canonical_01.md"
    path.write_text(path.read_text(encoding="utf-8").replace("Erster Satz", "Geänderter Satz"), encoding="utf-8")
    commit(repo, "modify", seq=2)
    modified = ET.fromstring(app.build_feed())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    old = {e.findtext("a:id", namespaces=ns): e for e in before.findall("a:entry", ns)}
    new = {e.findtext("a:id", namespaces=ns): e for e in modified.findall("a:entry", ns)}
    key = "urn:ibr-leitsaetze:Canonical_01"
    assert key in new and new[key].findtext("a:updated", namespaces=ns) > old[key].findtext("a:updated", namespaces=ns)
    (repo / "Leitsätze" / "LS-2.md").unlink()
    commit(repo, "delete", seq=3)
    deleted = ET.fromstring(app.build_feed())
    assert "urn:ibr-leitsaetze:LS-2" not in {e.findtext("a:id", namespaces=ns) for e in deleted.findall("a:entry", ns)}
