#!/bin/sh
set -eu
script_dir=$(cd "$(dirname "$0")" && pwd)
service_dir=$(cd "$script_dir/.." && pwd)
root=$(mktemp -d)
port=18743
cleanup() {
  docker rm -f ibr-feed-smoke >/dev/null 2>&1 || true
  rm -rf "$root"
}
trap cleanup EXIT
mkdir -p "$root/Leitsätze" "$root/Entscheidungsgründe" "$root/data"
cd "$root"
git init -b main >/dev/null
git config user.name Fixture
git config user.email fixture@example.invalid
cat > 'Leitsätze/Canon_1.md' <<'EOF'
---
title: "VG Berlin, Urteil"
AZ: 1 A 1/24
---

# VG Berlin, Urteil

1. Alpha.
2. Beta.
EOF
cat > 'Leitsätze/LS-1.md' <<'EOF'
# OVG Hamburg, Beschluss

Legacy text.
EOF
cat > 'Leitsätze/Canon_2.md' <<'EOF'
---
title: "VG Köln, Urteil"
---

# VG Köln, Urteil

Third.
EOF
cat > 'Leitsätze/LS-2.md' <<'EOF'
LG Bonn, Urteil

Fourth.
EOF
echo ignored > 'Entscheidungsgründe/ignored.md'
git add .
git commit -m fixture >/dev/null
docker build -f "$service_dir/Dockerfile" -t ibr-feed-smoke "$service_dir"
# The container runs as the host uid: the read-only, host-owned fixture is
# then same-uid (git refuses to serve foreign-owned repos: "dubious
# ownership"), and everything the app writes under the /data bind is owned
# by the invoking user, so cleanup needs no root. No host path is written.
docker run -d --name ibr-feed-smoke --user "$(id -u):$(id -g)" -e IBR_GITHUB_PAT=dummy -e IBR_FEED_TOKEN=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa -e IBR_REPO_URL="file:///fixture" -e IBR_DATA_DIR=/data -v "$root:/fixture:ro" -v "$root/data:/data" -p "127.0.0.1:$port:8000" ibr-feed-smoke >/dev/null
served=""
for attempt in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$port/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/leitsaetze.atom" -o "$root/feed.atom"; then served=1; break; fi
  sleep 1
done
if [ -z "$served" ]; then
  echo "smoke FAIL: feed not served; container logs:"
  docker logs ibr-feed-smoke 2>&1 | tail -20
  exit 1
fi
python3 - "$root/feed.atom" <<'PY'
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
ns = {"a": "http://www.w3.org/2005/Atom"}
entries = root.findall("a:entry", ns)
assert len(entries) == 4, len(entries)
assert all(e.find("a:id", ns) is not None and e.find("a:content", ns) is not None for e in entries)
assert "<ol>" in (entries[-1].findtext("a:content", namespaces=ns) or "") or any("<ol>" in (e.findtext("a:content", namespaces=ns) or "") for e in entries)
print(f"Atom smoke PASS: {len(entries)} entries; XML valid; content present")
PY
wrong=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/wrong/leitsaetze.atom")
test "$wrong" = 404
printf 'wrong-token HTTP %s PASS\n' "$wrong"
# Regression check for the B1 finding (t_e21e9a51): the access log must never
# contain the feed token; uvicorn has to log /<redacted>/leitsaetze.atom.
logs=$(docker logs ibr-feed-smoke 2>&1)
tok=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
case "$logs" in
  *"$tok"*) echo "smoke FAIL: feed token visible in container access log"; exit 1 ;;
esac
# uvicorn percent-quotes the path ('<' -> '%3C'), so decode before matching.
decoded=$(printf '%s' "$logs" | sed 's/%3C/</g; s/%3E/>/g')
case "$decoded" in
  *"/<redacted>/leitsaetze.atom"*) ;;
  *) echo "smoke FAIL: redacted path missing from container access log; last lines:"; printf '%s\n' "$logs" | tail -5; exit 1 ;;
esac
printf 'access-log redaction PASS\n'
