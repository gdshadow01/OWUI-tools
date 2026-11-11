# IBR Leitsätze Atom feed

A small, token-gated Atom endpoint over only the `Leitsätze/` directory of the private Vergaberechtsprechung repository. Stable entry IDs use the filename stem, preserving FreshRSS read state.

## Run

Copy `.env.example` to `.env`, set `IBR_GITHUB_PAT` (read-only Contents access to the one source repository) and generate `IBR_FEED_TOKEN` with `openssl rand -hex 24`. The PAT is supplied to Git through `GIT_ASKPASS`; do not put it in repository URLs or logs. Run using Docker Compose. The named volume persists the clone and feed cache. The compose file deliberately publishes no host ports and defines no shared network: deployment wiring must attach the service to the FreshRSS network, where FreshRSS can access `http://ibr-feed:8000/<token>/leitsaetze.atom`.

`IBR_REPO_URL` and `IBR_BRANCH` can point at a fixture repository for tests. `IBR_SYNC_INTERVAL` defaults to 900 seconds. The application requires both secrets at startup; `/healthz` has no feed data and no token requirement.

## Rotation

Replace `IBR_GITHUB_PAT` in the deployment environment with a newly scoped, read-only repository token and restart the service. Revoke the old PAT. To rotate `IBR_FEED_TOKEN`, update the FreshRSS feed URL and container environment together, then restart. Never commit `.env`.

## Scope and tests

The directory name `Leitsätze` is intentionally hardcoded. No other corpus paths are read. Run mandatory tests in Docker with `docker compose run --rm --build ibr-feed pytest -q`; the integration smoke test uses a temporary fixture Git repository and publishes a temporary localhost port only for the test. Access-log redaction is enforced by tests and smoke: uvicorn must never log the feed token, only the `/<redacted>/leitsaetze.atom` path (percent-encoded by uvicorn as `/%3Credacted%3E/`).
