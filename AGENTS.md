# AGENTS.md — OWUI-tools

Mono-repo of containerized tools around Open WebUI, composed via Docker
Compose include-trees. Python-first. Deploy target: .209.

## Layout

- Each subdir is a self-contained service (own Dockerfile, compose, README).
- `deprecated/` holds unmaintained variants — do not extend them.
- Root `docker-compose.yml` — tracked include-composition of all subdir
  services (its header comment claiming git-exclusion is stale).
- `local.docker-compose.yml` and `local2.docker-compose.yml` — git-excluded
  smoke overrides (`local.*` gitignore rule / `.git/info/exclude`). These
  files DEFINE the `ollama-tools` and `searxng` networks.

## Hard rules

- **Branch `dev` only — never commit to `main`.** Deployment pulls `dev`.
- **Docker-only tooling.** The host stays clean: no pip installs, no host
  venvs. Host `python3` stdlib scripts (no third-party imports) are allowed.
- **Run compose from the repo ROOT.** Subdir compose files reference the
  `ollama-tools` network without defining it, so `docker compose` inside a
  subdir fails on the undefined network. Use
  `docker compose -f local.docker-compose.yml up -d --build <service>`.
- Local commits only; **no push without explicit user instruction**.
- Never commit secrets — `.env` is gitignored everywhere; every service
  ships `.env.example`.

## Conventions

- Conventional commits: `feat|fix|chore|docs|test(scope): summary`.
- Subdir services join the shared `ollama-tools` network.
- Production on .209 uses `/home/shadow01/docker-compose.yml` (open-webui
  stack), which includes the same subdir compose files.
- websearch-mcp serves reddit.com URLs through `reddit_backend.py`
  (anonymous .rss Atom feeds, rate-gated + cached); never route reddit
  through the browser/stealth escalation tiers — they cannot read comments.

## Repo topology & RAG (fleet reference, verified 2026-11-03)

- **Remotes:** `origin` = `git@github.com:adabot-lab/OWUI-tools.git` — the
  authoritative remote per live `git remote -v`. `gdshadow01/OWUI-tools`
  exists as `upstream` (fork parent); older notes naming `gdshadow01` as the
  repo remote are stale.
- **Branch truth:** local `dev` is authoritative and may run AHEAD of
  `origin/dev` by design (local-only commits; no push without explicit user
  instruction). Before trusting the remote-tracking ref, check live with
  `git ls-remote origin dev`.
- **Sparse checkout:** the local working tree excludes parts of the tracked
  tree (`core.sparseCheckout=true`; e.g. all of `RAG/01_My_Little_RAG_Ingestion/`
  is absent locally). Read excluded files via `git show dev:<path>`.
- **`litellm/config/strip_hermes.py`:** LiteLLM pre-call hook that rewrites
  "Hermes Agent"/"Hermes" -> "the agent" in system prompts before requests
  to zai/glm models are forwarded (case-sensitive on purpose).
- **RAG services** (include-trees pulled in by the root compose):
  - `RAG/01_My_Little_RAG_Ingestion/` — one-shot ingestion service
    `my-rag-ingest-only` (run from repo root:
    `docker compose run --rm my-rag-ingest-only`; shared env `RAG/.env`,
    symlinked as `RAG/01/.env`; batch-size knob `QDRANT_BATCH_SIZE`,
    default 64, read in `qdrant_ops.py`).
  - `RAG/02_My_Little_RAG/` — production retrieval service
    (`my_little_rag-retrieval`; MCP + OpenAPI transports; reranker add-on
    `reranker.py`, fails open).
  - `RAG/03_My_Litte_RAG_Laws/` — laws corpus pipeline (the dir-name typo
    "Litte" is real — keep it).
  - `RAG/04_My_Little_RAG_qdrant/` — Qdrant storage service.
- **Qdrant collections (live 2026-11-03):** `vergaberechtsprechung` 14,085
  points; `vault` 14,363; `user_manual_vergabeplattform` 1,894;
  `vergabe_berlin-regelungen` 1,537; `openwebui-docs` 1,501;
  `vergabe_howtos` 238; `distrobox-docs` 113;
  `berlinovo_betriebsvereinbarungen` 79.
