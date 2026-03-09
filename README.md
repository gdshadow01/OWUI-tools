# my_little_mcp_tools

Collection of tools for Open WebUI.

## Tools

- **websearch** - Web search and page fetch functionality with Google, DuckDuckGo, and SearxNG support
- **RAG/01_My_Little_RAG_Ingestion** - RAG ingestion 
- **RAG/02_My_Little_RAG** - RAG retrieval 
- **RAG/03_My_Litte_RAG_Laws** - returns legislative text (only tested with german laws)
- **RAG/04_My_Little_RAG_qdrant** - docker-compose.yml for [Qdrant](https://github.com/qdrant/qdrant) (database for RAG tools)  
- **openterminal** - docker-compose.yml for [Open Terminal](https://github.com/open-webui/open-terminal) (provides terminal access)
- **tika** - docker-compose.yml for [Apache Tika](https://github.com/apache/tika) (document parser for websearch and Open Webui)
- **searxng** - docker-compose.yml and settings for [SearXNG](https://github.com/searxng/searxng) (metasearch engine)

## Setup

### Prerequisites

All tools must run on the same Docker network (default: `ollama-tools`). OpenWebUI must also be on this network.

The Docker network is not defined in this repository's docker-compose.yml, as it's expected to be managed by a main compose file.

### Example Main Docker Compose

Create a main docker-compose.yml at `/home/user/docker-compose.yml`:

```yaml
include:
  - path: ./open-webui/docker-compose.yml
  - path: ./my_little_mcp_tools/docker-compose.yml

networks:
  ollama-tools:
    name: ollama-tools
    driver: bridge
  # other networks (reverse proxy, ...)  
```

Example `/home/user/open-webui/docker-compose.yml`:

```yaml
services:
  ollama-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: ollama-webui
    volumes:
      - ./data:/app/backend/data
    ports:
      - 3000:8080
    environment:
      - OLLAMA_BASE_URL=http(s)://IP:Port_or_URL
      - ENV=dev
      - WEBUI_AUTH=True
      - WEBUI_NAME=localGPT
      - WEBUI_URL=http://localhost:8080
      - WEBUI_SECRET_KEY=somesecretstring
      - ADMIN_EMAIL=something@something.com
      - TZ=Continent/City
    restart: unless-stopped
    networks:
      - ollama-tools
    # - other networks (reverse proxy, ...)  
```

The `docker-compose.yml` from this repository should be at `/home/user/my_little_mcp_tools/docker-compose.yml`.

### Configuration

Copy the relevant `.env.example` files from each tool's directory to `.env` and configure the required environment variables:

```bash
cp websearch/.env.example websearch/.env
cp RAG/.env.example RAG/.env
# Copy and configure other .env files as needed
```

Edit each `.env` file with your specific configuration (API keys, URLs, etc.).

**RAG Tools Configuration:**

For tools under the `RAG` folder (01-04), each tool needs the `.env` file. You have three options:

1. **Copy manually** to each folder:
   ```bash
   cp RAG/.env RAG/01_My_Little_RAG_Ingestion/.env
   cp RAG/.env RAG/02_My_Little_RAG/.env
   cp RAG/.env RAG/03_My_Litte_RAG_Laws/.env
   cp RAG/.env RAG/04_My_Little_RAG_qdrant/.env
   ```

2. **Create symlinks manually** in each folder pointing to the RAG folder's `.env`

3. **Use the provided script** from the RAG directory:
   ```bash
   cd RAG
   chmod +x create-symlinks.sh
   ./create-symlinks.sh
   ```

### Selecting Tools

Comment out the tools you don't need in the main `docker-compose.yml`:

```yaml
include:

# web search
  - path: ./websearch/docker-compose.yml # :8011

# RAG
  - path: ./RAG/02_My_Little_RAG/docker-compose.yml # 8020
  - path: ./RAG/03_My_Litte_RAG_Laws/docker-compose.yml # 8023
  - path: ./RAG/04_My_Little_RAG_qdrant/docker-compose.yml  # 6777

# openterminal
  - path: ./openterminal/docker-compose.yml  # 8000

# tika
  - path: ./tika/docker-compose.yml # 9998

# searngx
  - path: ./searxng/docker-compose.yml  # 8040
```

### Running

From your main docker-compose directory:

```bash
docker-compose up -d
```

To stop:

```bash
docker-compose down
```

# Open WebUI setup
comming soon

## Default Ports
- **websearch:** port 8011
- **RAG/02_My_Little_RAG:** port 8020
- **RAG/03_My_Litte_RAG_Laws:** 8023
- **RAG/04_My_Little_RAG_qdrant:** port 6777
- **openterminal:** port 8000
- **tika:** port 9998
- **searxng:** port: 8040
