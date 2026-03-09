FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged user
RUN useradd -m -u 1000 mcp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/

RUN mkdir /workspace && chown -R mcp:mcp /workspace /app

USER mcp

ENV WORKSPACE_DIR=/workspace

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
