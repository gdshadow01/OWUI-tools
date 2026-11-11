FROM docker.io/python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl tar \
    && curl -fsSL https://github.com/pimalaya/himalaya/releases/download/v2.0.0/himalaya.x86_64-linux.tgz \
       | tar xz -C /usr/local/bin himalaya \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY attachments.py .

EXPOSE 9201

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9201"]
