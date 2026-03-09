FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY utils/ ./utils/

RUN mkdir -p /workspace/uploads /workspace/outputs && useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app /workspace

USER mcpuser

ENV WORKSPACE_DIR=/workspace
ENV PORT=8015

EXPOSE 8015

CMD ["python", "server.py"]
