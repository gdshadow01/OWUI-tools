FROM docker.io/python:3.13-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py reddit_backend.py ./

# Expose MCP Streamable HTTP port
EXPOSE 9200

# Run the MCP Streamable HTTP server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9200"]
