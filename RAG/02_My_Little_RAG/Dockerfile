FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies
# This layer will be cached unless requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
# Changes here won't trigger a reinstall of dependencies
COPY retrieval_api.py ./
COPY retrieval_engine.py ./
COPY reranker.py ./
COPY openapi_routes.py ./
COPY main.py ./
# COPY .env ./

# Set environment variable to disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Expose the port for the MCP server
EXPOSE 8000

# Default command to run the application; docker-compose files override TRANSPORT_TYPE to select mcp vs openapi mode
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
