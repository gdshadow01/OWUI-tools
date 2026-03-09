FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy only the requirements file first
COPY requirements.txt .

# Install dependencies
# This layer will be cached unless requirements.txt changes
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data during build to avoid runtime downloads
# RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('punkt_tab', download_dir='/usr/local/nltk_data')"

# Copy the rest of the application code
# Changes here won't trigger a reinstall of dependencies
COPY ingest_index.py ./

# Set environment variable to disable Python output buffering
ENV PYTHONUNBUFFERED=1

# Default entrypoint is defined in docker-compose
CMD ["bash"]
