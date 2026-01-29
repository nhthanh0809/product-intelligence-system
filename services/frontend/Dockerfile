FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Source code will be mounted as volume in dev mode
# COPY . .  # Not needed - mounted as volume

ENV PYTHONUNBUFFERED=1

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit with file watcher for hot reloading
# --server.runOnSave=true enables auto-reload when files change
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.runOnSave=true", \
     "--server.fileWatcherType=poll"]
