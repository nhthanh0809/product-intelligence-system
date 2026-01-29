FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (for better caching)
COPY pyproject.toml .

# Create empty src directory for editable install
RUN mkdir -p src

# Install dependencies in editable mode
# This installs deps but source code comes from volume mount
RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app/src:/app
ENV PYTHONUNBUFFERED=1

# Source code will be mounted as volume in dev mode
# COPY src/ ./src/  # Not needed - mounted as volume

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/health || exit 1

EXPOSE 8010

# --reload enables hot reloading when source files change
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8010", "--reload"]
