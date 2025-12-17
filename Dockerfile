FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY backend/ ./backend/

# Create data directory
RUN mkdir -p /app/backend/data

# Expose the port
EXPOSE 8888

# Environment variables with defaults
ENV BANDIT_STATE_PATH=/app/backend/data/bandit_state.json
ENV BANDIT_QUEUE_DB_PATH=/app/backend/data/queue.sqlite3
ENV BANDIT_ANALYTICS_DB_PATH=/app/backend/data/analytics.sqlite3
ENV BANDIT_SNAPSHOT_DIR=/app/backend/data/snapshots

# Default command - start the API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8888"]

