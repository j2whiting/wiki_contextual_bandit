#!/bin/bash
# Fresh start - wipe all data and start with a clean slate

set -e

if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
elif docker compose version >/dev/null 2>&1; then
  COMPOSE="docker compose"
else
  echo "ERROR: Docker Compose not found. Install Docker Desktop (recommended) or docker-compose."
  exit 1
fi

echo "🗑️  Stopping containers and removing data volume..."
$COMPOSE down -v 2>/dev/null || true

echo "🔨 Building and starting fresh..."
$COMPOSE up --build
