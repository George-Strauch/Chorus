#!/usr/bin/env bash
# Host-side startup script for Chorus.
# Usage: ./deploy/run.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# --- Pre-flight checks ---
if [ ! -f .env ]; then
    echo "ERROR: .env not found. Copy .env.example and fill in your keys:"
    echo "  cp .env.example .env"
    exit 1
fi

# --- Auto-detect IDs ---
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"

# Docker socket GID (needed so container user can talk to Docker)
if [ -S /var/run/docker.sock ]; then
    export DOCKER_GID="$(stat -c '%g' /var/run/docker.sock)"
else
    echo "WARNING: /var/run/docker.sock not found — agents won't be able to self-rebuild"
    export DOCKER_GID=999
fi

# Host home mount (default: $HOME)
export SCOPE_PATH="${SCOPE_PATH:-$HOME}"
export HOST_USER="$(whoami)"

echo "Starting Chorus..."
echo "  UID=$HOST_UID  GID=$HOST_GID  DOCKER_GID=$DOCKER_GID"
echo "  SCOPE_PATH=$SCOPE_PATH"

docker compose build && docker compose up -d

echo "Chorus is running. View logs with: docker compose logs -f"
