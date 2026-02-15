#!/usr/bin/env bash
# Agent self-rebuild script — works from the host or inside the container.
# The Docker daemon runs on the host, so the rebuild continues even after
# this container is replaced.
#
# Usage (from host):            bash deploy/rebuild.sh
# Usage (from agent bash tool): bash /app/deploy/rebuild.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Try to locate docker-compose.yml: script's parent dir first, then container path
if [ -f "$SCRIPT_DIR/../docker-compose.yml" ]; then
    COMPOSE_FILE="$(cd "$SCRIPT_DIR/.." && pwd)/docker-compose.yml"
elif [ -f "/mnt/host/PycharmProjects/Chorus/docker-compose.yml" ]; then
    COMPOSE_FILE="/mnt/host/PycharmProjects/Chorus/docker-compose.yml"
else
    echo "ERROR: docker-compose.yml not found"
    echo "Searched: $SCRIPT_DIR/../docker-compose.yml"
    echo "         /mnt/host/PycharmProjects/Chorus/docker-compose.yml"
    exit 1
fi

echo "Using compose file: $COMPOSE_FILE"

# Resolve git commit hash for the build context
COMPOSE_DIR="$(dirname "$COMPOSE_FILE")"
GIT_COMMIT="$(git -C "$COMPOSE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
export GIT_COMMIT
echo "Building from commit: $GIT_COMMIT"

echo "Triggering rebuild — this container will restart..."
nohup docker compose -f "$COMPOSE_FILE" up -d --build > /tmp/rebuild.log 2>&1 &
echo "Rebuild started (PID $!). Container will restart shortly."
