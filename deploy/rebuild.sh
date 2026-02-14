#!/usr/bin/env bash
# Agent self-rebuild script — called from INSIDE the container.
# The Docker daemon runs on the host, so the rebuild continues even after
# this container is replaced.
#
# Usage (from agent bash tool): bash /app/deploy/rebuild.sh
set -euo pipefail

COMPOSE_FILE="/mnt/host/PycharmProjects/Chorus/docker-compose.yml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "ERROR: docker-compose.yml not found at $COMPOSE_FILE"
    echo "Is the host home mounted at /mnt/host?"
    exit 1
fi

echo "Triggering rebuild — this container will restart..."
nohup docker compose -f "$COMPOSE_FILE" up -d --build > /tmp/rebuild.log 2>&1 &
echo "Rebuild started (PID $!). Container will restart shortly."
