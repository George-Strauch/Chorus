#!/usr/bin/env bash
# Rebuild the Chorus container with the latest source.
#
# From the host:  runs docker compose build+up directly.
# From container: builds the image first, then does a quick container swap.
#                 The swap (stop old + start new) is fast enough that compose
#                 finishes before the old container is killed.
#
# Usage (from host):            bash deploy/rebuild.sh
# Usage (from agent bash tool): bash /app/deploy/rebuild.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Detect environment ---
IN_CONTAINER=false
if [ -f /.dockerenv ] || grep -qsm1 'docker\|containerd' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=true
fi

# --- Locate docker-compose.yml ---
if [ "$IN_CONTAINER" = false ] && [ -f "$SCRIPT_DIR/../docker-compose.yml" ]; then
    COMPOSE_FILE="$(cd "$SCRIPT_DIR/.." && pwd)/docker-compose.yml"
elif [ -f "/mnt/host/PycharmProjects/Chorus/docker-compose.yml" ]; then
    COMPOSE_FILE="/mnt/host/PycharmProjects/Chorus/docker-compose.yml"
else
    echo "ERROR: docker-compose.yml not found"
    echo "Searched: $SCRIPT_DIR/../docker-compose.yml"
    echo "         /mnt/host/PycharmProjects/Chorus/docker-compose.yml"
    exit 1
fi
COMPOSE_DIR="$(dirname "$COMPOSE_FILE")"
echo "Using compose file: $COMPOSE_FILE"

# --- Resolve build metadata ---
GIT_COMMIT="$(git -C "$COMPOSE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
export GIT_COMMIT
echo "Building from commit: $GIT_COMMIT"

# --- Set env vars that docker-compose.yml expects ---
# run.sh sets these on first launch; rebuild needs them too.
export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"
if [ -S /var/run/docker.sock ]; then
    export DOCKER_GID="${DOCKER_GID:-$(stat -c '%g' /var/run/docker.sock)}"
else
    export DOCKER_GID="${DOCKER_GID:-999}"
fi

if [ "$IN_CONTAINER" = true ]; then
    # Recover the host-side SCOPE_PATH from the mount that backs /mnt/host.
    HOST_SCOPE="$(docker inspect "$(hostname)" \
        --format '{{range .Mounts}}{{if eq .Destination "/mnt/host"}}{{.Source}}{{end}}{{end}}' 2>/dev/null || true)"
    if [ -z "$HOST_SCOPE" ]; then
        echo "WARNING: Could not detect host SCOPE_PATH, falling back to /home/$(id -un)"
        HOST_SCOPE="/home/$(id -un)"
    fi
    export SCOPE_PATH="$HOST_SCOPE"
    export HOME="$HOST_SCOPE"
    echo "Detected host SCOPE_PATH=$SCOPE_PATH"
else
    export SCOPE_PATH="${SCOPE_PATH:-$HOME}"
fi

echo "  UID=$HOST_UID  GID=$HOST_GID  DOCKER_GID=$DOCKER_GID  SCOPE_PATH=$SCOPE_PATH"

# --- Rebuild ---
if [ "$IN_CONTAINER" = true ]; then
    # Build the new image first — this is safe and doesn't touch running containers.
    echo "Building new image..."
    docker compose -f "$COMPOSE_FILE" build

    # Now swap the container. This stops the old one (us) and starts the new one.
    # nohup keeps the process alive briefly — the swap is fast since the image
    # is already built (no --build needed).
    echo "Swapping container — this container will restart..."
    nohup docker compose -f "$COMPOSE_FILE" up -d --force-recreate --no-build \
        > /tmp/rebuild.log 2>&1 &
    echo "Swap initiated (PID $!). Check 'docker logs chorus-bot-1' for status."
else
    echo "Rebuilding..."
    docker compose -f "$COMPOSE_FILE" up -d --build --force-recreate
    echo "Rebuild complete."
fi
