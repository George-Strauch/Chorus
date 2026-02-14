FROM python:3.12-slim

WORKDIR /app

# Install git, curl, ca-certificates, openssh-client (needed for agent operations + Docker CLI + git push)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI (static binary — no daemon, just the client)
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.5.1.tgz \
    | tar xz --strip-components=1 -C /usr/local/bin docker/docker

# Install docker-compose standalone
RUN curl -fsSL -o /usr/local/bin/docker-compose \
        "https://github.com/docker/compose/releases/download/v2.33.1/docker-compose-linux-x86_64" \
    && chmod +x /usr/local/bin/docker-compose \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && ln -s /usr/local/bin/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose

COPY pyproject.toml .
COPY src/ src/
COPY template/ template/
COPY deploy/ deploy/

RUN pip install --no-cache-dir .

# Container runs as host user (UID/GID passed at build time)
ARG UID=1000
ARG GID=1000
ARG DOCKER_GID=999

RUN groupadd -g ${GID} appuser 2>/dev/null || true \
    && useradd -u ${UID} -g ${GID} -m appuser \
    && groupadd -g ${DOCKER_GID} dockerhost 2>/dev/null || true \
    && usermod -aG ${DOCKER_GID} appuser \
    && mkdir -p /home/appuser/.chorus-agents/db \
    && chown -R appuser:appuser /home/appuser/.chorus-agents
USER appuser

ENV CHORUS_HOME=/home/appuser/.chorus-agents
ENV CHORUS_TEMPLATE_DIR=/app/template
ENV CHORUS_SCOPE_PATH=/mnt/host

CMD ["python", "-m", "chorus"]
