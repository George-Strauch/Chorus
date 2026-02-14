FROM python:3.12-slim

WORKDIR /app

# Install git (needed for agent workspace operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY template/ template/

RUN pip install --no-cache-dir .

# Data directory — mapped to host ~/.chorus-agents via docker-compose
# Container runs as host user (UID/GID passed at build time)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser 2>/dev/null || true \
    && useradd -u ${UID} -g ${GID} -m appuser \
    && mkdir -p /home/appuser/.chorus-agents/db \
    && chown -R appuser:appuser /home/appuser/.chorus-agents
USER appuser

ENV CHORUS_HOME=/home/appuser/.chorus-agents
ENV CHORUS_TEMPLATE_DIR=/app/template

CMD ["python", "-m", "chorus"]
