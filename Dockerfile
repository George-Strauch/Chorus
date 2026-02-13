FROM python:3.12-slim

WORKDIR /app

# Install git (needed for agent workspace operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY template/ template/

RUN pip install --no-cache-dir .

# Create non-root user and pre-create data directory with correct ownership
RUN useradd --create-home chorus \
    && mkdir -p /home/chorus/.chorus-agents/db \
    && chown -R chorus:chorus /home/chorus/.chorus-agents
USER chorus

ENV CHORUS_HOME=/home/chorus/.chorus-agents
ENV CHORUS_TEMPLATE_DIR=/app/template

CMD ["python", "-m", "chorus"]
