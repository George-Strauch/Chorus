FROM python:3.12-slim

WORKDIR /app

# Install git (needed for agent workspace operations)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY template/ template/

RUN pip install --no-cache-dir .

# Create non-root user for running the bot
RUN useradd --create-home chorus
USER chorus

ENV CHORUS_HOME=/home/chorus/.chorus-agents

CMD ["python", "-m", "chorus"]
