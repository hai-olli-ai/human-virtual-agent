FROM dailyco/pipecat-base:latest

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY bot.py .
COPY config.py .
COPY persona.py .
COPY api_client.py .
COPY scene_context.py .

# S64c — Canvas Protocol modules
COPY tools tools
COPY context context
COPY services services
