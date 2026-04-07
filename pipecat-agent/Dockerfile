FROM dailyco/pipecat-base:latest

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY bot.py .
COPY canvas_actions.py .
COPY config.py .
COPY persona.py .
COPY api_client.py .
COPY scene_context.py .
