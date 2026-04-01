FROM dailyco/pipecat-base:latest

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY . .

CMD ["python", "bot.py"]
