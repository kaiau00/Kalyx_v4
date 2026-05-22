FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements-kalshi.txt .
RUN pip install --no-cache-dir -r requirements-kalshi.txt

COPY . .

CMD ["python", "run_kalshi_bot.py", "--asset", "BTC", "--live"]
