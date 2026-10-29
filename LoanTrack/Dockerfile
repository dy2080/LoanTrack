FROM python:3.11-slim

WORKDIR /app

# Install system deps for sklearn/xgboost
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train model at build time (so the image ships with a ready model)
RUN python train.py || true

EXPOSE 8000

CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
