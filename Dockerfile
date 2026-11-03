#   docker build -t sentiment-api .
#   docker run -p 8000:80 -e APP_API_KEY=dev -e GROQ_API_KEY=... -e HF_MODEL_REPO=... sentiment-api
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py inference.py llm_sentiment.py absa.py model_training.py sentiment_rules.py ./
COPY prompts/ prompts/
COPY models/linearsvc.joblib models/

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-80}"]
