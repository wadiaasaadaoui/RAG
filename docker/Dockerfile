FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# docker-compose.yml
version: '3'
services:
  rag-assistant:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/models
      - VECTOR_STORE_PATH=/app/data/vector_store