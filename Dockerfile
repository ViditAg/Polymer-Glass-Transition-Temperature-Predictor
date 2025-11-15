# Dockerfile for Polymer-Glass-Transition-Temperature-Predictor
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install chemprop

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "predict_Tg.py"]
