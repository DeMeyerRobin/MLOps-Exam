FROM python:3.10-slim

WORKDIR /app

COPY Fastapi/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Fastapi/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]