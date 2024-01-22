FROM tiangolo/uvicorn-gunicorn:python3.8-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV MODULE_NAME=main

# Gunicorn configuration env variables
ENV LOG_LEVEL=info
ENV MAX_WORKERS=3
ENV PORT=8000

# Give the workers enough time to load the language model (30s is not enough)
ENV TIMEOUT=60

# Install all the other required python dependencies
COPY ./requirements.txt /app
RUN pip install -r /app/requirements.txt \
    && rm -rf /root/.cache 

COPY ./scripts /app
COPY ./data /app
COPY ./main.py /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"]

