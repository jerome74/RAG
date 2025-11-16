# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
# Add estimate.csv
RUN mkdir archive
COPY archive/estimate.csv /app/archive/estimate.csv
# Install python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Add app
COPY main.py /app/main.py

# Use a non-root user (optional hardening)
RUN useradd -m appuser
USER appuser

# Run the ingestion script
ENTRYPOINT ["python3", "-u", "main.py"]