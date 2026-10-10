FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PHISH_BACKEND=rules \
    PHISH_DATA_DIR=/app/data
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt "gunicorn>=23,<24" \
    && groupadd --system phishguard \
    && useradd --system --gid phishguard phishguard \
    && mkdir -p /app/data \
    && chown phishguard:phishguard /app/data

COPY --chown=phishguard:phishguard . .
USER phishguard

EXPOSE 8080

# One worker shares the in-process cache lock; SQLite lives in the mounted volume.
CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:8080", "--timeout", "60", "api_service:app"]
