FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 INSTANCE_DIR=/data
WORKDIR /app
# xgboost needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
# first start trains the model (about a minute) and loads demo data into /data, then serves
CMD ["sh", "-c", "python manage.py init && gunicorn -w 2 -b 0.0.0.0:8000 --timeout 300 'app:create_app()'"]
