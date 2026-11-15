FROM python:3.11-slim

WORKDIR /app

COPY starter/requirements.txt .
# lightgbm: needed by lynch (the starter models do not use it)
# libgomp1: OpenMP runtime required by lightgbm (missing in the slim image)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt lightgbm

COPY starter/ ./starter/
COPY lynch/ ./lynch/
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["lynch-v8"]
