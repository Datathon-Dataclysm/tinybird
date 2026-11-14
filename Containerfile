
FROM python:3.11-slim

WORKDIR /app

COPY starter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY starter/ ./starter/
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["woodcock"]
