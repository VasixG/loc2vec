FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libspatialite7 \
    libsqlite3-mod-spatialite \
    spatialite-bin \
    && pip install -r requirements.txt

CMD ["python", "app/main.py"]
