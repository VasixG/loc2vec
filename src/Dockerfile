FROM python:3.9-slim

WORKDIR /src

COPY . .

RUN pip install -r requirements.txt

RUN chmod +x entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
