FROM python:3.9-slim

ENV FLASK_APP=main.py
ENV FLASK_ENV=production

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
CMD ["flask", "run", "--host=0.0.0.0"]

EXPOSE 5000