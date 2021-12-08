FROM python:3.8.0-buster
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR  /PA2Code
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && \
  apt-get install -y openjdk-11-jdk-headless
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
COPY /PA2Code .
CMD ["python3", "./EnriquezPA2.py"]

