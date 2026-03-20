FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11
RUN pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Upgrade typing_extensions so torch>=2.1 can import TypeIs (needs >=4.10.0)
RUN pip install --no-cache-dir "typing_extensions>=4.10.0"

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY dags/ /app/dags/

ENV PYTHONPATH=/app/src
ENV AIRFLOW_HOME=/opt/airflow
