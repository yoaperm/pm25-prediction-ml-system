FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgomp1 gcc curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG INSTALL_ML_DEPS=true

COPY requirements-base.txt requirements-ml.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-base.txt
RUN if [ "$INSTALL_ML_DEPS" = "true" ]; then \
      pip install --no-cache-dir -r requirements-ml.txt; \
    fi

ARG AIRFLOW_VERSION=2.8.1
ARG PYTHON_VERSION=3.11
# RUN pip install --no-cache-dir \
#     "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
#     --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
RUN PYTHON_SHORT_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    pip install --no-cache-dir \
    "apache-airflow[postgres]==${AIRFLOW_VERSION}" \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"

# Upgrade typing_extensions only when the optional torch stack is installed.
RUN if [ "$INSTALL_ML_DEPS" = "true" ]; then \
      pip install --no-cache-dir "typing_extensions>=4.10.0"; \
    fi

COPY src/ /app/src/
COPY configs/ /app/configs/
COPY dags/ /app/dags/

ENV PYTHONPATH=/app/src
ENV AIRFLOW_HOME=/opt/airflow
