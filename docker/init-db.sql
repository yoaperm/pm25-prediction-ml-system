CREATE USER airflow WITH PASSWORD 'airflow';
CREATE DATABASE airflow OWNER airflow;
CREATE USER mlflow WITH PASSWORD 'mlflow';
CREATE DATABASE mlflow OWNER mlflow;
CREATE DATABASE pm25 OWNER postgres;
ALTER DATABASE pm25 SET timezone TO 'Asia/Bangkok';
