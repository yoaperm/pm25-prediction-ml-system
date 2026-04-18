# C4 Architecture — PM2.5 Prediction ML System

## Overview

ระบบทำนายค่าฝุ่น PM2.5 ล่วงหน้า 24 ชั่วโมง สำหรับ 5 สถานีตรวจวัดในกรุงเทพฯ (Station 56, 57, 58, 59, 61) โดยใช้ข้อมูลรายชั่วโมงจาก AirBKK API ผ่านระบบ ML Pipeline อัตโนมัติ พร้อม auto-retrain เมื่อตรวจพบ drift

---

## Level 1: Context Diagram

แสดงขอบเขตของระบบ ผู้ใช้งาน และระบบภายนอกที่เกี่ยวข้อง

```mermaid
C4Context
    title PM2.5 Prediction ML System — Context Diagram

    Person(user, "End User / Public", "ผู้ใช้งานทั่วไปที่ต้องการ<br/>ดูค่าพยากรณ์ PM2.5")
    Person(mleng, "ML Engineer / Admin", "ดูแลระบบ, ติดตาม model<br/>performance, trigger retrain")

    System(pm25sys, "PM2.5 Prediction System", "ระบบพยากรณ์ค่า PM2.5<br/>ล่วงหน้า 24 ชม.<br/>สำหรับ 5 สถานีในกรุงเทพฯ")

    System_Ext(airbkk, "AirBKK API", "ข้อมูล PM2.5 รายชั่วโมง<br/>จากกรมควบคุมมลพิษ<br/>ประเทศไทย")
    System_Ext(github, "GitHub", "Source code repository<br/>CI/CD via GitHub Actions")
    System_Ext(ec2, "AWS EC2", "Production server<br/>Docker Compose deployment")

    Rel(user, pm25sys, "ดูค่าพยากรณ์ PM2.5", "HTTPS")
    Rel(mleng, pm25sys, "จัดการ pipeline,<br/>ดู experiment results", "HTTPS")
    Rel(pm25sys, airbkk, "ดึงข้อมูล PM2.5<br/>รายชั่วโมง", "HTTP REST API")
    Rel(github, ec2, "Auto-deploy on push<br/>to main branch", "SSH")
    Rel(mleng, github, "Push code changes", "HTTPS")
```

### Design Decisions — Level 1:
- **AirBKK API เป็น single data source** — ใช้ข้อมูลจากแหล่งเดียวที่เป็น official ของไทย ทำให้ข้อมูลมี consistency
- **AWS EC2 deployment** — เลือก EC2 เพราะต้องรัน Triton Inference Server ที่ต้องการ GPU/CPU แรง ไม่เหมาะกับ serverless
- **GitHub Actions CI/CD** — auto-deploy เมื่อ push main ลดขั้นตอน manual deployment

---

## Level 2: Container Diagram

แสดง components หลักของระบบ และ technology ที่ใช้

```mermaid
C4Container
    title PM2.5 Prediction System — Container Diagram

    Person(user, "End User")
    Person(mleng, "ML Engineer")
    System_Ext(airbkk, "AirBKK API")

    Container_Boundary(pm25sys, "PM2.5 Prediction System") {

        Container(streamlit, "Streamlit Dashboard", "Python, Streamlit 1.41", "UI สำหรับ predict,<br/>ดูผลเปรียบเทียบ model,<br/>monitoring dashboard")
        Container(api, "FastAPI Service", "Python, FastAPI 0.115", "REST API สำหรับ inference<br/>/predict, /actual, /retrain<br/>X-API-Key authentication")
        Container(triton, "Triton Inference Server", "NVIDIA Triton 24.08, ONNX Runtime", "High-performance inference<br/>5-10ms latency, auto-reload<br/>model ทุก 30 วินาที")
        Container(airflow, "Apache Airflow", "Airflow 2.10.3", "Orchestrate: ingest, train,<br/>monitor, retrain DAGs")
        Container(mlflow, "MLflow Tracking", "MLflow 2.16", "Log experiments, params,<br/>metrics, artifacts")
        ContainerDb(postgres, "PostgreSQL 15", "Database", "pm25_raw_hourly (96K+ rows)<br/>pm25_api_daily_predictions<br/>Airflow metadata, MLflow backend")
        Container(models, "Model Repository", "File System", "ONNX models, active_model.json<br/>feature_columns.json<br/>triton_model_repo/")
    }

    Rel(user, streamlit, "ดูค่าพยากรณ์", "HTTPS :8501")
    Rel(mleng, airflow, "จัดการ pipeline", "HTTPS :8080")
    Rel(mleng, mlflow, "ดู experiments", "HTTPS :5001")
    Rel(streamlit, api, "เรียก /predict", "HTTP :8000")
    Rel(api, triton, "Inference request", "HTTP/gRPC :8000")
    Rel(api, models, "Load active_model.json<br/>& feature_columns.json", "File I/O")
    Rel(triton, models, "Load ONNX models<br/>poll ทุก 30s", "File I/O")
    Rel(airflow, postgres, "Read/Write hourly data", "SQL :5432")
    Rel(airflow, airbkk, "Fetch hourly PM2.5", "HTTP REST")
    Rel(airflow, mlflow, "Log experiments", "HTTP :5000")
    Rel(airflow, models, "Save ONNX, update<br/>active_model.json", "File I/O")
    Rel(mlflow, postgres, "Store metadata", "SQL :5432")
    Rel(api, postgres, "Read predictions table", "SQL :5432")
```

### Design Decisions — Level 2:
- **Triton Inference Server แยกจาก FastAPI** — Triton จัดการ batching, concurrency, model versioning ได้ดีกว่า onnxruntime ตรงๆ; FastAPI ทำหน้าที่เป็น API gateway + feature engineering
- **ONNX-only deployment** — ทุก model (sklearn, XGBoost, PyTorch LSTM) ถูก export เป็น ONNX ทำให้ inference ไม่ต้องพึ่ง training framework
- **PostgreSQL shared instance** — Airflow metadata, MLflow backend, และ time-series data อยู่ DB เดียวกัน (แยก schema) ลด operational overhead
- **File-based model repository** — ใช้ file system แทน model registry เพราะ Triton poll จาก directory ตรงๆ; active_model.json เป็น pointer ไป ONNX file ปัจจุบัน
- **MLflow สำหรับ experiment tracking** — ไม่ได้ใช้ MLflow Model Registry เพราะใช้ ONNX + Triton path แทน

---

## Level 3: Component Diagram (ML Pipeline Container)

แสดง internal structure ของ ML pipeline ที่ orchestrate โดย Airflow

```mermaid
C4Component
    title ML Pipeline — Component Diagram (Airflow Orchestrated)

    Container_Ext(airbkk, "AirBKK API")
    ContainerDb_Ext(postgres, "PostgreSQL")
    Container_Ext(mlflow, "MLflow")
    Container_Ext(triton_repo, "Triton Model Repo")

    Container_Boundary(ml_pipeline, "ML Pipeline (Airflow DAGs + src/)") {

        Component(ingest, "Hourly Ingest", "pm25_hourly_ingest_dag.py<br/>+ airbkk_client.py + airflow_db.py", "ดึง PM2.5 รายชั่วโมง<br/>จาก AirBKK API<br/>validate & store to DB")

        Component(data_loader, "Data Loader", "src/data_loader.py", "Query data จาก PostgreSQL<br/>หรือ load จาก CSV/Excel<br/>แบ่ง train/val/test")

        Component(preprocess, "Preprocessing", "src/preprocessing.py", "ffill/bfill missing values<br/>clip outliers [0, 500] µg/m³<br/>sort by timestamp")

        Component(feat_eng, "Feature Engineering", "src/feature_engineering.py", "สร้าง 19 features:<br/>Lags [1,2,3,6,12,24h]<br/>Rolling mean/std [6,12,24h]<br/>Diff [1h,24h], Time features<br/>ใช้ shift(1) ป้องกัน leakage")

        Component(trainer, "Model Trainer", "src/train.py + src/lstm_model.py", "Train 5 models:<br/>Linear, Ridge, RF,<br/>XGBoost, LSTM<br/>GridSearchCV + TimeSeriesSplit(3)")

        Component(evaluator, "Evaluator", "src/evaluate.py", "Compute MAE, RMSE, R²<br/>เปรียบเทียบ new vs production<br/>deploy ถ้า MAE ดีกว่า")

        Component(exporter, "ONNX Exporter", "src/export_onnx.py", "Export sklearn → skl2onnx<br/>XGBoost → onnxmltools<br/>LSTM → torch.onnx.export")

        Component(publisher, "Triton Publisher", "src/triton_utils.py", "สร้าง config.pbtxt<br/>copy ONNX → triton_model_repo<br/>Triton auto-reload ทุก 30s")

        Component(monitor, "Drift Monitor", "src/monitor.py<br/>+ src/airflow_monitor.py", "Rolling 14-day RMSE<br/>PSI drift detection<br/>Sensor quality checks<br/>Trigger retrain ถ้า degraded")
    }

    Rel(ingest, airbkk, "Fetch hourly data", "HTTP")
    Rel(ingest, postgres, "INSERT pm25_raw_hourly", "SQL")
    Rel(data_loader, postgres, "SELECT hourly data", "SQL")
    Rel(data_loader, preprocess, "raw DataFrame", "in-memory")
    Rel(preprocess, feat_eng, "cleaned DataFrame", "in-memory")
    Rel(feat_eng, trainer, "X_train, y_train,<br/>X_val, y_val", "Parquet files")
    Rel(trainer, mlflow, "Log params, metrics", "HTTP")
    Rel(trainer, evaluator, "trained models", "in-memory")
    Rel(evaluator, exporter, "best model object", "in-memory")
    Rel(exporter, publisher, "ONNX file path", "File I/O")
    Rel(publisher, triton_repo, "model.onnx +<br/>config.pbtxt", "File I/O")
    Rel(monitor, postgres, "Query predictions<br/>& actuals", "SQL")
    Rel(monitor, trainer, "Trigger retrain<br/>if degraded", "Airflow DAG trigger")
```

### Design Decisions — Level 3:
- **5 competing models** — Linear, Ridge, RF, XGBoost, LSTM ถูก train ทุกครั้ง แล้วเลือก model ที่ MAE ต่ำสุดบน test set; ทำให้ระบบ adaptive ต่อ data pattern ที่เปลี่ยน
- **TimeSeriesSplit(n_splits=3)** — ใช้แทน random k-fold เพื่อรักษาลำดับเวลา ป้องกัน data leakage
- **shift(1) on all features** — critical design choice เพื่อไม่ให้ feature ใช้ข้อมูลของวันที่จะ predict
- **PSI (Population Stability Index)** — ใช้วัด distribution shift ระหว่าง predicted vs actual; PSI > 0.2 = significant drift → trigger retrain
- **ONNX export per model type** — แต่ละ framework มี export path ต่างกัน: sklearn ใช้ skl2onnx, XGBoost ใช้ onnxmltools, LSTM ใช้ torch.onnx.export
- **Versioned ONNX files** — ไม่ลบ model เก่า เก็บ `{model}_{train_start}_{train_end}.onnx` สำหรับ rollback

---

## Level 4: Code-Level Diagram (ML Training & Inference)

### 4a: Training Flow — Class & Function Level

```mermaid
classDiagram
    direction LR

    class DataLoader {
        +load_config(path) dict
        +load_station_data(file, sheet, station_id) DataFrame
        +load_train_test_data(config) tuple~DataFrame~
    }

    class Preprocessing {
        +handle_missing_values(df, method) DataFrame
        +remove_outliers(df, lower, upper) DataFrame
        +preprocess_pipeline(df, fill_method) DataFrame
    }

    class FeatureEngineering {
        +create_lag_features(df, lag_days) DataFrame
        +create_rolling_features(df, windows) DataFrame
        +create_time_features(df) DataFrame
        +create_change_features(df) DataFrame
        +build_features(df, lag_days, windows) DataFrame
        +get_feature_columns(df) list~str~
    }

    class ModelTrainer {
        +train_all_models(config) dict
        -_train_linear(X, y) LinearRegression
        -_train_ridge(X, y, params) Ridge
        -_train_rf(X, y, params) RandomForest
        -_train_xgb(X, y, params) XGBRegressor
        -_train_lstm(X, y, params) NeuralNetRegressor
    }

    class LSTMNet {
        -lstm: LSTM(input=19, hidden=units)
        -dropout: Dropout(p)
        -fc1: Linear(units, 32)
        -fc2: Linear(32, 1)
        +forward(x) Tensor
    }

    class Evaluator {
        +evaluate_model(y_true, y_pred) dict
        +print_metrics(name, metrics) void
        +compare_models(path) DataFrame
    }

    class OnnxExporter {
        +export_sklearn(model, name, dir, n_feat) str
        +export_xgboost(model, dir, n_feat) str
        +export_lstm(model, dir, n_feat) str
    }

    class TritonPublisher {
        +publish_to_triton(onnx_path, repo, is_lstm) void
        -_generate_config_pbtxt(model_name, inputs, outputs) str
    }

    class DriftMonitor {
        +compute_psi(expected, actual, bins) float
        +psi_status(psi) str
        +run_monitoring(config) dict
    }

    DataLoader --> Preprocessing : raw data
    Preprocessing --> FeatureEngineering : cleaned data
    FeatureEngineering --> ModelTrainer : X, y matrices
    ModelTrainer --> LSTMNet : LSTM training
    ModelTrainer --> Evaluator : trained models
    Evaluator --> OnnxExporter : best model
    OnnxExporter --> TritonPublisher : ONNX file
    DriftMonitor --> ModelTrainer : trigger retrain
```

### 4b: Inference Flow — Sequence Diagram

```mermaid
sequenceDiagram
    actor User
    participant Streamlit as Streamlit Dashboard
    participant API as FastAPI Service
    participant FeatEng as Feature Engineering
    participant Triton as Triton Server
    participant ONNX as ONNX Model
    participant DB as PostgreSQL
    participant Log as CSV Logs

    User->>Streamlit: เปิด Dashboard, Login
    Streamlit->>Streamlit: Session-based Auth

    User->>Streamlit: กรอก PM2.5 history (≥15 วัน)
    Streamlit->>API: POST /predict {history: [...]}
    
    Note over API: Validate X-API-Key header

    API->>FeatEng: Build features from history
    Note over FeatEng: Lags [1,2,3,5,7]<br/>Rolling mean/std [3,7,14]<br/>Time features<br/>shift(1) applied

    FeatEng-->>API: Feature vector (1 × 17)

    alt INFERENCE_BACKEND = triton
        API->>Triton: HTTP inference request
        Triton->>ONNX: Run ONNX model
        ONNX-->>Triton: prediction float
        Triton-->>API: prediction result
    else INFERENCE_BACKEND = onnxruntime
        API->>ONNX: onnxruntime.InferenceSession
        ONNX-->>API: prediction float
    end

    API->>Log: Append predictions_log.csv
    API-->>Streamlit: {predicted_pm25, model, date}
    Streamlit-->>User: แสดงผลค่าพยากรณ์ + Air Quality Level
```

### 4c: Auto-Retrain Flow — Sequence Diagram

```mermaid
sequenceDiagram
    participant Cron as Airflow Scheduler
    participant Monitor as Drift Monitor
    participant DB as PostgreSQL
    participant TrainDAG as Training DAG
    participant Models as 5 Models
    participant Eval as Evaluator
    participant ONNX as ONNX Exporter
    participant Triton as Triton Server
    participant MLflow as MLflow

    Cron->>Monitor: Daily check (01:00 UTC)
    Monitor->>DB: Query predictions + actuals<br/>(rolling 14-day window)
    
    Monitor->>Monitor: Compute RMSE
    Monitor->>Monitor: Compute PSI (10 bins)

    alt RMSE > 13.0 OR PSI > 0.2
        Monitor->>TrainDAG: Trigger pm25_24h_training

        TrainDAG->>DB: Load 3.5 years hourly data
        TrainDAG->>TrainDAG: Feature Engineering (19 features)
        
        par Train all 5 models
            TrainDAG->>Models: Linear Regression
            TrainDAG->>Models: Ridge (GridSearchCV)
            TrainDAG->>Models: Random Forest (GridSearchCV)
            TrainDAG->>Models: XGBoost (GridSearchCV)
            TrainDAG->>Models: LSTM (RandomizedSearchCV)
        end

        Models-->>Eval: 5 trained models
        Eval->>Eval: Evaluate on test set (MAE, RMSE, R²)
        Eval->>Eval: Compare with production model

        alt new_MAE < production_MAE
            Eval->>ONNX: Export best model
            ONNX->>Triton: Publish to triton_model_repo
            Note over Triton: Auto-reload within 30s
            Eval->>MLflow: Log experiment + metrics
            Eval->>DB: Update active_model.json
        else new_MAE >= production_MAE
            Note over Eval: Keep production model<br/>Log comparison result
        end
    else Performance OK
        Note over Monitor: No action needed<br/>Log monitoring result
    end
```

### 4d: Data Ingestion Flow

```mermaid
sequenceDiagram
    participant Cron as Airflow (Hourly)
    participant Client as AirBKK Client
    participant AirBKK as AirBKK API
    participant Validator as Data Validator
    participant DB as PostgreSQL
    participant QualMon as Quality Monitor
    participant Log as Metrics CSV

    Cron->>Client: Trigger fetch (every hour)
    Client->>AirBKK: GET measurements<br/>stations: 56,57,58,59,61
    
    Note over Client: Convert Thai Buddhist year<br/>(2569 → 2026)
    
    AirBKK-->>Client: PM2.5, PM10, Temp, RH, WS, WD

    Client->>Validator: Raw records
    Validator->>Validator: Range checks:<br/>PM2.5 ∈ [0, 500]<br/>RH ∈ [0, 100]<br/>WS ≥ 0
    
    Validator->>DB: INSERT pm25_raw_hourly<br/>(ON CONFLICT DO NOTHING)
    Note over DB: UNIQUE(station_id, timestamp)<br/>prevents duplicates

    DB-->>Validator: inserted_count, duplicate_count

    Validator->>QualMon: Check data quality
    QualMon->>DB: Query recent 24h data
    QualMon->>QualMon: Check null_rate (alert if >50%)<br/>Check outliers (alert if >10%)<br/>Detect sensor drift (1h vs 7d baseline)
    QualMon->>Log: Append hourly_ingestion_metrics.csv
```

### 4e: System Deployment Architecture

```mermaid
graph TB
    subgraph "AWS EC2 Instance (Production)"
        subgraph "Docker Compose Stack"
            subgraph "Data Layer"
                PG[(PostgreSQL 15<br/>Port 5432)]
            end

            subgraph "Orchestration Layer"
                AF_WEB[Airflow Webserver<br/>Port 8080]
                AF_SCH[Airflow Scheduler]
                MLF[MLflow Server<br/>Port 5001]
            end

            subgraph "ML Inference Layer"
                TRITON[Triton Server<br/>Port 8010/8011]
                API[FastAPI Service<br/>Port 8001]
            end

            subgraph "Presentation Layer"
                ST[Streamlit Dashboard<br/>Port 8501]
            end

            subgraph "Shared Volumes"
                MODELS[/models/<br/>ONNX files/]
                TRITON_REPO[/triton_model_repo/]
                RESULTS[/results/<br/>CSV logs/]
                DATA[/data/<br/>raw data/]
            end
        end
    end

    subgraph "External"
        GH[GitHub<br/>CI/CD]
        AIRBKK[AirBKK API]
        USERS[End Users]
    end

    USERS -->|HTTPS| ST
    USERS -->|HTTPS| API
    GH -->|SSH deploy| AF_SCH
    AIRBKK -->|HTTP| AF_SCH

    ST -->|HTTP| API
    API -->|gRPC/HTTP| TRITON
    API -->|read| MODELS
    TRITON -->|read| TRITON_REPO
    AF_SCH -->|SQL| PG
    AF_SCH -->|HTTP| MLF
    AF_SCH -->|write| MODELS
    AF_SCH -->|write| TRITON_REPO
    MLF -->|SQL| PG
    API -->|write| RESULTS

    style PG fill:#336791,color:#fff
    style TRITON fill:#76b900,color:#fff
    style API fill:#009688,color:#fff
    style ST fill:#FF4B4B,color:#fff
    style AF_WEB fill:#017CEE,color:#fff
    style AF_SCH fill:#017CEE,color:#fff
    style MLF fill:#0194E2,color:#fff
```

---

## Trade-offs Summary

| Decision | Benefit | Trade-off |
|----------|---------|-----------|
| ONNX-only inference | Framework-agnostic, fast, portable | ต้อง export ทุก model type ต่างกัน |
| Triton Server | Low latency (5-10ms), auto-batching | เพิ่ม complexity, ต้องจัดการ model repo |
| 5 models compete | Adaptive — เลือก model ที่ดีที่สุดตาม data | Training ช้ากว่า (15-20 ชม. ต่อ 5 สถานี) |
| PostgreSQL shared | ลด ops overhead, single backup point | ถ้า DB ล่ม ทั้งระบบหยุด |
| shift(1) on features | ป้องกัน data leakage 100% | สูญเสีย 1 row ต่อ feature |
| PSI + MAE monitoring | จับได้ทั้ง distribution drift และ accuracy drop | ต้องมี actual data มา match |
| Docker Compose | Deploy ง่าย, reproducible | ไม่ scale horizontally เหมือน K8s |
| File-based model versioning | เรียบง่าย, Triton poll ตรงๆ | ไม่มี model registry UI |
