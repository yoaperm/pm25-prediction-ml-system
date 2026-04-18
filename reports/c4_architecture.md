# Part 1: System Architecture Overview — C4 Model

**System:** FoonAlert PM2.5 Prediction System  
**Deployment:** `43.209.207.187` (AWS EC2)  
**Goal:** Predict next-day (24-hour) PM2.5 concentration (µg/m³) for Station 145 — Bangkhuntien, Bangkok

---

## Level 1 — Context Diagram

> System scope, users, and external systems.

```mermaid
C4Context
    title Context Diagram — FoonAlert PM2.5 Prediction System

    Person(user, "End User / Analyst", "Views PM2.5 predictions and monitoring trends via the FoonAlert Dashboard")
    Person(engineer, "ML Engineer / Data Scientist", "Manages training pipelines, monitors model performance, triggers retraining")

    System(system, "FoonAlert PM2.5 Prediction System", "Predicts next-day PM2.5 for Station 145, Bangkok. Includes automated monitoring and retraining.")

    System_Ext(pcd, "Thailand PCD Air Quality API", "กรมควบคุมมลพิษ — Provides hourly PM2.5 readings from 96 monitoring stations across Thailand")
    System_Ext(mlflow_ext, "MLflow Experiment Registry", "Remote artifact store and experiment tracker (hosted within the system boundary)")

    Rel(user, system, "Views predictions & monitoring dashboard", "HTTPS / Browser")
    Rel(engineer, system, "Triggers DAGs, inspects MLflow experiments, reviews metrics", "HTTPS / Airflow UI / API")
    Rel(system, pcd, "Ingests hourly PM2.5 data", "HTTP REST (airbkk_client)")
    Rel(system, mlflow_ext, "Logs model metrics, params, and artifacts", "HTTP REST")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

**Design Decisions:**
- The system is designed as a self-contained ML platform. MLflow is deployed inside the same Docker Compose stack rather than an external SaaS to avoid data egress costs and keep experiment history reproducible.
- Data ingestion is decoupled from prediction via Airflow DAGs, so the external PCD API outage does not block inference.

---

## Level 2 — Container Diagram

> Major system components and technologies.

```mermaid
C4Container
    title Container Diagram — FoonAlert PM2.5 Prediction System

    Person(user, "End User / Analyst")
    Person(engineer, "ML Engineer")

    System_Ext(pcd, "Thailand PCD Air Quality API")

    System_Boundary(sys, "FoonAlert PM2.5 Prediction System — Docker Compose on AWS EC2") {

        Container(streamlit, "Streamlit Dashboard", "Python / Streamlit", "Interactive web UI: submit PM2.5 history, view model comparison, monitor MAE/PSI trends. Port 8501.")
        Container(api, "Prediction API", "Python / FastAPI", "REST API: /predict, /actual, /retrain, /health. Handles inference routing and logs results. Port 8001.")
        Container(triton, "Triton Inference Server", "NVIDIA Triton / ONNX", "Serves ONNX-exported models (Random Forest, XGBoost, LSTM). Primary inference backend. Port 8010.")
        Container(airflow, "Airflow Scheduler + Webserver", "Apache Airflow 2.8 / LocalExecutor", "Orchestrates hourly ingest, daily training, and monitoring/retraining DAGs. Port 8080.")
        Container(mlflow, "MLflow Tracking Server", "MLflow / Gunicorn", "Logs experiment runs, hyperparameters, metrics and model artifacts. Port 5001.")
        Container(postgres, "PostgreSQL Database", "PostgreSQL 15", "Stores Airflow metadata, MLflow backend, and PM2.5 app data (predictions_log, actuals_log). Port 5432.")
        ContainerDb(modelstore, "Model Store", "Filesystem / JSON / ONNX", "Persists trained model files (.onnx), feature column configs, and active model metadata.")
    }

    Rel(user, streamlit, "Views dashboard", "HTTPS :8501")
    Rel(engineer, airflow, "Manages and triggers DAGs", "HTTPS :8080")
    Rel(engineer, mlflow, "Inspects experiments", "HTTPS :5001")

    Rel(streamlit, api, "Calls /predict, /actual", "HTTP :8001")
    Rel(api, triton, "Forwards feature vectors for inference", "HTTP :8010 (tritonclient)")
    Rel(api, modelstore, "Reads feature columns & active model metadata", "Filesystem")
    Rel(api, postgres, "Reads/writes predictions_log & actuals_log", "Port 5432 / SQL")

    Rel(airflow, pcd, "Ingests hourly PM2.5 readings", "HTTP REST")
    Rel(airflow, postgres, "Persists DAG state & task metadata", "SQL")
    Rel(airflow, mlflow, "Logs training runs and metrics", "HTTP :5000")
    Rel(airflow, modelstore, "Saves trained & exported ONNX models", "Filesystem")
    Rel(airflow, triton, "Hot-reloads updated model (poll interval 30 s)", "HTTP polling")

    Rel(mlflow, postgres, "Backend store for runs/experiments", "SQL")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

```mermaid
flowchart TB
    %% =========================================
    %% Container Diagram — FoonAlert PM2.5 Prediction System
    %% =========================================

    %% Actors
    user["👤 End User / Analyst"]
    engineer["👤 ML Engineer"]

    %% External System
    pcd["🌐 Thailand PCD Air Quality API"]

    %% System Boundary
    subgraph sys["FoonAlert PM2.5 Prediction System — Docker Compose on AWS EC2"]

        %% Top Layer (UI)
        streamlit["Streamlit Dashboard<br/>Python / Streamlit<br/>Port 8501"]

        %% Middle Layer (API)
        api["Prediction API<br/>Python / FastAPI<br/>Port 8001"]

        %% ML Layer
        triton["Triton Inference Server<br/>NVIDIA Triton / ONNX<br/>Port 8010"]
        mlflow["MLflow Tracking Server<br/>MLflow / Gunicorn<br/>Port 5001"]

        %% Orchestration
        airflow["Airflow Scheduler + Webserver<br/>Apache Airflow 2.8<br/>Port 8080"]

        %% Data Layer
        postgres[("PostgreSQL Database<br/>Port 5432")]
        modelstore[("Model Store<br/>Filesystem / ONNX")]
    end

    %% =========================
    %% Clean Top-Down Flow
    %% =========================

    user -->|Views dashboard<br/>HTTPS :8501| streamlit
    streamlit -->|Calls /predict, /actual<br/>HTTP :8001| api
    api -->|Inference<br/>HTTP :8010| triton

    %% =========================
    %% Side Connections (kept clean)
    %% =========================

    api -->|Logs / reads| postgres
    api -->|Reads model config| modelstore

    %% =========================
    %% ML / Training Flow
    %% =========================

    engineer -->|Manages DAGs<br/>HTTPS :8080| airflow
    engineer -->|Inspects experiments<br/>HTTPS :5001| mlflow

    airflow -->|Ingest PM2.5| pcd
    airflow -->|Store metadata| postgres
    airflow -->|Log experiments| mlflow
    airflow -->|Save models| modelstore

    mlflow -->|Backend store| postgres
```

**Design Decisions & Trade-offs:**

| Decision | Rationale |
|---|---|
| Triton as primary inference backend | Handles ONNX models uniformly; supports GPU acceleration for LSTM without code changes. Falls back to local ONNX Runtime if Triton is unavailable. |
| LocalExecutor (no Celery/k8s) | Simpler ops for a single-node deployment; sufficient for daily training cadence. |
| Shared PostgreSQL | Reduces infrastructure cost; Airflow, MLflow, and app DB coexist in separate schemas. |
| ONNX export for all models | Decouples training framework (sklearn, XGBoost, PyTorch) from the runtime; Triton can serve all formats via a single interface. |

---

## Level 3 — Component Diagram (ML Pipeline Container)

> Internal structure of the Airflow ML Pipeline.

```mermaid
flowchart LR
    %% ── External Systems (right column) ────────────────────────────
    PCD(["🌐 PCD API\nhourly PM2.5"])
    PG[("🛢 PostgreSQL\npm25_raw\npredictions_log")]
    MLFLOW(["📊 MLflow\nTracking Server"])
    TRITON(["⚡ Triton\nInference :8010"])

    %% ── DAG 1: Hourly Ingest ───────────────────────────────────────
    subgraph INGEST["🕐 DAG 1 — pm25_hourly_ingest  (every hour)"]
        direction LR
        I1["airbkk_client.py\nPoll PCD API"]
        I2["Validate & parse\nhourly PM2.5"]
        I3["INSERT to\npm25_raw table"]
        I1 --> I2 --> I3
    end

    %% ── DAG 2: Training Pipeline ───────────────────────────────────
    subgraph TRAIN["🏋️ DAG 2 — pm25_training_dag  (on-demand)"]
        direction LR
        T1["data_loader.py\nLoad data"]
        T2["preprocessing.py\nffill · clip 0–500"]
        T3["feature_engineering.py\n17 features · shift(1)"]
        T4["train.py\nRidge · RF · XGBoost\nLSTM (skorch)"]
        T5["evaluate.py\nMAE · RMSE · R²"]
        T6{"Champion\nlowest MAE"}
        T7["export_onnx.py\n→ models/onnx/"]
        T1 --> T2 --> T3 --> T4 --> T5 --> T6 --> T7
    end

    %% ── DAG 3: Monitoring & Auto-Retrain ───────────────────────────
    subgraph MONITOR["📈 DAG 3 — pm25_pipeline_dag  (daily 01:00 UTC)"]
        direction LR
        M1["export_data\nJoin pred + actual logs"]
        M2{"check_mae_and_psi\n30-day rolling window\nBranchPythonOperator"}
        M3["✅ done\nhealthy"]
        M4["⚠️ trigger_training\nTriggerDagRunOperator"]
        M1 --> M2
        M2 -- "MAE ≤ 6 & PSI ≤ 0.2" --> M3
        M2 -- "MAE > 6 OR PSI > 0.2" --> M4
    end

    %% ── Data / Trigger Connections ─────────────────────────────────
    PCD -- "GET hourly\nHTTP REST" --> I1
    I3 -- "SQL INSERT" --> PG
    T1 -- "Read\npm25_raw" --> PG
    T5 -- "Log metrics\n& artifacts" --> MLFLOW
    T7 -- "Write ONNX\nhot-reload 30s" --> TRITON
    M1 -- "Read\nlogs" --> PG
    M4 -- "Triggers\nretrain" --> TRAIN

    %% ── Styles ─────────────────────────────────────────────────────
    classDef ext fill:#f0f0f0,stroke:#999,color:#333,font-style:italic
    classDef db  fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
    classDef act fill:#d1fae5,stroke:#059669,color:#064e3b
    classDef dec fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef trg fill:#fce7f3,stroke:#db2777,color:#831843

    class PCD,MLFLOW,TRITON ext
    class PG db
    class I1,I2,I3,T1,T2,T3,T4,T5,T7,M1,M3 act
    class T6,M2 dec
    class M4 trg
```

**Design Decisions:**

- **BranchPythonOperator** is used in `pm25_pipeline_dag` so Airflow skips downstream tasks cleanly when no retrain is needed — avoids false "failed" states.
- `shift(1)` applied to all rolling/lag features ensures no future data leaks into training.
- Champion model selection is metric-driven (lowest MAE on test split); ONNX export uses the same champion path for all algorithm families.

---

## Level 4 — Code-Level Diagram (ML Component)

> Key implementation structures inside the ML component.

### 4a. LSTM Model Architecture (`src/lstm_model.py`)

```mermaid
classDiagram
    class LSTMNet {
        +int input_size = 17
        +LSTM lstm
        +Dropout dropout
        +Linear fc1
        +ReLU relu
        +Linear fc2
        +forward(x: Tensor) Tensor
        %% Input shape: (batch, timesteps=1, features=17)
        %% Output shape: (batch,)
    }

    class NeuralNetRegressor {
        <<skorch wrapper>>
        +module: LSTMNet
        +optimizer: Adam
        +criterion: MSELoss
        +max_epochs: int
        +batch_size: int
        +fit(X, y)
        +predict(X) ndarray
    }

    class RandomizedSearchCV {
        <<sklearn>>
        +estimator: NeuralNetRegressor
        +param_distributions: dict
        +cv: TimeSeriesSplit(n_splits=3)
        +scoring: neg_mean_absolute_error
        +fit(X_3d, y)
    }

    LSTMNet --> NeuralNetRegressor : wrapped by
    NeuralNetRegressor --> RandomizedSearchCV : tuned by
```

### 4b. Training Pipeline Data Flow (`src/train.py` + `src/feature_engineering.py`)

```mermaid
flowchart TD
    A[Raw PM2.5 CSV\nStation 145\n2024-01 → 2026-03] --> B[data_loader.py\nload_station_data]
    B --> C[preprocessing.py\nffill / bfill\nclip 0–500 µg/m³]
    C --> D[feature_engineering.py\nbuild_features]

    subgraph Features["17 Input Features (all shifted by 1 day)"]
        D --> D1[Lag features\npm25_lag_1/2/3/5/7]
        D --> D2[Rolling stats\nmean & std over 3/7/14 days]
        D --> D3[Calendar\nday_of_week, month,\nday_of_year, is_weekend]
        D --> D4[Target\npm25 next day]
    end

    D1 & D2 & D3 --> E[Train / Val / Test Split\nTimeSeriesSplit — no shuffle]

    subgraph Split["Dataset Split"]
        E --> E1["Train\n2024-01 → 2025-09"]
        E --> E2["Validation\n2025-10 → 2025-12"]
        E --> E3["Test\n2026-01 → 2026-03"]
    end

    E1 --> F1[Ridge\nGridSearchCV]
    E1 --> F2[RandomForest\nGridSearchCV]
    E1 --> F3[XGBoost\nGridSearchCV]
    E1 --> F4[LSTM\nRandomizedSearchCV\nskorch + PyTorch]

    F1 & F2 & F3 & F4 --> G[evaluate.py\nMAE / RMSE / R²\non test split]
    G --> H{Champion\nlowest MAE}
    H --> I[export_onnx.py\nCONVERT to ONNX]
    I --> J[models/onnx/lstm.onnx\nactive_model.json]
    J --> K[Triton Model Repository\nhot-reload every 30 s]
    G --> L[MLflow\nlog metrics + artifacts]
```

### 4c. Online Inference Flow (`src/api.py`)

```mermaid
sequenceDiagram
    actor Analyst
    participant Streamlit as Streamlit Dashboard<br/>:8501
    participant API as FastAPI Prediction API<br/>:8001
    participant Triton as Triton Inference Server<br/>:8010
    participant Log as predictions_log.csv

    Analyst->>Streamlit: Submit 15+ daily PM2.5 readings
    Streamlit->>API: POST /predict {history: [...], station_id}
    API->>API: Verify X-API-Key header
    API->>API: build_features() → 17-dim feature vector
    API->>Triton: InferenceRequest (float_input tensor)
    Triton-->>API: InferenceResponse (predicted_pm25)
    API->>Log: Append {date, predicted_pm25, model_version}
    API-->>Streamlit: {predicted_pm25, model_name, features_used}
    Streamlit-->>Analyst: Display prediction + confidence range

    Note over API,Triton: Fallback: if Triton unavailable,<br/>API uses local ONNX Runtime session
```

### 4d. Monitoring & Auto-Retrain Flow (`dags/pm25_pipeline_dag.py`)

```mermaid
flowchart LR
    T[Daily Trigger\n01:00 UTC] --> A[export_data\nJoin predictions_log\n+ actuals_log]
    A --> B{check_mae_and_psi\n30-day rolling window}
    B -- "MAE ≤ 6 µg/m³\nPSI ≤ 0.2\n✅ Healthy" --> C[done\nNo retrain needed]
    B -- "MAE > 6 µg/m³\nOR PSI > 0.2\n⚠️ Degraded" --> D[trigger_training\nTriggerDagRunOperator]
    D --> E[pm25_training_dag\nFull retrain cycle]
    E --> F[New champion ONNX\ndeployed to Triton]
    F --> C
```

---

## Summary of Architecture Decisions

| Concern | Decision | Trade-off |
|---|---|---|
| **Inference latency** | Triton serves ONNX; FastAPI is stateless | Adds one network hop vs in-process inference, but enables horizontal scaling |
| **Model flexibility** | All models exported to ONNX | Loses PyTorch dynamic graph features; static input shape required |
| **Data leakage prevention** | `shift(1)` on all lag/rolling features | Slightly reduces short-term signal but is production-safe |
| **Retraining trigger** | Dual signal: MAE + PSI | PSI catches distribution shift before MAE degrades; may trigger unnecessary retrains |
| **Data split** | Strict temporal (no shuffle) | Realistic evaluation; smaller effective training set vs random split |
| **Auth** | API key header (`X-API-Key`) | Simple to implement; suitable for internal / dashboard-only calls |
