# Progress Report 2: PM2.5 Prediction ML System

**วิชา:** ML Systems
**หัวข้อโปรเจกต์:** PM2.5 Prediction ML System
**วันที่:** มีนาคม 2026

---

## 1. สรุปภาพรวม Report 2

Report 1 ครอบคลุม: dataset, feature engineering, baseline + candidate models, offline evaluation

Report 2 ครอบคลุม:
- **Model Serving** — FastAPI inference service
- **Pipeline Orchestration** — Apache Airflow (Training DAG + Monitoring DAG)
- **Experiment Tracking** — MLflow
- **Model Monitoring** — MAE + PSI (Population Stability Index)
- **Auto-Retraining** — trigger อัตโนมัติเมื่อ performance ตก
- **Infrastructure** — Docker Compose (6 services)
- **ONNX Export** — portable model format

---

## 2. Model Serving — FastAPI Inference Service

### 2.1 สถาปัตยกรรม API (`src/api.py`)

โมเดลที่ดีที่สุด (Random Forest) ถูก deploy เป็น REST API ด้วย **FastAPI + uvicorn** รันบน Docker container พอร์ต 8001

| Endpoint | Method | หน้าที่ |
|----------|--------|---------|
| `/health` | GET | Liveness check |
| `/model/info` | GET | ชื่อโมเดล + feature list |
| `/predict` | POST | ทำนาย PM2.5 วันถัดไป + log ลง `predictions_log.csv` |
| `/actual` | POST | บันทึกค่าจริง + log ลง `actuals_log.csv` |
| `/retrain` | POST | คำนวณ MAE จาก logs + trigger Airflow DAG ถ้า MAE > threshold |

### 2.2 Input / Output ของ `/predict`

**Input** — ต้องส่ง ≥ 15 วันของข้อมูล PM2.5 ย้อนหลัง:
```json
{
  "history": [
    {"date": "2025-06-01", "pm25": 42.1},
    {"date": "2025-06-02", "pm25": 38.5},
    "... (≥15 วัน)"
  ]
}
```

**Output:**
```json
{
  "prediction_date": "2025-06-16",
  "predicted_pm25": 34.21,
  "unit": "µg/m³",
  "model": "random_forest"
}
```

### 2.3 Feature Engineering ใน API

API สร้าง 17 features เดียวกับ training pipeline โดยใช้ `shift(1)` ป้องกัน data leakage และโหลด `models/feature_columns.json` เพื่อให้ feature order ตรงกับตอน train:

```python
for lag in [1, 2, 3, 5, 7]:
    df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
for window in [3, 7, 14]:
    df[f"pm25_rolling_mean_{window}"] = df["pm25"].shift(1).rolling(window).mean()
    df[f"pm25_rolling_std_{window}"]  = df["pm25"].shift(1).rolling(window).std()
df["day_of_week"]       = df["date"].dt.dayofweek
df["month"]             = df["date"].dt.month
df["day_of_year"]       = df["date"].dt.dayofyear
df["is_weekend"]        = (df["day_of_week"] >= 5).astype(int)
df["pm25_diff_1"]       = df["pm25"].shift(1).diff(1)
df["pm25_pct_change_1"] = df["pm25"].shift(1).pct_change(1)
```

### 2.4 Prediction Logging Pattern

ทุก prediction จะถูก log พร้อม timestamp เพื่อใช้ใน monitoring ภายหลัง:

```
วันที่ t:   POST /predict  →  predictions_log.csv  (predicted_pm25, prediction_date, model, created_at)
วันที่ t+1: POST /actual   →  actuals_log.csv       (date, pm25_actual, recorded_at)
monitoring: join ทั้งสอง   →  คำนวณ MAE + PSI บน 30 วันล่าสุด
```

---

## 3. Experiment Tracking — MLflow

### 3.1 การใช้งาน MLflow

ใช้ **MLflow** track ทุก training run โดยบันทึก:
- Hyperparameters ที่ดีที่สุดจาก GridSearchCV
- Metrics (MAE, RMSE, R²) บน test set
- CV results ทั้งหมดเป็น artifact (.csv)

| Environment | MLflow URI |
|-------------|-----------|
| ใน Docker network | `http://mlflow:5000` |
| เข้าถึงจาก host | `http://localhost:5001` |

Backend store: **PostgreSQL** (`mlflow` database ใน shared Postgres container)

### 3.2 ผลลัพธ์ Experiment ล่าสุด

Station **10T** | Train: 2024 (359 วัน) → Test: 2025 (174 วัน)

| Model | MAE ↓ | RMSE ↓ | R² ↑ | Best Parameters |
|-------|--------|--------|------|-----------------|
| Linear Regression (Baseline) | 5.1348 | 6.7493 | 0.7726 | — |
| Ridge Regression | 4.8286 | 6.5294 | 0.7871 | α=100.0 |
| **Random Forest** ⭐ | **4.5494** | **6.5148** | **0.7881** | n_estimators=100, max_depth=10, min_samples_split=2 |
| XGBoost | 4.6853 | 6.7955 | 0.7695 | n_estimators=100, max_depth=3, lr=0.05 |
| LSTM | 6.2750 | 8.0277 | 0.6783 | units=64, dropout=0.1, lr=0.001 |

**โมเดลที่ deploy:** Random Forest (MAE = 4.55 µg/m³)

### 3.3 Hyperparameter Grid ที่ใช้

**Random Forest** (expanded grid):
```yaml
n_estimators: [100, 200, 300, 500]
max_depth: [5, 10, 15, 20, null]
min_samples_split: [2, 5, 10]
min_samples_leaf: [1, 2, 4]
max_features: ["sqrt", "log2", 0.5]
```

**XGBoost** (balanced grid — 72 combinations):
```yaml
n_estimators: [100, 200, 300]
max_depth: [3, 5, 7]
learning_rate: [0.01, 0.05, 0.1]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]
```

ทุกโมเดลใช้ **GridSearchCV + TimeSeriesSplit(n_splits=3)** scoring=`neg_mean_absolute_error`

---

## 4. Pipeline Orchestration — Apache Airflow

### 4.1 DAG 1: `pm25_training_pipeline`

Trigger: **Manual** (ไม่มี schedule — รันเมื่อต้องการ train หรือ retrain)

```
feature_engineering
    ├── train_baseline      (Linear Regression)
    ├── train_ridge         (GridSearchCV)
    ├── train_random_forest (GridSearchCV)
    ├── train_xgboost       (GridSearchCV)
    └── train_lstm          (PyTorch + skorch)
          └── (ทุก train tasks รอกัน) → evaluate → export_onnx
```

**Data Sharing ระหว่าง Tasks:**
- `feature_engineering` เขียน `data/processed/train_features.parquet` และ `test_features.parquet`
- ทุก train task อ่าน parquet เดียวกัน → train อิสระกัน (parallel)
- `evaluate` รวมผล → เขียน `results/experiment_results.csv`
- `export_onnx` export ทุกโมเดลเป็น ONNX

**MLflow Tracking ต่อ Task:**
```python
with mlflow.start_run(run_name="RandomForest"):
    model, best = train_with_tuning(...)
    mlflow.log_params(best)
    mlflow.log_metrics(metrics)
    joblib.dump(model, f"{MODELS_DIR}/random_forest.joblib")
```

### 4.2 DAG 2: `pm25_pipeline`

Schedule: **ทุกวัน 01:00 UTC** (หรือ trigger manual ได้)

```
export_data
    └── check_mae_and_psi  (BranchPythonOperator)
              ├── needs_retrain → pm25_training_pipeline (TriggerDagRunOperator) → clear_logs
              └── healthy       → (ไม่ทำอะไร)
```

**Task รายละเอียด:**

| Task | ประเภท | หน้าที่ |
|------|--------|---------|
| `export_data` | PythonOperator | join predictions_log + actuals_log → นับ matched pairs |
| `check_mae_and_psi` | BranchPythonOperator | คำนวณ MAE + PSI → return branch name |
| `needs_retrain` | TriggerDagRunOperator | trigger pm25_training_pipeline, wait_for_completion=True |
| `clear_logs` | PythonOperator | ลบ predictions_log.csv + actuals_log.csv |
| `healthy` | EmptyOperator | ไม่ทำอะไร |

**XCom keys ที่ push โดย `check_mae_and_psi`:**
```
evaluated_pairs, mae, mae_degraded, psi, psi_status, psi_degraded, needs_retraining
```

### 4.3 Airflow Configuration

```
Executor:      LocalExecutor
Metadata DB:   PostgreSQL (airflow database)
Auth Backend:  airflow.api.auth.backend.basic_auth
DAGs folder:   /app/dags (mounted volume)
```

---

## 5. Model Monitoring

### 5.1 Metrics ที่ใช้ Monitor (`src/monitor.py`)

#### Metric 1: MAE — วัด Prediction Accuracy

```python
mae = round(float(np.mean(np.abs(actual - predicted))), 4)
mae_degraded = mae > mae_threshold   # default: 6.0 µg/m³
```

| MAE | สถานะ |
|-----|-------|
| ≤ 6.0 µg/m³ | Healthy |
| > 6.0 µg/m³ | Degraded → Retrain |

#### Metric 2: PSI — วัด Distribution Shift

Population Stability Index วัดการเปลี่ยนแปลงของ distribution ระหว่าง **predicted** vs **actual** PM2.5

```python
def compute_psi(expected, actual, bins=10):
    breakpoints = np.percentile(np.concatenate([expected, actual]),
                                np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual,   bins=breakpoints)[0] / len(actual)

    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)
```

| PSI | Status | Action |
|-----|--------|--------|
| < 0.1 | Stable | ไม่ทำอะไร |
| 0.1 – 0.2 | Moderate shift | Monitor ต่อ |
| > 0.2 | Significant shift | Retrain |

### 5.2 Rolling Window Evaluation

ใช้เฉพาะ **30 วันล่าสุด** ของ matched pairs:
```python
cutoff = merged["prediction_date"].max() - pd.Timedelta(days=30)
merged = merged[merged["prediction_date"] >= cutoff]
```

ต้องมีอย่างน้อย **7 matched pairs** จึงจะประเมิน — ถ้าน้อยกว่านี้จะ skip

### 5.3 ตัวอย่างผลลัพธ์ Monitoring

**Normal mode** (ข้อมูลปกติ):
```
[Monitor] MAE=2.1042  threshold=6.0  [OK]
[Monitor] PSI=0.0412  threshold=0.2  status=stable  [OK]
[Monitor] needs_retraining=False
```

**Degraded mode** (PM2.5 สูง 2–3× ผิดปกติ):
```
[Monitor] MAE=42.8311  threshold=6.0  [DEGRADED]
[Monitor] PSI=0.8923   threshold=0.2  status=significant  [DEGRADED]
[Monitor] needs_retraining=True
```

ผลลัพธ์ทุก run บันทึกลง `results/monitoring_results.csv` (append mode)

---

## 6. Auto-Retraining Pipeline

### 6.1 Flow การ Retrain อัตโนมัติ

```
1. pm25_pipeline ทำงาน (daily 01:00 UTC)
2. export_data   → join logs, นับ matched pairs
3. check_mae_and_psi → MAE > 6.0 หรือ PSI > 0.2?
   ├── YES → needs_retrain
   │         → TriggerDagRunOperator (wait_for_completion=True)
   │         → รอ pm25_training_pipeline เสร็จ
   │         → clear_logs (ลบ predictions_log.csv + actuals_log.csv)
   └── NO  → healthy (จบ)
```

### 6.2 การแก้ปัญหา Retrain Loop

**ปัญหา:** หลัง retrain แล้ว log ยังมี degraded data → monitoring run ครั้งถัดไปเห็น data เดิม → trigger retrain อีก → loop ไม่หยุด

**วิธีแก้:**
1. เปลี่ยน `wait_for_completion=False` → `True` — รอ training เสร็จก่อน
2. เพิ่ม `clear_logs` task หลัง retrain — ลบ log ทั้งสองไฟล์

```python
def _clear_logs(**context):
    for path in [f"{RESULTS_DIR}/predictions_log.csv",
                 f"{RESULTS_DIR}/actuals_log.csv"]:
        if os.path.exists(path):
            os.remove(path)
```

หลังจากนี้ monitoring run ครั้งถัดไปจะไม่พบ log → skip (not enough pairs) → predictions ใหม่จากโมเดลที่ retrain แล้วจะสะสมใหม่

### 6.3 Trigger Retrain ด้วยตนเอง

```bash
# ผ่าน API endpoint
curl -X POST http://localhost:8001/retrain \
  -H "Content-Type: application/json" \
  -d '{"threshold": 6.0, "min_pairs": 7}'

# ผ่าน Airflow REST API
curl -X POST http://localhost:8080/api/v1/dags/pm25_pipeline/dagRuns \
  -u admin:admin -H "Content-Type: application/json" \
  -d '{"dag_run_id": "manual_monitor_1"}'
```

---

## 7. Infrastructure — Docker Compose

### 7.1 Services

| Service | Image | Port | หน้าที่ |
|---------|-------|------|---------|
| `postgres` | postgres:15 | — | Airflow metadata + MLflow backend store |
| `mlflow` | ghcr.io/mlflow/mlflow:latest | 5001 | Experiment tracking UI + artifact store |
| `airflow-init` | custom Dockerfile | — | DB migrate + สร้าง admin user (รันครั้งเดียว) |
| `airflow-webserver` | custom Dockerfile | 8080 | Airflow UI + REST API |
| `airflow-scheduler` | custom Dockerfile | — | DAG scheduling |
| `api` | custom Dockerfile.api | 8001 | FastAPI inference service |

### 7.2 สองอิมเมจ แยกหน้าที่

| Image | ใช้กับ | Dependencies | ขนาด |
|-------|--------|-------------|------|
| `Dockerfile` | Airflow containers | Python 3.11, Airflow, PyTorch, skorch, XGBoost, MLflow | ~2.8 GB |
| `Dockerfile.api` | API container | FastAPI, uvicorn, scikit-learn, XGBoost, joblib (lightweight) | ~1.2 GB |

### 7.3 Volume Mounts

```yaml
# Airflow (shared ทั้ง webserver + scheduler):
./dags      → /app/dags        # DAG files
./src       → /app/src         # Python modules
./configs   → /app/configs     # config.yaml
./data      → /app/data        # raw + processed data
./models    → /app/models      # saved models
./results   → /app/results     # logs + monitoring results

# API:
./models    → /app/models      # โหลด model + feature_columns.json
./results   → /app/results     # เขียน predictions_log, actuals_log
```

---

## 8. ONNX Export (`src/export_onnx.py`)

โมเดลทุกตัวถูก export เป็น ONNX หลัง training เสร็จ เพื่อ portable inference บน environment ที่ไม่มี Python/scikit-learn:

| Model | ONNX File |
|-------|-----------|
| Linear Regression | `models/onnx/baseline_linear_regression.onnx` |
| Ridge Regression | `models/onnx/ridge_regression.onnx` |
| Random Forest | `models/onnx/random_forest.onnx` |
| XGBoost | `models/onnx/xgboost.onnx` |
| LSTM | `models/onnx/lstm.onnx` |

---

## 9. End-to-End Testing — Mock Pipeline (`scripts/mock_pipeline.py`)

### 9.1 Mock Data Modes

```
normal   → PM2.5 = base + N(0, 2)           → MAE ต่ำ → ไม่ retrain
degraded → PM2.5 = base × U(2.0, 3.0)       → MAE สูง → retrain
drift    → PM2.5 = base + (i/days×35) + N    → PSI สูง → retrain
```

### 9.2 สิ่งที่ Script ทำ (อัตโนมัติ)

```
1. generate_mock(mode, days)         → สร้าง DataFrame
2. for i in range(15, days):
     POST /predict (history=15 วัน)  → ได้ prediction
     POST /actual  (actual_pm25)     → บันทึกค่าจริง
3. PATCH /api/v1/dags/pm25_pipeline  → unpause DAG
4. POST  /api/v1/dags/pm25_pipeline/dagRuns → trigger
5. Poll state ทุก 15 วินาที จนกว่า success/failed
6. อ่าน monitoring_results.csv → print summary
```

---

## 10. ปัญหาที่พบและวิธีแก้

| ปัญหา | สาเหตุ | วิธีแก้ |
|-------|--------|---------|
| Retrain loop ไม่หยุด | Log เก่าไม่ถูกลบหลัง retrain | เพิ่ม `clear_logs` task + `wait_for_completion=True` |
| `predictions_log.csv` ไม่ปรากฏบน host | ไม่ได้ mount `./results` ใน api container | เพิ่ม `- ./results:/app/results` ใน docker-compose |
| Airflow 401 UNAUTHORIZED | REST API basic auth ไม่ได้เปิด | เพิ่ม `AIRFLOW__API__AUTH_BACKENDS: "airflow.api.auth.backend.basic_auth"` |
| MLflow "Invalid Host header" | `--allowed-hosts` ไม่ครอบคลุม localhost:5001 | เพิ่ม `localhost:5001` ใน mlflow command |
| Pydantic field name clash | `date: date` ชนกับ type name | เปลี่ยนเป็น `import datetime` และใช้ `datetime.date` |
| Old DAG crash (`pm25_monitoring_dag.py`) | อ้างอิง key `performance_degraded` ที่ไม่มีแล้ว | ลบ DAG เก่า ใช้เฉพาะ `pm25_pipeline_dag.py` |

---

## 11. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose Stack                        │
│                                                                  │
│  ┌───────────┐    ┌────────────────┐    ┌────────────────────┐  │
│  │ PostgreSQL │    │    MLflow      │    │      Airflow       │  │
│  │  :5432    │◄───│    :5001       │    │      :8080         │  │
│  │           │    │ (artifact store│    │  webserver +       │  │
│  │ airflow DB│    │  + PostgreSQL  │    │  scheduler         │  │
│  │ mlflow DB │    │   backend)     │    └────────┬───────────┘  │
│  └───────────┘    └────────────────┘             │               │
│                                                  │ trigger DAG   │
│  ┌───────────────────────────────────────────────▼───────────┐  │
│  │                  FastAPI API  :8001                        │  │
│  │  POST /predict → build features → predict → log           │  │
│  │  POST /actual  → log actual                               │  │
│  │  POST /retrain → compute MAE → trigger Airflow            │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Airflow DAGs:                                                   │
│  ├── pm25_training_pipeline  (manual trigger)                    │
│  │     feature_eng → [5 models parallel] → evaluate → onnx      │
│  └── pm25_pipeline  (daily 01:00 UTC)                            │
│        export → check_mae_and_psi → retrain/healthy              │
│                       ↓ (if retrain)                             │
│              trigger pm25_training_pipeline                       │
│                       ↓ (after complete)                         │
│                   clear_logs                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ภาคผนวก

### A. วิธี Run ระบบ (Quick Start)

```bash
# 1. Start services
docker compose up -d

# 2. Train initial model (Airflow UI → pm25_training_pipeline → ▶)

# 3. Test end-to-end
python scripts/mock_pipeline.py --mode degraded --days 25

# 4. Check results
cat results/monitoring_results.csv
```

### B. Technology Stack

| Layer | เครื่องมือ | วัตถุประสงค์ |
|-------|-----------|-------------|
| ML | scikit-learn, XGBoost, PyTorch, skorch | Training + tuning |
| Serving | FastAPI, uvicorn, httpx | REST inference API |
| Orchestration | Apache Airflow (LocalExecutor) | DAG scheduling |
| Tracking | MLflow + PostgreSQL | Experiment logging |
| Infrastructure | Docker Compose | Containerized deployment |
| Export | ONNX, onnxruntime | Portable model format |
| Data | pandas, numpy | Data manipulation |
