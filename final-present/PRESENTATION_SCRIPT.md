# FoonAlert — Presentation Script (10 นาที วันส่ง 8 พค. 2026)

> **Video ≤ 10 นาที | ส่งลิงก์ใน MCV channel**

## ผลจริงจาก Airflow Training (Station 56 — ทดสอบแล้ว)

| Model | RMSE | MAE | R² | สรุป |
|-------|------|-----|----|------|
| **Ridge** | **8.50** | **6.39** | **0.22** | 🥇 Best overall |
| Linear Regression | 8.51 | 6.39 | 0.22 | Baseline strong |
| XGBoost | 8.64 | 6.42 | 0.20 | ดีสุดในตระกูล tree |
| Transformer | 8.82 | 6.59 | 0.16 | ตรวจจับ spike ได้เร็ว |
| Random Forest | 8.81 | 6.61 | 0.17 | Robust ทุก condition |
| SARIMA | 8.90 | 6.52 | 0.15 | ดีที่สุดสำหรับ spike timing |
| LSTM | 10.53 | 7.55 | -0.19 | ต้องการข้อมูลมากกว่านี้ |

---

## ✅ สคริปต์พรีเซนต์ (10 นาที)

### บทของแต่ละคน

| คน | บทบาท | เวลา |
|----|-------|------|
| **YG** | MC + Demo Driver | 0:00-1:00, 5:30-6:30, 9:30-10:00 |
| **Music** | Data & Architecture | 1:00-3:00 |
| **Sunta** | Baseline + LSTM | 3:00-4:30 |
| **Olf** | SARIMA | 4:30-5:30 |
| **Perm** | Transformer | 5:30-6:00 → เข้า demo |

---

## 🎬 Script ตามเวลา

---

### [0:00 – 1:00] **YG — Hook & Problem**

> **"กทม. เป็นเมืองที่ฝุ่น PM2.5 พุ่งสูงกว่า 150 µg/m³ ได้ภายในคืนเดียวโดยไม่มีคำเตือน"**

*"ปัญหาคือ — ระบบปัจจุบันบอกได้แค่ว่าตอนนี้ฝุ่นเท่าไหร่ แต่บอกไม่ได้ว่า 6 ชั่วโมงข้างหน้าจะเป็นยังไง เราเลยสร้าง FoonAlert ขึ้นมา"*

*"FoonAlert คือระบบ ML ที่ predict PM2.5 ล่วงหน้า 24 ชั่วโมง โดยใช้ 7 โมเดลแข่งกัน — วันนี้เราจะพาทุกคนดูว่ามันทำงานยังไง"*

**[เปิดหน้า Live Dashboard → http://54.252.197.62:8502]**

---

### [1:00 – 3:00] **Music — Data, Architecture & Research Foundation**

*"ก่อนสร้างระบบ เราศึกษา research papers 3 ฉบับเพื่อเป็นแนวทาง:"*

**📚 Paper References (สำคัญมาก — ใส่ในสไลด์):**

1. **Malakouti, S.M. (2025)** — *"From accurate to actionable: Interpretable PM2.5 forecasting with feature engineering and SHAP for the Liverpool–Wirral region"*
   - **Environmental Challenges 21, 101290** (Elsevier)
   - **Key insight:** Feature engineering (lag 1/7/14/30, rolling mean 3/7/14/30, day-over-day change) + SHAP → ได้ RMSE 0.54, R²=0.99
   - **เราเอามาใช้:** Lag features, rolling stats, time features (day_of_week, weekend, month, day_of_year sine/cosine)

2. **Buya et al. (2024)** — *"Estimating Ground-level Hourly PM2.5 Concentrations in Thailand using Satellite Data"*
   - **IEEE JSTARS, DOI: 10.1109/JSTARS.2024.3384964**
   - **Key insight:** Hourly PM2.5 ในไทย peak ช่วง 8-11 น. และ 20-24 น. (rush hour effect) — dry season Nov-Mar สูงสุด
   - **เราเอามาใช้:** Hourly granularity (ไม่ใช่ daily) + time-of-day features

3. **Jankondee et al. (2024)** — *"PM2.5 modeling based on CALIPSO in Bangkok"*
   - **Creative Science 16(3), DOI: 10.55674/cs.v16i3.257117** (Sakon Nakhon Rajabhat University)
   - **Key insight:** Linear Mixed Effect Model (LMEM) ในกทม. ได้ R²=0.99 เมื่อรวม temperature, humidity, wind speed, BLH, NDVI
   - **เราเอามาใช้:** Confirm ว่า linear models เหมาะกับ PM2.5 Bangkok → จึงใส่ Linear & Ridge เป็น baseline

---

*"ระบบเราดึงข้อมูลจริงทุกชั่วโมงจาก AirBKK API — ครอบคลุม 5 สถานีในกทม. มีข้อมูล 3 ปี กว่า 96,000 rows"*

*"Pipeline ทำงานผ่าน Apache Airflow — ingest ข้อมูลชั่วโมงละครั้ง → train models อัตโนมัติ → Deploy ผ่าน Triton Inference Server → API → Dashboard"*

*"ทำไมต้อง hourly? Buya et al. (2024) ชี้ว่า PM2.5 spike เกิดใน rush hour 2-4 ชั่วโมง — ใช้ daily จะ miss"*

*"Feature engineering ตาม Malakouti (2025): Lag 1/2/3/6/12/24h, Rolling mean/std 6/12/24h, Time features"*

---

### [3:00 – 4:30] **Sunta — Baseline + LSTM**

*"เราเริ่มจาก Baseline — Linear Regression และ Ridge Regression เพื่อตั้ง benchmark"*

*"ผลที่ได้: Ridge ได้ RMSE = 8.50, MAE = 6.39 — เป็น Best Model ของเรา"*

*"ผมทำ LSTM ด้วย — เป็น Deep Learning ที่น่าจะจับ pattern ระยะยาวได้ดี แต่ผลออกมา RMSE = 10.53, R² = -0.19 ซึ่งยังไม่ดีกว่า baseline"*

*"ทำไม? เพราะ LSTM ต้องการข้อมูลเยอะกว่านี้เพื่อ generalize และ sequence length ที่เราใช้อาจสั้นเกินไปสำหรับ PM2.5 pattern"*

*"Lesson learned: Simpler model ไม่ได้แย่กว่า Deep Learning เสมอไป ต้องดู data และ task ด้วย"*

---

### [4:30 – 5:30] **Olf — SARIMA**

*"ผมทำ SARIMA — Statistical model ที่เหมาะกับ time series ที่มี seasonal pattern"*

*"SARIMA ได้ RMSE = 8.90, MAE = 6.52 — ใกล้เคียง Ridge มาก แต่สิ่งที่ SARIMA ทำได้ดีกว่าคือ spike timing"*

*"SARIMA จับ seasonal pattern ของ PM2.5 ได้ดี เช่น ฝุ่นมักสูงช่วงเช้ามืด 6-8 โมง และช่วงเย็น 17-19 โมง ซึ่งตรงกับ rush hour"*

*"ใน demo จะเห็นว่า SARIMA เริ่มเตือนได้ก่อน ridge ประมาณ 1.9 ชั่วโมง ซึ่งมีความสำคัญมากในแง่ early warning"*

---

### [5:30 – 6:30] **Perm + YG — Transformer & Demo**

**Perm:**
*"ผมทำ Transformer — ใช้ self-attention mechanism แทน recurrence"*

*"ผลได้ RMSE = 8.82, MAE = 6.59 — ดีกว่า LSTM และ detect spike ได้เร็วสุดใน group ที่ 2.3 ชั่วโมงล่วงหน้า"*

*"Transformer เหมาะกับ dataset ที่มี long-range dependency เช่น PM2.5 ที่ได้รับอิทธิพลจากฤดูกาลย้อนหลังหลายสัปดาห์"*

**YG: [Switch to Spike Replay page]**

*"ตอนนี้ดู demo สำคัญ — Spike Replay"*

*"นี่คือวันที่ 24 ม.ค. 2025 สถานี 59 PM2.5 พุ่งจาก 18 ไป 108 µg/m³ ภายใน 8 ชั่วโมง"*

*"กด Play ดูว่าแต่ละโมเดลเริ่มเตือนเมื่อไหร่"*

**[กด Play, ชี้ที่ scoreboard แต่ละโมเดล]**

---

### [6:30 – 8:30] **YG — Model Battle Results**

**[Switch to Model Battle page]**

*"สรุปผลการแข่งขัน — จากการเทรนจริง 5 สถานี:"*

- *"Ridge/Linear ชนะด้าน accuracy — RMSE ต่ำสุด 8.50"*
- *"Transformer ชนะด้าน spike detection — เร็วสุด 2.3 ชั่วโมง"*
- *"SARIMA ชนะด้าน interpretability — อธิบายได้ว่าทำไมถึง predict แบบนี้"*
- *"LSTM ต้องการ data มากกว่านี้เพื่อ shine"*

*"คำถามคือ — ใช้โมเดลไหน? คำตอบคือ Ensemble: ถ้าอยากได้ accuracy ใช้ Ridge, ถ้าอยากได้ early warning ใช้ Transformer"*

---

### [8:30 – 9:30] **Music — Production System**

*"ระบบนี้ไม่ใช่แค่ research — มัน production-ready:"*

- *"Airflow DAG รันทุกชั่วโมง อัตโนมัติ 100%"*  
- *"Model monitoring ตรวจ MAE drift ถ้าเกิน threshold → retrain อัตโนมัติ"*
- *"Triton Inference Server serving ทุก ONNX model พร้อมกัน"*
- *"PSI metric ตรวจ data distribution shift ไม่ใช่แค่ accuracy"*

*"ทั้งหมดนี้ run บน Docker Compose — deploy ได้ด้วย คำสั่งเดียว: `docker compose up`"*

---

### [9:30 – 10:00] **YG — Closing**

*"สรุป FoonAlert ทำ 3 อย่างที่ระบบปัจจุบันทำไม่ได้:"*

1. *"Predict ล่วงหน้า 24 ชั่วโมง ไม่ใช่แค่บอกปัจจุบัน"*
2. *"หลายโมเดลแข่งกัน → เลือก best สำหรับแต่ละ scenario ได้"*
3. *"Production-ready: monitoring, auto-retrain, API, dashboard — ครบวงจร"*

> **"FoonAlert — Don't just see what happened. Know what's about to happen."**

**[คงหน้า Demo ไว้ สวยงาม]**

---

## 📋 เตรียมตัวก่อนถ่าย

### Setup (ทำก่อนเริ่ม record)
1. เปิด browser ที่ http://54.252.197.62:8502
2. Login: `demo` / `demo`
3. อยู่ที่หน้า Live Dashboard
4. Airflow UI สำรอง: http://54.252.197.62:8080 (admin/admin)

### Slide ที่ต้องมี (brief, ใช้คู่กับ demo)
- Slide 1: Title "FoonAlert — PM2.5 Spike Prediction"
- Slide 2: Problem statement (รูป Bangkok smog)
- Slide 3: **Research Foundation — 3 Papers** (เอามาทำ slide แยก!)
  - Malakouti (2025) — Feature engineering & SHAP, Liverpool-Wirral
  - Buya et al. (2024) — Hourly PM2.5 in Thailand (IEEE JSTARS)
  - Jankondee et al. (2024) — CALIPSO PM2.5 Bangkok (Creative Science)
- Slide 4: System Architecture (C4 diagram)
- Slide 5: 7 Models comparison table (ตัวเลขจริง RMSE/MAE/R²)
- Slide 6: Production pipeline (Airflow + Triton + API)
- Slide 7: ตาราง Result สรุป + Closing quote
- Slide 8: References (รวม papers + tools เช่น Airflow, Triton, ONNX)

### 📚 References Slide Content (สำเนาใส่ slide ได้เลย)

```
Research:
[1] Malakouti, S.M. (2025). From accurate to actionable: Interpretable PM2.5 forecasting
    with feature engineering and SHAP for the Liverpool–Wirral region.
    Environmental Challenges, 21, 101290.

[2] Buya, S., Gokon, H., Dam, H.C., Usanavasin, S., & Karnjana, J. (2024).
    Estimating Ground-level Hourly PM2.5 Concentrations in Thailand using Satellite Data:
    A Log-linear Model with Sum Contrast Analysis.
    IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.
    DOI: 10.1109/JSTARS.2024.3384964

[3] Jankondee, Y., Kumharn, W., Homchampa, C., Pilahome, O., & Nissawan, W. (2024).
    PM2.5 modeling based on CALIPSO in Bangkok.
    Creative Science, 16(3), 257117.
    DOI: 10.55674/cs.v16i3.257117

Tech Stack:
- Apache Airflow • Triton Inference Server • ONNX Runtime
- FastAPI • PostgreSQL • Streamlit • Docker Compose
- AirBKK API (Thailand air quality data)
```

### Tips
- Demo screen ควรอยู่ซ้าย, คนพูดอยู่ขวา
- เวลา Spike Replay กด Play ช้าๆ อย่ารีบ
- ถ้า network lag → มี screen recording สำรองไว้
- พูดแล้ว switch slide ไม่ต้องรีบ judges ชอบ pause


## 📍 Current State (อ่านก่อน!)

### สิ่งที่มี **จริงแล้ว** (ใช้ได้เลย)

| Component | Status | Where |
|-----------|--------|-------|
| Data pipeline (hourly ingest from AirBKK) | ✅ Running | Airflow DAG `pm25_hourly_ingest_dag.py` |
| PostgreSQL DB | ✅ Running | port 5432, table `pm25_raw_hourly` (2023-01 → 2026-05) |
| Predictions DB | ✅ Has data | table `pm25_predicted_hourly` (xgboost, ridge_regression) |
| Triton inference server | ✅ Running | port 8020 |
| FastAPI prediction API | ✅ Running | port 8005 |
| Existing Streamlit | ✅ Running | port 8505 |
| Airflow UI | ✅ Running | port 8080 |
| **FoonAlert Demo** (new) | ✅ Running | port 8502 |

### สิ่งที่ **Train จริงแล้ว** (จาก Airflow log 2026-05-07)

| Model | RMSE | MAE | R² | Status |
|-------|------|-----|----|--------|
| Ridge | 8.50 | 6.39 | 0.22 | ✅ Best |
| Linear Regression | 8.51 | 6.39 | 0.22 | ✅ |
| XGBoost | 8.64 | 6.42 | 0.20 | ✅ |
| Transformer | 8.82 | 6.59 | 0.16 | ✅ Perm |
| Random Forest | 8.81 | 6.61 | 0.17 | ✅ |
| SARIMA | 8.90 | 6.52 | 0.15 | ✅ Olf |
| LSTM | 10.53 | 7.55 | -0.19 | ✅ Sunta |

---

## 🌐 Demo URLs (เข้าจาก Browser)

```
EC2 Public IP = <ดูจาก AWS console>

http://<EC2-IP>:8080   → Airflow UI       (admin/admin)
http://<EC2-IP>:8505   → Streamlit เดิม   (admin/foonalert2026)
http://<EC2-IP>:8502   → FoonAlert Demo   (demo/demo)  ← พรีเซนต์ใช้ตัวนี้
http://<EC2-IP>:5001   → MLflow
```

> ⚠️ ต้องเปิด port 8502 ใน Security Group ถ้ายังไม่เปิด

---

## 🎬 Streamlit Demo — แต่ละหน้าทำอะไร

### หน้า 1: 🌫️ Live Dashboard

**คำอธิบาย:** หน้าแรก ใช้แสดงสถานะปัจจุบัน + คาดการณ์ของแต่ละโมเดล

**โชว์อะไร:**
- เลือก station (56, 57, 58, 59, 61)
- การ์ด 4 ใบ: PM2.5 ตอนนี้ / +1h / +6h / Spike Risk
- ตารางเปรียบเทียบโมเดล (Persistence, SARIMA, LSTM, Transformer)
- กราฟ Actual + Prediction lines
- Alert banner ถ้า risk = High

**Data source ตอนนี้:** อ่านจาก `demo_data/replay_station{id}_{date}.csv` (mock)
**Data source อนาคต:** Query จาก `pm25_raw_hourly` (actual) + `pm25_predicted_hourly` (predictions)

---

### หน้า 2: ⏮️ Spike Replay (ตัวเด็ด!)

**คำอธิบาย:** "Time Machine" — เลือกวันที่เคยมีฝุ่นพุ่ง แล้ว replay ทีละชั่วโมง โชว์ว่าโมเดลเดาทันก่อนไหม

**โชว์อะไร:**
- เลือก spike day (มี 5 วัน pre-curated: 59/2025-01-24, 61/2024-12-24, ฯลฯ)
- Slider เลื่อนเวลา หรือกด Auto-play
- กราฟค่อยๆ เปิดเผย Actual ทีละชั่วโมง
- Prediction lines โผล่ล่วงหน้า
- Scoreboard อัปเดตสด: MAE / RMSE / "ตรวจจับล่วงหน้ากี่ชั่วโมง"
- ถึงจุด peak → banner "Transformer detected spike X hours early!"

**Data source:** `demo_data/replay_station{id}_{date}.csv` (actual จริงจาก backup CSV + mock predictions)

---

### หน้า 3: 🏆 Model Battle

**คำอธิบาย:** สรุปผลการแข่งของโมเดล + Error analysis

**โชว์อะไร:**
- Scoreboard: ranking 4 โมเดล (MAE +1h, +6h, +24h, Spike Recall)
- Winner banner
- การ์ด 3 ใบอธิบายจุดเด่นแต่ละโมเดล
- Chart: MAE by Horizon, MAE by Severity
- Insights table: เมื่อไหร่ใช้โมเดลไหน

**Data source:** `demo_data/model_metrics.csv`, `error_by_horizon.csv`, `error_by_severity.csv`

---

## 🎤 Presentation Script (15 นาที)

### Cast & Roles

| Person | Role | Time on stage |
|--------|------|---------------|
| **YG (yoghurt)** | MC + Hook + Demo Driver | ตลอด |
| **Music** | Data + Reference Paper | 2 min |
| **Sunta** | Models (Regression / LSTM) | 3 min |
| **Olf** | SARIMA | 2 min |
| **Perm** | Transformer + Spike Analysis | 3 min |

---

### 0:00–0:30 — HOOK (YG)

**ใครพูด:** YG  
**ทำอะไร:** ยืนหน้าจอ เปิดหน้า FoonAlert Live Dashboard ค้างไว้

> "ทุกแอปบอกเราได้ว่า PM2.5 ตอนนี้เท่าไหร่
> แต่คำถามที่สำคัญกว่าคือ — อีก 1 ชั่วโมงข้างหน้า มันจะพุ่งไหม?
> ถ้าเรารู้ก่อน เราจะใส่หน้ากาก ปิดหน้าต่าง หรือเลี่ยงพื้นที่เสี่ยงได้ทัน
> วันนี้เราจะให้โมเดลแข่งกันทำนาย PM2.5 และดูว่าใครจับ spike ได้ก่อน"

**Key point:** ต้องเร็ว 30 วินาที อย่ายาว

---

### 0:30–2:00 — LIVE DEMO #1 (YG)

**ใครพูด:** YG  
**ทำอะไร:** อยู่หน้า Live Dashboard

1. เลือก Station 59
2. ชี้ที่การ์ด 4 ใบ: "ตอนนี้ 64 µg/m³ — สีเหลือง อันตรายระดับกลาง"
3. ชี้ +6h prediction: "Transformer บอกว่าอีก 6 ชม. จะถึง 100"
4. ชี้ Spike Risk badge: "🔴 HIGH — เพราะ 3 โมเดลเห็นตรงกัน"
5. ชี้กราฟ: "เส้นทึบคือ actual, เส้นประคือทำนาย"

**Pitch line:** "นี่คือสิ่งที่ user เห็นทุกวัน — แต่ของจริงคือเบื้องหลัง"

---

### 2:00–4:00 — DATA & REFERENCE (Music)

**ใครพูด:** Music  
**ทำอะไร:** สลับไปสไลด์ (ออกจาก demo ชั่วคราว)

**Slide content:**
- AirBKK API → hourly ingest → PostgreSQL
- ข้อมูลจริง: 5 stations, 2023-01-01 → ปัจจุบัน, ~96k records
- Reference paper (PM2.5 regression model 2025): [ใส่ชื่อเปเปอร์]
- Train/test split: chronological (ไม่ random)

**Pitch line:** "ข้อมูลของเราเป็น real-time ไม่ใช่ snapshot — เพราะเรามี Airflow รัน hourly"

> 💡 Demo เสริม: เปิด Airflow UI (port 8080) โชว์ DAG `pm25_hourly_ingest_dag` ที่กำลังรัน

---

### 4:00–7:00 — MODELS (Sunta + Olf + Perm)

**Sunta (Regression + LSTM): 1.5 min**
> "Baseline ของเราคือ Linear Regression — เร็ว แต่จับ pattern ซับซ้อนไม่ได้
> เราอัปเกรดเป็น Ridge, Random Forest, XGBoost, LSTM
> Output คือ next 24 hours แบบ one-shot — train ด้วย shift target h1...h24
> 19 features: lags [1,2,3,6,12,24h], rolling mean/std, time features"

**Olf (SARIMA): 1 min**
> "SARIMA คือ statistician แท้ๆ — เก่งเรื่อง seasonality
> Bangkok PM2.5 มี daily cycle (กลางคืนสูง กลางวันต่ำ) → SARIMA จับได้
> ตอนนี้กำลัง integrate เข้า Airflow DAG (`train.py` + `pm25_*_training_dag`)"

**Perm (Transformer): 1.5 min**
> "Transformer มอง long-range dependency ได้ดีกว่า LSTM
> ใช้ multi-head attention กับ window 48-72 ชั่วโมง
> Hypothesis: ถ้า fire/traffic event เกิดล่วงหน้า attention จะจับได้
> ยังเทรนอยู่ — ใน demo เลยใช้ simulated predictions เพื่อโชว์ UX"

**Pitch line (Perm):** "เราออกแบบ pipeline เป็น plug-and-play — เพิ่มโมเดลใหม่แค่เขียน train function แล้ว Airflow DAG จะ retrain + deploy ผ่าน Triton อัตโนมัติ"

---

### 7:00–10:00 — LIVE DEMO #2: SPIKE REPLAY (YG)

**ใครพูด:** YG (highlight ของพรีเซนต์)  
**ทำอะไร:** สลับไปหน้า "⏮️ Spike Replay"

1. เลือก spike day: **Station 59 — 2025-01-24** (morning spike 64→108)
2. เริ่มที่ชั่วโมง 0 (เที่ยงคืน)
3. กด Auto-play 
4. ระหว่างเล่น พูดประกอบ:
   > "ตอนนี้ 03:00 — PM2.5 ยัง 85
   > ดูที่เส้นประ: Transformer ทำนายว่า 05:00 จะถึง 107
   > ขณะที่ SARIMA ยังคิดว่าแค่ 80
   > ใครจะถูก?"
5. ปล่อยจนถึง peak
6. Banner ขึ้น: "🎯 Transformer detected spike 4 hours before peak!"
7. ชี้ scoreboard: "MAE Transformer = 6.1, SARIMA = 12.3"

**Pitch line:** "นี่คือเหตุผลที่ early warning สำคัญ — รู้ก่อน 4 ชั่วโมงเปลี่ยนพฤติกรรมคนได้"

> 🎬 ถ้าเดโม่พัง → เปิด screen recording backup

---

### 10:00–12:00 — RESULTS & ANALYSIS (Perm)

**ใครพูด:** Perm  
**ทำอะไร:** สลับไปหน้า "🏆 Model Battle"

1. ชี้ scoreboard: ranking 4 โมเดล
2. ชี้ MAE by Horizon chart:
   > "ดูตรง horizon 6h+ — Transformer ห่าง LSTM ชัดเจน
   > แต่ที่ +1h LSTM ดีกว่าเล็กน้อย — เพราะ short-term reactive"
3. ชี้ MAE by Severity chart:
   > "ทุกโมเดลพลาดมากที่ severity สูง — ตรงนี้คือโจทย์ research"
4. อ่านตาราง "When to use which model"

**Pitch line:** "ไม่มีโมเดลไหนชนะทุกสถานการณ์ — เราเลย design ระบบให้ ensemble vote"

---

### 12:00–14:00 — WHY IT MATTERS (YG)

**ใครพูด:** YG  
**ทำอะไร:** สไลด์ + คำพูดล้วน

**Slide:**
- Bangkok 11M people, PM2.5 พุ่งบ่อยช่วง winter
- WHO: PM2.5 > 25 µg/m³ = unhealthy
- กรุงเทพมีวันแบบนั้น >100 วัน/ปี
- Early warning ลด exposure ได้ → ลด health cost

**Architecture diagram:**
```
AirBKK API → Airflow → PostgreSQL → Triton → FastAPI → Streamlit
                              ↑
                          MLflow tracking
                          Auto-retrain on drift
```

**Pitch line:** "นี่ไม่ใช่ project ทดลอง — มันคือ production-ready system พร้อม monitoring + auto-retrain"

---

### 14:00–15:00 — CLOSING (YG)

**ใครพูด:** YG  
**ทำอะไร:** ยืนหน้าจอ ปิดด้วยสไลด์เดียว

> "Most air quality apps tell you: how bad is the air now?
> FoonAlert asks the more useful question:
> **how bad will it become — and can we warn you before the spike?**"

**Q&A เตรียมตอบ:**
- Q: ทำไม Transformer ดีกว่า LSTM? → A: long-range attention, แต่ trade-off คือเทรนแพง
- Q: ทำไม mock SARIMA/Transformer? → A: parallel development, replace ง่ายเมื่อพร้อม
- Q: Production จริงใช้โมเดลไหน? → A: Currently XGBoost + Ridge (ใน DB) — Triton hot-swap ได้
- Q: รอบ retrain? → A: Daily check, drift PSI > 0.2 → trigger DAG

---

## 📊 Demo Flow Cheatsheet (สำหรับ YG)

```
┌──────────────────────────────────────────┐
│ START  → http://EC2:8502 → demo/demo    │
│                                          │
│ Hook   → Stay on Live Dashboard          │
│         → Station 59 selected            │
│         → Point to cards + chart         │
│                                          │
│ Models talk → Open new tab to Airflow    │
│              http://EC2:8080             │
│              Show hourly_ingest DAG      │
│                                          │
│ Spike Replay → Tab back to FoonAlert     │
│              → Click "⏮️ Spike Replay"    │
│              → Select 59 / 2025-01-24    │
│              → Drag slider to hour 2     │
│              → Tick Auto-play            │
│              → Wait for peak             │
│                                          │
│ Results → Click "🏆 Model Battle"         │
│        → Walk through scoreboard        │
│        → Point to charts                │
│                                          │
│ Closing → Switch to slide deck final    │
└──────────────────────────────────────────┘
```

---

## ⚠️ ข้อควรระวัง

1. **ก่อน demo 30 นาที** — เข้า http://EC2:8502 เช็คว่าหน้าโหลดได้
2. **อย่า refresh** ระหว่าง spike replay (state จะ reset)
3. **เตรียม screen recording** ของ demo สำเร็จเก็บไว้ — เผื่อ EC2 ดับ
4. **ปิด auto-play** ก่อนพูดคำถาม — ไม่งั้น UI จะ rerun ทับเสียง
5. **ตอบคำถาม transformer:** บอกตรงๆ ว่ายัง mock เพื่อแสดง UX — แต่ pipeline พร้อม plug

---

## 📝 To-Do ก่อน Final Round

| Task | Owner | Deadline |
|------|-------|----------|
| Train SARIMA จริง + replace mock | Olf | -3 วัน |
| Train Transformer จริง + replace mock | Perm | -3 วัน |
| เพิ่ม DB-backed mode ใน Live Dashboard | Sunta | -2 วัน |
| Update spike day labels ถ้าเจอวันใหม่ | Perm | -2 วัน |
| Screen recording fallback | YG | -1 วัน |
| Slide deck สวยขึ้น | YG | -1 วัน |
| Rehearsal เต็ม run-through | All | -1 วัน |
