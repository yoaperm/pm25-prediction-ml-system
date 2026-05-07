# FoonAlert — Presentation Plan & Demo Script

> **Tournament round — must impress judges. Hook fast, demo live, finish strong.**

---

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

### สิ่งที่ **ยังเป็น Mock** ในเดโม่

| Item | Status | Owner |
|------|--------|-------|
| SARIMA model | 🚧 Mock predictions in CSV | Olf |
| Transformer model | 🚧 Mock predictions in CSV | Perm |
| Real LSTM/Ridge predictions for spike days | 🚧 ใช้ mock อยู่ | Sunta (ดึงจาก DB ใน demo) |

**ทำไมเป็น mock?** เพราะวันที่ทำ demo ยังไม่ได้ train SARIMA/Transformer แต่ต้องการ UI พร้อมก่อน เพื่อแบ่งงาน parallel กัน เมื่อ Olf/Perm train เสร็จ → แค่แทนค่าใน CSV ที่ `demo_data/` UI ไม่ต้องแก้

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
