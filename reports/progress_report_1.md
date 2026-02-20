# Progress Report 1: PM2.5 Prediction ML System

**วิชา:** ML Systems  
**หัวข้อโปรเจกต์:** PM2.5 Prediction ML System  
**วันที่:** กุมภาพันธ์ 2026

---

## 1. Project Overview

### 1.1 สรุปปัญหา

ปัญหามลพิษทางอากาศจากฝุ่นละอองขนาดเล็ก PM2.5 (Particulate Matter ≤ 2.5 µm) เป็นปัญหาสำคัญด้านสาธารณสุขของประเทศไทย โดยเฉพาะในช่วงฤดูแล้ง (พฤศจิกายน–เมษายน) ที่ค่า PM2.5 มักสูงเกินมาตรฐาน การทำนายค่า PM2.5 ล่วงหน้าสามารถช่วยให้หน่วยงานที่เกี่ยวข้องและประชาชนเตรียมรับมือได้อย่างทันท่วงที

### 1.2 เป้าหมายของระบบ ML

สร้างระบบ Machine Learning สำหรับทำนายค่าความเข้มข้นของ PM2.5 (หน่วย µg/m³) รายวัน โดยใช้ข้อมูลย้อนหลังจากสถานีตรวจวัดคุณภาพอากาศ

**ลักษณะงาน:** Regression (Time-Series) — ทำนายค่า PM2.5 ต่อเนื่องจากข้อมูลในอดีต  
**สถานีที่เลือกศึกษา:** สถานี 10T — เคหะชุมชนคลองจั่น, เขตบางกะปิ, กทม.

---

## 2. Dataset & Data Description

### 2.1 แหล่งที่มาของข้อมูล

ข้อมูลได้มาจาก **กรมควบคุมมลพิษ (Pollution Control Department)** กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม ซึ่งเป็นข้อมูลการตรวจวัดคุณภาพอากาศสาธารณะ (public environmental monitoring data)

### 2.2 ประเภทและขนาดข้อมูล

ข้อมูลอยู่ในรูปแบบ Excel (.xlsx) จำนวน 2 ไฟล์:

| ไฟล์ | ช่วงเวลา | จำนวนแถว | จำนวนสถานี | ขนาดไฟล์ |
|------|----------|----------|------------|----------|
| PM2.5(2024).xlsx | ม.ค. – ธ.ค. 2024 | 366 วัน | 96 สถานี | 248 KB |
| PM2.5(2025).xlsx | ม.ค. – มิ.ย. 2025 | 181 วัน | 100 สถานี | 129 KB |

**โครงสร้างข้อมูลแต่ละไฟล์:**

- **Sheet "Data":** แถวแรก = หัวตาราง (Date, 02T, 05T, 10T, ...), แต่ละแถว = ค่า PM2.5 รายวันต่อสถานี
- **Sheet "รายละเอียดจุดตรวจวัด":** ข้อมูล metadata ของสถานี ได้แก่ ลำดับ, รหัสสถานี, ชื่อสถานี, รายละเอียดจุดติดตั้ง

**สถานีที่เลือก (10T):**

| รายละเอียด | ข้อมูล |
|-----------|--------|
| รหัสสถานี | 10T |
| ชื่อสถานี | แขวงคลองจั่น เขตบางกะปิ กทม. |
| จุดติดตั้ง | เคหะชุมชนคลองจั่น |
| ข้อมูล 2024 | 355 valid / 11 missing (จาก 366 วัน) |
| ข้อมูล 2025 | 181 valid / 0 missing |
| ค่าเฉลี่ย 2024 | 20.0 µg/m³ (SD = 11.8) |
| ค่าเฉลี่ย 2025 | 25.2 µg/m³ (SD = 14.2) |

### 2.3 วิธีการแบ่งข้อมูล (Train / Validation / Test)

เนื่องจากข้อมูลเป็น time-series จึงใช้ **temporal split** แทน random split เพื่อป้องกัน data leakage:

| ชุดข้อมูล | ที่มา | ช่วงเวลา | จำนวนวัน |
|----------|-------|----------|----------|
| **Training** | PM2.5(2024).xlsx | ม.ค. – ต.ค. 2024 | ~298 วัน |
| **Validation** | PM2.5(2024).xlsx | พ.ย. – ธ.ค. 2024 | ~61 วัน |
| **Test** | PM2.5(2025).xlsx | ม.ค. – มิ.ย. 2025 | 174 วัน* |

*\*หลัง feature engineering (ตัด NaN จาก lag features แล้ว)*

> **หมายเหตุ:** ใช้ TimeSeriesSplit (n_splits=3) สำหรับ cross-validation ภายใน training set เพื่อ hyperparameter tuning ซึ่งจะ respect ลำดับเวลาของข้อมูล

### 2.4 Privacy, PDPA, and Ethical Considerations

**ข้อมูลส่วนบุคคล:**
- ข้อมูลชุดนี้ **ไม่มีข้อมูลส่วนบุคคล** (Personally Identifiable Information) แต่อย่างใด เป็นข้อมูลการตรวจวัดคุณภาพอากาศสาธารณะ ไม่มีข้อมูลชื่อ ที่อยู่ หรือรายละเอียดที่สามารถระบุตัวบุคคลได้

**ข้อมูลอ่อนไหว:**
- ไม่มีข้อมูลอ่อนไหว (sensitive data) — เป็นข้อมูลสิ่งแวดล้อม (environmental data) ที่กรมควบคุมมลพิษเผยแพร่สู่สาธารณะ

**แนวทาง Privacy / การใช้ข้อมูลอย่างรับผิดชอบ:**
- อ้างอิงแหล่งที่มาข้อมูลอย่างชัดเจน (กรมควบคุมมลพิษ)
- ไม่ใช้ข้อมูลเชิงพาณิชย์โดยไม่ได้รับอนุญาต
- ข้อมูลใช้เพื่อการศึกษาและวิจัยเท่านั้น

**ผลกระทบที่อาจเกิดขึ้นหากระบบทำนายผิด:**
- **False Low Prediction:** หากระบบทำนายค่า PM2.5 ต่ำกว่าความเป็นจริง ประชาชนอาจไม่ป้องกันตนเอง (เช่น ไม่สวมหน้ากาก) ในวันที่อากาศจริงๆ แย่ → เสี่ยงต่อสุขภาพ
- **False High Prediction:** หากทำนายสูงกว่าจริง อาจทำให้เกิดความตื่นตระหนกโดยไม่จำเป็น หรือส่งผลกระทบต่อกิจกรรมทางเศรษฐกิจ
- ระบบนี้ควรใช้เป็น **เครื่องมือสนับสนุนการตัดสินใจ** (decision support) ไม่ใช่แทนที่การตรวจวัดจริง

---

## 3. Data Engineering & Training Data Considerations

### 3.1 แนวทางเตรียมข้อมูลและจัดการ Missing/Noisy Data

**Missing Values:**
- ข้อมูลดิบมี missing values (NaN) จำนวน 11 จุดในสถานี 10T (ปี 2024)
- ใช้ **Forward-Fill (ffill)** เป็นวิธีหลัก — เหมาะกับ time-series เพราะค่า PM2.5 วันก่อนหน้ามักใกล้เคียงกับวันถัดไป
- ตามด้วย **Backward-Fill (bfill)** สำหรับ NaN ที่อยู่ต้นชุดข้อมูล
- ไม่ใช้ mean imputation เพราะจะทำลาย temporal pattern

**Outlier Handling:**
- กำหนดขอบเขตที่สมเหตุสมผลทางกายภาพ: PM2.5 ∈ [0, 500] µg/m³
- ค่าติดลบหรือค่าสูงกว่า 500 ถูกตัดออก (sensor error)
- ในชุดข้อมูลจริงของสถานี 10T ไม่พบ outliers หลังการ preprocessing

**Data Cleaning:**
- ตัดแถวท้ายไฟล์ที่ไม่ใช่ข้อมูล (เช่น แถว "หมายเหตุ : N/A ไม่มีข้อมูล")
- แปลงคอลัมน์ Date เป็น datetime, PM2.5 เป็น float

### 3.2 Class Imbalance / Sampling Strategy

เนื่องจากงานนี้เป็น **regression** (ไม่ใช่ classification) จึงไม่มีปัญหา class imbalance โดยตรง อย่างไรก็ตาม ค่า PM2.5 มี **right-skewed distribution** (ค่าสูงมากมีน้อย) ซึ่งอาจทำให้โมเดลทำนายค่าสูงได้ไม่ดี ข้อนี้สังเกตได้จากการที่ RMSE > MAE ค่อนข้างมาก (ลงโทษ large errors)

### 3.3 แหล่งที่มาของ Label และความเสี่ยงของ Label Noise

- **Label (Target Variable):** ค่า PM2.5 รายวันจากเซ็นเซอร์ตรวจวัด ณ สถานี → เป็น continuous value (µg/m³)
- **Label Noise Risk:** เป็นไปได้ที่เซ็นเซอร์อาจมีความคลาดเคลื่อน (sensor drift, calibration error) แต่เนื่องจากเป็นข้อมูลจากกรมควบคุมมลพิษ ซึ่งมีมาตรฐานการดูแลรักษาเครื่องมือ ความเสี่ยงจึงอยู่ในระดับต่ำ

### 3.4 สมมติฐานสำคัญเกี่ยวกับข้อมูล

1. **Temporal Autocorrelation:** ค่า PM2.5 ของวันนี้มีความสัมพันธ์กับค่าในวันก่อนหน้า (lag dependency)
2. **Seasonality:** ค่า PM2.5 มี pattern ตามฤดูกาล (สูงในฤดูแล้ง, ต่ำในฤดูฝน)
3. **Stationarity (Local):** แม้ข้อมูลอาจไม่ stationary ทั้งหมด แต่ในช่วงสั้น (1-2 สัปดาห์) มี pattern ที่ค่อนข้างสม่ำเสมอ
4. **No Major Structural Break:** สมมติว่าไม่มีเหตุการณ์ผิดปกติขนาดใหญ่ (เช่น ไฟป่าใหญ่) ที่ทำให้ pattern เปลี่ยนแปลงฉับพลัน

---

## 4. Feature Engineering Plan

### 4.1 Feature หลักที่ใช้ (17 features)

| ประเภท | Features | จำนวน | เหตุผล |
|--------|----------|-------|--------|
| **Lag Features** | `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`, `pm25_lag_5`, `pm25_lag_7` | 5 | จับ temporal autocorrelation — ค่า PM2.5 วันก่อนหน้าเป็น predictor ที่แข็งแกร่งที่สุด |
| **Rolling Mean** | `pm25_rolling_mean_3`, `pm25_rolling_mean_7`, `pm25_rolling_mean_14` | 3 | จับ short/medium-term trend — ค่าเฉลี่ยเคลื่อนที่สะท้อนทิศทางโดยรวม |
| **Rolling Std** | `pm25_rolling_std_3`, `pm25_rolling_std_7`, `pm25_rolling_std_14` | 3 | จับความผันผวน — ช่วงที่มี volatility สูง มักบ่งชี้การเปลี่ยนแปลงสภาพอากาศ |
| **Time Features** | `day_of_week`, `month`, `day_of_year`, `is_weekend` | 4 | จับ seasonality และ weekly pattern (เช่น traffic pattern ต่างกัน weekday/weekend) |
| **Change Features** | `pm25_diff_1`, `pm25_pct_change_1` | 2 | จับ momentum — อัตราการเปลี่ยนแปลงบอก direction of trend |

> **การป้องกัน Data Leakage:** lag features และ rolling statistics ใช้ `shift(1)` ก่อนคำนวณ เพื่อให้มั่นใจว่าไม่ใช้ข้อมูลจากวันปัจจุบัน (ซึ่งเป็น target) ในการสร้าง feature

### 4.2 Features ที่ตั้งใจไม่ใช้

| Feature ไม่ใช้ | เหตุผล |
|----------------|--------|
| ข้อมูลจากสถานีอื่น | เพื่อความเรียบง่ายในการทดลอง — สามารถเพิ่มในอนาคตเป็น spatial features |
| ข้อมูลสภาพอากาศ (อุณหภูมิ, ลม, ฝน) | ไม่มีอยู่ใน dataset — สามารถ integrate จาก external sources ในอนาคต |
| Holiday flag | ข้อมูลวันหยุดไม่มีอยู่ใน dataset (ใช้ `is_weekend` แทน) |

---

## 5. Baseline Model

### 5.1 Baseline ที่เลือก

**Linear Regression** (Ordinary Least Squares)

### 5.2 เหตุผลที่เลือก

- เป็นโมเดลที่ง่ายที่สุดสำหรับ regression — เหมาะเป็น baseline
- Interpretable — สามารถดู coefficients เพื่อเข้าใจความสัมพันธ์ได้
- ไม่มี hyperparameters ที่ต้อง tune → ผลลัพธ์ reproducible
- เป็น benchmark สำหรับวัดว่า candidate models ดีขึ้นจริงหรือไม่

### 5.3 ผลการทดลอง Baseline

| Metric | ค่า | ความหมาย |
|--------|-----|---------|
| **MAE** | 5.1348 | ทำนายผิดเฉลี่ย ~5.13 µg/m³ |
| **RMSE** | 6.7493 | ค่า error ใหญ่ถูกลงโทษมากขึ้น (penalizes large errors) |
| **R²** | 0.7726 | อธิบาย variance ของข้อมูลได้ ~77.3% |

---

## 6. Candidate Models (3 Models)

### 6.1 Ridge Regression

| รายละเอียด | ข้อมูล |
|-----------|--------|
| **ชื่อโมเดล** | Ridge Regression (L2 Regularization) |
| **เหตุผลที่เลือก** | เพิ่ม regularization บน Linear Regression เพื่อลด overfitting จาก multicollinearity ของ lag features ที่มีความสัมพันธ์กันสูง |
| **ความแตกต่างจาก baseline** | เพิ่ม L2 penalty term (α · Σw²) บน coefficients → ทำให้ weights ไม่สุดโต่ง, generalize ดีกว่า |
| **Best Hyperparameter** | α = 100.0 |

### 6.2 Random Forest Regressor

| รายละเอียด | ข้อมูล |
|-----------|--------|
| **ชื่อโมเดล** | Random Forest Regressor |
| **เหตุผลที่เลือก** | จับ non-linear relationships ที่ Linear models ไม่สามารถจับได้, robust ต่อ outliers, ไม่ต้อง feature scaling |
| **ความแตกต่างจาก baseline** | เป็น ensemble of decision trees ใช้ bagging → ลด variance, สามารถจับ complex interactions ระหว่าง features |
| **Best Hyperparameters** | n_estimators=50, max_depth=5, min_samples_split=10 |

### 6.3 XGBoost Regressor

| รายละเอียด | ข้อมูล |
|-----------|--------|
| **ชื่อโมเดล** | XGBoost (Extreme Gradient Boosting) |
| **เหตุผลที่เลือก** | State-of-the-art สำหรับ tabular data, มักให้ผลดีที่สุดใน competitions, มี built-in regularization |
| **ความแตกต่างจาก baseline** | ใช้ gradient boosting — สร้าง trees ลำดับต่อเนื่อง โดยแต่ละ tree แก้ไข errors ของ tree ก่อนหน้า → จับ residual patterns |
| **Best Hyperparameters** | n_estimators=50, max_depth=3, learning_rate=0.05 |

---

## 7. Experiment Setup

### 7.1 การตั้งค่า Experiment ให้เปรียบเทียบกันอย่างเป็นธรรม (Fair Comparison)

เพื่อให้การเปรียบเทียบโมเดลมีความยุติธรรม ได้ใช้แนวทางดังนี้:

1. **Same Data Split:** ทุกโมเดลใช้ข้อมูลชุดเดียวกัน — Train: 2024, Test: 2025
2. **Same Preprocessing:** ใช้ preprocessing pipeline เดียวกัน (forward-fill, outlier removal) สำหรับทุกโมเดล
3. **Same Features:** ทุกโมเดลใช้ 17 features เดียวกัน (lag, rolling, time, change features)
4. **Same Evaluation:** ทุกโมเดลวัดด้วย metrics เดียวกัน (MAE, RMSE, R²) บน test set เดียวกัน

### 7.2 Data Split และ Preprocessing ที่ใช้เหมือนกัน

```
PM2.5(2024).xlsx → Preprocess → Feature Engineering → X_train (359, 17), y_train (359,)
PM2.5(2025).xlsx → Preprocess → Feature Engineering → X_test  (174, 17), y_test  (174,)
```

**Preprocessing Pipeline (เหมือนกันทุกโมเดล):**
1. Load data → แปลง Date เป็น datetime, PM2.5 เป็น float
2. Forward-fill → Backward-fill สำหรับ missing values
3. Remove outliers (PM2.5 ∉ [0, 500])
4. Create 17 features (lag, rolling, time, change)
5. Drop NaN rows จาก lag features

### 7.3 แนวทาง Hyperparameter Tuning

| โมเดล | วิธี Tuning | CV Strategy |
|-------|------------|-------------|
| Linear Regression | ไม่มี hyperparameters | — |
| Ridge Regression | GridSearchCV (α) | TimeSeriesSplit(n_splits=3) |
| Random Forest | GridSearchCV (n_estimators, max_depth, min_samples_split) | TimeSeriesSplit(n_splits=3) |
| XGBoost | GridSearchCV (n_estimators, max_depth, learning_rate) | TimeSeriesSplit(n_splits=3) |

> **หมายเหตุ:** ใช้ `TimeSeriesSplit` แทน `KFold` เพราะเป็น time-series data — ป้องกัน data leakage จากอนาคต  
> **Scoring:** `neg_mean_absolute_error` (MAE) — เลือกโมเดลที่ MAE ต่ำที่สุด

---

## 8. Offline Evaluation & Results

### 8.1 Metrics ที่ใช้

| Metric | ประเภท | ความหมาย |
|--------|--------|---------|
| **MAE** (Mean Absolute Error) | Primary | ค่าเฉลี่ยของ absolute error — interpretable, หน่วยเดียวกับ PM2.5 (µg/m³) |
| **RMSE** (Root Mean Squared Error) | Secondary | ลงโทษ large errors มากขึ้น — สำคัญเพราะการทำนาย PM2.5 ผิดมากๆ อันตราย |
| **R²** (Coefficient of Determination) | Secondary | สัดส่วน variance ที่โมเดลอธิบายได้ (0 ถึง 1, สูงกว่า = ดีกว่า) |

### 8.2 ตารางเปรียบเทียบผลลัพธ์

ทดสอบบน **test set** (PM2.5 ปี 2025, สถานี 10T, 174 วัน):

| Model | MAE ↓ | RMSE ↓ | R² ↑ |
|-------|-------|--------|------|
| **Linear Regression (Baseline)** | 5.1348 | 6.7493 | 0.7726 |
| **Ridge Regression** | 4.8286 | 6.5294 | 0.7871 |
| **Random Forest** | **4.5702** | **6.5961** | **0.7828** |
| **XGBoost** | 4.9735 | 7.3464 | 0.7305 |

> *↓ = ยิ่งต่ำยิ่งดี, ↑ = ยิ่งสูงยิ่งดี*  
> **ตัวหนา** = ค่าดีที่สุดในแต่ละ metric

---

## 9. Analysis & Model Selection

### 9.1 โมเดลใดดีกว่า Baseline

ทุก candidate models มี **MAE ต่ำกว่า baseline** (Linear Regression):

| Model | MAE Improvement vs Baseline |
|-------|---------------------------|
| Ridge Regression | -0.31 (↓6.0%) |
| **Random Forest** | **-0.56 (↓11.0%)** |
| XGBoost | -0.16 (↓3.1%) |

**Random Forest** ให้ผลดีที่สุดใน MAE — ทำนายผิดเฉลี่ยเพียง 4.57 µg/m³

### 9.2 Trade-offs ที่พบ

| Trade-off | รายละเอียด |
|-----------|-----------|
| **MAE vs RMSE** | Random Forest มี MAE ดีที่สุด แต่ Ridge Regression มี RMSE ดีที่สุด (6.53) → Ridge ทำนาย extreme values ได้ดีกว่าเล็กน้อย |
| **R² Comparison** | Ridge มี R² สูงสุด (0.787) ขณะที่ Random Forest R² ใกล้เคียง (0.783) → ทั้งสองอธิบาย variance ได้ดีพอกัน |
| **Interpretability** | Linear/Ridge interpretable สูง (ดู coefficients ได้) vs Random Forest เป็น black-box มากกว่า |
| **Complexity** | Linear Regression ง่ายที่สุด, Random Forest ซับซ้อนปานกลาง, XGBoost ซับซ้อนที่สุดแต่กลับให้ผลไม่ดีที่สุดในกรณีนี้ |
| **XGBoost Performance** | XGBoost ให้ผลแย่กว่าที่คาด (MAE=4.97, R²=0.73) — อาจเกิดจากข้อมูลมีน้อย (~359 training samples) ทำให้ boosting overfits |

### 9.3 เลือกโมเดลที่จะใช้ต่อ

**โมเดลที่เลือก: Random Forest Regressor**

**เหตุผลเชิงระบบ (System-Level Reasoning):**

1. **Performance:** MAE ดีที่สุด (4.57 µg/m³) — ลดการทำนายผิดเฉลี่ยได้ 11% จาก baseline
2. **Robustness:** Random Forest มี variance ต่ำจาก bagging, robust ต่อ outliers ในข้อมูลใหม่
3. **No Feature Scaling Required:** ไม่ต้อง normalize features → ลดขั้นตอนใน inference pipeline
4. **Inference Speed:** ทำนายเร็ว (parallel across trees), เหมาะกับการ deploy ระบบ real-time
5. **Model Size:** เก็บเป็น .joblib ขนาดเล็ก, deploy ง่าย
6. **Reproducibility:** ตั้ง `random_state=42` ในทุกขั้นตอน, ใช้ config file กลาง (`configs/config.yaml`) สำหรับกำหนดพารามิเตอร์ทั้งหมด

**แยก Training / Inference Pipeline:**
- `src/train.py` — สำหรับ training (อ่านข้อมูลดิบ → preprocess → train → save model)
- `src/predict.py` — สำหรับ inference (โหลด saved model → รับ input → ทำนาย)
- Feature columns บันทึกไว้ใน `models/feature_columns.json` เพื่อให้มั่นใจว่า inference ใช้ features เดียวกับ training

---

## ภาคผนวก

### A. Repository Structure

```
pm25-prediction-ml-system/
├── configs/config.yaml           # Configuration กลาง
├── data/raw/                     # ข้อมูลดิบ (.xlsx)
├── data/processed/               # ข้อมูลที่ผ่าน preprocessing
├── src/
│   ├── data_loader.py            # โหลดข้อมูล
│   ├── preprocessing.py          # จัดการ missing values, outliers
│   ├── feature_engineering.py    # สร้าง features
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation metrics
│   └── predict.py                # Inference pipeline (แยก)
├── models/                       # Saved models (.joblib)
├── results/                      # ผลลัพธ์การทดลอง
├── reports/                      # รายงาน
├── notebooks/                    # EDA notebooks
└── tests/                        # Unit tests
```

### B. วิธีรันระบบ

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# Train ทุกโมเดล
PYTHONPATH=src python src/train.py

# ดูผลลัพธ์
PYTHONPATH=src python src/evaluate.py

# รัน Inference
PYTHONPATH=src python src/predict.py
```

### C. Technology Stack

| เครื่องมือ | วัตถุประสงค์ |
|-----------|------------|
| Python 3.14 | ภาษาหลัก |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models, evaluation, CV |
| XGBoost | Gradient boosting |
| matplotlib / seaborn | Visualization |
| PyYAML | Configuration management |
| joblib | Model serialization |
| Git | Version control |
