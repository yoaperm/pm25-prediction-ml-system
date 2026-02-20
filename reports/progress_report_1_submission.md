# Progress Report 1
# PM2.5 Prediction ML System

> **วิชา:** ML Systems  
> **หัวข้อโปรเจกต์:** ระบบทำนายค่า PM2.5 ด้วย Machine Learning  
> **วันที่ส่ง:** กุมภาพันธ์ 2026  
> **GitHub:** `pm25-prediction-ml-system`

---

## 1. Project Overview

> *สรุปปัญหาและเป้าหมายของระบบ ML โดยย่อ*

**ปัญหา (Problem Statement):**  
ฝุ่นละอองขนาดเล็ก PM2.5 (Particulate Matter ≤ 2.5 µm) เป็นปัญหาด้านสาธารณสุขที่สำคัญของประเทศไทย โดยเฉพาะในช่วงฤดูแล้ง (พฤศจิกายน–เมษายน) ค่า PM2.5 สูงเกินมาตรฐานอย่างต่อเนื่อง ส่งผลกระทบต่อสุขภาพประชาชน เช่น โรคระบบทางเดินหายใจ โรคหัวใจ และภูมิแพ้ ปัจจุบันการแจ้งเตือนส่วนใหญ่เป็นแบบ reactive (แจ้งหลังจากค่าสูงแล้ว) แทนที่จะเป็น proactive (ทำนายล่วงหน้า)

**เป้าหมายของระบบ ML (ML System Objective):**  
สร้างระบบ Machine Learning สำหรับ **ทำนายค่าความเข้มข้นของ PM2.5 (µg/m³) รายวัน** โดยใช้ข้อมูลย้อนหลังจากสถานีตรวจวัดคุณภาพอากาศ เพื่อให้หน่วยงานที่เกี่ยวข้องและประชาชนสามารถเตรียมรับมือได้อย่างทันท่วงที

- **ลักษณะงาน (Task Type):** Regression — ทำนายค่า PM2.5 เป็นตัวเลขต่อเนื่อง (continuous target)
- **ลักษณะข้อมูล (Data Nature):** Time-Series — ข้อมูลมีลำดับเวลา ค่าวันนี้ขึ้นอยู่กับค่าในอดีต
- **สถานีที่เลือกศึกษา:** สถานี **10T** — เคหะชุมชนคลองจั่น, เขตบางกะปิ, กรุงเทพฯ

---

## 2. Dataset & Data Description

### 2.1 แหล่งที่มาของข้อมูล (Data Source)

> *แหล่งที่มาของข้อมูล*

ข้อมูลได้มาจาก **กรมควบคุมมลพิษ (Pollution Control Department, PCD)** กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม ซึ่งเป็นข้อมูลการตรวจวัดคุณภาพอากาศ **สาธารณะ** (public environmental monitoring data) ที่เผยแพร่เพื่อประโยชน์สาธารณะ

### 2.2 ประเภทและขนาดข้อมูล (Data Type & Size)

> *ประเภทและขนาดข้อมูล*

ข้อมูลอยู่ในรูปแบบ **Excel (.xlsx)** จำนวน 2 ไฟล์:

| ไฟล์ | ช่วงเวลา | จำนวนแถว (วัน) | จำนวนสถานี | ขนาดไฟล์ |
|------|----------|----------------|------------|----------|
| `PM2.5(2024).xlsx` | ม.ค. – ธ.ค. 2024 | 366 วัน | 96 สถานี | 248 KB |
| `PM2.5(2025).xlsx` | ม.ค. – มิ.ย. 2025 | 181 วัน | 100 สถานี | 129 KB |

**โครงสร้างข้อมูลภายในแต่ละไฟล์:**

- **Sheet "Data":** คอลัมน์แรกคือ `Date` (วันที่), คอลัมน์ที่เหลือคือรหัสสถานี (02T, 05T, 10T, ...) แต่ละแถวคือค่า PM2.5 รายวันของแต่ละสถานี (หน่วย µg/m³)
- **Sheet "รายละเอียดจุดตรวจวัด":** metadata ของสถานี ได้แก่ ลำดับ, รหัสสถานี, ชื่อสถานี, รายละเอียดจุดติดตั้ง

**สถานีที่เลือกศึกษา:**

| รายละเอียด | ข้อมูล |
|-----------|--------|
| รหัสสถานี | 10T |
| ชื่อสถานี | แขวงคลองจั่น เขตบางกะปิ กทม. |
| จุดติดตั้ง | เคหะชุมชนคลองจั่น |
| ข้อมูลปี 2024 | 355 valid / 11 missing (จาก 366 วัน) |
| ข้อมูลปี 2025 | 181 valid / 0 missing |
| ค่าเฉลี่ย PM2.5 (2024) | 20.0 µg/m³ (SD = 11.8) |
| ค่าเฉลี่ย PM2.5 (2025) | 25.2 µg/m³ (SD = 14.2) |

**เหตุผลในการเลือกสถานี 10T:**  
สถานีนี้ตั้งอยู่ในเขตเมือง (กรุงเทพฯ) ซึ่งมีผู้ได้รับผลกระทบจำนวนมาก มีข้อมูลค่อนข้างครบ (missing เพียง ~3% ในปี 2024) และเป็นตัวแทนสภาพอากาศเขตเมืองได้ดี

### 2.3 วิธีการแบ่งข้อมูล (Data Splitting Strategy)

> *วิธีการแบ่งข้อมูล (train/validation/test)*

เนื่องจากข้อมูลเป็น **time-series** จึงใช้ **Temporal Split** (แบ่งตามลำดับเวลา) แทน random split เพื่อป้องกัน **data leakage** (การรั่วไหลของข้อมูลจากอนาคตไปยังอดีต):

| ชุดข้อมูล | ไฟล์ที่มา | ช่วงเวลา | จำนวนวัน | วัตถุประสงค์ |
|----------|----------|----------|----------|------------|
| **Training Set** | PM2.5(2024).xlsx | ม.ค. – ต.ค. 2024 | ~298 วัน | สอนโมเดล |
| **Validation Set** | PM2.5(2024).xlsx | พ.ย. – ธ.ค. 2024 | ~61 วัน | ปรับ hyperparameters |
| **Test Set** | PM2.5(2025).xlsx | ม.ค. – มิ.ย. 2025 | 174 วัน* | ประเมินผลสุดท้าย |

> *\*จำนวน 174 วันหลัง feature engineering เนื่องจากต้องตัด NaN rows จากการสร้าง lag features*

**ทำไมไม่ใช้ Random Split:**  
Random split จะทำให้ข้อมูลจากเดือน ธ.ค. 2024 อาจไปอยู่ใน training set ขณะที่ข้อมูลเดือน ม.ค. 2024 อยู่ใน test set → โมเดลจะเรียนรู้จากอนาคตเพื่อทำนายอดีต ซึ่งเป็น data leakage

**Cross-Validation:**  
ใช้ `TimeSeriesSplit(n_splits=3)` สำหรับ hyperparameter tuning ภายใน training set ซึ่ง respect ลำดับเวลา — fold สุดท้ายจะเป็นข้อมูลล่าสุดเสมอ

### 2.4 Privacy, PDPA, and Ethical Considerations

#### ข้อมูลเกี่ยวข้องกับข้อมูลส่วนบุคคลหรือไม่?

> **ไม่เกี่ยวข้อง** — ข้อมูลชุดนี้เป็นข้อมูลการตรวจวัดคุณภาพอากาศสาธารณะ ไม่มีข้อมูลชื่อ นามสกุล ที่อยู่ หมายเลขบัตรประชาชน หรือรายละเอียดใดๆ ที่สามารถระบุตัวบุคคลได้ (Personally Identifiable Information — PII) ข้อมูลทั้งหมดเป็นตัวเลขค่า PM2.5 กับตำแหน่งสถานีตรวจวัดเท่านั้น

#### มีข้อมูลอ่อนไหวหรือเสี่ยงต่อการระบุตัวบุคคลหรือไม่?

> **ไม่มี** — เป็นข้อมูลสิ่งแวดล้อม (environmental data) ที่กรมควบคุมมลพิษเผยแพร่สู่สาธารณะ ไม่มีข้อมูลอ่อนไหว (sensitive data) เช่น ข้อมูลสุขภาพ ข้อมูลทางการเงิน หรือข้อมูลชาติพันธุ์ ข้อมูลสถานีตรวจวัดเป็นจุดสาธารณะที่เปิดเผยตำแหน่งอยู่แล้ว

#### แนวทางลดความเสี่ยงด้าน Privacy / การใช้ข้อมูลอย่างรับผิดชอบ

แม้ข้อมูลจะไม่มีความเสี่ยงด้าน privacy โดยตรง แต่ยึดหลักดังนี้:
- **อ้างอิงแหล่งที่มา** อย่างชัดเจน (กรมควบคุมมลพิษ กระทรวงทรัพยากรธรรมชาติฯ)
- **ไม่ใช้ข้อมูลเพื่อการพาณิชย์** โดยไม่ได้รับอนุญาต
- **ใช้เพื่อการศึกษาและวิจัย** เท่านั้น
- **ไม่บิดเบือนผลลัพธ์** หรือนำไปใช้ในทางที่อาจทำให้เกิดความเข้าใจผิดเกี่ยวกับคุณภาพอากาศ

#### ผลกระทบที่อาจเกิดขึ้นหากระบบทำนายผิด

| ประเภทข้อผิดพลาด | ผลกระทบ |
|------------------|---------|
| **ทำนายต่ำกว่าจริง (False Low)** | ประชาชนอาจไม่ป้องกันตนเอง เช่น ไม่สวมหน้ากากอนามัย ไม่หลีกเลี่ยงกิจกรรมกลางแจ้ง → **เสี่ยงต่อสุขภาพ** (ร้ายแรงกว่า) |
| **ทำนายสูงกว่าจริง (False High)** | อาจทำให้เกิดความตื่นตระหนกโดยไม่จำเป็น หรือส่งผลกระทบต่อกิจกรรมทางเศรษฐกิจ เช่น การท่องเที่ยว |

**ข้อเสนอแนะ:** ระบบนี้ควรใช้เป็น **เครื่องมือสนับสนุนการตัดสินใจ (decision support tool)** ที่ทำงานร่วมกับข้อมูลการตรวจวัดจริง ไม่ใช่ระบบที่ทดแทนการตรวจวัดจริงทั้งหมด

---

## 3. Data Engineering & Training Data Considerations

### 3.1 แนวทางเตรียมข้อมูลและจัดการ Missing / Noisy Data

> *แนวทางเตรียมข้อมูลและจัดการ missing/noisy data*

**Missing Values Handling:**

| ขั้นตอน | วิธีการ | เหตุผล |
|---------|--------|--------|
| 1. ตัดแถวที่ไม่ใช่ข้อมูล | ตัดแถวท้ายไฟล์ที่เป็นหมายเหตุ (เช่น "หมายเหตุ : N/A ไม่มีข้อมูล") | ป้องกัน parsing error |
| 2. แปลง data type | แปลง Date → `datetime`, PM2.5 → `float`, ค่า n/a → `NaN` | ให้พร้อมประมวลผล |
| 3. Forward-Fill (ffill) | ใช้ค่า PM2.5 ของวันก่อนหน้าแทนค่า NaN | เหมาะกับ time-series เพราะค่า PM2.5 มีความต่อเนื่อง ค่าวันก่อนหน้ามักใกล้เคียงวันถัดไป |
| 4. Backward-Fill (bfill) | ใช้กรณี NaN อยู่ต้นชุดข้อมูล (ไม่มีวันก่อนหน้า) | เติมค่าที่เหลือจาก ffill |

> **ไม่ใช้ mean imputation** เพราะจะทำลาย temporal pattern ของข้อมูล time-series

**Noisy Data / Outlier Handling:**

- กำหนดขอบเขตทางกายภาพที่สมเหตุสมผล: **PM2.5 ∈ [0, 500] µg/m³**
- ค่าติดลบ → sensor error → ตัดออก
- ค่าสูงกว่า 500 µg/m³ → เหตุการณ์ผิดปกติสุดขีดหรือ sensor malfunction → ตัดออก
- ในข้อมูลจริงของสถานี 10T: **ไม่พบ outliers** หลัง preprocessing

**สรุปผลหลัง preprocessing:**
| ข้อมูล | ก่อน | หลัง |
|--------|-----|------|
| Train (2024) | 11 missing | 0 missing, 366 rows |
| Test (2025) | 0 missing | 0 missing, 181 rows |

### 3.2 Class Imbalance และ Sampling Strategy

> *class imbalance และ sampling strategy (ถ้ามี)*

เนื่องจากงานนี้เป็น **Regression** (ทำนายค่าต่อเนื่อง) **ไม่ใช่ Classification** จึงไม่มีปัญหา class imbalance โดยตรง

อย่างไรก็ตาม มีข้อสังเกตที่สำคัญ:

- ค่า PM2.5 มี **right-skewed distribution** — ค่าต่ำ (< 20 µg/m³) มีจำนวนมาก ขณะที่ค่าสูงมาก (> 50 µg/m³) มีน้อย
- ผลคือ: โมเดลอาจ **ทำนายค่าสูงได้ไม่แม่นยำ** เพราะมีตัวอย่างน้อย
- สังเกตได้จาก **RMSE > MAE** ค่อนข้างมากในทุกโมเดล (RMSE ลงโทษ large errors มากขึ้น)
- **Sampling strategy:** ไม่ได้ใช้ oversampling/undersampling เนื่องจากเป็น regression ใช้ข้อมูลทั้งหมดตามลำดับเวลา

### 3.3 แหล่งที่มาของ Label และความเสี่ยงของ Label Noise

> *แหล่งที่มาของ label และความเสี่ยงของ label noise*

**Label (Target Variable):**  
ค่า PM2.5 รายวัน (µg/m³) ที่วัดจาก **เซ็นเซอร์ตรวจวัดคุณภาพอากาศ** ณ สถานี เป็น **continuous value** ที่ได้จากการตรวจวัดจริง (measured, ไม่ใช่ annotated)

**ความเสี่ยง Label Noise:**

| ความเสี่ยง | ระดับ | รายละเอียด |
|-----------|-------|-----------|
| Sensor Drift | ต่ำ | เซ็นเซอร์อาจเสื่อมสภาพตามเวลา ค่าอาจคลาดเคลื่อนเล็กน้อย |
| Calibration Error | ต่ำ | การสอบเทียบที่ไม่สม่ำเสมออาจทำให้ค่าผิดพลาดชั่วคราว |
| Environmental Interference | ต่ำ-ปานกลาง | มลพิษเฉพาะจุด เช่น ไฟไหม้ใกล้สถานี อาจทำให้ค่าผิดปกติชั่วคราว |

> **ความเสี่ยงโดยรวม: ต่ำ** — เนื่องจากกรมควบคุมมลพิษมีมาตรฐานการดูแลรักษาเครื่องมือตรวจวัด

### 3.4 สมมติฐานสำคัญเกี่ยวกับข้อมูล

> *สมมติฐานสำคัญเกี่ยวกับข้อมูล*

1. **Temporal Autocorrelation:** ค่า PM2.5 ของวันนี้มีความสัมพันธ์กับค่าในวันก่อนหน้า → lag features จะเป็น predictors ที่มีพลัง
2. **Seasonality:** ค่า PM2.5 มี pattern ตามฤดูกาล — สูงในฤดูแล้ง (พ.ย.–เม.ย.), ต่ำในฤดูฝน (พ.ค.–ต.ค.) → month/day_of_year เป็น features ที่มีความหมาย
3. **Local Stationarity:** แม้ข้อมูลอาจไม่ stationary ในภาพรวม แต่ในช่วงสั้น (1-2 สัปดาห์) มี pattern ค่อนข้างสม่ำเสมอ → rolling statistics จับได้
4. **No Major Structural Break:** สมมติว่าไม่มีเหตุการณ์ผิดปกติขนาดใหญ่ (ไฟป่าใหญ่, การระบาด, lockdown) ที่ทำให้ pattern เปลี่ยนฉับพลัน
5. **Single-Station Sufficiency:** สมมติว่าข้อมูลจากสถานีเดียวมีข้อมูลเพียงพอสำหรับทำนาย PM2.5 ในพื้นที่นั้น (ไม่จำเป็นต้องใช้ข้อมูลข้ามสถานี)

---

## 4. Feature Engineering Plan

### 4.1 Feature หลักที่ใช้

> *feature หลักที่ใช้ (engineered / learned) + เหตุผลในการเลือก feature*

ใช้ **Engineered Features** ทั้งหมด 17 features (ไม่มี learned features เนื่องจากไม่ได้ใช้ deep learning):

| ประเภท | Features | จำนวน | เหตุผลในการเลือก |
|--------|----------|-------|----------------|
| **Lag Features** | `pm25_lag_1`, `pm25_lag_2`, `pm25_lag_3`, `pm25_lag_5`, `pm25_lag_7` | 5 | **จับ temporal autocorrelation** — จากสมมติฐานข้อ 1 ค่า PM2.5 วันก่อนหน้าเป็น predictor ที่แข็งแกร่งที่สุด ใช้ shift() เพื่อป้องกัน data leakage |
| **Rolling Mean** | `pm25_rolling_mean_3`, `pm25_rolling_mean_7`, `pm25_rolling_mean_14` | 3 | **จับ short/medium-term trend** — ค่าเฉลี่ยเคลื่อนที่สะท้อนทิศทางระยะสั้น/กลาง ช่วยให้โมเดลเห็น "แนวโน้ม" ไม่ใช่แค่ค่าจุดเดียว |
| **Rolling Std** | `pm25_rolling_std_3`, `pm25_rolling_std_7`, `pm25_rolling_std_14` | 3 | **จับความผันผวน (volatility)** — ช่วงที่ค่า PM2.5 ผันผวนสูงมักบ่งชี้การเปลี่ยนแปลงสภาพอากาศ |
| **Time Features** | `day_of_week`, `month`, `day_of_year`, `is_weekend` | 4 | **จับ seasonality/weekly pattern** — month จับ seasonal trend (ฤดูแล้ง vs ฤดูฝน), is_weekend จับ traffic pattern (จราจรน้อยลงวันหยุด) |
| **Change Features** | `pm25_diff_1`, `pm25_pct_change_1` | 2 | **จับ momentum/direction** — อัตราการเปลี่ยนแปลงบอกว่าค่ากำลังขึ้นหรือลง ช่วยทำนาย trend |

> **สำคัญ — การป้องกัน Data Leakage:** Lag features และ rolling statistics ใช้ `shift(1)` ก่อนคำนวณ เพื่อให้มั่นใจว่าไม่ใช้ข้อมูลจากวันปัจจุบัน (ซึ่งเป็น target variable) ในการสร้าง features

### 4.2 Features ที่ตั้งใจไม่ใช้

> *feature ที่ตั้งใจไม่ใช้ (ถ้ามี)*

| Feature ที่ไม่ใช้ | เหตุผล |
|------------------|--------|
| ข้อมูลจากสถานีอื่น (cross-station data) | เพื่อควบคุมความซับซ้อนของ experiment — สามารถเพิ่มเป็น spatial features ในอนาคต |
| ข้อมูลสภาพอากาศ (อุณหภูมิ, ความเร็วลม, ปริมาณฝน) | ไม่มีอยู่ใน dataset ปัจจุบัน — สามารถ integrate จาก external sources (TMD) ในอนาคต |
| Holiday flag (วันหยุดนักขัตฤกษ์) | ไม่มีข้อมูลวันหยุดอยู่ใน dataset — ใช้ `is_weekend` เป็น proxy แทน |
| Fourier transform features | เพิ่มความซับซ้อน ยังไม่จำเป็นในขั้นตอนนี้ — month/day_of_year จับ seasonality ได้เพียงพอ |

---

## 5. Baseline Model

### 5.1 Baseline Model ที่เลือกใช้

> *baseline model ที่เลือกใช้*

**Linear Regression** (Ordinary Least Squares — OLS)

### 5.2 เหตุผลที่เลือก

> *เหตุผลที่เลือก*

- **Simple & interpretable:** เป็นโมเดลที่ง่ายที่สุดสำหรับ regression ทำให้สามารถดู coefficients เพื่อเข้าใจว่า feature ไหนสำคัญ
- **No hyperparameters:** ไม่มี hyperparameters ที่ต้อง tune → ผลลัพธ์ reproducible 100%
- **Establishes benchmark:** เป็น "เส้นฐาน" ที่ candidate models ทุกตัวต้องเอาชนะ — ถ้า candidate model ไม่ดีกว่า Linear Regression แสดงว่าความซับซ้อนเพิ่มขึ้นไม่คุ้มค่า
- **เหมาะกับ linear relationships:** หากค่า PM2.5 มีความสัมพันธ์เชิงเส้นกับ lag features อยู่แล้ว Linear Regression จะให้ผลดีเป็น baseline ที่แข็งแกร่ง

### 5.3 ผลการทดลอง Baseline

> *ผลการทดลอง baseline (metrics หลัก)*

ทดสอบบน **Test Set** (PM2.5 ปี 2025, สถานี 10T, 174 วัน):

| Metric | ค่า | ความหมาย |
|--------|-----|---------|
| **MAE** (Primary) | **5.1348** | ทำนายผิดเฉลี่ย ~5.13 µg/m³ ต่อวัน |
| **RMSE** (Secondary) | **6.7493** | ค่า error ใหญ่ถูกลงโทษมากขึ้น (penalizes large errors) |
| **R²** (Secondary) | **0.7726** | อธิบาย variance ของข้อมูลได้ ~77.3% |

> **ตีความ:** MAE = 5.13 หมายความว่าโดยเฉลี่ย โมเดลทำนายค่า PM2.5 คลาดเคลื่อนประมาณ 5 µg/m³ ซึ่งถือว่าพอใช้ได้สำหรับ baseline แต่ยังมีที่ว่างสำหรับปรับปรุง

---

## 6. Candidate Models (3 Models)

> *สำหรับแต่ละโมเดล: ชื่อโมเดล / เหตุผลที่เลือก / ความแตกต่างจาก baseline*

### Model A: Ridge Regression

| หัวข้อ | รายละเอียด |
|-------|-----------|
| **ชื่อโมเดล** | Ridge Regression (L2 Regularization) |
| **เหตุผลที่เลือก** | Lag features หลายตัว (lag_1, lag_2, lag_3, ...) มี **multicollinearity สูง** (ค่า PM2.5 วันติดกันใกล้เคียงกัน) ทำให้ Linear Regression coefficients อาจสุดโต่ง → Ridge เพิ่ม regularization เพื่อลด overfitting |
| **ความแตกต่างจาก baseline** | เพิ่ม **L2 penalty term** (α · Σw²) บน coefficients → ทำให้ weights ไม่สุดโต่ง, ลด overfitting, generalize ดีกว่า linear regression ธรรมดา |
| **Tuned Hyperparameter** | α = 100.0 (ผ่าน GridSearchCV + TimeSeriesSplit) |

### Model B: Random Forest Regressor

| หัวข้อ | รายละเอียด |
|-------|-----------|
| **ชื่อโมเดล** | Random Forest Regressor (Ensemble of Decision Trees) |
| **เหตุผลที่เลือก** | จับ **non-linear relationships** ที่ linear models ทำไม่ได้ (เช่น PM2.5 อาจเพิ่มแบบ exponential ในบางช่วง), **robust ต่อ outliers**, ไม่ต้อง feature scaling |
| **ความแตกต่างจาก baseline** | ใช้ **bagging** (สร้าง decision trees หลายต้นจาก random subsets แล้ว average ผลลัพธ์) → ลด variance, จับ complex interactions ระหว่าง features ได้ เช่น "lag_1 สูง + month เป็นม.ค. → PM2.5 จะสูงมาก" |
| **Tuned Hyperparameters** | n_estimators=50, max_depth=5, min_samples_split=10 |

### Model C: XGBoost Regressor

| หัวข้อ | รายละเอียด |
|-------|-----------|
| **ชื่อโมเดล** | XGBoost — Extreme Gradient Boosting |
| **เหตุผลที่เลือก** | เป็น **state-of-the-art สำหรับ tabular data** มีชื่อเสียงในการแข่งขัน ML (Kaggle), มี built-in regularization (L1/L2), จัดการ missing values ได้ดี |
| **ความแตกต่างจาก baseline** | ใช้ **gradient boosting** — สร้าง trees ลำดับต่อเนื่อง โดยแต่ละ tree เรียนรู้จาก **residual errors** ของ tree ก่อนหน้า → ค่อยๆ แก้ไข errors จนทำนายดีขึ้น (ต่างจาก Random Forest ที่สร้าง trees พร้อมกันแบบ parallel) |
| **Tuned Hyperparameters** | n_estimators=50, max_depth=3, learning_rate=0.05 |

---

## 7. Experiment Setup

### 7.1 การตั้งค่า Experiment ให้เปรียบเทียบกันอย่างเป็นธรรม

> *การตั้งค่า experiment ให้เปรียบเทียบกันอย่างเป็นธรรม*

เพื่อให้การเปรียบเทียบโมเดลมีความยุติธรรม (**fair comparison**) ใช้หลักการ:

1. **Same Data Split:** ทุกโมเดลใช้ข้อมูล **ชุดเดียวกัน** — Train: 2024, Test: 2025
2. **Same Preprocessing:** ใช้ **preprocessing pipeline เดียวกัน** (forward-fill → outlier removal) สำหรับทุกโมเดล
3. **Same Features:** ทุกโมเดลใช้ **17 features เดียวกัน** (lag, rolling, time, change features)
4. **Same Evaluation:** ทุกโมเดลวัดด้วย **metrics เดียวกัน** (MAE, RMSE, R²) บน **test set เดียวกัน**
5. **Same Random Seed:** ใช้ `random_state=42` ทุกที่เพื่อ reproducibility

### 7.2 Data Split และ Preprocessing ที่ใช้เหมือนกัน

> *data split และ preprocessing ที่ใช้เหมือนกัน*

```
[Shared Pipeline — เหมือนกันทุกโมเดล]

PM2.5(2024).xlsx ──→ Load ──→ Forward-Fill ──→ Outlier Removal ──→ Feature Engineering ──→ X_train (359, 17)
PM2.5(2025).xlsx ──→ Load ──→ Forward-Fill ──→ Outlier Removal ──→ Feature Engineering ──→ X_test  (174, 17)
```

ทุกโมเดลได้รับ **input เดียวกันทุกประการ** ทำให้ความแตกต่างของ performance เกิดจากตัวโมเดลล้วนๆ

### 7.3 แนวทาง Hyperparameter Tuning

> *แนวทางการตั้งค่า/ปรับ hyperparameters (แบบย่อ)*

| โมเดล | วิธี Tuning | Grid | CV Strategy |
|-------|------------|------|-------------|
| Linear Regression | ไม่มี hyperparameters | — | — |
| Ridge Regression | `GridSearchCV` | α ∈ {0.01, 0.1, 1, 10, 100} | `TimeSeriesSplit(n_splits=3)` |
| Random Forest | `GridSearchCV` | n_estimators ∈ {50, 100, 200}, max_depth ∈ {5, 10, 15, None}, min_samples_split ∈ {2, 5, 10} | `TimeSeriesSplit(n_splits=3)` |
| XGBoost | `GridSearchCV` | n_estimators ∈ {50, 100, 200}, max_depth ∈ {3, 5, 7}, learning_rate ∈ {0.01, 0.05, 0.1} | `TimeSeriesSplit(n_splits=3)` |

> **ทำไมใช้ TimeSeriesSplit แทน KFold?**  
> KFold จะสุ่มข้อมูลทำให้ fold อาจมีข้อมูลจากอนาคตปนอยู่ → data leakage  
> TimeSeriesSplit ใช้เฉพาะข้อมูลในอดีตเป็น training ทุกรอบ → ป้องกัน data leakage
>
> **Scoring:** `neg_mean_absolute_error` — เลือกโมเดลที่ MAE ต่ำที่สุดจาก CV

---

## 8. Offline Evaluation & Results

### 8.1 Metric หลักและ Metric รอง

> *metric หลักและ metric รอง*

| Metric | ประเภท | สูตร | ความหมายในบริบท PM2.5 |
|--------|--------|------|----------------------|
| **MAE** (Mean Absolute Error) | **Primary** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | ทำนายผิดเฉลี่ยกี่ µg/m³ — **interpretable** หน่วยเดียวกับ PM2.5 |
| **RMSE** (Root Mean Squared Error) | Secondary | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | ลงโทษ **large errors** มากขึ้น — สำคัญเพราะค่า PM2.5 ที่ทำนายผิดมากๆ อาจเป็นอันตราย |
| **R²** (Coefficient of Determination) | Secondary | $1 - \frac{SS_{res}}{SS_{tot}}$ | สัดส่วน variance ที่โมเดลอธิบายได้ (1.0 = สมบูรณ์แบบ, 0 = แย่เท่า mean prediction) |

> **เหตุผลที่เลือก MAE เป็น primary:** MAE ให้ค่าที่ interpretable ตรงๆ — "โมเดลทำนายผิดเฉลี่ย X µg/m³" เข้าใจง่ายสำหรับผู้ใช้งานจริง

### 8.2 ตารางเปรียบเทียบผลลัพธ์

> *ตารางเปรียบเทียบผลระหว่าง Baseline / Model A / Model B / Model C*

ทดสอบบน **Test Set** — PM2.5 ปี 2025, สถานี 10T, จำนวน 174 วัน:

| | Model | MAE ↓ | RMSE ↓ | R² ↑ | Best Hyperparameters |
|---|-------|-------|--------|------|---------------------|
| Baseline | **Linear Regression** | 5.1348 | 6.7493 | 0.7726 | — (ไม่มี) |
| Model A | **Ridge Regression** | 4.8286 | **6.5294** | **0.7871** | α = 100.0 |
| Model B | **Random Forest** | **4.5702** | 6.5961 | 0.7828 | n_est=50, depth=5, split=10 |
| Model C | **XGBoost** | 4.9735 | 7.3464 | 0.7305 | n_est=50, depth=3, lr=0.05 |

> *↓ = ยิ่งต่ำยิ่งดี, ↑ = ยิ่งสูงยิ่งดี*  
> **ตัวหนา** = ค่าดีที่สุดในแต่ละ metric

**MAE Improvement vs Baseline:**

| Model | MAE | MAE ลดลงจาก Baseline |
|-------|-----|----------------------|
| Ridge Regression | 4.8286 | −0.31 (**↓ 6.0%**) |
| Random Forest | 4.5702 | −0.56 (**↓ 11.0%**) |
| XGBoost | 4.9735 | −0.16 (**↓ 3.1%**) |

---

## 9. Analysis & Model Selection

### 9.1 โมเดลใดดีกว่า Baseline และเพราะเหตุใด

> *โมเดลใดดีกว่า baseline และเพราะเหตุใด*

**ทุก candidate models ให้ผลดีกว่า baseline** (Linear Regression) ในทุก primary metric:

- **Random Forest ดีที่สุดใน MAE** (primary metric): ทำนายผิดเฉลี่ยเพียง 4.57 µg/m³ ดีกว่า baseline 11% เพราะสามารถจับ **non-linear patterns** ที่ Linear Regression ไม่สามารถเรียนรู้ได้ เช่น interaction effects ระหว่าง lag features กับ seasonal patterns
- **Ridge Regression ดีที่สุดใน RMSE และ R²**: มี RMSE ต่ำสุด (6.53) และ R² สูงสุด (0.787) แสดงว่าจัดการ **extreme errors** ได้ดีกว่า เพราะ L2 regularization ช่วยให้ coefficients ไม่สุดโต่ง
- **XGBoost ให้ผลแย่กว่าที่คาด**: MAE = 4.97 และ R² = 0.73 ต่ำกว่า Ridge — สาเหตุน่าจะเป็นเพราะข้อมูลมีน้อย (~359 training samples) ทำให้ gradient boosting มีแนวโน้ม **overfit** มากกว่า bagging-based model อย่าง Random Forest

### 9.2 Trade-offs ที่พบ

> *trade-offs ที่พบ (เช่น FP vs FN, performance vs complexity)*

| Trade-off | รายละเอียด |
|-----------|-----------|
| **MAE vs RMSE** | Random Forest มี MAE ดีที่สุด (4.57) แต่ Ridge มี RMSE ดีที่สุด (6.53) — หมายความว่า Random Forest ทำนาย **average case ได้ดีกว่า** แต่ Ridge ทำนาย **extreme values ได้ดีกว่าเล็กน้อย** |
| **Performance vs Interpretability** | Linear Regression/Ridge **interpretable** สูง (ดู coefficient ได้ว่า feature ไหนสำคัญ) vs Random Forest เป็น **black-box** มากกว่า — ต้องใช้ feature importance แทน |
| **Performance vs Complexity** | XGBoost มีความซับซ้อนสูงสุด (3 hyperparameters + boosting) แต่ **ผลกลับไม่ดีที่สุด** ในกรณีนี้ → ความซับซ้อนเพิ่มไม่ได้แปลว่าดีเสมอ โดยเฉพาะเมื่อข้อมูลน้อย |
| **Training Time vs Performance** | Linear Regression เร็วที่สุด, Random Forest ปานกลาง, XGBoost ช้าที่สุด (GridSearch บน 27 combinations) — แต่ performance ที่ดีขึ้น 11% ของ RF คุ้มค่ากับเวลาที่เพิ่ม |
| **Robustness** | Random Forest มี built-in robustness จาก bagging (เฉลี่ยจาก 50 trees) → แม้ข้อมูลใหม่จะมี noise ก็ยังทำนายได้ค่อนข้างเสถียร |

### 9.3 เลือกโมเดลที่จะใช้ต่อ พร้อมเหตุผลเชิงระบบ

> *เลือกโมเดลที่จะใช้ต่อ พร้อมเหตุผลเชิงระบบ*

### ✅ โมเดลที่เลือก: **Random Forest Regressor**

**เหตุผลเชิงระบบ (System-Level Reasoning):**

| ด้าน | เหตุผล |
|------|--------|
| **1. Best Primary Metric** | MAE ดีที่สุด (4.57 µg/m³) — ลดการทำนายผิดเฉลี่ยได้ 11% จาก baseline |
| **2. Robustness** | Bagging (เฉลี่ยจาก 50 trees) ให้ variance ต่ำ แม้ข้อมูลใหม่จะมี distribution shift เล็กน้อย |
| **3. No Feature Scaling** | ไม่ต้อง normalize/standardize features → ลดขั้นตอนใน inference pipeline, ลด preprocessing bugs |
| **4. Inference Speed** | ทำนายเร็ว (trees predict แบบ parallel) — เหมาะกับระบบที่ต้องทำนายรายวัน |
| **5. Model Persistence** | บันทึกเป็น `.joblib` ขนาดเล็ก, โหลดง่าย, ไม่ต้องพึ่งเฟรมเวิร์คหนัก |
| **6. Reproducibility** | ตั้ง `random_state=42` ทุกขั้นตอน + ใช้ config file กลาง (`configs/config.yaml`) |

**System Architecture:**
```
[Training Pipeline]  src/train.py     →  อ่านข้อมูลดิบ → preprocess → สร้าง features → train → save model (.joblib)
[Inference Pipeline] src/predict.py   →  load saved model → รับ input → preprocess → predict → output
[Evaluation]         src/evaluate.py  →  load results → แสดงตารางเปรียบเทียบ
[Configuration]      configs/config.yaml → กำหนด station, features, hyperparameters ทั้งหมด
```

> **Training กับ Inference แยกกัน** ตามหลัก ML Systems best practice — feature columns บันทึกใน `models/feature_columns.json` เพื่อให้มั่นใจว่า inference ใช้ features เดียวกับ training ทุกประการ

---

## ภาคผนวก

### A. Technology Stack

| เครื่องมือ | วัตถุประสงค์ |
|-----------|------------|
| Python 3.14 | ภาษาหลัก |
| pandas / numpy | Data manipulation |
| scikit-learn | ML models (LR, Ridge, RF), evaluation, GridSearchCV, TimeSeriesSplit |
| XGBoost | Gradient boosting |
| matplotlib / seaborn | Visualization (EDA notebook) |
| PyYAML | Configuration management |
| joblib | Model serialization |
| Git | Version control |

### B. Repository Structure

```
pm25-prediction-ml-system/
├── configs/config.yaml           # Configuration กลาง (station, features, hyperparameters)
├── data/
│   ├── raw/                      # ข้อมูลดิบ (.xlsx)
│   └── processed/                # ข้อมูลที่ผ่าน preprocessing (.csv)
├── src/
│   ├── data_loader.py            # โหลดข้อมูลจาก Excel
│   ├── preprocessing.py          # จัดการ missing values, outliers
│   ├── feature_engineering.py    # สร้าง lag, rolling, time features
│   ├── train.py                  # Training pipeline (ทั้ง 4 โมเดล)
│   ├── evaluate.py               # Evaluation metrics + comparison table
│   └── predict.py                # Inference pipeline (แยกจาก training)
├── models/                       # Saved models (.joblib) + feature_columns.json
├── results/                      # experiment_results.csv + plots
├── reports/                      # Progress Report
├── notebooks/01_eda.ipynb        # Exploratory Data Analysis
├── tests/                        # Unit tests
├── requirements.txt              # Dependencies
└── README.md                     # Project documentation
```

### C. วิธีรันระบบ (Reproducibility)

```bash
# ติดตั้ง dependencies
pip install -r requirements.txt

# Train ทุกโมเดล + บันทึกผล
PYTHONPATH=src python src/train.py

# ดูผลลัพธ์เปรียบเทียบ
PYTHONPATH=src python src/evaluate.py

# รัน Inference (โหลด saved model → ทำนาย)
PYTHONPATH=src python src/predict.py
```
