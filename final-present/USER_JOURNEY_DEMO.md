# FoonAlert Demo — User Journey Design

## Overview

**App Name:** FoonAlert: Real-Time PM2.5 Spike Forecasting  
**Tagline:** "Don't just see what happened — know what's about to happen."  
**Demo Duration:** 2–3 minutes (for presentation)

---

## User Journey Flow

```
Login → Station Select → Live Dashboard → Spike Replay Mode → Model Battle → Results
```

---

## Screen-by-Screen Design

### Screen 1: Login (3 seconds)

```
┌─────────────────────────────────────────┐
│  🌫️ FoonAlert                          │
│  Real-Time PM2.5 Spike Forecasting      │
│                                         │
│  [Username: ________]                   │
│  [Password: ________]                   │
│  [Login]                                │
└─────────────────────────────────────────┘
```

**Action:** Quick login → immediate redirect to dashboard

---

### Screen 2: Live Dashboard (Main Screen)

This is the "wow" screen. Shows current status + predictions side by side.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 🌫️ FoonAlert                    Station: [56 ▼]   Mode: [Live | Replay] │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Current PM2.5│  │   +1 Hour    │  │   +6 Hours   │  │  Spike Risk  │ │
│  │              │  │              │  │              │  │              │ │
│  │    47.3      │  │    52.1      │  │    78.4      │  │   🔴 HIGH    │ │
│  │  🟡 Moderate │  │  🟡 Moderate │  │  🟠 Unhealthy│  │              │ │
│  │  14:00 today │  │  15:00 est.  │  │  20:00 est.  │  │ +31 µg/m³   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │              Model Battle — Next Hour Predictions                    │ │
│  ├─────────────────────────────────────────────────────────────────────┤ │
│  │  Model         │ +1h  │ +6h  │ +24h │ Spike Risk │ Confidence     │ │
│  │  Persistence   │ 47.3 │ 47.3 │ 47.3 │ Low        │ —              │ │
│  │  SARIMA        │ 49.8 │ 62.1 │ 51.2 │ Medium     │ ±8.2           │ │
│  │  LSTM          │ 52.1 │ 72.8 │ 58.4 │ High       │ ±6.1           │ │
│  │  Transformer   │ 54.3 │ 78.4 │ 63.7 │ High       │ ±5.8           │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  [Actual vs Predicted Chart — last 48 hours + next 24h forecast]    │ │
│  │                                                                     │ │
│  │  PM2.5                                                              │ │
│  │  120┤                                                               │ │
│  │  100┤              ╭─╮                                              │ │
│  │   80┤           ╭─╯ ╰─╮    ┊ ╭── Transformer                       │ │
│  │   60┤     ╭───╯       ╰╮   ┊╭╯── LSTM                             │ │
│  │   40┤──╭─╯             ╰╮  ┊╯─── SARIMA                           │ │
│  │   20┤                   ╰──┊───── Persistence                      │ │
│  │     └──────────────────────┊────────────────────→ Time             │ │
│  │        Past (actual)      Now    Future (predicted)                 │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ⚠️  SPIKE ALERT: 2/3 models predict PM2.5 > 75 µg/m³ within 6 hours   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### Screen 3: Spike Replay Mode (The "Time Machine")

**This is the DEMO KILLER.** User selects a historical spike day and watches models "predict" it hour by hour.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 🌫️ FoonAlert — Spike Replay Mode                                        │
│                                                                           │
│  Station: [59 ▼]   Date: [2025-01-24 ▼]   Speed: [▶ Play | ⏸ Pause]    │
│  Current Hour: [======●==========] 05:00 / 23:00                         │
│                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐│
│  │  Replay Status                                                       ││
│  │                                                                      ││
│  │  Time: 05:00    Actual PM2.5: 107.6 µg/m³   🔴 Unhealthy            ││
│  │                                                                      ││
│  │  ┌─────────────────────────────────────────────────────────────────┐ ││
│  │  │ 🏆 Model Scoreboard (live during replay)                       │ ││
│  │  ├─────────────────────────────────────────────────────────────────┤ ││
│  │  │ Model        │ Predicted │ Actual │ Error │ Spike Alert?        │ ││
│  │  │ LSTM         │   98.2    │ 107.6  │  9.4  │ ✅ Yes (3h early)  │ ││
│  │  │ Transformer  │  101.5    │ 107.6  │  6.1  │ ✅ Yes (4h early)  │ ││
│  │  │ SARIMA       │   82.4    │ 107.6  │ 25.2  │ ✅ Yes (2h early)  │ ││
│  │  │ Persistence  │   64.0    │ 107.6  │ 43.6  │ ❌ No              │ ││
│  │  └─────────────────────────────────────────────────────────────────┘ ││
│  │                                                                      ││
│  │  [Chart: Progressive reveal — actual line extends hour by hour]      ││
│  │  [Prediction lines shown 6h ahead of current time pointer]           ││
│  │                                                                      ││
│  │  🎯 Transformer detected spike 4 HOURS before peak!                  ││
│  └──────────────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────────────┘
```

**Interaction Flow:**
1. User selects station + spike day (pre-curated list)
2. Auto-play or manual step (⏭ Next Hour button)
3. Graph progressively reveals: actual shows up to current hour, predictions show ahead
4. Warning cards change color as spike approaches
5. At peak: celebration banner "Model X detected spike N hours early!"

---

### Screen 4: Model Battle Results

Final scoreboard after replay or across all test data.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 🏆 Model Battle Results                                                   │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │ Rank │ Model        │ MAE  │ RMSE │ Spike Recall │ Best At          │ │
│  │  1   │ Transformer  │ 5.8  │ 8.2  │    87%       │ Long horizon     │ │
│  │  2   │ LSTM         │ 6.1  │ 8.9  │    82%       │ Short-term spike │ │
│  │  3   │ SARIMA       │ 8.4  │ 11.3 │    71%       │ Stable pattern   │ │
│  │  4   │ Persistence  │ 12.7 │ 16.8 │    23%       │ Very short-term  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  │
│  │ Error by Horizon    │  │ Error by Severity   │  │ Spike Detection │  │
│  │ [Bar chart]         │  │ [Grouped bar chart] │  │ [Timeline]      │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Spike Risk Logic

```python
def compute_spike_risk(current_pm25, predictions_6h, model_consensus):
    """
    Spike Risk Levels:
    - Low:    predicted < 37.5 OR increase < 10
    - Medium: 37.5 ≤ predicted ≤ 75 OR increase 10-20
    - High:   predicted > 75 OR increase > 20
    - Severe: predicted > 100 OR increase > 40
    
    Consensus boost: if 2/3 models agree on High → upgrade to High
    """
    max_predicted = max(predictions_6h)
    increase = max_predicted - current_pm25
    
    if max_predicted > 100 or increase > 40:
        return "Severe"
    elif max_predicted > 75 or increase > 20:
        return "High"
    elif max_predicted > 37.5 or increase > 10:
        return "Medium"
    else:
        return "Low"
```

---

## Pre-curated Spike Days for Demo

| Station | Date | Pattern | PM2.5 Range | Good For |
|---------|------|---------|-------------|----------|
| 59 | 2025-01-24 | Morning spike + evening rebound | 18→108 | **Primary demo** |
| 61 | 2024-12-24 | Night peak → drop → evening surge | 22→116 | Dramatic |
| 58 | 2025-03-14 | Low baseline → sudden spike | 16→192 | Extreme event |
| 57 | 2024-02-14 | Night peak sustained | 43→121 | High baseline |
| 56 | 2024-04-30 | Clean moderate spike | 14→82 | Moderate case |

---

## Demo Script (2.5 minutes)

### 0:00–0:10 — Hook
> "ทุกแอปบอกค่าฝุ่นตอนนี้ แต่เราจะบอกว่าอีก 1 ชั่วโมง ฝุ่นจะพุ่งไหม"

### 0:10–0:30 — Show Live Dashboard
- Select Station 59
- Point out current PM2.5, model predictions, spike risk badge
- "3 โมเดลเห็นตรงกัน: ฝุ่นกำลังจะพุ่ง"

### 0:30–1:30 — Spike Replay
- Switch to Replay Mode
- Select 2025-01-24 (Station 59: morning spike 64→108)
- Play animation: actual reveals hour by hour
- Models predicted spike 3-4 hours early
- Banner: "Transformer detected spike 4 hours before peak!"

### 1:30–2:00 — Model Battle Scoreboard
- Show final ranking
- "Transformer ชนะที่ long-range, LSTM ดีที่ short-term"

### 2:00–2:30 — Why It Matters
- "นี่คือ early warning system ไม่ใช่แค่ dashboard"
- "ถ้ารู้ก่อน 3 ชม. คนจะใส่หน้ากาก ปิดหน้าต่าง เลี่ยงพื้นที่เสี่ยงได้ทัน"

---

## Page Structure for Streamlit App

```
app_foonalert_demo.py
├── Page: 🌫️ Live Dashboard
│   ├── Station selector
│   ├── Current PM2.5 metric card
│   ├── Next 1h/6h prediction cards
│   ├── Spike Risk badge
│   ├── Model comparison table
│   └── Actual vs Predicted chart (past 48h + forecast)
│
├── Page: ⏮️ Spike Replay
│   ├── Station + Date selector (curated spike days)
│   ├── Play/Pause/Step controls
│   ├── Progressive reveal chart
│   ├── Model scoreboard (live updating)
│   └── Spike detection banner
│
└── Page: 🏆 Model Battle
    ├── Overall metrics scoreboard
    ├── Error by forecast horizon chart
    ├── Error by PM2.5 severity chart
    └── Spike detection timeline
```

---

## Data Requirements

### For Live Dashboard
- Recent 48h of actual PM2.5 from PostgreSQL (or CSV fallback)
- Model predictions (pre-computed or on-the-fly)

### For Spike Replay
- `demo_data/replay_{station}_{date}.csv` — hourly actual values
- `demo_data/predictions_{station}_{date}_{model}.csv` — model predictions at each hour

### For Model Battle
- `demo_data/model_metrics.csv` — MAE, RMSE, R², Spike Recall per model
- `demo_data/error_analysis.csv` — error breakdown by horizon and severity

---

## Color Scheme & AQI Levels

| Level | PM2.5 Range | Color | Emoji | Label |
|-------|-------------|-------|-------|-------|
| Good | 0–25 | 🟢 Green | 🟢 | Good |
| Moderate | 25.1–37.5 | 🟡 Yellow | 🟡 | Moderate |
| Unhealthy (Sensitive) | 37.6–75 | 🟠 Orange | 🟠 | Unhealthy for Sensitive |
| Unhealthy | 75.1–100 | 🔴 Red | 🔴 | Unhealthy |
| Very Unhealthy | 100.1–150 | 🟣 Purple | 🟣 | Very Unhealthy |
| Hazardous | >150 | 🟤 Maroon | ⚫ | Hazardous |

---

## Technical Notes

- All data pre-cached in `demo_data/` folder (no live DB dependency for demo)
- Fallback: if DB connection fails, load from CSV
- Replay animation uses `st.empty()` + loop with `time.sleep(0.5)` or slider
- Charts use Plotly for interactivity
- Keep demo under 30s load time even on slow EC2
