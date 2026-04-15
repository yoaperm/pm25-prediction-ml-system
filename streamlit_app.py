"""
FoonAlert — PM2.5 Prediction Dashboard
=======================================
Streamlit front-end for the PM2.5 prediction system.

Pages:
  1. Predict   — auto-load history → date picker → forecast chart
  2. Results   — model comparison
  3. Monitoring — MAE / PSI trends

History data loading (tried in order):
  1. GET {API_URL}/history          — if a /history endpoint is ever added to api.py
  2. results/actuals_log.csv        — actual PM2.5 values written by POST /actual
  3. results/predictions_log.csv    — predicted values used as proxy
  4. data/processed/*.parquet       — processed feature files (local dev only)
"""

import os
import glob
import datetime
import httpx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL     = os.environ.get("API_URL",     "http://localhost:8001")
API_KEY     = os.environ.get("API_KEY",     "foonalert-secret-key")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

VALID_USERS = {
    "admin":   "foonalert2026",
    "yoghurt": "yoghurt123",
    "nick":    "nick123",
    "sunta":   "sunta123",
    "boss":    "boss123",
    "perm":    "perm123",
}

STATIONS = [
    {"id": "10T",  "name_th": "เคหะชุมชนคลองจั่น",         "district": "เขตบางกะปิ, กทม."},
    {"id": "35T",  "name_th": "ริมถนนพหลโยธิน",             "district": "เขตจตุจักร, กทม."},
    {"id": "36T",  "name_th": "สวนลุมพินี",                  "district": "เขตปทุมวัน, กทม."},
    {"id": "52T",  "name_th": "ราษฎร์บูรณะ",                "district": "เขตราษฎร์บูรณะ, กทม."},
    {"id": "61T",  "name_th": "ดินแดง",                      "district": "เขตดินแดง, กทม."},
    {"id": "145",  "name_th": "บางขุนเทียน",                 "district": "เขตบางขุนเทียน, กทม."},
    {"id": "02T",  "name_th": "ริมถนนวิภาวดีรังสิต",         "district": "เขตดอนเมือง, กทม."},
    {"id": "11T",  "name_th": "โรงเรียนวัดราษฎร์บูรณะ",     "district": "สมุทรปราการ"},
    {"id": "76T",  "name_th": "เทศบาลเมืองเชียงราย",         "district": "เชียงราย"},
    {"id": "91T",  "name_th": "อุทยานแห่งชาติดอยอินทนนท์",  "district": "เชียงใหม่"},
]
STATION_LABELS = [f"{s['id']} — {s['name_th']}, {s['district']}" for s in STATIONS]


# ── Auth ──────────────────────────────────────────────────────────────────────

def check_login() -> bool:
    if st.session_state.get("authenticated"):
        return True
    st.markdown("## 🔐 Login to FoonAlert")
    with st.form("login_form"):
        username  = st.text_input("Username")
        password  = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"]      = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    return False


def api_headers() -> dict:
    return {"X-API-Key": API_KEY}


# ── Data loading — multi-source fallback ──────────────────────────────────────

def _try_api_history(end_date: datetime.date, num_days: int) -> pd.DataFrame:
    """
    Try GET {API_URL}/history.
    Returns DataFrame['date','pm25'] or empty DataFrame if endpoint missing / fails.
    NOTE: This endpoint does not exist in the default api.py.
    Add it there if you want this source to work.
    """
    try:
        resp = httpx.get(
            f"{API_URL}/history",
            params={"days": num_days, "end_date": str(end_date)},
            headers=api_headers(),
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("history") or data.get("data") or []
            if records:
                df = pd.DataFrame(records)
                # normalise column names
                df.columns = [c.lower() for c in df.columns]
                if "pm25" not in df.columns:
                    for alias in ("pm2_5", "pm25_actual", "value"):
                        if alias in df.columns:
                            df = df.rename(columns={alias: "pm25"})
                            break
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                return df[["date", "pm25"]].dropna()
    except Exception:
        pass
    return pd.DataFrame(columns=["date", "pm25"])


def _try_actuals_log(end_date: datetime.date, num_days: int) -> pd.DataFrame:
    """Read from results/actuals_log.csv (written by POST /actual)."""
    path = os.path.join(RESULTS_DIR, "actuals_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "pm25"])
    try:
        df = pd.read_csv(path)
        # column may be 'date' or 'actual_date', value may be 'pm25_actual' or 'pm25'
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        val_col  = next((c for c in df.columns if "pm25" in c.lower()), None)
        if date_col is None or val_col is None:
            return pd.DataFrame(columns=["date", "pm25"])
        df = df.rename(columns={date_col: "date", val_col: "pm25"})
        df["date"] = pd.to_datetime(df["date"])
        df = (
            df[df["date"].dt.date <= end_date]
            .sort_values("date")
            .tail(num_days)
        )
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[["date", "pm25"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["date", "pm25"])


def _try_predictions_log(end_date: datetime.date, num_days: int) -> pd.DataFrame:
    """Read from results/predictions_log.csv — uses predicted values as proxy history."""
    path = os.path.join(RESULTS_DIR, "predictions_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "pm25"])
    try:
        df = pd.read_csv(path)
        # Be explicit: prediction_date → date, predicted_pm25 → pm25
        # Generic fallback uses strict exclusions to avoid "predict" matching "prediction_date"
        date_col = next(
            (c for c in df.columns if c.lower() in ("prediction_date", "date", "ds")), None
        )
        if date_col is None:
            date_col = next((c for c in df.columns if "date" in c.lower()), None)

        val_col = next((c for c in df.columns if "pm25" in c.lower()), None)
        if val_col is None:
            # Only match "predict*" columns that are NOT date columns
            val_col = next(
                (c for c in df.columns
                 if "predict" in c.lower() and "date" not in c.lower()),
                None,
            )

        if date_col is None or val_col is None:
            return pd.DataFrame(columns=["date", "pm25"])

        df = df.rename(columns={date_col: "date", val_col: "pm25"})
        df["date"] = pd.to_datetime(df["date"])
        # prediction_date = the next day after the history ends.
        # Include predictions whose prediction_date <= end_date + 1 day.
        cutoff = end_date + datetime.timedelta(days=1)
        df = (
            df[df["date"].dt.date <= cutoff]
            .sort_values("date")
            .tail(num_days)
        )
        # Shift back by 1 day so "date" represents the last history day, not the prediction day
        df["date"] = (df["date"] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        return df[["date", "pm25"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["date", "pm25"])


def _try_parquet(end_date: datetime.date, num_days: int) -> pd.DataFrame:
    """
    Read processed Parquet files from data/processed/ (local dev / Airflow container).
    Looks for a 'pm25' column; skips feature-engineering files that don't have raw values.
    """
    patterns = [
        os.path.join("data", "processed", "*.parquet"),
        os.path.join("/app", "data", "processed", "*.parquet"),
    ]
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    if not files:
        return pd.DataFrame(columns=["date", "pm25"])
    for fpath in sorted(files):
        try:
            df = pd.read_parquet(fpath)
            df.columns = [c.lower() for c in df.columns]
            date_col = next((c for c in df.columns if c in ("date", "ds", "timestamp")), None)
            val_col  = next((c for c in df.columns if c in ("pm25", "pm2_5", "value", "pm25_actual")), None)
            if date_col is None or val_col is None:
                continue
            df = df.rename(columns={date_col: "date", val_col: "pm25"})
            df["date"] = pd.to_datetime(df["date"])
            df = (
                df[df["date"].dt.date <= end_date]
                .sort_values("date")
                .tail(num_days)
            )
            if not df.empty:
                df["date"] = df["date"].dt.strftime("%Y-%m-%d")
                return df[["date", "pm25"]].dropna()
        except Exception:
            continue
    return pd.DataFrame(columns=["date", "pm25"])


def load_history(
    end_date: datetime.date,
    num_days: int = 30,
) -> tuple[pd.DataFrame, str]:
    """
    Load PM2.5 history up to `end_date` (inclusive), trying sources in priority order.
    Returns (df, source_label).
    """
    steps = [
        (_try_api_history,      "API /history endpoint"),
        (_try_actuals_log,      "results/actuals_log.csv"),
        (_try_predictions_log,  "results/predictions_log.csv  *(predicted values used as proxy)*"),
        (_try_parquet,          "data/processed/*.parquet"),
    ]
    for fn, label in steps:
        df = fn(end_date, num_days)
        if not df.empty:
            return df, label
    return pd.DataFrame(columns=["date", "pm25"]), "none"


def get_available_date_range() -> tuple[datetime.date | None, datetime.date | None]:
    """
    Scan data sources that contain HISTORICAL PM2.5 readings (not forward predictions).
    predictions_log is intentionally excluded here because its 'prediction_date'
    is always a FUTURE date, causing the date-picker to show a range with no loadable history.
    """
    all_dates: list[datetime.date] = []

    # 1. actuals_log — actual measured PM2.5 values, the best source
    path = os.path.join(RESULTS_DIR, "actuals_log.csv")
    if os.path.exists(path):
        try:
            df       = pd.read_csv(path)
            date_col = next((c for c in df.columns if "date" in c.lower()), None)
            if date_col:
                dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                all_dates += dates.dt.date.tolist()
        except Exception:
            pass

    # 2. Parquet processed files (local dev / Airflow volume)
    for pat in [
        os.path.join("data", "processed", "*.parquet"),
        os.path.join("/app", "data", "processed", "*.parquet"),
    ]:
        for fpath in glob.glob(pat):
            try:
                df       = pd.read_parquet(fpath)
                df.columns = [c.lower() for c in df.columns]
                date_col = next((c for c in df.columns if c in ("date", "ds", "timestamp")), None)
                if date_col:
                    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                    all_dates += dates.dt.date.tolist()
            except Exception:
                pass

    # 3. predictions_log — ONLY for date-range discovery when no other source exists.
    #    Its 'prediction_date' is tomorrow's date, so subtract 1 day to approximate
    #    the last day of history that was available when the prediction was made.
    if not all_dates:
        path = os.path.join(RESULTS_DIR, "predictions_log.csv")
        if os.path.exists(path):
            try:
                df       = pd.read_csv(path)
                # explicitly prefer "prediction_date" over any other date-ish column
                date_col = next(
                    (c for c in df.columns if c.lower() in ("prediction_date", "date", "ds")),
                    None,
                )
                if date_col is None:
                    date_col = next((c for c in df.columns if "date" in c.lower()), None)
                if date_col:
                    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
                    # prediction_date is always "next day", so history ends 1 day before
                    hist_dates = [d.date() - datetime.timedelta(days=1) for d in dates]
                    all_dates += hist_dates
            except Exception:
                pass

    if all_dates:
        return min(all_dates), max(all_dates)
    return None, None


# ── API prediction call ────────────────────────────────────────────────────────

def _call_predict(history: list) -> dict | None:
    try:
        resp = httpx.post(
            f"{API_URL}/predict",
            json={"history": history},
            headers=api_headers(),
            timeout=30,
        )
        if resp.status_code == 401:
            st.error("API authentication failed.")
            return None
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        st.error("Cannot connect to prediction API. Is the API service running?")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.text}")
        return None


# ── Chart ─────────────────────────────────────────────────────────────────────

def build_forecast_chart(
    station_label: str,
    history: list,
    future_preds: list,
    is_today: bool = True,
    history_source: str = "results/actuals_log.csv",
) -> go.Figure:
    """
    3-line Plotly chart:
      ─  light green  solid  → Real PM2.5 past 24 hr  (when history_source = actuals/API)
                               Predicted PM2.5 proxy    (when history_source = predictions_log)
      ┄  light grey   dotted → Past predictions from predictions_log  (only when history is actuals)
      ┄  light blue   dotted → Forecast next 24 hr  (is_today=True)
      ┄  grey         dotted → Forecast for a past date (is_today=False)
    """
    FORECAST_COLOR  = "#ADD8E6" if is_today else "#A8A8A8"
    FORECAST_MARKER = "#5BC8F5" if is_today else "#808080"
    FORECAST_LABEL  = "Forecast — next 24 hr" if is_today else "Forecast (historical) — 24 hr"

    # History is "real" only if it came from measured actuals (API, actuals_log, or user-supplied)
    HISTORY_IS_ACTUAL = history_source in (
        "API /history endpoint",
        "results/actuals_log.csv",
        "manual",
    )

    recent  = history[-24:] if len(history) >= 24 else history
    hist_df = pd.DataFrame(recent)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    # ── Past predictions from log (grey dotted line) ───────────────────────────
    # Only shown when history is REAL actuals — otherwise it's the same data as history
    pred_log_path = os.path.join(RESULTS_DIR, "predictions_log.csv")
    past_pred_x, past_pred_y = [], []
    if HISTORY_IS_ACTUAL and os.path.exists(pred_log_path):
        try:
            plog     = pd.read_csv(pred_log_path)
            date_col = next(
                (c for c in plog.columns if c.lower() in ("prediction_date", "date", "ds")), None
            )
            if date_col is None:
                date_col = next((c for c in plog.columns if "date" in c.lower()), None)
            val_col = next((c for c in plog.columns if "pm25" in c.lower()), None)
            if val_col is None:
                val_col = next(
                    (c for c in plog.columns
                     if "predict" in c.lower() and "date" not in c.lower()),
                    None,
                )
            if date_col and val_col:
                plog = plog.rename(columns={date_col: "date", val_col: "pm25"})
                plog["date"] = pd.to_datetime(plog["date"])
                # Shift prediction_date back 1 day → aligns with history "last measured day"
                plog["date"] = plog["date"] - pd.Timedelta(days=1)
                hist_dates   = set(hist_df["date"].dt.strftime("%Y-%m-%d"))
                matched      = plog[plog["date"].dt.strftime("%Y-%m-%d").isin(hist_dates)].sort_values("date")
                if not matched.empty:
                    past_pred_x = matched["date"].tolist()
                    past_pred_y = matched["pm25"].tolist()
        except Exception:
            pass

    # Future predictions
    future_df = pd.DataFrame(future_preds) if future_preds else pd.DataFrame(columns=["date", "pm25"])
    if not future_df.empty:
        future_df["date"] = pd.to_datetime(future_df["date"])

    # ── Smart Y-axis range ─────────────────────────────────────────────────────
    all_vals = list(hist_df["pm25"].dropna())
    if not future_df.empty:
        all_vals += list(future_df["pm25"].dropna())
    if past_pred_y:
        all_vals += [v for v in past_pred_y if v is not None]

    if all_vals:
        data_max = max(all_vals)
        # Top: 25% padding above max, minimum ceiling of 60 so AQI bands are always visible
        y_top = max(data_max * 1.25, 60)
    else:
        y_top = 100

    # AQI-aligned tick marks up to y_top
    aqi_ticks = [0, 15, 25, 37, 50, 75, 100, 150, 200, 300]
    tickvals  = [v for v in aqi_ticks if v <= y_top]
    if not tickvals or tickvals[-1] < y_top * 0.9:
        tickvals.append(int(round(y_top)))

    fig = go.Figure()

    # AQI background bands (clipped to y_top)
    for y0, y1, fc in [
        (0,    15,   "rgba(144,238,144,0.10)"),
        (15,   25,   "rgba(255,255,100,0.08)"),
        (25,   37.5, "rgba(255,165,0,0.08)"),
        (37.5, 75,   "rgba(255,80,80,0.07)"),
        (75,   300,  "rgba(160,0,200,0.05)"),
    ]:
        if y0 < y_top:
            fig.add_hrect(y0=y0, y1=min(y1, y_top), fillcolor=fc, line_width=0, layer="below")

    # Line 1 — History PM2.5 (light green, solid ─)
    # Label changes based on whether history is real measured data or predicted proxy
    hist_line_label = (
        "Real PM2.5 — past 24 hr" if HISTORY_IS_ACTUAL
        else "Predicted PM2.5 — past 24 hr (proxy, ไม่มีข้อมูลจริง)"
    )
    fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["pm25"],
        mode="lines+markers",
        name=hist_line_label,
        line=dict(color="#90EE90", width=2.5, dash="solid"),
        marker=dict(size=5, color="#4CAF50"),
        hovertemplate="%{x|%Y-%m-%d}<br>"
                      + ("Actual" if HISTORY_IS_ACTUAL else "Proxy") +
                      ": <b>%{y:.1f} µg/m³</b><extra></extra>",
    ))

    # Line 2 — Past predictions (light grey, dotted ┄)
    if past_pred_x:
        fig.add_trace(go.Scatter(
            x=past_pred_x, y=past_pred_y,
            mode="lines+markers",
            name="Predicted — past 24 hr",
            line=dict(color="#C0C0C0", width=2, dash="dot"),
            marker=dict(size=4, color="#A0A0A0"),
            hovertemplate="%{x|%Y-%m-%d}<br>Past pred: <b>%{y:.1f} µg/m³</b><extra></extra>",
        ))

    # Line 3 — Forecast (dotted ┄, colour depends on is_today)
    if not future_df.empty:
        # thin connector from last real → first forecast
        fig.add_trace(go.Scatter(
            x=[hist_df["date"].iloc[-1], future_df["date"].iloc[0]],
            y=[hist_df["pm25"].iloc[-1], future_df["pm25"].iloc[0]],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=FORECAST_COLOR, width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=future_df["date"], y=future_df["pm25"],
            mode="lines+markers",
            name=FORECAST_LABEL,
            line=dict(color=FORECAST_COLOR, width=2.5, dash="dot"),
            marker=dict(size=5, color=FORECAST_MARKER),
            hovertemplate="%{x|%Y-%m-%d}<br>Forecast: <b>%{y:.1f} µg/m³</b><extra></extra>",
        ))

    # "Today" vertical divider — ISO string to avoid Timestamp bug
    if not hist_df.empty:
        today_str     = hist_df["date"].iloc[-1].strftime("%Y-%m-%d")
        divider_label = "Today" if is_today else today_str
        fig.add_shape(
            type="line",
            x0=today_str, x1=today_str,
            y0=0, y1=1, xref="x", yref="paper",
            line=dict(color="rgba(160,160,160,0.7)", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=today_str, y=1, xref="x", yref="paper",
            text=divider_label, showarrow=False,
            font=dict(size=11, color="gray"),
            xanchor="left", yanchor="bottom",
        )

    # ── X-axis locked to 24 hr before + 24 hr forecast ───────────────────────
    x_start = hist_df["date"].iloc[0].strftime("%Y-%m-%d")
    x_end   = (
        future_df["date"].iloc[-1].strftime("%Y-%m-%d")
        if not future_df.empty
        else hist_df["date"].iloc[-1].strftime("%Y-%m-%d")
    )

    fig.update_layout(
        title=dict(text=f"PM2.5 Forecast — {station_label}", font=dict(size=16)),
        xaxis_title="Date",
        yaxis_title="PM2.5 (µg/m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
            range=[x_start, x_end],    # ← lock to exactly 24 before + 24 after
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(200,200,200,0.2)",
            range=[0, y_top],
            tickvals=tickvals,
            ticktext=[str(v) for v in tickvals],
        ),
        margin=dict(t=80, b=40),
    )
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_predict():
    # ── Station selector (sidebar) ─────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📍 Station")
    selected_idx = st.sidebar.selectbox(
        "Select monitoring station",
        options=range(len(STATION_LABELS)),
        format_func=lambda i: STATION_LABELS[i],
        index=0,
        key="station_selector",
    )
    station       = STATIONS[selected_idx]
    station_id    = station["id"]
    station_label = STATION_LABELS[selected_idx]

    st.sidebar.markdown(
        f"""
        <div style="
            background:rgba(144,238,144,0.12);
            border:1px solid rgba(144,238,144,0.35);
            border-radius:8px;
            padding:10px 12px;
            font-size:12.5px;
            line-height:1.7;
        ">
            🏭 <b>Station {station_id}</b><br>
            {station['name_th']}<br>
            <span style="color:gray">{station['district']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**FoonAlert** v1.0 \nPM2.5 Prediction System")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.header("🌫️ PM2.5 Next-Day Prediction")
    st.caption(f"📍 {station_label}")

    # ── Input mode ─────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Input method",
        ["🔄 Auto-load from data", "✏️ Manual entry", "📂 Upload CSV"],
        horizontal=True,
    )

    history: list = []
    is_today: bool = True    # overridden in auto-load and after history is built
    source_used: str = "manual"  # manual/csv = user-supplied real data → treated as actual

    # ══ AUTO-LOAD mode ══════════════════════════════════════════════════════════
    if input_mode == "🔄 Auto-load from data":
        # Discover what dates are available across all sources
        min_date, max_date = get_available_date_range()

        if max_date is None:
            st.warning("⚠️ ยังไม่พบข้อมูลประวัติ PM2.5 จากแหล่งใดเลย")
            st.markdown(
                """
                **สาเหตุที่เป็นไปได้:**
                - `results/actuals_log.csv` ยังว่างอยู่ — ต้อง submit actuals ผ่าน `POST /actual` ก่อน หรือรัน `mock_pipeline.py`
                - `data/processed/*.parquet` ไม่ได้ mount เข้า Streamlit container (ตรวจ `docker-compose.yml`)

                **แก้ไขด่วน:** เปลี่ยนไปใช้ **✏️ Manual entry** หรือ **📂 Upload CSV** แทน
                """
            )
            # Debug: show which files were found
            with st.expander("🔍 ตรวจสอบไฟล์ข้อมูล"):
                for fname in ["actuals_log.csv", "predictions_log.csv", "monitoring_results.csv"]:
                    fpath = os.path.join(RESULTS_DIR, fname)
                    if os.path.exists(fpath):
                        try:
                            df_check = pd.read_csv(fpath)
                            st.success(f"✅ `{fname}` — {len(df_check)} rows, columns: {list(df_check.columns)}")
                        except Exception as e:
                            st.error(f"❌ `{fname}` exists but failed to read: {e}")
                    else:
                        st.info(f"📭 `{fname}` — ไม่พบไฟล์ใน `{RESULTS_DIR}/`")
            return

        col_date, col_days = st.columns([2, 1])
        with col_date:
            today = datetime.date.today()
            # Default to today if we have data for today, else the latest available date
            default_end = today if (min_date <= today <= max_date) else max_date
            if default_end != today:
                st.caption(
                    f"ℹ️ ไม่พบข้อมูลวันนี้ ({today}) — ใช้วันล่าสุดที่มีข้อมูล: **{max_date}**"
                )
            selected_end = st.date_input(
                "วันที่สิ้นสุด (วันสุดท้ายของช่วงข้อมูล)",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                help="เลือกวันที่ภายในช่วงที่มีข้อมูล",
            )
        with col_days:
            num_days = st.number_input(
                "Window (days)", min_value=15, max_value=90, value=30,
                help="Number of days ending on the selected date.",
            )

        # Determine whether the selected date is today (affects forecast line colour)
        is_today = (selected_end == datetime.date.today())

        # Load data with fallback chain
        with st.spinner("Loading data…"):
            df_loaded, source_used = load_history(selected_end, num_days)

        if df_loaded.empty:
            st.warning(
                f"ไม่พบข้อมูลในช่วงวันที่ **{selected_end - datetime.timedelta(days=num_days-1)}** "
                f"ถึง **{selected_end}** จากแหล่งข้อมูลใดเลย"
            )
            with st.expander("🔍 ตรวจสอบไฟล์"):
                for fname in ["actuals_log.csv", "predictions_log.csv"]:
                    fpath = os.path.join(RESULTS_DIR, fname)
                    if os.path.exists(fpath):
                        try:
                            df_check = pd.read_csv(fpath)
                            st.info(f"`{fname}` — {len(df_check)} rows, columns: {list(df_check.columns)}")
                        except Exception as e:
                            st.error(f"`{fname}`: {e}")
                    else:
                        st.info(f"`{fname}` — ไม่พบไฟล์")
            return

        history = df_loaded.to_dict("records")

        # Status banner — show which source was used
        source_icon = {
            "API /history endpoint":          "🌐",
            "results/actuals_log.csv":        "📄",
        }.get(source_used.split("*")[0].strip(), "📦")
        st.success(
            f"{source_icon} Loaded **{len(history)} days** from "
            f"`{source_used}` &nbsp;·&nbsp; "
            f"{history[0]['date']} → {history[-1]['date']}"
        )

        with st.expander("Preview loaded data"):
            preview_df = pd.DataFrame(history).rename(
                columns={"date": "Date", "pm25": "PM2.5 (µg/m³)"}
            )
            st.dataframe(preview_df, use_container_width=True)

    # ══ MANUAL ENTRY mode ═══════════════════════════════════════════════════════
    elif input_mode == "✏️ Manual entry":
        st.markdown("Enter at least **15 consecutive daily PM2.5 readings** (oldest first).")
        num_days   = st.number_input("Number of days", min_value=15, max_value=60, value=15)
        start_date = st.date_input(
            "Start date",
            value=datetime.date.today() - datetime.timedelta(days=int(num_days)),
        )
        cols = st.columns(3)
        for i in range(int(num_days)):
            d = start_date + datetime.timedelta(days=i)
            with cols[i % 3]:
                val = st.number_input(
                    f"{d.strftime('%Y-%m-%d')}",
                    min_value=0.0, max_value=500.0, value=25.0, step=0.1,
                    key=f"pm25_{i}",
                )
            history.append({"date": str(d), "pm25": val})

    # ══ CSV UPLOAD mode ══════════════════════════════════════════════════════════
    else:
        st.markdown("Upload a CSV with columns `date` and `pm25` (at least 15 rows).")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if "date" not in df.columns or "pm25" not in df.columns:
                    st.error("CSV must have `date` and `pm25` columns")
                    return
                df      = df.sort_values("date")
                history = df[["date", "pm25"]].to_dict("records")
                st.dataframe(df.tail(20), use_container_width=True)
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return

    if not history:
        return

    # Compute is_today from actual last date in history (works for ALL input modes)
    try:
        last_hist_date = datetime.date.fromisoformat(str(history[-1]["date"]))
        is_today = (last_hist_date >= datetime.date.today())
    except Exception:
        is_today = True

    # ── Predict button ─────────────────────────────────────────────────────────
    if st.button("🔮 Predict", type="primary", disabled=len(history) < 15):

        # Step 1 — next-day prediction
        with st.spinner("Calling prediction API…"):
            result = _call_predict(history)
        if result is None:
            return

        pm25_val = result["predicted_pm25"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction Date", result["prediction_date"])
        with col2:
            st.metric("Predicted PM2.5", f"{pm25_val:.1f} µg/m³")
        with col3:
            if   pm25_val <= 15:   level, icon = "Good",                 "🟢"
            elif pm25_val <= 25:   level, icon = "Moderate",              "🟡"
            elif pm25_val <= 37.5: level, icon = "Unhealthy (Sensitive)", "🟠"
            elif pm25_val <= 75:   level, icon = "Unhealthy",             "🔴"
            else:                  level, icon = "Very Unhealthy",        "🟣"
            st.metric("Air Quality", f"{icon} {level}")

        st.success(f"Model: **{result['model']}**")

        # Step 2 — 24-day rolling forecast (iterative API calls)
        future_preds: list = []
        rolling   = list(history)
        last_date = datetime.date.fromisoformat(str(history[-1]["date"]))

        prog = st.progress(0, text="Generating 24 hr forecast…")
        for i in range(24):
            next_date = last_date + datetime.timedelta(days=i + 1)
            r = _call_predict(rolling[-30:])
            if r is None:
                break
            pred_val = r["predicted_pm25"]
            future_preds.append({"date": str(next_date), "pm25": pred_val})
            rolling.append({"date": str(next_date), "pm25": pred_val})
            prog.progress((i + 1) / 24, text=f"Forecasting hr {i + 1}/24…")
        prog.empty()

        # Step 3 — chart
        st.markdown("---")
        st.subheader("📈 PM2.5 Forecast Chart")

        # ── Inline legend using real SVG dashed lines ──────────────────────────
        history_is_actual = source_used in (
            "API /history endpoint",
            "results/actuals_log.csv",
        )
        forecast_svg_color = "#ADD8E6" if is_today else "#A8A8A8"
        forecast_label     = "Forecast — next 24 hr" if is_today else "Forecast (historical) — 24 hr"
        green_label        = (
            "Real PM2.5 — past 24 hr" if history_is_actual
            else "Predicted PM2.5 — past 24 hr (proxy)"
        )
        st.markdown(
            f"""
            <div style="display:flex;gap:32px;align-items:center;
                        flex-wrap:wrap;margin-bottom:4px;font-size:13.5px;">
              <div style="display:flex;align-items:center;gap:8px;">
                <svg width="44" height="12">
                  <line x1="0" y1="6" x2="44" y2="6"
                        stroke="#90EE90" stroke-width="2.5"/>
                  <circle cx="8"  cy="6" r="3.5" fill="#4CAF50"/>
                  <circle cx="22" cy="6" r="3.5" fill="#4CAF50"/>
                  <circle cx="36" cy="6" r="3.5" fill="#4CAF50"/>
                </svg>
                <span>{green_label}</span>
              </div>
              {"" if not history_is_actual else '''
              <div style="display:flex;align-items:center;gap:8px;">
                <svg width="44" height="12">
                  <line x1="0" y1="6" x2="44" y2="6"
                        stroke="#C0C0C0" stroke-width="2"
                        stroke-dasharray="4 3"/>
                  <circle cx="8"  cy="6" r="3" fill="#A0A0A0"/>
                  <circle cx="22" cy="6" r="3" fill="#A0A0A0"/>
                  <circle cx="36" cy="6" r="3" fill="#A0A0A0"/>
                </svg>
                <span>Predicted — past 24 hr (from log)</span>
              </div>'''}
              <div style="display:flex;align-items:center;gap:8px;">
                <svg width="44" height="12">
                  <line x1="0" y1="6" x2="44" y2="6"
                        stroke="{forecast_svg_color}" stroke-width="2.5"
                        stroke-dasharray="4 3"/>
                  <circle cx="8"  cy="6" r="3.5" fill="{forecast_svg_color}"/>
                  <circle cx="22" cy="6" r="3.5" fill="{forecast_svg_color}"/>
                  <circle cx="36" cy="6" r="3.5" fill="{forecast_svg_color}"/>
                </svg>
                <span>{forecast_label}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Warning banner if not using real measured data
        if not history_is_actual:
            st.warning(
                "⚠️ **ไม่พบข้อมูล PM2.5 จริง** (`actuals_log.csv` ไม่มีข้อมูล) — "
                "กราฟเส้นเขียวแสดง **ค่าทำนายย้อนหลัง** จาก `predictions_log.csv` แทน  \n"
                "เส้นสีเทา *Predicted — past 24 hr* จะไม่แสดงเพราะข้อมูลเดียวกับเส้นเขียว  \n"
                "เพื่อเห็นค่าจริง: รัน `mock_pipeline.py` หรือ submit ข้อมูลจริงผ่าน `POST /actual`"
            )

        fig = build_forecast_chart(
            station_label, history, future_preds,
            is_today=is_today,
            history_source=source_used,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Step 4 — expandable forecast table
        if future_preds:
            with st.expander("📋 24 hr Forecast Table"):
                fdf = pd.DataFrame(future_preds)
                fdf.columns = ["Date", "Predicted PM2.5 (µg/m³)"]
                fdf["Air Quality"] = fdf["Predicted PM2.5 (µg/m³)"].apply(
                    lambda v:
                        "🟢 Good"                 if v <= 15   else
                        "🟡 Moderate"              if v <= 25   else
                        "🟠 Unhealthy (Sensitive)"  if v <= 37.5 else
                        "🔴 Unhealthy"             if v <= 75   else
                        "🟣 Very Unhealthy"
                )
                st.dataframe(fdf, use_container_width=True)


# ── Results page ───────────────────────────────────────────────────────────────

def page_results():
    st.header("📊 Model Comparison Results")

    results_path = os.path.join(RESULTS_DIR, "experiment_results_station145.csv")
    if not os.path.exists(results_path):
        results_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    if not os.path.exists(results_path):
        st.warning("No experiment results found. Run the training pipeline first.")
        return

    df = pd.read_csv(results_path)
    st.markdown("### Experiment Results")
    st.dataframe(
        df.style.highlight_min(subset=["MAE", "RMSE"], color="#90EE90"),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="model", y="MAE",
                     title="MAE by Model (lower is better)",
                     color="MAE", color_continuous_scale="RdYlGn_r")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x="model", y="R2",
                     title="R² by Model (higher is better)",
                     color="R2", color_continuous_scale="RdYlGn")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    best = df.loc[df["MAE"].idxmin()]
    st.info(
        f"🏆 **Champion Model:** {best['model']} — "
        f"MAE: {best['MAE']:.4f} µg/m³, RMSE: {best['RMSE']:.4f}, R²: {best['R2']:.4f}"
    )


# ── Monitoring page ────────────────────────────────────────────────────────────

def page_monitoring():
    st.header("📈 Monitoring Dashboard")

    pred_path       = os.path.join(RESULTS_DIR, "predictions_log.csv")
    actual_path     = os.path.join(RESULTS_DIR, "actuals_log.csv")
    monitoring_path = os.path.join(RESULTS_DIR, "monitoring_results.csv")

    if os.path.exists(monitoring_path):
        st.markdown("### MAE & PSI Trends")
        mon_df = pd.read_csv(monitoring_path)

        if "date" in mon_df.columns or "timestamp" in mon_df.columns:
            date_col = "date" if "date" in mon_df.columns else "timestamp"
            mon_df[date_col] = pd.to_datetime(mon_df[date_col])

            if "mae" in mon_df.columns:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    x=mon_df[date_col], y=mon_df["mae"],
                    mode="lines+markers", name="MAE",
                    line=dict(color="#FF6B6B", width=2),
                ))
                fig_mae.add_hline(y=6.0, line_dash="dash", line_color="red",
                                  annotation_text="Retrain threshold (6.0)")
                fig_mae.update_layout(title="Rolling MAE Over Time",
                                      xaxis_title="Date", yaxis_title="MAE (µg/m³)")
                st.plotly_chart(fig_mae, use_container_width=True)

            if "psi" in mon_df.columns:
                fig_psi = go.Figure()
                fig_psi.add_trace(go.Scatter(
                    x=mon_df[date_col], y=mon_df["psi"],
                    mode="lines+markers", name="PSI",
                    line=dict(color="#4ECDC4", width=2),
                ))
                fig_psi.add_hline(y=0.2, line_dash="dash", line_color="red",
                                  annotation_text="Drift threshold (0.2)")
                fig_psi.add_hline(y=0.1, line_dash="dot",  line_color="orange",
                                  annotation_text="Monitor threshold (0.1)")
                fig_psi.update_layout(title="PSI Over Time",
                                      xaxis_title="Date", yaxis_title="PSI")
                st.plotly_chart(fig_psi, use_container_width=True)

            if "retrain_triggered" in mon_df.columns:
                retrain_events = mon_df[mon_df["retrain_triggered"] == True]
                if not retrain_events.empty:
                    st.warning(f"⚠️ **{len(retrain_events)} retrain(s) triggered**")
                    st.dataframe(retrain_events, use_container_width=True)
                else:
                    st.success("✅ No retrains triggered — model is stable")

            st.markdown("### Full Monitoring Log")
            st.dataframe(mon_df, use_container_width=True)
    else:
        st.info("No monitoring results yet. Run the `pm25_pipeline` DAG first.")

    st.markdown("---")
    st.markdown("### Predictions vs Actuals Log")

    if os.path.exists(pred_path) and os.path.exists(actual_path):
        preds_df   = pd.read_csv(pred_path,   parse_dates=["prediction_date"])
        actuals_df = pd.read_csv(actual_path, parse_dates=["date"])
        merged = preds_df.merge(
            actuals_df, left_on="prediction_date", right_on="date", how="inner"
        ).sort_values("prediction_date")

        if not merged.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                mae = (merged["pm25_actual"] - merged["predicted_pm25"]).abs().mean()
                st.metric("Current MAE", f"{mae:.2f} µg/m³")
            with col2:
                st.metric("Matched Pairs", len(merged))
            with col3:
                bias = (merged["predicted_pm25"] - merged["pm25_actual"]).mean()
                st.metric("Prediction Bias", f"{bias:+.2f} µg/m³")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["pm25_actual"], y=merged["predicted_pm25"],
                mode="markers", name="Predictions",
                marker=dict(color="#6C5CE7", size=8, opacity=0.7),
            ))
            max_val = max(merged["pm25_actual"].max(), merged["predicted_pm25"].max()) + 5
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val], mode="lines",
                name="Perfect", line=dict(color="gray", dash="dash"),
            ))
            fig.update_layout(title="Predicted vs Actual PM2.5",
                              xaxis_title="Actual (µg/m³)", yaxis_title="Predicted (µg/m³)")
            st.plotly_chart(fig, use_container_width=True)

            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=merged["prediction_date"], y=merged["pm25_actual"],
                mode="lines+markers", name="Actual", line=dict(color="#00B894", width=2),
            ))
            fig_ts.add_trace(go.Scatter(
                x=merged["prediction_date"], y=merged["predicted_pm25"],
                mode="lines+markers", name="Predicted", line=dict(color="#6C5CE7", width=2),
            ))
            fig_ts.update_layout(title="Actual vs Predicted Over Time",
                                 xaxis_title="Date", yaxis_title="PM2.5 (µg/m³)")
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No matched pairs found yet.")
    else:
        st.info("No logs found. Use the Predict page or run `mock_pipeline.py` to generate data.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="FoonAlert — PM2.5 Prediction",
        page_icon="🌫️",
        layout="wide",
    )

    if not check_login():
        return

    st.sidebar.markdown(f"👤 Logged in as **{st.session_state.get('username', '?')}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🌫️ Predict", "📊 Model Results", "📈 Monitoring"],
    )

    if page == "🌫️ Predict":
        page_predict()
    elif page == "📊 Model Results":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**FoonAlert** v1.0 \nPM2.5 Prediction System")
        page_results()
    elif page == "📈 Monitoring":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**FoonAlert** v1.0 \nPM2.5 Prediction System")
        page_monitoring()


if __name__ == "__main__":
    main()