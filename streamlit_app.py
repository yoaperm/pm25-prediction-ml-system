"""
FoonAlert — PM2.5 Prediction Dashboard
=======================================
Streamlit front-end for the PM2.5 prediction system.

Pages:
  1. Predict  — submit PM2.5 history → get next-day prediction
  2. Results  — model comparison from experiment results
  3. Monitoring — MAE / PSI trends and retrain history
"""

import os
import datetime

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://localhost:8001")
API_KEY = os.environ.get("API_KEY", "foonalert-secret-key")
RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

VALID_USERS = {
    "admin": "foonalert2026",
    "yoghurt": "yoghurt123",
    "nick": "nick123",
    "sunta": "sunta123",
    "boss": "boss123",
    "perm": "perm123",
}


# ── Auth ──────────────────────────────────────────────────────────────────────
def check_login():
    """Simple session-based authentication."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown("## 🔐 Login to FoonAlert")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username in VALID_USERS and VALID_USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid username or password")
    return False


def api_headers():
    return {"X-API-Key": API_KEY}


# ── Pages ─────────────────────────────────────────────────────────────────────
def page_predict():
    import importlib
    from urllib.parse import urlparse

    station_ids = [56, 57, 58, 59, 61]
    db_candidates = [
        os.environ.get("PM25_DB_URL"),
        "postgresql://admin:admin@postgres:5432/pm25",
        "postgresql://admin:admin@postgres:5432/postgres",
    ]
    db_candidates = [candidate for candidate in db_candidates if candidate]

    psycopg2 = None
    try:
        psycopg2 = importlib.import_module("psycopg2")
    except ModuleNotFoundError:
        psycopg2 = None

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_station_options():
        last_error = None
        station_map = {sid: f"Station {sid}" for sid in station_ids}

        station_query = """
            SELECT
                station_id,
                COALESCE(NULLIF(MAX(station_name), ''), 'Station ' || station_id::text) AS station_name
            FROM pm25_raw_hourly
            WHERE station_id = ANY(%s)
            GROUP BY station_id
            ORDER BY station_id
        """

        for db_url in db_candidates:
            try:
                with _connect_postgres(db_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(station_query, (station_ids,))
                        rows = cur.fetchall()
                for station_id, station_name in rows:
                    station_map[int(station_id)] = station_name
                return db_url, station_map, None
            except Exception as exc:
                last_error = exc

        return None, station_map, last_error

    @st.cache_data(ttl=300, show_spinner=False)
    def _load_station_series(db_url: str, station_id: int):
        actual_query = """
            SELECT
                DATE(timestamp AT TIME ZONE 'Asia/Bangkok') AS reading_date,
                ROUND(AVG(pm25)::numeric, 2) AS actual_pm25
            FROM pm25_raw_hourly
            WHERE station_id = %(station_id)s
              AND pm25 IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """
        predicted_query = """
            SELECT
                prediction_date,
                predicted_pm25
            FROM pm25_api_daily_predictions
            WHERE source_station_id = %(station_id)s
            ORDER BY prediction_date
        """

        with _connect_postgres(db_url) as conn:
            actual_df = pd.read_sql(actual_query, conn, params={"station_id": station_id})
            predicted_df = pd.read_sql(predicted_query, conn, params={"station_id": station_id})

        if not actual_df.empty:
            actual_df["reading_date"] = pd.to_datetime(actual_df["reading_date"])
            actual_df = actual_df.rename(columns={"reading_date": "date"})

        if not predicted_df.empty:
            predicted_df["prediction_date"] = pd.to_datetime(predicted_df["prediction_date"])
            predicted_df = predicted_df.rename(columns={"prediction_date": "date"})

        return actual_df, predicted_df

    def _connect_postgres(db_url: str):
        if psycopg2 is None:
            raise RuntimeError(
                "PostgreSQL driver is not installed. Please install `psycopg2-binary` "
                "or add a supported PostgreSQL client to this Streamlit environment."
            )
        parsed = urlparse(db_url)
        return psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            dbname=(parsed.path or "").lstrip("/"),
            user=parsed.username,
            password=parsed.password,
        )

    def _pm25_level_color(value):
        if pd.isna(value):
            return "#9E9E9E"
        if value <= 15:
            return "#2E8B57"
        if value <= 25:
            return "#D4A017"
        if value <= 37.5:
            return "#FF8C42"
        if value <= 75:
            return "#D62828"
        return "#7B2CBF"

    st.header("🌫️ PM2.5 Next-Day Prediction")
    st.markdown("เลือกสถานีเพื่อดูกราฟ Actual และ Predicted จาก PostgreSQL")

    if psycopg2 is None:
        st.warning(
            "ยังไม่สามารถโหลดกราฟจาก PostgreSQL ได้ เพราะ environment นี้ไม่มี `psycopg2-binary` "
            "หรือ PostgreSQL driver อื่น ๆ"
        )

    db_url, station_map, db_error = _load_station_options()
    station_options = [
        {"station_id": sid, "label": f"{sid} [{station_map.get(sid, f'Station {sid}')}]"}
        for sid in station_ids
    ]

    selected_station = st.selectbox(
        "Station",
        station_options,
        index=0,
        format_func=lambda option: option["label"],
    )
    selected_station_id = selected_station["station_id"]
    selected_station_name = station_map.get(selected_station_id, f"Station {selected_station_id}")

    if psycopg2 is None:
        db_url = None
        db_error = RuntimeError("missing_postgres_driver")

    if not db_url:
        if psycopg2 is not None:
            st.error(
                "ไม่สามารถเชื่อมต่อ PostgreSQL ได้จากหน้า Predict "
                f"({type(db_error).__name__}: {db_error})"
            )
        else:
            st.info("ข้ามส่วนกราฟจาก PostgreSQL ชั่วคราว และยังใช้งานการทำนายแบบเดิมด้านล่างได้")
    else:
        try:
            actual_df, predicted_df = _load_station_series(db_url, selected_station_id)
        except Exception as exc:
            st.error(f"โหลดข้อมูลจาก PostgreSQL ไม่สำเร็จ: {type(exc).__name__}: {exc}")
            actual_df = pd.DataFrame()
            predicted_df = pd.DataFrame()

        if not actual_df.empty or not predicted_df.empty:
            latest_actual_date = None
            if not actual_df.empty:
                latest_actual_date = actual_df["date"].max().normalize()
            latest_predicted_date = None
            if not predicted_df.empty:
                latest_predicted_date = predicted_df["date"].max().normalize()

            available_dates = []
            if not actual_df.empty:
                available_dates.extend(actual_df["date"].dropna().tolist())
            if not predicted_df.empty:
                available_dates.extend(predicted_df["date"].dropna().tolist())

            min_available_date = None
            max_allowed_date = None
            if available_dates:
                min_available_date = pd.Timestamp(min(available_dates)).normalize()
                max_available_date = pd.Timestamp(max(available_dates)).normalize()
                if latest_actual_date is not None:
                    max_allowed_date = max(latest_actual_date + pd.Timedelta(days=1), max_available_date)
                else:
                    max_allowed_date = max_available_date

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Station ID", selected_station_id)
            with col2:
                st.metric("Station Name", selected_station_name)
            with col3:
                st.metric(
                    "Latest Actual Date",
                    latest_actual_date.strftime("%Y-%m-%d") if latest_actual_date is not None else "-",
                )

            selected_display_date = None
            if latest_actual_date is not None:
                selected_display_date = pd.Timestamp(
                    st.date_input(
                        "Display date",
                        value=latest_actual_date.date(),
                        min_value=min_available_date.date() if min_available_date is not None else None,
                        max_value=max_allowed_date.date() if max_allowed_date is not None else None,
                    )
                ).normalize()

            fig = go.Figure()

            if not actual_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=actual_df["date"],
                        y=actual_df["actual_pm25"],
                        mode="lines+markers",
                        name="Actual Data",
                        line=dict(color="#2E8B57", width=3),
                        marker=dict(
                            size=9,
                            color=[_pm25_level_color(value) for value in actual_df["actual_pm25"]],
                            line=dict(color="#FFFFFF", width=1),
                        ),
                        hovertemplate="Date=%{x|%Y-%m-%d}<br>Actual=%{y:.2f} µg/m³<extra></extra>",
                    )
                )

            if not predicted_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=predicted_df["date"],
                        y=predicted_df["predicted_pm25"],
                        mode="lines+markers",
                        name="Predicted Data",
                        line=dict(color="#4A4A4A", width=3, dash="dash"),
                        marker=dict(
                            size=9,
                            color=[_pm25_level_color(value) for value in predicted_df["predicted_pm25"]],
                            line=dict(color="#FFFFFF", width=1),
                        ),
                        hovertemplate="Date=%{x|%Y-%m-%d}<br>Predicted=%{y:.2f} µg/m³<extra></extra>",
                    )
                )

            xaxis_range = None
            if selected_display_date is not None:
                if selected_display_date == latest_actual_date:
                    range_start = selected_display_date - pd.Timedelta(days=7)
                    range_end = selected_display_date + pd.Timedelta(days=1)
                else:
                    range_start = selected_display_date - pd.Timedelta(days=7)
                    range_end = selected_display_date + pd.Timedelta(days=7)

                if min_available_date is not None:
                    range_start = max(range_start, min_available_date)
                if max_allowed_date is not None:
                    range_end = min(range_end, max_allowed_date)

                xaxis_range = [range_start, range_end]

            fig.update_layout(
                title=f"PM2.5 Actual vs Predicted - Station {selected_station_id} [{selected_station_name}]",
                xaxis_title="Date",
                yaxis_title="PM2.5 (µg/m³)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    type="date",
                    range=xaxis_range,
                    tickformat="%Y-%m-%d",
                    dtick="D1",
                ),
                height=520,
            )

            st.plotly_chart(fig, use_container_width=True)

            latest_pred = pd.DataFrame()
            if not predicted_df.empty and selected_display_date is not None:
                if selected_display_date == latest_actual_date:
                    compare_start = selected_display_date - pd.Timedelta(days=7)
                    compare_end = selected_display_date + pd.Timedelta(days=1)
                else:
                    compare_start = selected_display_date - pd.Timedelta(days=7)
                    compare_end = selected_display_date + pd.Timedelta(days=7)

                if min_available_date is not None:
                    compare_start = max(compare_start, min_available_date)
                if max_allowed_date is not None:
                    compare_end = min(compare_end, max_allowed_date)

                latest_pred = predicted_df[
                    (predicted_df["date"] >= compare_start) & (predicted_df["date"] <= compare_end)
                ].copy()

            if not latest_pred.empty:
                latest_pred["date"] = latest_pred["date"].dt.strftime("%Y-%m-%d")
                latest_pred = latest_pred.rename(columns={"date": "prediction_date"})
                st.markdown("##### Predicted values around latest actual date")
                st.dataframe(latest_pred, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        "Enter at least **15 consecutive daily PM2.5 readings** (oldest first) "
        "to predict the next day's PM2.5 concentration."
    )

    # Input mode selection
    input_mode = st.radio("Input method", ["Manual entry", "Upload CSV"], horizontal=True)

    history = []

    if input_mode == "Manual entry":
        st.markdown("##### Enter daily PM2.5 readings")
        num_days = st.number_input("Number of days", min_value=15, max_value=60, value=15)
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
                    min_value=0.0,
                    max_value=500.0,
                    value=25.0,
                    step=0.1,
                    key=f"pm25_{i}",
                )
                history.append({"date": str(d), "pm25": val})

    else:
        st.markdown("Upload a CSV with columns `date` and `pm25` (at least 15 rows).")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if "date" not in df.columns or "pm25" not in df.columns:
                    st.error("CSV must have `date` and `pm25` columns")
                    return
                df = df.sort_values("date")
                st.dataframe(df.tail(20), use_container_width=True)
                history = df[["date", "pm25"]].to_dict("records")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                return

    # Predict button
    if st.button("🔮 Predict", type="primary", disabled=len(history) < 15):
        with st.spinner("Calling prediction API..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/predict",
                    json={"history": history},
                    headers=api_headers(),
                    timeout=30,
                )
                if resp.status_code == 401:
                    st.error("API authentication failed. Check API key.")
                    return
                resp.raise_for_status()
                result = resp.json()

                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction Date", result["prediction_date"])
                with col2:
                    pm25_val = result["predicted_pm25"]
                    st.metric("Predicted PM2.5", f"{pm25_val:.1f} µg/m³")
                with col3:
                    # AQI level indicator
                    if pm25_val <= 15:
                        level, color = "Good", "🟢"
                    elif pm25_val <= 25:
                        level, color = "Moderate", "🟡"
                    elif pm25_val <= 37.5:
                        level, color = "Unhealthy (Sensitive)", "🟠"
                    elif pm25_val <= 75:
                        level, color = "Unhealthy", "🔴"
                    else:
                        level, color = "Very Unhealthy", "🟣"
                    st.metric("Air Quality", f"{color} {level}")

                st.success(f"Model: **{result['model']}**")

            except httpx.ConnectError:
                st.error("Cannot connect to prediction API. Is the API service running?")
            except httpx.HTTPStatusError as e:
                st.error(f"API error: {e.response.status_code} — {e.response.text}")


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
    st.dataframe(df.style.highlight_min(subset=["MAE", "RMSE"], color="#90EE90"), use_container_width=True)

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df, x="model", y="MAE",
            title="MAE by Model (lower is better)",
            color="MAE",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            df, x="model", y="R2",
            title="R² by Model (higher is better)",
            color="R2",
            color_continuous_scale="RdYlGn",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Champion model highlight
    best = df.loc[df["MAE"].idxmin()]
    st.info(
        f"🏆 **Champion Model:** {best['model']} — "
        f"MAE: {best['MAE']:.4f} µg/m³, "
        f"RMSE: {best['RMSE']:.4f}, "
        f"R²: {best['R2']:.4f}"
    )


def page_monitoring():
    st.header("📈 Monitoring Dashboard")

    # --- Predictions Log ---
    pred_path = os.path.join(RESULTS_DIR, "predictions_log.csv")
    actual_path = os.path.join(RESULTS_DIR, "actuals_log.csv")
    monitoring_path = os.path.join(RESULTS_DIR, "monitoring_results.csv")

    # --- Monitoring Results (MAE + PSI trends) ---
    if os.path.exists(monitoring_path):
        st.markdown("### MAE & PSI Trends")
        mon_df = pd.read_csv(monitoring_path)

        if "date" in mon_df.columns or "timestamp" in mon_df.columns:
            date_col = "date" if "date" in mon_df.columns else "timestamp"
            mon_df[date_col] = pd.to_datetime(mon_df[date_col])

            # MAE trend
            if "mae" in mon_df.columns:
                fig_mae = go.Figure()
                fig_mae.add_trace(go.Scatter(
                    x=mon_df[date_col], y=mon_df["mae"],
                    mode="lines+markers", name="MAE",
                    line=dict(color="#FF6B6B", width=2),
                ))
                fig_mae.add_hline(
                    y=6.0, line_dash="dash", line_color="red",
                    annotation_text="Retrain threshold (6.0)",
                )
                fig_mae.update_layout(
                    title="Rolling MAE Over Time",
                    xaxis_title="Date",
                    yaxis_title="MAE (µg/m³)",
                )
                st.plotly_chart(fig_mae, use_container_width=True)

            # PSI trend
            if "psi" in mon_df.columns:
                fig_psi = go.Figure()
                fig_psi.add_trace(go.Scatter(
                    x=mon_df[date_col], y=mon_df["psi"],
                    mode="lines+markers", name="PSI",
                    line=dict(color="#4ECDC4", width=2),
                ))
                fig_psi.add_hline(
                    y=0.2, line_dash="dash", line_color="red",
                    annotation_text="Drift threshold (0.2)",
                )
                fig_psi.add_hline(
                    y=0.1, line_dash="dot", line_color="orange",
                    annotation_text="Monitor threshold (0.1)",
                )
                fig_psi.update_layout(
                    title="Population Stability Index (PSI) Over Time",
                    xaxis_title="Date",
                    yaxis_title="PSI",
                )
                st.plotly_chart(fig_psi, use_container_width=True)

            # Retrain events
            if "retrain_triggered" in mon_df.columns:
                retrain_events = mon_df[mon_df["retrain_triggered"]]
                if not retrain_events.empty:
                    st.warning(f"⚠️ **{len(retrain_events)} retrain(s) triggered** in monitoring history")
                    st.dataframe(retrain_events, use_container_width=True)
                else:
                    st.success("✅ No retrains triggered — model is stable")

        st.markdown("### Full Monitoring Log")
        st.dataframe(mon_df, use_container_width=True)

    else:
        st.info("No monitoring results yet. Run the monitoring pipeline (`pm25_pipeline` DAG) first.")

    # --- Predictions vs Actuals ---
    st.markdown("---")
    st.markdown("### Predictions vs Actuals Log")

    if os.path.exists(pred_path) and os.path.exists(actual_path):
        preds_df = pd.read_csv(pred_path, parse_dates=["prediction_date"])
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

            # Scatter: predicted vs actual
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=merged["pm25_actual"], y=merged["predicted_pm25"],
                mode="markers", name="Predictions",
                marker=dict(color="#6C5CE7", size=8, opacity=0.7),
            ))
            max_val = max(merged["pm25_actual"].max(), merged["predicted_pm25"].max()) + 5
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines", name="Perfect prediction",
                line=dict(color="gray", dash="dash"),
            ))
            fig.update_layout(
                title="Predicted vs Actual PM2.5",
                xaxis_title="Actual PM2.5 (µg/m³)",
                yaxis_title="Predicted PM2.5 (µg/m³)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Time series
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Scatter(
                x=merged["prediction_date"], y=merged["pm25_actual"],
                mode="lines+markers", name="Actual",
                line=dict(color="#00B894", width=2),
            ))
            fig_ts.add_trace(go.Scatter(
                x=merged["prediction_date"], y=merged["predicted_pm25"],
                mode="lines+markers", name="Predicted",
                line=dict(color="#6C5CE7", width=2),
            ))
            fig_ts.update_layout(
                title="PM2.5: Actual vs Predicted Over Time",
                xaxis_title="Date",
                yaxis_title="PM2.5 (µg/m³)",
            )
            st.plotly_chart(fig_ts, use_container_width=True)
        else:
            st.info("No matched prediction-actual pairs found yet.")
    else:
        st.info(
            "No prediction/actual logs found yet. "
            "Use the Predict page or run `mock_pipeline.py` to generate data."
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="FoonAlert — PM2.5 Prediction",
        page_icon="🌫️",
        layout="wide",
    )

    if not check_login():
        return

    # Sidebar
    st.sidebar.markdown(f"👤 Logged in as **{st.session_state.get('username', '?')}**")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["🌫️ Predict", "📊 Model Results", "📈 Monitoring"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**FoonAlert** v1.0  \n"
        "PM2.5 Prediction System  \n"
        "Station 145 — Bangkhuntien, Bangkok"
    )

    if page == "🌫️ Predict":
        page_predict()
    elif page == "📊 Model Results":
        page_results()
    elif page == "📈 Monitoring":
        page_monitoring()


if __name__ == "__main__":
    main()
