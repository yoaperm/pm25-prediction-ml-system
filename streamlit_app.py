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
    st.header("🌫️ PM2.5 Next-Day Prediction")
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

    else:  # Upload CSV
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
                retrain_events = mon_df[mon_df["retrain_triggered"] == True]
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
