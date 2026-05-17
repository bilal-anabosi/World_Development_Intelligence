"""
pages/8_time_series.py — Time Series Analysis

Works with the dataset created in Data Builder.

Fixes:
- Prevents PACF lag errors on small samples.
- Explains when forecasting is not possible.
- Supports future year prediction using ARIMA.
- Requires All years panel data for meaningful forecasting.
"""

import sys
import os
import warnings

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import (
    CSS,
    render_sidebar,
    init_state,
    page_header,
    step_bar,
    stat_row,
)

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Time Series · World Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CSS, unsafe_allow_html=True)
init_state()
render_sidebar()
step_bar(6)


# ─────────────────────────────────────────────────────────────────────────────
# Small UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def info_box(text: str):
    st.markdown(
        f"""
        <div class="info-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def ok_box(text: str):
    st.markdown(
        f"""
        <div class="ok-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def warn_box(text: str):
    st.markdown(
        f"""
        <div class="warn-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def error_box(text: str):
    st.markdown(
        f"""
        <div class="warn-strip">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def get_active_df():
    if st.session_state.get("clean_df") is not None:
        return st.session_state.clean_df.copy()

    if st.session_state.get("raw_df") is not None:
        return st.session_state.raw_df.copy()

    return None


def is_numeric_year(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() > 0


def prepare_series(df, country_col, year_col, value_col, selected_country):
    sub = df[df[country_col] == selected_country].copy()

    sub[year_col] = pd.to_numeric(sub[year_col], errors="coerce")
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")

    sub = sub.dropna(subset=[year_col, value_col])
    sub[year_col] = sub[year_col].astype(int)

    sub = (
        sub.groupby(year_col, as_index=False)[value_col]
        .mean()
        .sort_values(year_col)
        .reset_index(drop=True)
    )

    return sub


def adf_test(series):
    try:
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Stationary": result[1] < 0.05,
        }
    except Exception:
        return None


def safe_acf_pacf_values(series, requested_lags):
    """
    This is the main fix.

    PACF requires nlags < 50% of sample size.
    ACF can handle more, but we keep both safe and consistent.
    """

    clean = pd.Series(series).dropna()

    n = len(clean)

    if n < 4:
        return None, None, 0, "Not enough points for ACF/PACF. Need at least 4 valid values."

    max_pacf_lags = max(1, (n // 2) - 1)
    max_acf_lags = max(1, n - 2)

    safe_lags = min(int(requested_lags), max_pacf_lags, max_acf_lags)

    if safe_lags < 1:
        return None, None, 0, "Not enough data after differencing to compute ACF/PACF."

    try:
        ac_values = acf(clean, nlags=safe_lags, fft=False)
        pc_values = pacf(clean, nlags=safe_lags, method="yw")
        return ac_values, pc_values, safe_lags, None

    except Exception as e:
        return None, None, 0, str(e)


def fit_forecast_arima(series_df, year_col, value_col, order, future_steps):
    """
    Fits ARIMA and predicts future years.
    """

    y = series_df[value_col].astype(float).values
    years = series_df[year_col].astype(int).values

    model = ARIMA(y, order=order)
    fitted = model.fit()

    forecast_result = fitted.get_forecast(steps=future_steps)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int()

    last_year = int(years.max())
    future_years = list(range(last_year + 1, last_year + future_steps + 1))

    forecast_df = pd.DataFrame(
        {
            "year": future_years,
            "forecast": forecast_mean,
            "lower_bound": conf_int[:, 0],
            "upper_bound": conf_int[:, 1],
        }
    )

    fitted_values = fitted.fittedvalues

    fitted_df = pd.DataFrame(
        {
            "year": years,
            "actual": y,
            "fitted": fitted_values,
        }
    )

    return fitted, fitted_df, forecast_df


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

page_header(
    "📈",
    "Time Series Analysis",
    "Analyze how a selected development indicator changes over time and forecast future years using ARIMA.",
    tags=[
        ("Trend", "blue"),
        ("Stationarity", "purple"),
        ("ACF / PACF", "green"),
        ("Forecasting", "amber"),
    ],
    accent="linear-gradient(90deg,#22C55E,#14B8A6)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

df = get_active_df()

if df is None or df.empty:
    error_box("No dataset found. Please build a dataset first from the Data Builder page.")
    st.stop()

required_cols = ["country", "year"]

missing_required = [c for c in required_cols if c not in df.columns]

if missing_required:
    error_box(
        "Time Series requires country and year columns. "
        f"Missing: {', '.join(missing_required)}"
    )
    st.stop()

if not is_numeric_year(df["year"]):
    error_box(
        "The year column is not numeric. This usually happens when you built the dataset using "
        "'Average over range'. For time series forecasting, rebuild the data using 'All years panel'."
    )
    st.stop()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ["year"]]

if not numeric_cols:
    error_box("No numeric indicators found for time series analysis.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Controls
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">1 — Select Series</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">🌍 Country</div>',
        unsafe_allow_html=True,
    )

    countries = sorted(df["country"].dropna().unique().tolist())

    selected_country = st.selectbox(
        "Country",
        countries,
        index=0,
    )

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">📊 Indicator</div>',
        unsafe_allow_html=True,
    )

    selected_indicator = st.selectbox(
        "Indicator",
        numeric_cols,
        format_func=safe_label,
    )

    st.markdown("</div>", unsafe_allow_html=True)


series_df = prepare_series(
    df=df,
    country_col="country",
    year_col="year",
    value_col=selected_indicator,
    selected_country=selected_country,
)

if series_df.empty:
    error_box("No valid numeric time series data found for this country and indicator.")
    st.stop()

n_points = len(series_df)
year_min = int(series_df["year"].min())
year_max = int(series_df["year"].max())

stat_row(
    [
        ("Data Points", f"{n_points:,}", "valid yearly records", "#2563EB"),
        ("Start Year", str(year_min), "first available year", "#059669"),
        ("End Year", str(year_max), "latest available year", "#D97706"),
        ("Missing Years", str((year_max - year_min + 1) - n_points), "gaps in period", "#7C3AED"),
    ]
)

if n_points < 6:
    warn_box(
        "⚠ This series has very few yearly observations. Trend plotting works, "
        "but ACF/PACF and forecasting may be limited. For better forecasting, choose "
        "'All years panel' in Data Builder and a wider year range."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trend
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">2 — Trend Over Time</div>', unsafe_allow_html=True)

fig_trend = px.line(
    series_df,
    x="year",
    y=selected_indicator,
    markers=True,
    title=f"{safe_label(selected_indicator)} over time — {selected_country}",
)

fig_trend.update_layout(
    template="plotly_dark",
    height=430,
    margin=dict(l=20, r=20, t=70, b=20),
    xaxis_title="Year",
    yaxis_title=safe_label(selected_indicator),
)

st.plotly_chart(fig_trend, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Stationarity
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">3 — Stationarity Check</div>', unsafe_allow_html=True)

if not STATSMODELS_AVAILABLE:
    error_box("statsmodels is not installed. Install it using: pip install statsmodels")
    st.stop()

adf_original = adf_test(series_df[selected_indicator])

if adf_original is not None:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("ADF Statistic", f"{adf_original['ADF Statistic']:.4f}")

    with col2:
        st.metric("p-value", f"{adf_original['p-value']:.4f}")

    with col3:
        st.metric(
            "Stationary?",
            "Yes" if adf_original["Stationary"] else "No",
        )

    if adf_original["Stationary"]:
        ok_box("✅ The selected series appears stationary based on the ADF test.")
    else:
        warn_box(
            "⚠ The selected series may not be stationary. Differencing can help ARIMA work better."
        )
else:
    warn_box("Could not run ADF stationarity test on this series.")


# ─────────────────────────────────────────────────────────────────────────────
# ACF and PACF
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">4 — ACF and PACF Identification</div>', unsafe_allow_html=True)

acf_input_col, diff_input_col = st.columns([1, 1])

with acf_input_col:
    requested_lags = st.slider(
        "Requested lags",
        min_value=1,
        max_value=20,
        value=8,
        help="The app will automatically reduce this if the sample size is too small.",
    )

with diff_input_col:
    use_diff = st.checkbox(
        "Use first difference for ACF/PACF",
        value=True,
        help="Differencing is useful when the original series is not stationary.",
    )

if use_diff:
    acf_series = series_df[selected_indicator].diff().dropna()
    acf_title_suffix = "First Difference"
else:
    acf_series = series_df[selected_indicator].dropna()
    acf_title_suffix = "Original Series"

ac_values, pc_values, safe_lags, acf_error = safe_acf_pacf_values(
    acf_series,
    requested_lags,
)

if acf_error:
    warn_box(f"ACF/PACF could not be computed: {acf_error}")
else:
    if safe_lags < requested_lags:
        warn_box(
            f"Requested lags were reduced from {requested_lags} to {safe_lags} because "
            "PACF can only use lags smaller than half of the available sample size."
        )

    lag_index = list(range(len(ac_values)))

    acf_df = pd.DataFrame(
        {
            "lag": lag_index,
            "ACF": ac_values,
            "PACF": pc_values,
        }
    )

    col_acf, col_pacf = st.columns(2)

    with col_acf:
        fig_acf = px.bar(
            acf_df,
            x="lag",
            y="ACF",
            title=f"ACF — {acf_title_suffix}",
        )

        fig_acf.add_hline(y=0, line_dash="dash")

        fig_acf.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig_acf, use_container_width=True)

    with col_pacf:
        fig_pacf = px.bar(
            acf_df,
            x="lag",
            y="PACF",
            title=f"PACF — {acf_title_suffix}",
        )

        fig_pacf.add_hline(y=0, line_dash="dash")

        fig_pacf.update_layout(
            template="plotly_dark",
            height=380,
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig_pacf, use_container_width=True)

    info_box(
        "Use ACF/PACF as a rough guide: PACF helps suggest AR order p, "
        "ACF helps suggest MA order q. For small datasets, keep p and q small."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Forecasting
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-header">5 — Future Forecasting</div>', unsafe_allow_html=True)

if n_points < 8:
    warn_box(
        "Forecasting is disabled because this series has fewer than 8 data points. "
        "Go back to Data Builder and choose 'All years panel' with a wider year range."
    )
    st.stop()

forecast_left, forecast_right = st.columns([1, 1], gap="large")

with forecast_left:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">⚙️ ARIMA Settings</div>',
        unsafe_allow_html=True,
    )

    p = st.number_input(
        "p — AR order",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
    )

    d = st.number_input(
        "d — differencing",
        min_value=0,
        max_value=2,
        value=1,
        step=1,
    )

    q = st.number_input(
        "q — MA order",
        min_value=0,
        max_value=5,
        value=1,
        step=1,
    )

    future_steps = st.slider(
        "Future years to predict",
        min_value=1,
        max_value=20,
        value=5,
    )

    st.markdown("</div>", unsafe_allow_html=True)


with forecast_right:
    st.markdown(
        '<div class="g-card"><div class="g-card-title">🧠 Recommendation</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        For your project demo, a simple ARIMA setting is usually enough:

        - Use **ARIMA(1,1,1)** for most non-stationary indicators.
        - Use **ARIMA(1,0,1)** if the ADF test says the series is stationary.
        - Avoid large p and q values when the data has few years.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


run_forecast = st.button("🔮 Predict Future Years", type="primary")

if run_forecast:
    try:
        fitted, fitted_df, forecast_df = fit_forecast_arima(
            series_df=series_df,
            year_col="year",
            value_col=selected_indicator,
            order=(int(p), int(d), int(q)),
            future_steps=int(future_steps),
        )

        ok_box("✅ Forecast generated successfully.")

        actual_trace = go.Scatter(
            x=series_df["year"],
            y=series_df[selected_indicator],
            mode="lines+markers",
            name="Actual",
        )

        # Remove unstable first fitted values that destroy the graph axis
        fitted_plot_df = fitted_df.copy()

        # ARIMA with differencing often gives bad/zero fitted values at the beginning
        actual_min = series_df[selected_indicator].min()
        actual_max = series_df[selected_indicator].max()

        # Keep fitted values only if they are reasonably close to the actual data range
        value_range = actual_max - actual_min
        lower_allowed = actual_min - value_range
        upper_allowed = actual_max + value_range

        fitted_plot_df = fitted_plot_df[
            (fitted_plot_df["fitted"] >= lower_allowed)
            & (fitted_plot_df["fitted"] <= upper_allowed)
        ].copy()

        fitted_trace = go.Scatter(
            x=fitted_plot_df["year"],
            y=fitted_plot_df["fitted"],
            mode="lines",
            name="Fitted",
        )

        forecast_trace = go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
        )

        upper_trace = go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["upper_bound"],
            mode="lines",
            name="Upper bound",
            line=dict(width=0),
            showlegend=False,
        )

        lower_trace = go.Scatter(
            x=forecast_df["year"],
            y=forecast_df["lower_bound"],
            mode="lines",
            name="Confidence interval",
            fill="tonexty",
            line=dict(width=0),
        )

        fig_forecast = go.Figure()

        fig_forecast.add_trace(actual_trace)
        fig_forecast.add_trace(fitted_trace)
        fig_forecast.add_trace(forecast_trace)
        fig_forecast.add_trace(upper_trace)
        fig_forecast.add_trace(lower_trace)

        fig_forecast.update_layout(
            title=f"Forecast for {safe_label(selected_indicator)} — {selected_country}",
            template="plotly_dark",
            height=500,
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis_title="Year",
            yaxis_title=safe_label(selected_indicator),
        )

        st.plotly_chart(fig_forecast, use_container_width=True)

        st.markdown('<div class="sec-header">Forecast Table</div>', unsafe_allow_html=True)

        forecast_display = forecast_df.copy()
        forecast_display["forecast"] = forecast_display["forecast"].round(4)
        forecast_display["lower_bound"] = forecast_display["lower_bound"].round(4)
        forecast_display["upper_bound"] = forecast_display["upper_bound"].round(4)

        st.dataframe(forecast_display, use_container_width=True)

        csv_buffer = io.StringIO()
        forecast_display.to_csv(csv_buffer, index=False)

        st.download_button(
            "⬇️ Download Forecast CSV",
            csv_buffer.getvalue(),
            file_name=f"{selected_country}_{selected_indicator}_forecast.csv".replace(" ", "_"),
            mime="text/csv",
        )

        with st.expander("Model Summary"):
            st.text(str(fitted.summary()))

    except Exception as e:
        error_box(
            "Forecasting failed. Try a simpler ARIMA order like ARIMA(1,1,1), "
            "ARIMA(1,1,0), or ARIMA(0,1,1)."
        )
        st.exception(e)