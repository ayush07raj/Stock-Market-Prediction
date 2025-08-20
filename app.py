
# app.py
# Streamlit Stock Market Prediction App
#
# Features:
# - Fetches historical data from Yahoo Finance.
# - Builds features (moving averages, RSI, MACD, rolling volatility, lagged returns).
# - Trains a time-series ML model (Ridge or Random Forest) with proper TimeSeriesSplit.
# - Backtests on the most recent fold and reports MAE / RMSE / R^2.
# - Recursively forecasts the next N days and overlays the forecast on the price chart.
# - Saves/loads models per-ticker to speed up iteration.
#
# How to run (after installing requirements):
#   streamlit run app.py
#
# Notes:
# - This is an educational tool, not financial advice.
# - Forecasts are uncertain; use responsibly.

import warnings
warnings.filterwarnings("ignore")

import os
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Market Prediction App")
st.caption("Educational demo — not financial advice.")

# --------------------------
# Utilities
# --------------------------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df["Return"] = np.log(df["Close"]).diff()
    return df.dropna()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill")

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Moving averages
    for w in [5, 10, 20, 50]:
        out[f"MA_{w}"] = out["Close"].rolling(w).mean()
    # Volatility (rolling std of returns)
    out["Volatility_10"] = out["Return"].rolling(10).std()
    # RSI & MACD
    out["RSI_14"] = rsi(out["Close"], 14)
    macd_line, signal_line = macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd_line
    out["MACD_signal"] = signal_line
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
    # Lagged returns
    for l in [1, 3, 5, 10]:
        out[f"Ret_lag_{l}"] = out["Return"].shift(l)
    out = out.dropna()
    # Target: next-day log return
    out["Target"] = out["Return"].shift(-1)
    out = out.dropna()
    return out

def train_evaluate(X: pd.DataFrame, y: pd.Series, model_name: str, n_splits: int = 5):
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=n_splits)
    last_fold_metrics = None
    fitted_model = None

    # Define model
    if model_name == "Ridge (Linear)":
        model = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0, random_state=42))])
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported model")

    # Walk forward fit; evaluate on final split
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        last_fold_metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "R2": r2_score(y_test, y_pred),
            "y_test": y_test,
            "y_pred": pd.Series(y_pred, index=y_test.index),
        }
        fitted_model = model  # keep last fitted (most recent training window)

    return fitted_model, last_fold_metrics

def recursive_forecast(df_full: pd.DataFrame, model, horizon: int) -> pd.DataFrame:
    """
    Recursively forecast next `horizon` daily returns and prices.
    We update Close using predicted log returns and recompute indicators.
    """
    future = df_full.copy()
    last_close = future["Close"].iloc[-1]
    forecasts = []

    for step in range(1, horizon + 1):
        # Rebuild features on the fly to include latest synthetic Close
        feats_df = build_features(future.tail(400))  # keep memory small
        X_cols = [c for c in feats_df.columns if c not in ["Target"]]
        X = feats_df.iloc[[-1]][[c for c in X_cols if c != "Return"]]  # model expects current state to predict next return

        # Align columns for pipelines
        try:
            yhat_ret = float(model.predict(X)[0])
        except Exception:
            # Fallback: if pipeline expects certain columns that are missing (rare)
            X = feats_df.drop(columns=["Target"]).iloc[[-1]]
            yhat_ret = float(model.predict(X)[0])

        # Update price using predicted log-return
        last_close = float(np.exp(np.log(last_close) + yhat_ret))
        next_date = future.index[-1] + pd.tseries.offsets.BDay(1)
        new_row = future.iloc[[-1]].copy()
        new_row.index = [next_date]
        new_row["Close"] = last_close
        # approximate OHLC around Close for indicators (visual only)
        new_row["Open"] = last_close
        new_row["High"] = last_close
        new_row["Low"] = last_close
        # keep volume as rolling median to avoid zeros
        new_row["Volume"] = future["Volume"].rolling(20).median().iloc[-1] if "Volume" in future.columns else 0
        new_row["Return"] = np.log(new_row["Close"]).diff()

        future = pd.concat([future, new_row]).copy()
        forecasts.append({"date": next_date, "pred_return": yhat_ret, "pred_close": last_close})

    return pd.DataFrame(forecasts).set_index("date")

def plot_price_with_forecast(hist: pd.DataFrame, forecast: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    hist["Close"].plot(ax=ax, label="Historical Close")
    if forecast is not None and not forecast.empty:
        forecast["pred_close"].plot(ax=ax, label="Forecast", linestyle="--")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

def plot_pred_vs_actual(y_test: pd.Series, y_pred: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, alpha=0.6)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Actual next-day log return")
    ax.set_ylabel("Predicted next-day log return")
    st.pyplot(fig)

# --------------------------
# Sidebar Controls
# --------------------------
st.sidebar.header("Configuration")
default_ticker = "AAPL"
ticker = st.sidebar.text_input("Ticker (Yahoo Finance)", value=default_ticker).strip().upper()
col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start date", pd.to_datetime("2015-01-01"))
end = col2.date_input("End date", pd.to_datetime("today"))
horizon = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1)
model_name = st.sidebar.selectbox("Model", ["Ridge (Linear)", "Random Forest"], index=1)
n_splits = st.sidebar.slider("Backtest folds (TimeSeriesSplit)", 3, 8, 5)
run_button = st.sidebar.button("Run", type="primary")

st.sidebar.caption("Tip: Try tickers like MSFT, GOOGL, TSLA, INFY.NS, TCS.NS, RELIANCE.NS")

# --------------------------
# Main Flow
# --------------------------
if run_button:
    with st.spinner("Loading data..."):
        df = load_data(ticker, str(start), str(end))

    if df.empty or len(df) < 200:
        st.error("Not enough data. Try a different date range or ticker.")
        st.stop()

    st.subheader(f"Data Preview: {ticker}")
    st.dataframe(df.tail(10))

    with st.spinner("Engineering features..."):
        feats = build_features(df)

    feature_cols = [c for c in feats.columns if c not in ["Target"]]
    X = feats[feature_cols].copy()
    y = feats["Target"].copy()

    st.write(f"Total samples for modeling: {len(X):,}")

    with st.spinner("Training & backtesting..."):
        model, metrics = train_evaluate(X, y, model_name, n_splits=n_splits)

    if metrics is None:
        st.error("Unable to compute metrics.")
        st.stop()

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("MAE", f"{metrics['MAE']:.6f}")
    colm2.metric("RMSE", f"{metrics['RMSE']:.6f}")
    colm3.metric("R²", f"{metrics['R2']:.3f}")

    plot_pred_vs_actual(metrics["y_test"], metrics["y_pred"], "Backtest: Predicted vs Actual (last fold)")

    # Save model
    model_dir = ".models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{ticker}_{model_name.replace(' ', '')}.joblib")
    try:
        joblib.dump({"model": model, "features": feature_cols}, model_path)
        st.success(f"Model saved to {model_path}")
    except Exception as e:
        st.warning(f"Could not save model: {e}")

    with st.spinner("Forecasting..."):
        forecast_df = recursive_forecast(df, model, horizon=horizon)

    st.subheader("Price & Forecast")
    plot_price_with_forecast(df, forecast_df, f"{ticker} Close with {horizon}-day Forecast")

    st.subheader("Forecast Table")
    st.dataframe(forecast_df.assign(pred_return_pct=np.exp(forecast_df["pred_return"]) - 1))

    # Feature importances if RF
    if model_name == "Random Forest" and hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        importances.iloc[::-1].plot(kind="barh", ax=ax)
        ax.set_title("Top Feature Importances (Random Forest)")
        st.pyplot(fig)

    st.info("Reminder: Forecasts are simulated and uncertain; do not use for real trading decisions.")
else:
    st.markdown(
        """
        **How to use**
        1. Enter a Yahoo Finance ticker (e.g., `AAPL`, `MSFT`, `INFY.NS`).
        2. Choose a date range with enough history (>= ~3 years recommended).
        3. Pick a model and backtest folds.
        4. Click **Run** to train, evaluate, and forecast.

        **What it does**
        - Predicts next-day log returns and converts them into future price projections.
        - Builds a compact feature set from technical indicators and lagged returns.
        - Shows backtest quality and a 1-30 day forward projection.
        """
    )
