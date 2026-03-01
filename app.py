from __future__ import annotations

from datetime import date, timedelta
import importlib.util

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Stock Predictor (LSTM)", layout="wide")

st.title("Stock Predictor (LSTM time-series forecasting)")
st.caption("Trains an LSTM from scratch on historical data. Not financial advice.")


def _missing_deps() -> list[str]:
    """
    Detect missing optional dependencies so Streamlit can render a helpful message
    instead of crashing on import.
    """

    required = {
        "yfinance": "yfinance",
        "sklearn": "scikit-learn",
        "tensorflow": "tensorflow",
    }
    missing: list[str] = []
    for module_name, label in required.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(label)
    return missing


missing = _missing_deps()
if missing:
    st.error(
        "Missing Python dependencies: "
        + ", ".join(missing)
        + ".\n\nInstall them with: `pip install -r requirements.txt`"
    )
    st.stop()


from data_fetch_train import (  # noqa: E402
    TrainForecastResult,
    train_and_forecast,
    train_and_forecast_from_close,
)


def plot_prediction(result: TrainForecastResult, *, tail_days: int = 900) -> go.Figure:
    """
    Plot historical close prices, test predictions, and the next-week forecast.

    Parameters
    ----------
    result:
        Output from train_and_forecast().
    tail_days:
        Plot only the last N rows of history for readability.
    """

    close = result.close.tail(tail_days)
    test_pred = result.test_predictions.loc[close.index.min() :]  # align to plotted range

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close (history)",
            line={"color": "#1f77b4"},
        )
    )

    # Show test-period predictions as a dashed line to distinguish from history.
    fig.add_trace(
        go.Scatter(
            x=test_pred.index,
            y=test_pred["predicted"],
            mode="lines",
            name="Predicted (test)",
            line={"color": "#2ca02c", "dash": "dash"},
        )
    )

    # Show future forecast as a line with markers.
    fig.add_trace(
        go.Scatter(
            x=result.forecast.index,
            y=result.forecast["predicted"],
            mode="lines+markers",
            name="Forecast (next 7 days)",
            line={"color": "#ff7f0e"},
        )
    )

    fig.update_layout(
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="Date",
        yaxis_title="Price",
        legend={"orientation": "h"},
    )
    return fig


def prepare_close_series_from_dataframe(
    df: pd.DataFrame,
    *,
    date_col: str,
    close_col: str,
    fill_business_days: bool = True,
) -> pd.Series:
    """
    Convert a user-provided CSV DataFrame into a clean close-price series.

    This keeps preprocessing explicit and predictable:
    - parses dates
    - converts close values to floats
    - sorts by date and removes duplicates
    - optionally reindexes to business days and forward-fills gaps
    """

    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col!r}")
    if close_col not in df.columns:
        raise ValueError(f"Missing close column: {close_col!r}")

    date_values = df[date_col]
    close_values = df[close_col]

    # Pandas returns a DataFrame when column names are duplicated; reduce to a single series.
    if isinstance(date_values, pd.DataFrame):
        date_values = date_values.iloc[:, 0]
    if isinstance(close_values, pd.DataFrame):
        if close_values.shape[1] == 1:
            close_values = close_values.iloc[:, 0]
        else:
            raise ValueError(
                f"Column {close_col!r} matches multiple columns in the CSV. Rename columns to be unique."
            )

    data = pd.DataFrame({date_col: date_values, close_col: close_values}).copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[close_col] = pd.to_numeric(data[close_col], errors="coerce")
    data = data.dropna(subset=[date_col, close_col])

    data = data.sort_values(date_col)
    data = data.drop_duplicates(subset=[date_col], keep="last")
    close = data.set_index(date_col)[close_col].astype(float)
    close = close[close > 0.0]
    close.name = "close"

    if close.empty:
        raise ValueError("No usable close prices after cleaning the CSV.")

    if fill_business_days:
        full_index = pd.bdate_range(close.index.min(), close.index.max())
        close = close.reindex(full_index).ffill().dropna()

    return close


with st.sidebar:
    st.header("Inputs")
    data_source = st.radio(
        "Data source",
        options=["Download (Yahoo Finance)", "Upload CSV"],
    )

    ticker = st.text_input("Ticker", value="MSFT", help="Examples: MSFT, AAPL, GOOGL")

    start = None
    end = None
    uploaded_file = None
    fill_business_days = True

    if data_source.startswith("Download"):
        start = st.date_input("Start date", value=date.today() - timedelta(days=365 * 5))
        end = st.date_input("End date", value=date.today())
    else:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        fill_business_days = st.checkbox(
            "Fill missing business days",
            value=True,
            help="Reindex to business-day frequency and forward-fill gaps.",
        )

    st.header("Model settings")
    lookback = st.slider("Lookback window (days)", min_value=20, max_value=200, value=60, step=5)
    train_ratio = st.slider("Train split", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=15, step=1)
    batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=32, step=8)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)


st.subheader("Train locally and forecast next 7 trading days")
st.write(
    "The app trains an LSTM from scratch (no pre-trained model downloads) and predicts the next 7 business days."
)

close_from_csv: pd.Series | None = None
if data_source.startswith("Upload"):
    if uploaded_file is None:
        st.info("Upload a CSV from the sidebar to continue.")
    else:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            st.stop()

        if df.empty:
            st.error("Uploaded CSV is empty.")
            st.stop()

        st.write("CSV preview:")
        st.dataframe(df.head(20), use_container_width=True)

        columns = list(df.columns)
        if not columns:
            st.error("Uploaded CSV has no columns.")
            st.stop()

        # Heuristics for better UX: pre-select common column names when possible.
        guessed_date = "Date" if "Date" in columns else columns[0]
        guessed_close = "Close" if "Close" in columns else columns[-1]

        col1, col2 = st.columns(2)
        with col1:
            date_col = st.selectbox(
                "Date column",
                options=columns,
                index=columns.index(guessed_date),
                key="csv_date_col",
            )
        with col2:
            close_col = st.selectbox(
                "Close column",
                options=columns,
                index=columns.index(guessed_close),
                key="csv_close_col",
            )

        try:
            close_from_csv = prepare_close_series_from_dataframe(
                df,
                date_col=date_col,
                close_col=close_col,
                fill_business_days=fill_business_days,
            )
        except Exception as exc:
            st.error(f"Could not prepare close-price series from CSV: {exc}")
            st.stop()

train_clicked = st.button("Train & forecast", type="primary")
if not train_clicked:
    st.info("Adjust settings and click **Train & forecast**.")
    st.stop()

with st.spinner("Downloading data, training LSTM, and generating forecast..."):
    try:
        if data_source.startswith("Download"):
            if start is None or end is None:
                raise ValueError("Start/end dates are required for downloads.")
            if start >= end:
                raise ValueError("Start date must be earlier than end date.")

            result = train_and_forecast(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                lookback=int(lookback),
                train_ratio=float(train_ratio),
                horizon_days=7,
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed),
            )
        else:
            if close_from_csv is None:
                raise ValueError("Upload a CSV and select columns first.")
            result = train_and_forecast_from_close(
                close_from_csv,
                ticker=ticker or "CUSTOM",
                lookback=int(lookback),
                train_ratio=float(train_ratio),
                horizon_days=7,
                epochs=int(epochs),
                batch_size=int(batch_size),
                seed=int(seed),
            )
    except Exception as exc:
        st.error(f"Could not train/forecast: {exc}")
        st.stop()

st.success(f"Done. Ticker: {result.ticker}")

rows_used = int(result.close.shape[0])
data_start = str(result.close.index.min().date())
data_end = str(result.close.index.max().date())
last_close = float(result.close.iloc[-1])
metrics = dict(result.metrics)

st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows used", f"{rows_used:,}")
c2.metric("Data start", data_start)
c3.metric("Data end (last trading day)", data_end)
c4.metric("Last close", f"{last_close:,.2f}")

if data_source.startswith("Download") and start is not None and end is not None:
    requested_start = start.isoformat()
    requested_end = end.isoformat()
    if requested_start != data_start or requested_end != data_end:
        st.caption(
            f"Requested range: {requested_start} -> {requested_end}. "
            f"Returned data range: {data_start} -> {data_end} "
            "(markets close on weekends/holidays; latest available trading day is shown)."
        )
        if end.weekday() >= 5:
            st.info(
                f"Your selected end date ({requested_end}) is a {end.strftime('%A')}, "
                f"so there is no trading data for that day. The latest trading day returned is {data_end}."
            )

st.subheader("Test metrics (holdout)")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{metrics.get('mae', float('nan')):,.2f}")
m2.metric("RMSE", f"{metrics.get('rmse', float('nan')):,.2f}")
m3.metric("MAPE", f"{metrics.get('mape', float('nan')):,.2f}%")

with st.expander("Raw output (JSON)", expanded=False):
    st.json(
        {
            "rows_used": rows_used,
            "data_start": data_start,
            "data_end": data_end,
            "last_close": last_close,
            "test_metrics": metrics,
        }
    )

st.plotly_chart(plot_prediction(result), use_container_width=True)

st.subheader("Next 7-day forecast")
forecast = result.forecast.copy()
forecast.index.name = "date"
st.dataframe(forecast, use_container_width=True)

st.download_button(
    "Download forecast CSV",
    data=forecast.to_csv().encode("utf-8"),
    file_name=f"{result.ticker}_forecast_7d.csv",
    mime="text/csv",
)
