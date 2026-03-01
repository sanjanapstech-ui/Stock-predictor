"""
End-to-end utilities to fetch stock data, train an LSTM, and forecast the next week.

This module:
- downloads historical prices with yfinance
- preprocesses the close-price series (scaling + sliding windows)
- trains an LSTM model from scratch (no pre-trained model downloads)
- predicts a 7-day ahead forecast via iterative next-step prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model_lstm import build_lstm_model, set_global_seed, train_lstm_model


DateLike = Union[str, date, pd.Timestamp]


@dataclass(frozen=True)
class TrainForecastResult:
    """
    Container for a single training + forecasting run.

    Attributes
    ----------
    ticker:
        The ticker used for the run.
    history:
        Raw OHLCV history as returned by yfinance (after basic cleanup).
    close:
        The cleaned close-price series used for modeling.
    test_predictions:
        DataFrame indexed by date with columns: actual, predicted.
    forecast:
        DataFrame indexed by future business dates with column: predicted.
    metrics:
        Simple regression metrics on the held-out test period.
    model:
        The trained Keras model instance.
    """

    ticker: str
    history: pd.DataFrame
    close: pd.Series
    test_predictions: pd.DataFrame
    forecast: pd.DataFrame
    metrics: Dict[str, float]
    model: Any


def normalize_ticker(ticker: str) -> str:
    """
    Normalize user input into a yfinance-friendly ticker.

    Raises
    ------
    ValueError:
        If the ticker is empty after normalization.
    """

    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        raise ValueError("Ticker symbol cannot be empty.")
    # yfinance treats comma/whitespace as multiple tickers; this project expects one.
    if "," in cleaned or len(cleaned.split()) != 1:
        raise ValueError("Enter a single ticker symbol (example: MSFT).")
    return cleaned


def download_history(
    ticker: str,
    *,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download historical OHLCV data for a single ticker using yfinance.

    Error handling:
    - Invalid/unavailable ticker typically returns an empty DataFrame.
    - Network failures can raise exceptions from inside yfinance/requests.
    """

    import yfinance as yf

    ticker = normalize_ticker(ticker)
    start_str = None
    end_str = None

    if start is not None:
        start_ts = pd.Timestamp(start).tz_localize(None)
        start_str = start_ts.date().isoformat()

    if end is not None:
        # yfinance treats `end` as exclusive; make the UI-provided end date inclusive for daily data.
        end_ts = pd.Timestamp(end).tz_localize(None) + pd.Timedelta(days=1)
        end_str = end_ts.date().isoformat()

    df = yf.download(
        ticker,
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"No data returned for ticker {ticker!r}. Check the symbol and try again.")

    # yfinance uses the index as the date; ensure it's a plain DatetimeIndex with no timezone.
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df.dropna(axis=0, how="all")

    if df.empty:
        raise ValueError(f"No usable rows returned for ticker {ticker!r}.")

    return df


def extract_close_series(
    history: pd.DataFrame,
    *,
    price_col: str = "Close",
    ticker: Optional[str] = None,
) -> pd.Series:
    """
    Extract a clean close-price series from the downloaded history.

    Raises
    ------
    ValueError:
        If the requested close column is missing or results in an empty series.
    """

    if history.empty:
        raise ValueError("History DataFrame is empty.")

    close_raw: object

    # yfinance can return MultiIndex columns (especially when it interprets the input as multiple tickers).
    if isinstance(history.columns, pd.MultiIndex):
        level0 = set(map(str, history.columns.get_level_values(0)))
        level1 = set(map(str, history.columns.get_level_values(1)))

        if price_col in level0:
            close_raw = history[price_col]
        elif price_col in level1:
            close_raw = history.xs(price_col, level=1, axis=1)
        else:
            raise ValueError(
                f"Expected column {price_col!r} not found in MultiIndex columns. "
                f"Level0 sample: {sorted(list(level0))[:10]} / Level1 sample: {sorted(list(level1))[:10]}"
            )

        if isinstance(close_raw, pd.DataFrame):
            # If multiple tickers were returned, close_raw has one column per ticker.
            if close_raw.shape[1] == 1:
                close_raw = close_raw.iloc[:, 0]
            else:
                if ticker is None:
                    raise ValueError(
                        "Downloaded data contains multiple tickers. Enter a single ticker (example: MSFT)."
                    )
                ticker = normalize_ticker(ticker)
                if ticker in close_raw.columns:
                    close_raw = close_raw[ticker]
                else:
                    raise ValueError(
                        "Downloaded data contains multiple tickers, but the requested ticker column "
                        f"{ticker!r} was not found. Columns: {list(close_raw.columns)[:10]}"
                    )
    else:
        if price_col not in history.columns:
            raise ValueError(
                f"Expected column {price_col!r} not found in data. Columns: {list(history.columns)}"
            )
        close_raw = history[price_col]

        # Duplicate column names can also cause a DataFrame here; handle similarly.
        if isinstance(close_raw, pd.DataFrame):
            if close_raw.shape[1] == 1:
                close_raw = close_raw.iloc[:, 0]
            else:
                raise ValueError(
                    "Multiple close columns were returned. This usually means multiple tickers were downloaded. "
                    "Enter a single ticker (example: MSFT)."
                )

    close = pd.to_numeric(close_raw, errors="coerce").astype(float).dropna()
    close = close[close > 0.0]  # basic sanity check; stock prices should be positive
    close.name = "close"

    if close.empty:
        raise ValueError("Close price series is empty after cleaning.")

    return close


def create_sliding_windows(scaled_values: np.ndarray, *, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D scaled series into (X, y) samples using a sliding window.

    Example:
        X[t] = values[t-lookback : t]
        y[t] = values[t]
    """

    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if scaled_values.ndim != 2 or scaled_values.shape[1] != 1:
        raise ValueError("scaled_values must have shape (n, 1)")
    if len(scaled_values) <= lookback:
        raise ValueError("Not enough data to create even one sliding window.")

    X: list[np.ndarray] = []
    y: list[float] = []

    # Build windows sequentially; do not shuffle (time-series ordering matters).
    for i in range(lookback, len(scaled_values)):
        X.append(scaled_values[i - lookback : i, 0])
        y.append(float(scaled_values[i, 0]))

    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, lookback, 1)
    y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
    return X_arr, y_arr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a few simple regression metrics (MAE, RMSE, MAPE)."""

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    denom = np.where(y_true == 0.0, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def prepare_train_test_data(
    close: pd.Series,
    *,
    lookback: int = 60,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, np.ndarray, int]:
    """
    Scale the close-price series and create sliding-window train/test datasets.

    Returns
    -------
    X_train, y_train, X_test, y_test:
        Arrays ready for Keras.
    scaler:
        MinMaxScaler fitted on the training portion only.
    scaled_all:
        The full scaled series (shape: n x 1), useful for future forecasting windows.
    split_index:
        Index in the original series where the test segment starts.
    """

    if not (0.5 <= train_ratio < 0.95):
        raise ValueError("train_ratio must be between 0.5 and 0.95")

    values = close.to_numpy(dtype=float).reshape(-1, 1)
    n = int(values.shape[0])
    if n < lookback + 30:
        raise ValueError(
            f"Not enough history to train reliably (need at least {lookback + 30} rows, got {n})."
        )

    split_index = int(n * train_ratio)
    if split_index <= lookback:
        raise ValueError("Train split is too small for the selected lookback window.")

    # Fit scaler only on training values to avoid leaking test statistics.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(values[:split_index])

    scaled_all = scaler.transform(values).astype(np.float32)

    train_scaled = scaled_all[:split_index]
    # Include a lookback overlap so the first test window has enough history.
    test_scaled = scaled_all[split_index - lookback :]

    X_train, y_train = create_sliding_windows(train_scaled, lookback=lookback)
    X_test, y_test = create_sliding_windows(test_scaled, lookback=lookback)

    return X_train, y_train, X_test, y_test, scaler, scaled_all, split_index


def forecast_next_days(
    model: Any,
    *,
    scaled_all: np.ndarray,
    scaler: MinMaxScaler,
    lookback: int,
    horizon_days: int = 7,
) -> np.ndarray:
    """
    Forecast the next `horizon_days` close prices by iteratively predicting one day ahead.

    We predict in scaled space, append the prediction to the window, and repeat.
    Finally, we inverse-transform back to price space.
    """

    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if scaled_all.ndim != 2 or scaled_all.shape[1] != 1:
        raise ValueError("scaled_all must have shape (n, 1)")
    if len(scaled_all) < lookback:
        raise ValueError("Not enough data to build the final lookback window.")

    window = scaled_all[-lookback:].astype(np.float32)
    preds_scaled: list[float] = []

    for _ in range(horizon_days):
        x_in = window.reshape(1, lookback, 1)
        next_scaled = float(model.predict(x_in, verbose=0)[0, 0])
        preds_scaled.append(next_scaled)

        # Slide the window forward by one day.
        window = np.vstack([window[1:], [[next_scaled]]]).astype(np.float32)

    preds = scaler.inverse_transform(np.asarray(preds_scaled, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    return preds.astype(float)


def train_and_forecast(
    ticker: str,
    *,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    lookback: int = 60,
    train_ratio: float = 0.8,
    horizon_days: int = 7,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> TrainForecastResult:
    """
    High-level helper: download -> preprocess -> train -> predict (test) -> forecast (future).
    """

    ticker = normalize_ticker(ticker)

    history = download_history(ticker, start=start, end=end, interval="1d")
    close = extract_close_series(history, price_col="Close", ticker=ticker)

    return train_and_forecast_from_close(
        close,
        ticker=ticker,
        history=history,
        lookback=lookback,
        train_ratio=train_ratio,
        horizon_days=horizon_days,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )


def train_and_forecast_from_close(
    close: pd.Series,
    *,
    ticker: str = "CUSTOM",
    history: Optional[pd.DataFrame] = None,
    lookback: int = 60,
    train_ratio: float = 0.8,
    horizon_days: int = 7,
    epochs: int = 10,
    batch_size: int = 32,
    seed: int = 42,
) -> TrainForecastResult:
    """
    Train and forecast using an already-prepared close-price series.

    This is useful for offline workflows where you upload/provide your own CSV history.
    """

    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.index is None:
        raise ValueError("close series must have a DatetimeIndex")

    ticker = normalize_ticker(ticker)
    close = close.copy()
    close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None)
    close = close.dropna()
    close = close.astype(float)

    if history is None:
        # Keep a minimal history DataFrame so the UI can still display/download it if needed.
        history = pd.DataFrame({"Close": close})

    X_train, y_train, X_test, y_test, scaler, scaled_all, split_index = prepare_train_test_data(
        close,
        lookback=lookback,
        train_ratio=train_ratio,
    )

    # Build and train the model from scratch.
    set_global_seed(seed)
    model = build_lstm_model(lookback=lookback, n_features=1)
    train_lstm_model(
        model,
        X_train=X_train,
        y_train=y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # Predict the held-out test segment (for a sanity-check chart + basic metrics).
    y_pred_test_scaled = model.predict(X_test, verbose=0)
    y_pred_test = scaler.inverse_transform(y_pred_test_scaled).reshape(-1)
    y_true_test = scaler.inverse_transform(y_test).reshape(-1)

    test_dates = close.index[split_index:]
    test_predictions = pd.DataFrame(
        {"actual": y_true_test.astype(float), "predicted": y_pred_test.astype(float)},
        index=pd.to_datetime(test_dates),
    )

    metrics = compute_metrics(y_true_test, y_pred_test)

    # Forecast next week (7 business days by default).
    future_prices = forecast_next_days(
        model,
        scaled_all=scaled_all,
        scaler=scaler,
        lookback=lookback,
        horizon_days=horizon_days,
    )

    last_date = pd.to_datetime(close.index[-1])
    future_dates = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon_days)
    forecast = pd.DataFrame({"predicted": future_prices}, index=future_dates)

    return TrainForecastResult(
        ticker=ticker,
        history=history,
        close=close,
        test_predictions=test_predictions,
        forecast=forecast,
        metrics=metrics,
        model=model,
    )
