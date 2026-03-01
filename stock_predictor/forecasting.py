from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    method: str
    order: Optional[Tuple[int, int, int]]
    aic: Optional[float]
    forecast: pd.DataFrame


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower().strip() if ch.isalnum())


def guess_date_column(columns: Iterable[str]) -> Optional[str]:
    normalized = {col: _normalize_column_name(col) for col in columns}
    for col, norm in normalized.items():
        if norm in {"date", "datetime", "timestamp", "time"}:
            return col
    for col, norm in normalized.items():
        if "date" in norm or "time" in norm:
            return col
    return None


def guess_price_column(columns: Iterable[str]) -> Optional[str]:
    normalized = {col: _normalize_column_name(col) for col in columns}
    for target in ("adjclose", "adjustedclose", "close", "price", "last"):
        for col, norm in normalized.items():
            if norm == target:
                return col
    for col, norm in normalized.items():
        if "close" in norm or "price" in norm:
            return col
    return None


def prepare_price_series(
    df: pd.DataFrame,
    *,
    date_col: str,
    price_col: str,
    business_days: bool = True,
) -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col!r}")
    if price_col not in df.columns:
        raise ValueError(f"Missing price column: {price_col!r}")

    data = df[[date_col, price_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[price_col] = pd.to_numeric(data[price_col], errors="coerce")
    data = data.dropna(subset=[date_col, price_col])

    data = data.sort_values(date_col)
    data = data.drop_duplicates(subset=[date_col], keep="last")
    series = data.set_index(date_col)[price_col].astype(float)
    series.name = "price"

    if business_days:
        full_index = pd.bdate_range(series.index.min(), series.index.max())
        series = series.reindex(full_index).ffill().dropna()

    if len(series) < 10:
        raise ValueError("Not enough usable rows after cleaning (need at least 10).")

    return series


def _try_fit_arima(series: pd.Series, order: tuple[int, int, int]):
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ARIMA(series, order=order)
        return model.fit()


def select_arima_order(
    series: pd.Series,
    *,
    max_p: int = 3,
    max_q: int = 3,
    d_values: Sequence[int] = (0, 1),
) -> Tuple[Tuple[int, int, int], float]:
    best_order: Optional[Tuple[int, int, int]] = None
    best_aic = math.inf

    for d in d_values:
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                order = (p, d, q)
                try:
                    fitted = _try_fit_arima(series, order)
                except Exception:
                    continue

                aic = float(getattr(fitted, "aic", math.inf))
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best_order = order

    if best_order is None:
        raise RuntimeError("ARIMA order search failed for all candidate orders.")

    return best_order, best_aic


def fit_and_forecast(
    series: pd.Series,
    *,
    horizon: int = 5,
    max_p: int = 3,
    max_q: int = 3,
    d_values: Sequence[int] = (0, 1),
    alpha: float = 0.05,
) -> ForecastResult:
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")

    if len(series) < 30:
        return drift_forecast(series, horizon=horizon)

    try:
        order, aic = select_arima_order(series, max_p=max_p, max_q=max_q, d_values=d_values)
        fitted = _try_fit_arima(series, order)
        frame = fitted.get_forecast(steps=horizon).summary_frame(alpha=alpha)
        forecast = pd.DataFrame(
            {
                "yhat": frame["mean"].astype(float),
                "yhat_lower": frame["mean_ci_lower"].astype(float),
                "yhat_upper": frame["mean_ci_upper"].astype(float),
            }
        )
        return ForecastResult(method="arima", order=order, aic=aic, forecast=forecast)
    except Exception:
        return drift_forecast(series, horizon=horizon)


def drift_forecast(series: pd.Series, *, horizon: int = 5, z: float = 1.96) -> ForecastResult:
    last_value = float(series.iloc[-1])
    diffs = series.diff().dropna().astype(float)
    mu = float(diffs.mean()) if not diffs.empty else 0.0
    sigma = float(diffs.std(ddof=1)) if len(diffs) >= 2 else 0.0

    last_date = pd.to_datetime(series.index[-1])
    future_index = pd.bdate_range(last_date + pd.offsets.BDay(1), periods=horizon)
    steps = np.arange(1, horizon + 1, dtype=float)
    mean = last_value + steps * mu
    std = np.sqrt(steps) * sigma

    forecast = pd.DataFrame(
        {
            "yhat": mean,
            "yhat_lower": mean - z * std,
            "yhat_upper": mean + z * std,
        },
        index=future_index,
    ).astype(float)
    return ForecastResult(method="drift", order=None, aic=None, forecast=forecast)


def backtest_arima(
    series: pd.Series,
    *,
    order: Tuple[int, int, int],
    test_size: int = 30,
) -> Dict[str, float]:
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be between 1 and len(series)-1")

    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    fitted = _try_fit_arima(train, order)
    pred = fitted.get_forecast(steps=test_size).predicted_mean
    pred = pd.Series(np.asarray(pred, dtype=float), index=test.index)

    err = pred - test
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    denom = np.where(np.asarray(test, dtype=float) == 0.0, np.nan, np.asarray(test, dtype=float))
    mape = float(np.nanmean(np.abs(np.asarray(err, dtype=float) / denom)) * 100.0)

    return {"mae": mae, "rmse": rmse, "mape": mape}
