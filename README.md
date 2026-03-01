# Stock Predictor (LSTM Time Series Forecasting)

Train an **LSTM model from scratch** (no pre-trained model downloads) and predict the next **7 trading days** of closing prices for a stock.

## What you get
- A simple UX (Streamlit) to enter a ticker (e.g., `MSFT`)
- Downloads historical prices from Yahoo Finance via `yfinance`
- Option to upload your own historical CSV (offline-friendly)
- Trains an LSTM locally (MinMaxScaler + sliding window)
- Forecasts the next **7 business days** (≈ 1 trading week)

## Quickstart
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Data format (for CSV upload)
Your CSV must contain:
- a date column (e.g., `Date`)
- a price column (e.g., `Close` or `Adj Close`)

Common Yahoo Finance CSV columns work out of the box:
`Date, Open, High, Low, Close, Adj Close, Volume`

You can also start with the included template: `data/sample_template.csv`.

## Notes / limitations
- This is not financial advice.
- Stock prices are noisy; forecasts will be uncertain.
- “7 days” is interpreted as **business/trading days** (markets are closed on weekends; holidays aren’t accounted for).
- First run may take a while because the model is trained locally.
