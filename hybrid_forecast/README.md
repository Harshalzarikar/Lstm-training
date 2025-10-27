Hybrid LSTM + Prophet forecast with GARCH volatility

Overview
- This project provides a runnable script to build a hybrid forecasting pipeline for financial time series (stocks/crypto).
- Hybrid approach: Prophet models trend/seasonality; an LSTM models Prophet residuals; a GARCH(1,1) model is used to produce volatility forecasts that are fed as features to the LSTM.

Files
- `data_loader.py` — download data (yfinance) and basic preprocessing
- `garch.py` — fit GARCH(1,1) and forecast variance
- `prophet_model.py` — fit Prophet and produce forecasts
- `lstm_model.py` — utility for training/predicting with a Keras LSTM
- `utils.py` — evaluation metrics (RMSE, MAPE, directional accuracy) and plotting helpers
- `hybrid_train.py` — main script that runs the full pipeline

Quick start (Windows PowerShell)
1. Create and activate a Python environment (recommended Python 3.10+):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the training script (example for BTC-USD):

```powershell
python hybrid_train.py --ticker BTC-USD --start 2018-01-01 --end 2024-12-31 --horizon 90 --epochs 30
```

Notes
- The script will download data using yfinance. For long runs or heavy experimentation, consider caching data.
- Prophet's training may take time depending on data size.
- Tune LSTM hyperparameters (epochs, batch size, sequence length) for better performance.

Files are intentionally minimal and documented inline. See function docstrings for usage details.
