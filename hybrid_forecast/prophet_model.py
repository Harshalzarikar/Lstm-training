from prophet import Prophet
import pandas as pd


def fit_prophet(df: pd.DataFrame, weekly_seasonality: bool = True, yearly_seasonality: bool = True) -> Prophet:
    """Fit Prophet model on a dataframe with columns ds (date) and y (value)."""
    m = Prophet(weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality)
    m.fit(df)
    return m


def forecast_prophet(model: Prophet, periods: int, freq: str = "D") -> pd.DataFrame:
    """Return Prophet forecast dataframe (contains ds, yhat, yhat_lower, yhat_upper)."""
    future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
    forecast = model.predict(future)
    return forecast
