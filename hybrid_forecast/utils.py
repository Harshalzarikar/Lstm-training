import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # avoid division by zero
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def directional_accuracy(y_true, y_pred):
    # compare sign of returns (day-to-day changes)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # compute day-over-day change
    true_delta = np.sign(np.diff(y_true))
    pred_delta = np.sign(np.diff(y_pred))
    # align lengths
    min_len = min(len(true_delta), len(pred_delta))
    if min_len == 0:
        return np.nan
    return (true_delta[:min_len] == pred_delta[:min_len]).mean() * 100


def series_to_dates(values, start_date, freq='D'):
    # helper to create pandas Series with dates for plotting
    idx = pd.date_range(start=start_date, periods=len(values), freq=freq)
    return pd.Series(values, index=idx)
