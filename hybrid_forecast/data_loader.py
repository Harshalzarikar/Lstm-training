import yfinance as yf
import pandas as pd


def download_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Download OHLC data for a ticker using yfinance.

    Returns a DataFrame with Date index and columns: Open, High, Low, Close, Adj Close, Volume
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} from {start} to {end}")
    
    # Handle MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df.index = pd.to_datetime(df.index)
    return df


def prepare_prophet_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert price dataframe to Prophet expected format (ds, y).

    Uses 'Close' column as target.
    """
    df2 = df.copy()

    # If already in prophet format, use directly
    if "ds" in df2.columns and "y" in df2.columns:
        out = df2[["ds", "y"]].copy()
    else:
        # build ds from Date column if present, otherwise from the index
        if "Date" in df2.columns:
            ds = pd.to_datetime(df2["Date"])
        else:
            ds = pd.to_datetime(df2.index)

        # prefer Close, fallback to Adj_Close
        if "Close" in df2.columns:
            col = df2["Close"]
        elif "Adj_Close" in df2.columns:
            col = df2["Adj_Close"]
        else:
            raise ValueError("DataFrame must contain 'Close' or 'Adj_Close' column for Prophet target")

        # ensure a 1-D numeric series
        y = pd.to_numeric(pd.Series(col.values.ravel()), errors="coerce")
        out = pd.DataFrame({"ds": ds, "y": y})

    # drop missing and reset index
    out = out.dropna().reset_index(drop=True)
    return out


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple return and log-return features to df.
    Expects 'Close' column.
    """
    df = df.copy()
    df["return"] = df["Close"].pct_change()
    df["log_return"] = (df["Close"].shift(1) / df["Close"])
    return df
