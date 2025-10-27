import numpy as np
import pandas as pd
import ta
from typing import Union

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical analysis features to price DataFrame.
    
    Args:
        df: DataFrame with at least Close column and preferably OHLCV
    
    Returns:
        DataFrame with additional technical features
    """
    df = df.copy()
    
    # Ensure all price columns are 1D Series with float values
    def prepare_series(s):
        if isinstance(s, pd.Series):
            return pd.Series(s.values.ravel(), index=df.index, dtype=float)
        elif isinstance(s, np.ndarray):
            return pd.Series(s.ravel(), index=df.index, dtype=float)
        else:
            return pd.Series(np.asarray(s).ravel(), index=df.index, dtype=float)
    
    # Prepare base price series
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        high = prepare_series(df['High'])
        low = prepare_series(df['Low'])
        close = prepare_series(df['Close'])
        
        # ADX
        adx = ta.trend.ADXIndicator(high, low, close)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(high, low)
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        
        # MACD
        macd = ta.trend.MACD(close)
        df['macd'] = pd.Series(macd.macd().values.ravel(), index=df.index)
        df['macd_signal'] = pd.Series(macd.macd_signal().values.ravel(), index=df.index)
        df['macd_diff'] = pd.Series(macd.macd_diff().values.ravel(), index=df.index)
        
        # Parabolic SAR
        psar = ta.trend.PSARIndicator(high, low, close)
        df['psar'] = pd.Series(psar.psar().values.ravel(), index=df.index)
        
        # KST (Know Sure Thing)
        kst = ta.trend.KSTIndicator(close)
        df['kst'] = pd.Series(kst.kst().values.ravel(), index=df.index)
        df['kst_sig'] = pd.Series(kst.kst_sig().values.ravel(), index=df.index)
        df['kst_diff'] = pd.Series(kst.kst_diff().values.ravel(), index=df.index)
    
    # Moving Averages & Price Channels
    windows = [5, 10, 20, 50]
    for w in windows:
        # Calculate indicators and properly reshape them
        sma_vals = ta.trend.sma_indicator(close, window=w).values.ravel()
        ema_vals = ta.trend.ema_indicator(close, window=w).values.ravel()
        df[f'sma_{w}'] = pd.Series(sma_vals, index=df.index)
        df[f'ema_{w}'] = pd.Series(ema_vals, index=df.index)
        
        # Price Channels
        df[f'high_{w}'] = pd.Series(high.rolling(window=w).max().values.ravel(), index=df.index)
        df[f'low_{w}'] = pd.Series(low.rolling(window=w).min().values.ravel(), index=df.index)
        
        # Distance from MA
        df[f'dist_sma_{w}'] = pd.Series((close.values - sma_vals) / sma_vals, index=df.index)
        df[f'dist_ema_{w}'] = pd.Series((close.values - ema_vals) / ema_vals, index=df.index)
    
    # Momentum indicators with prepared series
    # RSI and variants
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['stoch_rsi'] = ta.momentum.StochRSIIndicator(close).stochrsi()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Rate of Change
    for p in [1, 2, 5, 10, 20]:
        roc_vals = ta.momentum.ROCIndicator(close, window=p).roc().values.ravel()
        df[f'roc_{p}'] = pd.Series(roc_vals, index=df.index)
    
    # Ultimate Oscillator
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        uo_vals = ta.momentum.UltimateOscillator(high, low, close).ultimate_oscillator().values.ravel()
        df['uo'] = pd.Series(uo_vals, index=df.index)
    
    if 'Volume' in df.columns:
        # Volume indicators with prepared series
        volume = prepare_series(df['Volume'])
        
        # Calculate volume indicators with proper reshaping
        obv_vals = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().values.ravel()
        adi_vals = ta.volume.AccDistIndexIndicator(high, low, close, volume).acc_dist_index().values.ravel()
        cmf_vals = ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow().values.ravel()
        fi_vals = ta.volume.ForceIndexIndicator(close, volume).force_index().values.ravel()
        em_vals = ta.volume.EaseOfMovementIndicator(high, low, volume).ease_of_movement().values.ravel()
        
        # Assign to DataFrame with proper Series construction
        df['obv'] = pd.Series(obv_vals, index=df.index)
        df['adi'] = pd.Series(adi_vals, index=df.index)
        df['cmf'] = pd.Series(cmf_vals, index=df.index)
        df['fi'] = pd.Series(fi_vals, index=df.index)
        df['em'] = pd.Series(em_vals, index=df.index)
        df['vwap'] = pd.Series((close.values * volume.values).cumsum() / volume.values.cumsum(), index=df.index)
    
    # Volatility
    if all(col in df.columns for col in ['High', 'Low', 'Close']):
        # Bollinger Bands
        for w in [10, 20]:
            bb = ta.volatility.BollingerBands(close)
            bb_high = pd.Series(bb.bollinger_hband().values.ravel(), index=df.index)
            bb_low = pd.Series(bb.bollinger_lband().values.ravel(), index=df.index)
            bb_mavg = pd.Series(bb.bollinger_mavg().values.ravel(), index=df.index)
            
            df[f'bb_high_{w}'] = bb_high
            df[f'bb_low_{w}'] = bb_low
            df[f'bb_mavg_{w}'] = bb_mavg
            df[f'bb_width_{w}'] = (bb_high - bb_low) / bb_mavg
            df[f'bb_pct_{w}'] = (close - bb_low) / (bb_high - bb_low)
        
        # ATR and variants
        for w in [10, 20]:
            atr = ta.volatility.AverageTrueRange(high, low, close, window=w)
            atr_vals = pd.Series(atr.average_true_range().values.ravel(), index=df.index)
            df[f'atr_{w}'] = atr_vals
            df[f'atr_pct_{w}'] = atr_vals / close  # ATR as percentage of price
    
    # Price relatives and returns
    lags = [1, 2, 3, 5, 10, 20]
    for lag in lags:
        df[f'close_prev_{lag}'] = df['Close'].shift(lag)
        df[f'returns_{lag}d'] = df['Close'].pct_change(lag)
        df[f'log_returns_{lag}d'] = np.log1p(df[f'returns_{lag}d'])
        if lag > 1:
            # Rolling stats on returns
            df[f'returns_std_{lag}'] = df['returns_1d'].rolling(lag).std()
            df[f'returns_skew_{lag}'] = df['returns_1d'].rolling(lag).skew()
            df[f'returns_kurt_{lag}'] = df['returns_1d'].rolling(lag).kurt()
    
    # Custom momentum features
    df['momentum_1d'] = close / df['close_prev_1'] - 1
    df['momentum_2d'] = close / df['close_prev_2'] - 1
    df['momentum_5d'] = close / df['close_prev_5'] - 1
    
    # Clean up
    df = df.replace([np.inf, -np.inf], 0)  # Replace inf with 0 instead of NaN
    df = df.fillna(0)  # Fill remaining NaNs with 0
    
    return df