import argparse
import numpy as np
import pandas as pd
from data_loader import download_data, prepare_prophet_df
from prophet_model import fit_prophet, forecast_prophet
from garch import fit_garch11, forecast_volatility
from lstm_model import make_sequences, train_lstm, scale_data
from utils import rmse, mape, directional_accuracy
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from features import add_technical_features


def main(ticker, start, end, horizon=30, seq_len=60, epochs=200):
    print(f"Loading {ticker} from {start} to {end}")
    df = download_data(ticker, start, end)
    
    # Fix MultiIndex columns
    new_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.columns = new_cols

    # Handle MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    
    # Split data first before adding features
    train_raw = df.iloc[:-horizon].copy()
    test_raw = df.iloc[-horizon:].copy()
    
    # Keep original close prices
    train_close = train_raw['Close'].copy()
    test_close = test_raw['Close'].copy()

    # Fit Prophet on raw training data
    prophet_df = prepare_prophet_df(train_raw)
    m = fit_prophet(prophet_df)
    prophet_forecast = forecast_prophet(m, periods=horizon, freq='D')
    prophet_pred = prophet_forecast['yhat'].values

    # Now add technical features for LSTM
    df_features = add_technical_features(df)
    print(f"Added {len(df_features.columns)} technical features")
    train = df_features.iloc[:-horizon].copy()
    test = df_features.iloc[-horizon:].copy()

    # Get GARCH vol features
    train_returns = train_raw['Close'].pct_change().dropna()
    if len(train_returns) < 30:
        print("Not enough data for GARCH; skipping volatility feature")
        vol_forecast = np.zeros(horizon)
        train_vol = np.zeros(len(train_raw))
    else:
        garch_res = fit_garch11(train_returns)
        vol_forecast = forecast_volatility(garch_res, horizon=horizon)
        # convert variance percent back to stddev in price returns space approximately
        vol_forecast = np.sqrt(vol_forecast) / 100.0
        vol_forecast = np.asarray(vol_forecast).ravel()
        # in-sample volatility using rolling std
        train_vol = train_raw['Close'].pct_change().rolling(window=5).std().bfill().fillna(0).values

    # In-sample Prophet fit to compute residuals for training LSTM
    # Predict on training dates to get residuals
    train_dates = pd.DataFrame({'ds': train_raw.index})
    train_prophet_preds = m.predict(train_dates)["yhat"].values
    train_close = train_raw['Close'].values.flatten()  # Flatten to 1D
    residuals = train_close - train_prophet_preds
    print(f"Training data points available: {len(residuals)}")

    # Check residuals stationarity
    from statsmodels.tsa.stattools import adfuller
    adf_result = adfuller(residuals)
    print('\nResiduals Augmented Dickey-Fuller Test:')
    print(f'ADF Statistic: {adf_result[0]:.4f}')
    print(f'p-value: {adf_result[1]:.4f}')
    print('Critical values:')
    for key, value in adf_result[4].items():
        print(f'\t{key}: {value:.4f}')
    
    # If residuals are not stationary, take first difference
    if adf_result[1] > 0.05:  # If p-value > 0.05, residuals are non-stationary
        print("\nResiduals are non-stationary. Taking first difference...")
        residuals = np.diff(residuals)
        residuals = np.insert(residuals, 0, 0)  # Add back first element
        
        # Check stationarity again
        adf_result = adfuller(residuals)
        print('\nDifferenced Residuals Augmented Dickey-Fuller Test:')
        print(f'ADF Statistic: {adf_result[0]:.4f}')
        print(f'p-value: {adf_result[1]:.4f}')
        print('Critical values:')
        for key, value in adf_result[4].items():
            print(f'\t{key}: {value:.4f}')

    # Prepare LSTM dataset with residuals and technical features
    train_features = train.copy()  # Use train that has features
    
    # Drop Close price as it's already used in residuals
    feature_cols = [col for col in train_features.columns if col != 'Close']
    print(f"Feature columns found: {len(feature_cols)}")
    print(f"Train features shape: {train_features.shape}")
    print(f"Sample columns: {train_features.columns[:5].tolist()}")
    features = train_features[feature_cols].fillna(0).values  # Fill NaN with 0
    
    X = []
    y = []
    
    # Ensure all arrays are properly shaped
    res = residuals
    
    # Create sequences for LSTM training
    # First ensure we have enough data
    print(f"Features shape: {features.shape}, Residuals shape: {res.shape}")
    
    # Drop any NaN values in features
    valid_indices = ~np.isnan(features).any(axis=1)
    features = features[valid_indices]
    res = res[valid_indices]
    
    min_len = min(len(res), len(features))
    print(f"Clean data points available: {min_len}")
    
    if min_len > seq_len:
        # Use strided array operations for faster sequence creation
        n_sequences = min_len - seq_len
        # Create sequence indices
        idx = np.arange(seq_len)[None, :] + np.arange(n_sequences)[:, None]
        
        # Create feature sequences
        seq_feat = features[idx]
        seq_res = res[idx].reshape(n_sequences, seq_len, 1)
        
        # Combine features with residuals
        X = np.concatenate([seq_res, seq_feat], axis=2)
        y = res[seq_len:min_len]
        
        print(f"Created {len(X)} sequences for training")
    else:
        X = np.array([])
        y = np.array([])
        print("Not enough clean data points for sequence creation")
        
    if len(X) == 0:
        print("Not enough data to train LSTM on residuals; returning Prophet-only forecasts")
        final_pred = prophet_pred
    else:
        # Split data first
        split = int(0.9 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        # Scale all data properly using our enhanced scaling
        X_tr_scaled, X_val_scaled, y_tr_scaled, y_val_scaled, scalers, feature_groups = scale_data(X_tr, X_val, y_tr, y_val)
        
        # Train LSTM model
        model, y_scaler = train_lstm(X_tr_scaled, y_tr_scaled, X_val_scaled, y_val_scaled, epochs=epochs)

        # Store scalers for prediction
        price_scaler = scalers['price']
        volume_scaler = scalers['volume']
        tech_scaler = scalers['tech']
        price_features = feature_groups['price']
        volume_features = feature_groups['volume']
        tech_features = feature_groups['tech']

        # Prepare input for forecasting residuals for horizon days
        last_res = np.array(res[-seq_len:]).ravel()
        last_features = features[-seq_len:].copy()
        
        # Initialize prediction arrays
        preds_res = []
        cur_res_seq = last_res.copy()
        cur_features = last_features.copy()
        n_features = 1 + cur_features.shape[1]  # 1 for residual + number of technical features
        
        for h in range(horizon):
            try:
                # Prepare feature array with proper structure (matching training structure)
                current_sequence = np.column_stack([
                    cur_res_seq.reshape(-1, 1),  # residual
                    cur_features
                ])
                print(f"Debug - Current sequence shape before scaling: {current_sequence.shape}")
                print(f"Debug - Feature groups: price={price_features}, vol={volume_features}, tech={tech_features}")
                
                # Scale features
                scaled_sequence = np.zeros((1, seq_len, current_sequence.shape[1]))

                # Scale residuals (first column)
                scaled_sequence[0, :, 0:1] = y_scaler.transform(np.log1p(np.abs(current_sequence[:, 0:1])))

                # Scale price features
                if price_features:
                    feat_price = np.log1p(np.abs(current_sequence[:, price_features]))
                    # Reshape feat_price to (30, n_price_features) if needed
                    feat_price_scaled = price_scaler.transform(feat_price)
                    for i, idx in enumerate(price_features):
                        scaled_sequence[0, :, idx] = feat_price_scaled[:, i]

                # Scale volume features
                if volume_features:
                    feat_vol = np.log1p(np.abs(current_sequence[:, volume_features]))
                    # Reshape feat_vol to (30, n_volume_features) if needed
                    feat_vol_scaled = volume_scaler.transform(feat_vol)
                    for i, idx in enumerate(volume_features):
                        scaled_sequence[0, :, idx] = feat_vol_scaled[:, i]

                # Scale technical features
                if tech_features:
                    tech_data = current_sequence[:, tech_features]
                    # Reshape tech_data to (30, n_tech_features) if needed
                    tech_scaled = tech_scaler.transform(tech_data)
                    for i, idx in enumerate(tech_features):
                        scaled_sequence[0, :, idx] = tech_scaled[:, i]

                print(f"Debug - Scaled sequence shape: {scaled_sequence.shape}")

                # Make prediction using properly scaled features
                pred_scaled = model.predict(scaled_sequence, verbose=0)[0]
                
                # Reshape to 2D before inverse transform
                pred_log = y_scaler.inverse_transform(pred_scaled.reshape(1, -1))
                pred_res = float(np.expm1(pred_log)[0, 0])
                
            except Exception as e:
                print(f"Warning: Error in prediction step {h}: {str(e)}")
                print("Using last valid prediction or 0")
                pred_res = preds_res[-1] if preds_res else 0
            
            preds_res.append(pred_res)
            
            # Update sequences for next prediction - handle shape properly
            cur_res_seq = np.roll(cur_res_seq, -1)
            cur_res_seq[-1] = pred_res
            
            # Update technical features using forecast info where possible
            # For volatility features, use GARCH forecast
            if len(vol_forecast) > h:
                # Find volatility feature indices (assuming they're named with 'vol' or 'volatility')
                vol_indices = [i for i, col in enumerate(feature_cols) if 'vol' in col.lower()]
                for idx in vol_indices:
                    last_row = cur_features[:, idx].copy()
                    last_row = np.roll(last_row, -1)
                    last_row[-1] = vol_forecast[h]
                    cur_features[:, idx] = last_row
            
            # Other features: carry forward last value
            # In a more sophisticated version, you could forecast these too

        preds_res = np.array(preds_res)
        # final prediction = prophet_pred + predicted residuals
        final_pred = prophet_pred + preds_res

    # Evaluate
    y_true = test_close.values.flatten()  # Use test_close for true values
    # Keep final_pred as is - it was either set in the LSTM section or defaulted to prophet_pred above
    
    print(f"\nEvaluation on horizon={horizon} days:")
    if len(y_true) > 0 and len(final_pred) > 0:
        print(f"RMSE: {rmse(y_true, final_pred):.4f}")
        print(f"MAPE: {mape(y_true, final_pred):.2f}%")
        print(f"Directional accuracy: {directional_accuracy(y_true, final_pred):.2f}%")
        
        # Save a simple plot
        dates = test.index
        if len(dates) > 0 and len(y_true) == len(prophet_pred):
            plt.figure(figsize=(10, 5))
            plt.plot(dates, y_true, label='Actual')
            plt.plot(dates, prophet_pred, label='Prophet')
            if len(final_pred) == len(prophet_pred):  # In case final_pred is different from prophet_pred
                plt.plot(dates, final_pred, label='Hybrid (Prophet + LSTM residual)')
            plt.legend()
            plt.title(f"{ticker} - Forecast vs Actual")
            plt.tight_layout()
            plt.savefig('forecast_plot.png')
            print("Saved plot to forecast_plot.png")
    else:
        print("No predictions available for evaluation.")

    # Save trained models
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save Prophet model
    import pickle
    with open('models/prophet_model.pkl', 'wb') as f:
        pickle.dump(m, f)
    
    # Save LSTM and scalers if we trained one
    if len(X) > 0:
        model.save('models/lstm_model.keras')  # use .keras extension as recommended
        # Save all scalers
        with open('models/scalers.pkl', 'wb') as f:
            pickle.dump({
                'price_scaler': price_scaler,
                'volume_scaler': volume_scaler,
                'tech_scaler': tech_scaler,
                'target_scaler': y_scaler,
                'feature_groups': feature_groups
            }, f)
    
    print("\nModel Architecture Summary:")
    model.summary()
    print("\nSaved models and scalers to models/ directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, default='BTC-USD')
    parser.add_argument('--start', type=str, default='2018-01-01')
    parser.add_argument('--end', type=str, default='2024-12-31')
    parser.add_argument('--horizon', type=int, default=90)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    main(args.ticker, args.start, args.end, horizon=args.horizon, seq_len=args.seq_len, epochs=args.epochs)