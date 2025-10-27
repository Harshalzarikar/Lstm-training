import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.layers import GRU, Conv1D, GlobalAveragePooling1D, Input, Concatenate, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler, RobustScaler


def make_sequences(values: np.ndarray, seq_len: int, step: int = 1):
    """Create sequences with optional stride for increased data diversity.
    
    Args:
        values: Input array of shape (n_samples, n_features)
        seq_len: Length of sequences to create
        step: Stride between sequences (default=1)
    """
    X, y = [], []
    for i in range(0, len(values) - seq_len, step):
        X.append(values[i : i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, dropout_rate=0.1):
        super(AttentionLayer, self).__init__()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.scale = None
        
    def build(self, input_shape):
        # Calculate scale based on the key dimension
        key_dim = input_shape[1][-1]  # Assumes key is the second input
        self.scale = tf.cast(tf.sqrt(tf.cast(key_dim, tf.float32)), tf.float32)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, inputs, training=None):
        query, key, value = inputs
        
        # Scale for numerical stability
        score = tf.matmul(query, key, transpose_b=True) / self.scale
        
        # Apply attention masking
        attention_weights = tf.nn.softmax(score, axis=-1)
        
        # Apply dropout during training
        attention_weights = self.dropout(attention_weights, training=training)
        
        context_vector = tf.matmul(attention_weights, value)
        return [context_vector, attention_weights]


def create_model(input_shape, units: int = 128, dropout: float = 0.3):
    """Create an enhanced model with feature-group attention."""
    input_layer = Input(shape=input_shape)
    
    # Split input into feature groups
    price_input = Lambda(lambda x: x[:, :, :4])(input_layer)
    volume_input = Lambda(lambda x: x[:, :, 4:5])(input_layer)
    tech_input = Lambda(lambda x: x[:, :, 5:])(input_layer)
    
    # Process each feature group with CNN first
    price_conv = Conv1D(filters=units//2, kernel_size=3, padding='same', activation='relu')(price_input)
    volume_conv = Conv1D(filters=units//4, kernel_size=3, padding='same', activation='relu')(volume_input)
    tech_conv = Conv1D(filters=units, kernel_size=3, padding='same', activation='relu')(tech_input)
    
    # Apply LSTM to processed features
    price_lstm = Bidirectional(LSTM(units//2, return_sequences=True))(price_conv)
    volume_lstm = Bidirectional(LSTM(units//4, return_sequences=True))(volume_conv)
    tech_lstm = Bidirectional(LSTM(units, return_sequences=True))(tech_conv)
    
    # Apply BatchNorm and Dropout after each LSTM
    price_lstm = BatchNormalization()(price_lstm)
    price_lstm = Dropout(dropout)(price_lstm)
    volume_lstm = BatchNormalization()(volume_lstm)
    volume_lstm = Dropout(dropout)(volume_lstm)
    tech_lstm = BatchNormalization()(tech_lstm)
    tech_lstm = Dropout(dropout)(tech_lstm)
    
    # Apply attention to processed sequences
    attention = AttentionLayer()
    price_att, _ = attention([price_lstm, price_lstm, price_lstm])
    volume_att, _ = attention([volume_lstm, volume_lstm, volume_lstm])
    tech_att, _ = attention([tech_lstm, tech_lstm, tech_lstm])
    
    # Global feature extraction
    price_global = GlobalAveragePooling1D()(price_lstm)
    volume_global = GlobalAveragePooling1D()(volume_lstm)
    tech_global = GlobalAveragePooling1D()(tech_lstm)
    
    # Concatenate global features
    concat = Concatenate()([price_global, volume_global, tech_global])
    
    # Add skip connection from raw features
    raw_global = GlobalAveragePooling1D()(input_layer)
    raw_dense = Dense(units//2, activation='relu')(raw_global)
    raw_dense = BatchNormalization()(raw_dense)
    raw_dense = Dropout(dropout/2)(raw_dense)
    
    # Combine processed features with raw features
    combined = Concatenate()([concat, raw_dense])
    
    # Final dense layers with residual connections
    dense1 = Dense(units, activation='relu',
                  kernel_regularizer=L1L2(l1=1e-6, l2=1e-4))(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(dropout)(dense1)
    
    # Second dense layer
    dense2 = Dense(units//2, activation='relu',
                  kernel_regularizer=L1L2(l1=1e-6, l2=1e-4))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(dropout/2)(dense2)
    
    # Add skip connection
    final = Concatenate()([dense1, dense2])
    
    # Final prediction layer
    output = Dense(1, activation='linear',
                  kernel_regularizer=L1L2(l1=1e-6, l2=1e-4))(final)
    
    return Model(inputs=input_layer, outputs=output)
    
    # Output layer
    output = Dense(1, activation='linear',
                  kernel_regularizer=L1L2(l1=1e-6, l2=1e-4))(final)
    
    return Model(inputs=input_layer, outputs=output)

def build_lstm(input_shape, units: int = 128, dropout: float = 0.3):
    """Build an enhanced hybrid model with LSTM, GRU, and attention mechanisms.
    
    Args:
        input_shape: Shape of input sequences (seq_len, n_features)
        units: Number of LSTM units in first layer
        dropout: Dropout rate for regularization
        
    Returns:
        A compiled Keras model ready for training
    """
    # Create the base model
    model = create_model(input_shape, units, dropout)
    
    # Custom loss combining Huber loss with directional accuracy penalty
    def combined_loss(y_true, y_pred):
        # Huber loss for value accuracy
        huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        
        # Direction loss
        diff_true = y_true[1:] - y_true[:-1]
        diff_pred = y_pred[1:] - y_pred[:-1]
        direction_match = tf.cast(tf.sign(diff_true) == tf.sign(diff_pred), tf.float32)
        direction_loss = 1.0 - tf.reduce_mean(direction_match)
        
        # Combine losses with weight on directional accuracy
        return huber + 0.2 * direction_loss
    
    # Compile with optimizer and custom loss
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,  # Initial learning rate
        weight_decay=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Add gradient clipping
    optimizer.clipnorm = 1.0
    
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=['mae', 'mse']
    )
    
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, epochs: int = 200, batch_size: int = 32,
             patience: int = 20, initial_lr: float = 0.001, warmup_epochs: int = 5, max_lr: float = 0.01):
    """Train the hybrid model with advanced training techniques.
    
    Args:
        X_train: Training sequences of shape (n_samples, seq_len, n_features)
        y_train: Training targets of shape (n_samples,)
        X_val: Validation sequences (optional)
        y_val: Validation targets (optional)
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        initial_lr: Initial learning rate
        warmup_epochs: Number of warmup epochs before early stopping
        
    Returns:
        tuple: (trained Keras model, target scaler for inverse transformation)
        
    Raises:
        ValueError: If input shapes are incompatible
    """
    # Validate inputs
    if len(X_train.shape) != 3:
        raise ValueError(f"Expected X_train to have shape (samples, seq_len, features), got {X_train.shape}")
    if len(y_train.shape) != 1:
        raise ValueError(f"Expected y_train to be 1D array, got shape {y_train.shape}")
    if len(y_train) != len(X_train):
        raise ValueError(f"X_train and y_train must have same number of samples. Got {len(X_train)} vs {len(y_train)}")
    if X_val is not None:
        if len(X_val.shape) != 3 or X_val.shape[1:] != X_train.shape[1:]:
            raise ValueError(f"X_val must have same shape as X_train except for samples dimension")
        if y_val is None or len(y_val) != len(X_val):
            raise ValueError("y_val must be provided with X_val and have matching samples")
    
    # Scale targets for better training dynamics
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    if y_val is not None:
        y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    
    # Build model with input shape matching training data
    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Automatic validation split if not provided
    if X_val is None:
        val_size = int(0.15 * len(X_train))
        X_val = X_train[-val_size:]
        y_val_scaled = y_train_scaled[-val_size:]
        X_train = X_train[:-val_size]
        y_train_scaled = y_train_scaled[:-val_size]
    
    # Learning rate warmup scheduler
    def lr_warmup_scheduler(epoch):
        if epoch < warmup_epochs:
            return initial_lr * (1 + epoch / warmup_epochs)
        return initial_lr
    
    # Training callbacks
    callbacks = [
        # Early stopping with restore best weights
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            restore_best_weights=True,
            min_delta=1e-4
        ),
        # Learning rate reduction on plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-6,
            mode='min',
            verbose=1
        ),
        # Model checkpointing
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
    ]
    
    # Train with callbacks
    model.fit(
        X_train, y_train_scaled,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_scaled),
        callbacks=callbacks,
        verbose=1
    )
    
    # Return model and scaler for inverse transformation of predictions
    return model, y_scaler
def prepare_sequences(data: pd.DataFrame, seq_len: int = 60, step: int = 1):
    """Prepare sequences from DataFrame for training.
    
    Args:
        data: Input DataFrame with features
        seq_len: Length of sequences to create
        step: Stride between sequences
        
    Returns:
        X: Array of input sequences, shape (n_samples, seq_len, n_features)
        y: Array of target values, shape (n_samples,)
    """
    # Convert to numpy array
    values = data.values
    
    # Create sequences
    X, y = make_sequences(values, seq_len, step)
    
    return X, y


def scale_data(X_train, X_test=None, y_train=None, y_test=None):
    """Scale features and targets using advanced normalization techniques.
    
    Args:
        X_train: Training sequences
        X_test: Test sequences (optional)
        y_train: Training targets (optional)
        y_test: Test targets (optional)
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
               scalers_dict, feature_groups_dict)
    """
    # Initialize scalers
    price_scaler = RobustScaler()
    volume_scaler = RobustScaler()
    tech_scaler = RobustScaler()
    y_scaler = RobustScaler() if y_train is not None else None
    
    # Get dimensions
    n_samples, n_steps, n_features = X_train.shape
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, n_features)  # Combine samples and time steps
    
    # Create feature groups - first identify price and volume features
    price_features = [i for i in range(n_features) if 'price' in str(i).lower() or 'close' in str(i).lower()]
    volume_features = [i for i in range(n_features) if 'volume' in str(i).lower()]
    
    # Then identify tech features as all remaining features
    tech_features = [i for i in range(n_features) if i not in price_features + volume_features]
    
    # Organize into dictionary
    feature_groups = {
        'price': price_features,
        'volume': volume_features,
        'tech': tech_features
    }
    X_train_scaled = np.zeros_like(X_train)
    if X_test is not None:
        X_test_scaled = np.zeros_like(X_test)
    
    # Identify feature groups based on feature indices
    # First 5 features are OHLCV
    price_features = list(range(0, 4))  # OHLC
    volume_features = [4]  # Volume is 5th feature
    # Rest are technical indicators
    tech_features = list(range(5, n_features))
    
    # Scale price features
    if price_features:
        X_price = np.log1p(np.abs(X_train[:, :, price_features].reshape(-1, len(price_features))))
        X_train_scaled[:, :, price_features] = price_scaler.fit_transform(X_price).reshape(n_samples, n_steps, -1)
        if X_test is not None:
            X_test_price = np.log1p(np.abs(X_test[:, :, price_features].reshape(-1, len(price_features))))
            X_test_scaled[:, :, price_features] = price_scaler.transform(X_test_price).reshape(X_test.shape[0], n_steps, -1)
    
    # Scale volume features
    if volume_features:
        X_vol = np.log1p(np.abs(X_train[:, :, volume_features].reshape(-1, len(volume_features))))
        X_train_scaled[:, :, volume_features] = volume_scaler.fit_transform(X_vol).reshape(n_samples, n_steps, -1)
        if X_test is not None:
            X_test_vol = np.log1p(np.abs(X_test[:, :, volume_features].reshape(-1, len(volume_features))))
            X_test_scaled[:, :, volume_features] = volume_scaler.transform(X_test_vol).reshape(X_test.shape[0], n_steps, -1)
    
    # Scale technical features
    if tech_features:
        X_tech = X_train[:, :, tech_features].reshape(-1, len(tech_features))
        X_train_scaled[:, :, tech_features] = tech_scaler.fit_transform(X_tech).reshape(n_samples, n_steps, -1)
        if X_test is not None:
            X_test_tech = X_test[:, :, tech_features].reshape(-1, len(tech_features))
            X_test_scaled[:, :, tech_features] = tech_scaler.transform(X_test_tech).reshape(X_test.shape[0], n_steps, -1)
    
    # Scale targets if provided
    if y_train is not None:
        # Use log transform for price targets
        y_train_log = np.log1p(np.abs(y_train.reshape(-1, 1)))
        y_train_scaled = y_scaler.fit_transform(y_train_log).ravel()
        if y_test is not None:
            y_test_log = np.log1p(np.abs(y_test.reshape(-1, 1)))
            y_test_scaled = y_scaler.transform(y_test_log).ravel()
    
    # Create return dictionaries
    scalers_dict = {
        'price': price_scaler,
        'volume': volume_scaler,
        'tech': tech_scaler,
        'target': y_scaler
    }
    
    feature_groups_dict = {
        'price': price_features,
        'volume': volume_features,
        'tech': tech_features
    }
    
    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, 
            scalers_dict, feature_groups_dict)


def predict_sequence(model, sequences, y_scaler=None, batch_size=32):
    """Make predictions for sequences.
    
    Args:
        model: Trained model
        sequences: Input sequences to predict, shape (n_samples, seq_len, n_features)
        y_scaler: Target scaler for inverse transformation
        batch_size: Batch size for prediction
        
    Returns:
        Array of predictions in original scale
        
    Raises:
        ValueError: If input shape doesn't match model's expected input
    """
    # Validate input shape
    if len(sequences.shape) != 3:
        raise ValueError(f"Expected sequences to have shape (samples, seq_len, features), got {sequences.shape}")
    
    expected_shape = model.input_shape[1:]
    if sequences.shape[1:] != expected_shape:
        raise ValueError(f"Sequences shape {sequences.shape[1:]} doesn't match model's expected shape {expected_shape}")
    
    # Make predictions in batches
    predictions = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        try:
            batch_pred = model.predict(batch, verbose=0)
            predictions.append(batch_pred)
        except Exception as e:
            raise RuntimeError(f"Error making predictions for batch {i//batch_size}: {str(e)}")
    
    # Concatenate batch predictions
    try:
        predictions = np.concatenate(predictions)
    except Exception as e:
        raise RuntimeError(f"Error concatenating predictions: {str(e)}")
    
    # Inverse transform if scaler provided
    if y_scaler is not None:
        try:
            predictions = y_scaler.inverse_transform(predictions).ravel()
        except Exception as e:
            raise RuntimeError(f"Error inverse transforming predictions: {str(e)}")
    
    return predictions
