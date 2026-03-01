"""
LSTM model utilities for time-series forecasting.

This module intentionally builds and trains a model from scratch (no pre-trained weights).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """
    Set random seeds for repeatable-ish runs.

    Notes:
    - Full determinism is hard with TensorFlow/GPU, but this improves consistency.
    - Call this before building/training the model.
    """

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Import tensorflow lazily so the rest of the project can still import without it installed.
    import tensorflow as tf

    tf.random.set_seed(seed)


def build_lstm_model(
    *,
    lookback: int,
    n_features: int = 1,
    lstm_units: int = 64,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """
    Build and compile an LSTM model for next-step prediction.

    The model learns: last `lookback` timesteps -> next timestep value.
    """

    if lookback < 2:
        raise ValueError("lookback must be >= 2")
    if n_features < 1:
        raise ValueError("n_features must be >= 1")

    import tensorflow as tf
    from tensorflow import keras

    inputs = keras.Input(shape=(lookback, n_features), name="window")

    # Stacked LSTM tends to work better than a single layer on noisy financial series.
    x = keras.layers.LSTM(lstm_units, return_sequences=True, name="lstm_1")(inputs)
    x = keras.layers.Dropout(dropout, name="dropout_1")(x)
    x = keras.layers.LSTM(lstm_units, name="lstm_2")(x)
    x = keras.layers.Dropout(dropout, name="dropout_2")(x)

    outputs = keras.layers.Dense(1, name="next_value")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="stock_lstm")

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")

    # Reduce TensorFlow noise in Streamlit logs.
    tf.get_logger().setLevel("ERROR")

    return model


def train_lstm_model(
    model,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    validation_split: float = 0.1,
    patience: int = 5,
    verbose: int = 0,
):
    """
    Train the LSTM model with early stopping.

    Parameters
    ----------
    model:
        A compiled Keras model.
    X_train, y_train:
        Training data produced by a sliding-window function.
    epochs, batch_size:
        Standard Keras training parameters.
    validation_split:
        Fraction of training data held out for validation.
    patience:
        EarlyStopping patience; helps avoid overfitting and long runs.
    verbose:
        Keras verbosity (0=silent, 1=progress bar).
    """

    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not (0.0 < validation_split < 0.5):
        raise ValueError("validation_split must be between 0 and 0.5")

    from tensorflow import keras

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,  # important for time-series: keep temporal ordering
    )

    return history
