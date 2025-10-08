"""LSTM model for time series price prediction"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base_model import BaseModel

logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")

    # Create dummy classes to prevent import errors
    class DummyKerasModel:
        pass

    keras = type("keras", (), {"Model": DummyKerasModel})()


class LSTMModel(BaseModel):
    """LSTM implementation for time series price prediction"""

    def __init__(self, model_name: str = "lstm_model", **params):
        super().__init__(model_name)

        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model")

        # Default LSTM parameters
        default_params = {
            "sequence_length": 60,  # Days of history to look at
            "lstm_units": [50, 50],  # LSTM layer sizes
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
            "reduce_lr_patience": 5,
        }

        default_params.update(params)
        self.params = default_params
        self.scaler = None
        self.sequence_length = self.params["sequence_length"]

    def _create_sequences(
        self, data: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential()

        # First LSTM layer
        model.add(
            layers.LSTM(
                self.params["lstm_units"][0],
                return_sequences=len(self.params["lstm_units"]) > 1,
                input_shape=input_shape,
            )
        )
        model.add(layers.Dropout(self.params["dropout"]))

        # Additional LSTM layers
        for i, units in enumerate(self.params["lstm_units"][1:], 1):
            return_sequences = i < len(self.params["lstm_units"]) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(self.params["dropout"]))

        # Output layer
        model.add(layers.Dense(1))

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params["learning_rate"]),
            loss="mse",
            metrics=["mae"],
        )

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train LSTM model"""

        logger.info(
            f"Training LSTM model with {len(X_train)} samples and {len(X_train.columns)} features"
        )

        try:
            # Store feature names
            self.feature_names = list(X_train.columns)

            # Normalize features
            from sklearn.preprocessing import MinMaxScaler

            self.scaler = MinMaxScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)

            # Create sequences
            X_seq, y_seq = self._create_sequences(X_train_scaled, y_train.values)

            if len(X_seq) == 0:
                raise ValueError(
                    f"Not enough data to create sequences. Need at least {self.sequence_length + 1} samples."
                )

            # Handle validation data
            X_val_seq, y_val_seq = None, None
            if X_val is not None and y_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                X_val_seq, y_val_seq = self._create_sequences(
                    X_val_scaled, y_val.values
                )

            # Build model
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            self.model = self._build_model(input_shape)

            # Setup callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=self.params["early_stopping_patience"],
                    restore_best_weights=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=self.params["reduce_lr_patience"], factor=0.5, min_lr=1e-7
                ),
            ]

            # Train model
            validation_data = (X_val_seq, y_val_seq) if X_val_seq is not None else None

            history = self.model.fit(
                X_seq,
                y_seq,
                batch_size=self.params["batch_size"],
                epochs=self.params["epochs"],
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=0,
            )

            self.is_trained = True

            # Store training history
            self.training_history = {
                "train_samples": len(X_seq),
                "val_samples": len(X_val_seq) if X_val_seq is not None else 0,
                "features": len(X_train.columns),
                "epochs_trained": len(history.history["loss"]),
                "final_train_loss": history.history["loss"][-1],
                "final_val_loss": history.history["val_loss"][-1]
                if "val_loss" in history.history
                else None,
            }

            # Evaluate
            train_pred = self.predict(X_train)
            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val) if X_val is not None else {}

            results = {
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "training_history": self.training_history,
                "keras_history": history.history,
            }

            logger.info(
                f"LSTM training complete. Train RMSE: {train_metrics['rmse']:.4f}"
            )
            if val_metrics:
                logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")

            return results

        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if not self.validate_features(X):
            raise ValueError("Feature validation failed")

        # Scale features
        X_scaled = self.scaler.transform(X[self.feature_names])

        # Create sequences
        if len(X_scaled) < self.sequence_length:
            # If we don't have enough data, pad with the first available values
            padding_needed = self.sequence_length - len(X_scaled)
            padding = np.repeat(X_scaled[0:1], padding_needed, axis=0)
            X_scaled = np.vstack([padding, X_scaled])

        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            # Return prediction based on last sequence
            last_sequence = X_scaled[-self.sequence_length :].reshape(
                1, self.sequence_length, -1
            )
            predictions = self.model.predict(last_sequence, verbose=0)
            return predictions.flatten()

        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """LSTM regressor doesn't have predict_proba, return predictions"""
        logger.warning(
            "LSTM regressor doesn't support predict_proba, returning predictions"
        )
        return self.predict(X)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """LSTM doesn't provide traditional feature importance"""
        logger.warning("LSTM models don't provide traditional feature importance")
        return None
