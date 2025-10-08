"""Machine learning models package for price prediction"""

from .base_model import BaseModel
from .ensemble_model import EnsembleModel
from .lightgbm_model import LightGBMModel
from .model_trainer import ModelTrainer
from .xgboost_model import XGBoostModel

# Import LSTM only if TensorFlow is available
try:
    from .lstm_model import LSTMModel
except ImportError:
    LSTMModel = None

__all__ = [
    "BaseModel",
    "XGBoostModel",
    "LightGBMModel",
    "LSTMModel",
    "EnsembleModel",
    "ModelTrainer",
]
