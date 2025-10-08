"""Base model interface for all ML models"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import joblib
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_history = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': self.__class__.__name__,
            'version': '1.0'
        }
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics and history
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classification models)
        
        Args:
            X: Features for prediction
            
        Returns:
            Probability predictions array
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance
        
        Args:
            X: Features
            y: True targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions = self.predict(X)
        
        # Calculate basic metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions)
        }
        
        # Add direction accuracy (for returns prediction)
        if len(y) > 0:
            actual_direction = (y > 0).astype(int)
            predicted_direction = (predictions > 0).astype(int)
            direction_accuracy = (actual_direction == predicted_direction).mean()
            metrics['direction_accuracy'] = direction_accuracy
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance scores
        
        Returns:
            Series with feature importance scores or None if not supported
        """
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_, 
                index=self.feature_names
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            return pd.Series(
                np.abs(self.model.coef_), 
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            logger.warning(f"Feature importance not available for {self.model_name}")
            return None
    
    def save_model(self, filepath: str) -> None:
        """Save model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model and metadata
            model_data = {
                'model': self.model,
                'model_name': self.model_name,
                'feature_names': self.feature_names,
                'training_history': self.training_history,
                'metadata': self.metadata,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load model from file
        
        Args:
            filepath: Path to load the model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.model_name = model_data['model_name']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']
            self.metadata = model_data['metadata']
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'metadata': self.metadata,
            'training_history': self.training_history
        }
    
    def validate_features(self, X: pd.DataFrame) -> bool:
        """Validate that input features match training features
        
        Args:
            X: Input features
            
        Returns:
            True if features are valid, False otherwise
        """
        if not self.feature_names:
            logger.warning("No feature names stored in model")
            return True
        
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features found (will be ignored): {extra_features}")
        
        return True