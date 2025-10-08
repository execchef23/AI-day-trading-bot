"""XGBoost model for price prediction"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost implementation for price prediction"""
    
    def __init__(self, model_name: str = "xgboost_model", **params):
        super().__init__(model_name)
        
        # Default XGBoost parameters optimized for financial time series
        default_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 50,
            'eval_metric': 'rmse'
        }
        
        # Update with user-provided parameters
        default_params.update(params)
        self.params = default_params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train XGBoost model"""
        
        logger.info(f"Training XGBoost model with {len(X_train)} samples and {len(X_train.columns)} features")
        
        try:
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Create XGBoost model
            self.model = xgb.XGBRegressor(**self.params)
            
            # Prepare evaluation set
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_train, y_train), (X_val, y_val)]
            else:
                eval_set = [(X_train, y_train)]
            
            # Train model
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            self.is_trained = True
            
            # Store training history
            self.training_history = {
                'train_samples': len(X_train),
                'val_samples': len(X_val) if X_val is not None else 0,
                'features': len(X_train.columns),
                'best_iteration': getattr(self.model, 'best_iteration', self.params['n_estimators']),
                'best_score': getattr(self.model, 'best_score', None)
            }
            
            # Evaluate on training data
            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val) if X_val is not None else {}
            
            results = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'training_history': self.training_history
            }
            
            logger.info(f"XGBoost training complete. Train RMSE: {train_metrics['rmse']:.4f}")
            if val_metrics:
                logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with XGBoost model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.validate_features(X):
            raise ValueError("Feature validation failed")
        
        # Select only the features used in training
        X_pred = X[self.feature_names]
        
        predictions = self.model.predict(X_pred)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """XGBoost regressor doesn't have predict_proba, return predictions"""
        logger.warning("XGBoost regressor doesn't support predict_proba, returning predictions")
        return self.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get XGBoost feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_scores = self.model.feature_importances_
        return pd.Series(
            importance_scores,
            index=self.feature_names
        ).sort_values(ascending=False)
    
    def plot_importance(self, max_features: int = 20):
        """Plot feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained to plot importance")
        
        try:
            import matplotlib.pyplot as plt
            
            importance = self.get_feature_importance().head(max_features)
            
            plt.figure(figsize=(10, 8))
            importance.plot(kind='barh')
            plt.title(f'Top {max_features} Feature Importance - {self.model_name}')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting importance: {e}")
    
    def update_params(self, **new_params):
        """Update model parameters"""
        self.params.update(new_params)
        if self.is_trained:
            logger.info("Parameters updated. Model needs to be retrained.")
            self.is_trained = False
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        return self.params.copy()