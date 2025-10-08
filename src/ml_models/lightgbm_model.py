"""LightGBM model for price prediction"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import lightgbm as lgb
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class LightGBMModel(BaseModel):
    """LightGBM implementation for price prediction"""
    
    def __init__(self, model_name: str = "lightgbm_model", **params):
        super().__init__(model_name)
        
        # Default LightGBM parameters optimized for financial time series
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'silent': True,
            'force_col_wise': True
        }
        
        # Update with user-provided parameters
        default_params.update(params)
        self.params = default_params
        self.n_estimators = params.get('n_estimators', 1000)
        self.early_stopping_rounds = params.get('early_stopping_rounds', 50)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train LightGBM model"""
        
        logger.info(f"Training LightGBM model with {len(X_train)} samples and {len(X_train.columns)} features")
        
        try:
            # Store feature names
            self.feature_names = list(X_train.columns)
            
            # Create training dataset
            train_data = lgb.Dataset(X_train, label=y_train)
            
            # Create validation dataset if provided
            valid_data = None
            if X_val is not None and y_val is not None:
                valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Set up callbacks
            callbacks = []
            if valid_data is not None:
                callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
                callbacks.append(lgb.log_evaluation(period=100))
            
            # Train model
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[train_data] + ([valid_data] if valid_data else []),
                valid_names=['train'] + (['valid'] if valid_data else []),
                callbacks=callbacks
            )
            
            self.is_trained = True
            
            # Store training history
            self.training_history = {
                'train_samples': len(X_train),
                'val_samples': len(X_val) if X_val is not None else 0,
                'features': len(X_train.columns),
                'best_iteration': self.model.best_iteration,
                'best_score': self.model.best_score.get('valid', {}).get('rmse', None) if hasattr(self.model, 'best_score') else None
            }
            
            # Evaluate on training data
            train_metrics = self.evaluate(X_train, y_train)
            val_metrics = self.evaluate(X_val, y_val) if X_val is not None else {}
            
            results = {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'training_history': self.training_history
            }
            
            logger.info(f"LightGBM training complete. Train RMSE: {train_metrics['rmse']:.4f}")
            if val_metrics:
                logger.info(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LightGBM model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if not self.validate_features(X):
            raise ValueError("Feature validation failed")
        
        # Select only the features used in training
        X_pred = X[self.feature_names]
        
        predictions = self.model.predict(X_pred, num_iteration=self.model.best_iteration)
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """LightGBM regressor doesn't have predict_proba, return predictions"""
        logger.warning("LightGBM regressor doesn't support predict_proba, returning predictions")
        return self.predict(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get LightGBM feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")
        
        importance_scores = self.model.feature_importance(importance_type='gain')
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
        if 'n_estimators' in new_params:
            self.n_estimators = new_params.pop('n_estimators')
        if 'early_stopping_rounds' in new_params:
            self.early_stopping_rounds = new_params.pop('early_stopping_rounds')
            
        self.params.update(new_params)
        if self.is_trained:
            logger.info("Parameters updated. Model needs to be retrained.")
            self.is_trained = False
    
    def get_params(self) -> Dict[str, Any]:
        """Get current model parameters"""
        params = self.params.copy()
        params['n_estimators'] = self.n_estimators
        params['early_stopping_rounds'] = self.early_stopping_rounds
        return params