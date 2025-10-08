"""Model trainer utility for automating model training and evaluation"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
import os
import json

from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .lstm_model import LSTMModel
from .ensemble_model import EnsembleModel
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Automated model training and evaluation system"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.trained_models = {}
        self.training_results = {}
        
        # Ensure models directory exists
        os.makedirs(models_dir, exist_ok=True)
    
    def train_single_model(self, model: BaseModel, 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None, 
                          y_val: Optional[pd.Series] = None,
                          save_model: bool = True) -> Dict[str, Any]:
        """Train a single model"""
        
        logger.info(f"Training {model.model_name}")
        
        try:
            # Train model
            start_time = datetime.now()
            results = model.train(X_train, y_train, X_val, y_val)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Add training time to results
            results['training_time_seconds'] = training_time
            results['model_name'] = model.model_name
            
            # Store trained model
            self.trained_models[model.model_name] = model
            self.training_results[model.model_name] = results
            
            # Save model if requested
            if save_model:
                model_path = os.path.join(self.models_dir, f"{model.model_name}.joblib")
                model.save_model(model_path)
                results['model_path'] = model_path
            
            logger.info(f"Successfully trained {model.model_name} in {training_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error training {model.model_name}: {e}")
            raise
    
    def train_multiple_models(self, model_configs: List[Dict[str, Any]], 
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None, 
                            y_val: Optional[pd.Series] = None) -> Dict[str, Dict[str, Any]]:
        """Train multiple models with different configurations"""
        
        logger.info(f"Training {len(model_configs)} models")
        
        all_results = {}
        
        for config in model_configs:
            try:
                model = self._create_model_from_config(config)
                results = self.train_single_model(model, X_train, y_train, X_val, y_val)
                all_results[model.model_name] = results
                
            except Exception as e:
                logger.error(f"Error with model config {config}: {e}")
                continue
        
        # Save training summary
        self._save_training_summary(all_results)
        
        return all_results
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> BaseModel:
        """Create model from configuration dictionary"""
        model_type = config.get('type', 'xgboost')
        model_name = config.get('name', f"{model_type}_model")
        params = config.get('params', {})
        
        if model_type == 'xgboost':
            return XGBoostModel(model_name, **params)
        elif model_type == 'lightgbm':
            return LightGBMModel(model_name, **params)
        elif model_type == 'lstm':
            return LSTMModel(model_name, **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_ensemble(self, model_names: List[str], 
                       weights: Optional[List[float]] = None,
                       ensemble_name: str = "ensemble_model") -> EnsembleModel:
        """Create ensemble from trained models"""
        
        models = []
        for name in model_names:
            if name not in self.trained_models:
                raise ValueError(f"Model {name} not found in trained models")
            models.append(self.trained_models[name])
        
        ensemble = EnsembleModel(models, ensemble_name, weights)
        return ensemble
    
    def get_default_model_configs(self) -> List[Dict[str, Any]]:
        """Get default model configurations for training"""
        return [
            {
                'type': 'xgboost',
                'name': 'xgb_default',
                'params': {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 1000,
                    'random_state': 42
                }
            },
            {
                'type': 'xgboost',
                'name': 'xgb_conservative',
                'params': {
                    'max_depth': 4,
                    'learning_rate': 0.05,
                    'n_estimators': 1500,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'random_state': 42
                }
            },
            {
                'type': 'lightgbm',
                'name': 'lgb_default',
                'params': {
                    'num_leaves': 31,
                    'learning_rate': 0.1,
                    'n_estimators': 1000,
                    'random_state': 42
                }
            },
            {
                'type': 'lightgbm',
                'name': 'lgb_aggressive',
                'params': {
                    'num_leaves': 63,
                    'learning_rate': 0.15,
                    'n_estimators': 800,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.9,
                    'random_state': 42
                }
            }
        ]
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare performance of all trained models"""
        
        if not self.trained_models:
            raise ValueError("No trained models to compare")
        
        comparison_data = []
        
        for model_name, model in self.trained_models.items():
            try:
                # Evaluate model
                metrics = model.evaluate(X_test, y_test)
                
                # Add model info
                metrics['model_name'] = model_name
                metrics['model_type'] = model.__class__.__name__
                
                # Add training time if available
                if model_name in self.training_results:
                    metrics['training_time'] = self.training_results[model_name].get('training_time_seconds', 0)
                
                comparison_data.append(metrics)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                continue
        
        if not comparison_data:
            raise ValueError("No models could be evaluated")
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('rmse')  # Sort by RMSE (lower is better)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'rmse', 
                      X_test: Optional[pd.DataFrame] = None, 
                      y_test: Optional[pd.Series] = None) -> Tuple[str, BaseModel]:
        """Get the best performing model"""
        
        if not self.trained_models:
            raise ValueError("No trained models available")
        
        if X_test is not None and y_test is not None:
            # Evaluate on test data
            comparison = self.compare_models(X_test, y_test)
            
            if metric == 'rmse':
                best_model_name = comparison.loc[comparison['rmse'].idxmin(), 'model_name']
            elif metric == 'direction_accuracy':
                best_model_name = comparison.loc[comparison['direction_accuracy'].idxmax(), 'model_name']
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        else:
            # Use validation results from training
            best_score = float('inf') if metric == 'rmse' else -float('inf')
            best_model_name = None
            
            for model_name, results in self.training_results.items():
                val_metrics = results.get('val_metrics', {})
                if metric in val_metrics:
                    score = val_metrics[metric]
                    
                    if metric == 'rmse' and score < best_score:
                        best_score = score
                        best_model_name = model_name
                    elif metric == 'direction_accuracy' and score > best_score:
                        best_score = score
                        best_model_name = model_name
            
            if best_model_name is None:
                raise ValueError(f"No model has {metric} in validation results")
        
        return best_model_name, self.trained_models[best_model_name]
    
    def _save_training_summary(self, results: Dict[str, Dict[str, Any]]):
        """Save training summary to JSON file"""
        try:
            summary_path = os.path.join(self.models_dir, "training_summary.json")
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'models_trained': len(results),
                'results': {}
            }
            
            # Extract key metrics for summary
            for model_name, result in results.items():
                summary['results'][model_name] = {
                    'train_rmse': result.get('train_metrics', {}).get('rmse', None),
                    'val_rmse': result.get('val_metrics', {}).get('rmse', None),
                    'train_direction_accuracy': result.get('train_metrics', {}).get('direction_accuracy', None),
                    'val_direction_accuracy': result.get('val_metrics', {}).get('direction_accuracy', None),
                    'training_time': result.get('training_time_seconds', None)
                }
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Training summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving training summary: {e}")
    
    def load_model(self, model_name: str, model_path: Optional[str] = None) -> BaseModel:
        """Load a saved model"""
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create a temporary model to load the saved model
        # We need to determine the model type from the saved file
        import joblib
        model_data = joblib.load(model_path)
        model_type = model_data.get('metadata', {}).get('model_type', 'XGBoostModel')
        
        if model_type == 'XGBoostModel':
            model = XGBoostModel(model_name)
        elif model_type == 'LightGBMModel':
            model = LightGBMModel(model_name)
        elif model_type == 'LSTMModel':
            model = LSTMModel(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_model(model_path)
        self.trained_models[model_name] = model
        
        logger.info(f"Loaded model {model_name} from {model_path}")
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of all training results"""
        return {
            'trained_models': list(self.trained_models.keys()),
            'training_results': self.training_results
        }