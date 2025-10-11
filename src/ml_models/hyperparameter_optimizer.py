"""
Hyperparameter optimization for ML models using Optuna
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import json
import os
from datetime import datetime

# Import model classes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.ml_models.xgboost_model import XGBoostModel
from src.ml_models.lightgbm_model import LightGBMModel

logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """Optimize hyperparameters for trading models"""
    
    def __init__(self, n_trials: int = 50, timeout: int = 300):
        """
        Args:
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = {}
        
    def objective_xgboost(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Objective function for XGBoost optimization"""
        
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        try:
            # Train model with suggested parameters
            model = XGBoostModel(params)
            model.train(X_train, y_train, X_val, y_val)
            
            # Get validation predictions and calculate RMSE
            val_predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            
            return rmse
            
        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return float('inf')
    
    def objective_lightgbm(self, trial, X_train, y_train, X_val, y_val) -> float:
        """Objective function for LightGBM optimization"""
        
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 0, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        
        try:
            # Train model with suggested parameters
            model = LightGBMModel(params)
            model.train(X_train, y_train, X_val, y_val)
            
            # Get validation predictions and calculate RMSE
            val_predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            
            return rmse
            
        except Exception as e:
            logger.warning(f"Trial failed with params {params}: {e}")
            return float('inf')
    
    def optimize_model(self, model_type: str, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model type"""
        
        logger.info(f"Starting hyperparameter optimization for {model_type}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Select objective function
        if model_type == 'xgboost':
            objective_func = lambda trial: self.objective_xgboost(trial, X_train, y_train, X_val, y_val)
        elif model_type == 'lightgbm':
            objective_func = lambda trial: self.objective_lightgbm(trial, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Run optimization
        study.optimize(
            objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Store results
        self.best_params[model_type] = study.best_params
        
        result = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'optimization_time': sum([trial.duration.total_seconds() for trial in study.trials if trial.duration]),
        }
        
        logger.info(f"Optimization complete for {model_type}. Best RMSE: {study.best_value:.4f}")
        
        return result
    
    def optimize_all_models(self, X_train, y_train, X_val, y_val) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all supported models"""
        
        results = {}
        
        # Optimize XGBoost
        results['xgboost'] = self.optimize_model('xgboost', X_train, y_train, X_val, y_val)
        
        # Optimize LightGBM
        results['lightgbm'] = self.optimize_model('lightgbm', X_train, y_train, X_val, y_val)
        
        # Save results
        self.save_optimization_results(results)
        
        return results
    
    def save_optimization_results(self, results: Dict[str, Dict[str, Any]], 
                                 filename: str = None) -> None:
        """Save optimization results to JSON file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hyperparameter_optimization_{timestamp}.json"
        
        # Ensure models directory exists
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        filepath = os.path.join(models_dir, filename)
        
        # Add metadata
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'n_trials_per_model': self.n_trials,
            'timeout_seconds': self.timeout
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filename: str) -> Dict[str, Dict[str, Any]]:
        """Load optimization results from JSON file"""
        
        filepath = os.path.join("models", filename)
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Load best parameters
        for model_type in ['xgboost', 'lightgbm']:
            if model_type in results:
                self.best_params[model_type] = results[model_type]['best_params']
        
        logger.info(f"Optimization results loaded from {filepath}")
        
        return results
    
    def get_optimized_model_configs(self) -> List[Dict[str, Any]]:
        """Get model configurations with optimized hyperparameters"""
        
        configs = []
        
        if 'xgboost' in self.best_params:
            configs.append({
                'type': 'xgboost',
                'name': 'xgb_optimized',
                'params': self.best_params['xgboost']
            })
        
        if 'lightgbm' in self.best_params:
            configs.append({
                'type': 'lightgbm', 
                'name': 'lgb_optimized',
                'params': self.best_params['lightgbm']
            })
        
        return configs


def run_hyperparameter_optimization(X_train, y_train, X_val, y_val, 
                                   n_trials: int = 50, timeout: int = 300) -> Dict[str, Dict[str, Any]]:
    """Run hyperparameter optimization for all models
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_trials: Number of trials per model
        timeout: Maximum time per model in seconds
        
    Returns:
        Dictionary with optimization results
    """
    
    optimizer = HyperparameterOptimizer(n_trials=n_trials, timeout=timeout)
    
    logger.info("🔧 Starting hyperparameter optimization...")
    logger.info(f"   Trials per model: {n_trials}")
    logger.info(f"   Timeout per model: {timeout} seconds")
    
    results = optimizer.optimize_all_models(X_train, y_train, X_val, y_val)
    
    logger.info("✅ Hyperparameter optimization complete!")
    
    return results


if __name__ == "__main__":
    # Test the optimization
    print("🧪 Testing hyperparameter optimization...")
    
    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randn(100)
    X_val = np.random.randn(20, 10)
    y_val = np.random.randn(20)
    
    # Run optimization with minimal settings for testing
    results = run_hyperparameter_optimization(
        X_train, y_train, X_val, y_val,
        n_trials=5, timeout=60
    )
    
    print("✅ Test completed!")
    print(f"Results: {results}")