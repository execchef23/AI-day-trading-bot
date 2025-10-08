"""Ensemble model combining multiple ML models"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class EnsembleModel(BaseModel):
    """Ensemble model that combines predictions from multiple models"""
    
    def __init__(self, models: List[BaseModel], 
                 model_name: str = "ensemble_model",
                 weights: Optional[List[float]] = None,
                 method: str = "weighted_average"):
        super().__init__(model_name)
        
        self.models = models
        self.method = method
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        self.individual_predictions = {}
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        results = {
            'individual_results': {},
            'ensemble_metrics': {}
        }
        
        try:
            # Train each model
            for i, model in enumerate(self.models):
                logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
                
                model_results = model.train(X_train, y_train, X_val, y_val, **kwargs)
                results['individual_results'][model.model_name] = model_results
            
            # All models are now trained
            self.is_trained = True
            self.feature_names = self.models[0].feature_names  # Assume all models use same features
            
            # Evaluate ensemble performance
            if X_val is not None and y_val is not None:
                ensemble_pred = self.predict(X_val)
                ensemble_metrics = {
                    'mse': np.mean((y_val - ensemble_pred) ** 2),
                    'rmse': np.sqrt(np.mean((y_val - ensemble_pred) ** 2)),
                    'mae': np.mean(np.abs(y_val - ensemble_pred)),
                    'direction_accuracy': np.mean((y_val > 0) == (ensemble_pred > 0))
                }
                results['ensemble_metrics'] = ensemble_metrics
                
                logger.info(f"Ensemble validation RMSE: {ensemble_metrics['rmse']:.4f}")
            
            # Store training history
            self.training_history = {
                'models_trained': len(self.models),
                'ensemble_method': self.method,
                'weights': self.weights,
                'individual_results': results['individual_results']
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            if not model.is_trained:
                raise ValueError(f"Model {model.model_name} is not trained")
            
            model_pred = model.predict(X)
            predictions.append(model_pred)
            
            # Store individual predictions for analysis
            self.individual_predictions[model.model_name] = model_pred
        
        predictions = np.array(predictions)
        
        # Combine predictions based on method
        if self.method == "weighted_average":
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        elif self.method == "simple_average":
            ensemble_pred = np.mean(predictions, axis=0)
        elif self.method == "median":
            ensemble_pred = np.median(predictions, axis=0)
        elif self.method == "best_model":
            # Use prediction from the model with highest weight
            best_model_idx = np.argmax(self.weights)
            ensemble_pred = predictions[best_model_idx]
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble predict_proba (for classification models)"""
        logger.warning("Ensemble predict_proba returns predictions for regression")
        return self.predict(X)
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get average feature importance across models"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained to get feature importance")
        
        importance_scores = []
        model_weights = []
        
        # Collect importance from models that support it
        for i, model in enumerate(self.models):
            importance = model.get_feature_importance()
            if importance is not None:
                importance_scores.append(importance)
                model_weights.append(self.weights[i])
        
        if not importance_scores:
            logger.warning("No models in ensemble provide feature importance")
            return None
        
        # Weighted average of importance scores
        total_weight = sum(model_weights)
        normalized_weights = [w / total_weight for w in model_weights]
        
        ensemble_importance = importance_scores[0] * normalized_weights[0]
        for i in range(1, len(importance_scores)):
            ensemble_importance += importance_scores[i] * normalized_weights[i]
        
        return ensemble_importance.sort_values(ascending=False)
    
    def get_individual_predictions(self) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        return self.individual_predictions.copy()
    
    def update_weights(self, new_weights: List[float]):
        """Update model weights"""
        if len(new_weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(new_weights)
        self.weights = [w / total_weight for w in new_weights]
        
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def add_model(self, model: BaseModel, weight: float = None):
        """Add a new model to the ensemble"""
        self.models.append(model)
        
        if weight is None:
            # Equal weight for all models
            new_weight = 1.0 / len(self.models)
            self.weights = [new_weight] * len(self.models)
        else:
            self.weights.append(weight)
            # Renormalize weights
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        
        # If ensemble was trained, the new model needs to be trained too
        if self.is_trained:
            logger.warning("New model added to trained ensemble. Ensemble needs to be retrained.")
            self.is_trained = False
        
        logger.info(f"Added model {model.model_name} to ensemble")
    
    def remove_model(self, model_name: str):
        """Remove a model from the ensemble"""
        model_indices = [i for i, model in enumerate(self.models) if model.model_name == model_name]
        
        if not model_indices:
            raise ValueError(f"Model {model_name} not found in ensemble")
        
        # Remove model and its weight
        model_idx = model_indices[0]
        self.models.pop(model_idx)
        self.weights.pop(model_idx)
        
        # Renormalize weights
        if self.weights:
            total_weight = sum(self.weights)
            self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Removed model {model_name} from ensemble")
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for each model in the ensemble"""
        performance = {}
        
        for model in self.models:
            if model.is_trained:
                performance[model.model_name] = model.evaluate(X, y)
        
        # Add ensemble performance
        if self.is_trained:
            ensemble_pred = self.predict(X)
            performance['ensemble'] = {
                'mse': np.mean((y - ensemble_pred) ** 2),
                'rmse': np.sqrt(np.mean((y - ensemble_pred) ** 2)),
                'mae': np.mean(np.abs(y - ensemble_pred)),
                'direction_accuracy': np.mean((y > 0) == (ensemble_pred > 0))
            }
        
        return performance