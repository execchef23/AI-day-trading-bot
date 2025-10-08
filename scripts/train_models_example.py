#!/usr/bin/env python3
"""Example script for training machine learning models"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_sources.data_manager import DataManager
from src.utils.data_preprocessor import DataPreprocessor
from src.ml_models.model_trainer import ModelTrainer
from src.ml_models.xgboost_model import XGBoostModel
from src.ml_models.lightgbm_model import LightGBMModel
from config.config import ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, TRADING_SYMBOLS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_or_fetch_data():
    """Load processed data or fetch and process new data"""
    
    # Check if we have processed data
    processed_data_dir = "data/processed"
    
    if os.path.exists(processed_data_dir):
        # Look for recent processed data files
        csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.csv')]
        
        if csv_files:
            # Use the most recent file
            latest_file = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(processed_data_dir, x)))
            data_path = os.path.join(processed_data_dir, latest_file)
            
            print(f"📂 Loading processed data from: {latest_file}")
            data = pd.read_csv(data_path)
            
            # Convert Date column back to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            return data, latest_file.split('_')[0]  # Extract symbol from filename
    
    # If no processed data, fetch and process new data
    print("📡 No processed data found. Fetching and processing new data...")
    
    # Initialize data manager
    data_manager = DataManager(
        alpha_vantage_key=ALPHA_VANTAGE_API_KEY,
        polygon_key=POLYGON_API_KEY
    )
    
    # Use first symbol from config
    symbol = TRADING_SYMBOLS[0]
    print(f"🔍 Fetching data for {symbol}...")
    
    # Fetch historical data
    raw_data = data_manager.get_historical_data(
        symbol=symbol,
        period="2y",  # Get more data for better training
        interval="1d"
    )
    
    if raw_data.empty:
        raise ValueError(f"Could not fetch data for {symbol}")
    
    # Process data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_pipeline(raw_data)
    
    # Save processed data
    os.makedirs(processed_data_dir, exist_ok=True)
    output_file = os.path.join(processed_data_dir, f"{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.csv")
    processed_data.to_csv(output_file, index=False)
    
    return processed_data, symbol

def main():
    """Main function to demonstrate model training"""
    
    print("🤖 AI Day Trading Bot - Model Training Example")
    print("=" * 50)
    
    try:
        # Load or fetch data
        data, symbol = load_or_fetch_data()
        print(f"✅ Data loaded successfully for {symbol}")
        print(f"   Shape: {data.shape}")
        print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
        
        # Prepare data for training
        print("\n🎯 Preparing data for machine learning...")
        preprocessor = DataPreprocessor()
        
        # Use preprocessor to prepare training data
        training_data = preprocessor.prepare_for_training(
            data,
            target_horizon=1,  # Predict 1 day ahead returns
            test_size=0.2,
            validation_size=0.1
        )
        
        print(f"✅ Training data prepared:")
        print(f"   Training: {len(training_data['X_train'])} samples")
        print(f"   Validation: {len(training_data['X_val'])} samples")
        print(f"   Test: {len(training_data['X_test'])} samples")
        print(f"   Features: {len(training_data['feature_names'])}")
        
        # Initialize model trainer
        print("\n🏋️  Initializing model trainer...")
        trainer = ModelTrainer(models_dir="models")
        
        # Get default model configurations
        model_configs = trainer.get_default_model_configs()
        print(f"   Will train {len(model_configs)} different models")
        
        # Train multiple models
        print("\n🚀 Starting model training...")
        training_results = trainer.train_multiple_models(
            model_configs,
            training_data['X_train'],
            training_data['y_train'],
            training_data['X_val'],
            training_data['y_val']
        )
        
        print(f"\n✅ Training completed for {len(training_results)} models!")
        
        # Compare model performance
        print("\n📊 Comparing model performance...")
        comparison = trainer.compare_models(
            training_data['X_test'],
            training_data['y_test']
        )
        
        print("\n🏆 Model Performance Comparison:")
        print(comparison[['model_name', 'rmse', 'mae', 'direction_accuracy', 'training_time']].round(4))
        
        # Get best model
        best_model_name, best_model = trainer.get_best_model(
            metric='rmse',
            X_test=training_data['X_test'],
            y_test=training_data['y_test']
        )
        
        print(f"\n🥇 Best Model: {best_model_name}")
        
        # Show feature importance for best model
        feature_importance = best_model.get_feature_importance()
        if feature_importance is not None:
            print(f"\n📈 Top 10 Most Important Features:")
            print(feature_importance.head(10))
        
        # Make sample predictions
        print(f"\n🔮 Sample Predictions (last 5 test samples):")
        sample_X = training_data['X_test'].tail(5)
        sample_y = training_data['y_test'].tail(5)
        predictions = best_model.predict(sample_X)
        
        for i, (actual, predicted) in enumerate(zip(sample_y, predictions)):
            direction_actual = "📈" if actual > 0 else "📉"
            direction_pred = "📈" if predicted > 0 else "📉"
            print(f"   Sample {i+1}: Actual: {actual:.4f} {direction_actual}, Predicted: {predicted:.4f} {direction_pred}")
        
        # Create ensemble model
        print(f"\n🤝 Creating ensemble model...")
        top_models = comparison.head(3)['model_name'].tolist()  # Top 3 models
        ensemble = trainer.create_ensemble(
            model_names=top_models,
            ensemble_name=f"ensemble_{symbol}_{datetime.now().strftime('%Y%m%d')}"
        )
        
        # Train ensemble (this just sets up the ensemble, individual models are already trained)
        ensemble.is_trained = True
        ensemble.feature_names = training_data['feature_names']
        
        # Evaluate ensemble
        ensemble_pred = ensemble.predict(training_data['X_test'])
        ensemble_metrics = {
            'rmse': np.sqrt(np.mean((training_data['y_test'] - ensemble_pred) ** 2)),
            'direction_accuracy': np.mean((training_data['y_test'] > 0) == (ensemble_pred > 0))
        }
        
        print(f"\n🎭 Ensemble Performance:")
        print(f"   RMSE: {ensemble_metrics['rmse']:.4f}")
        print(f"   Direction Accuracy: {ensemble_metrics['direction_accuracy']:.4f}")
        
        # Save ensemble
        ensemble_path = os.path.join("models", f"{ensemble.model_name}.joblib")
        ensemble.save_model(ensemble_path)
        print(f"   Ensemble saved to: {ensemble_path}")
        
        print(f"\n🎉 Model training example completed successfully!")
        print(f"\nTrained models saved in: models/")
        print(f"\nNext steps:")
        print(f"1. Check the training summary: models/training_summary.json")
        print(f"2. Run the dashboard to see predictions: streamlit run dashboard.py")
        print(f"3. Implement live trading signals with the trained models")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()