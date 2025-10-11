#!/usr/bin/env python3
"""
Enhanced AI Trading Bot - Model Training with Hyperparameter Optimization

This script provides a complete ML pipeline with:
- Data loading and preprocessing
- Feature engineering and validation
- Hyperparameter optimization using Optuna
- Model training with optimized parameters
- Model persistence and ensemble creation
- Performance evaluation and comparison
"""

import json
import logging
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

import numpy as np

# Core imports
import pandas as pd

from config.config import *
from src.data_sources.data_manager import DataManager
from src.ml_models.hyperparameter_optimizer import HyperparameterOptimizer
from src.ml_models.model_trainer import ModelTrainer
from src.utils.data_preprocessor import DataPreprocessor


def load_or_fetch_data():
    """Load existing processed data or fetch and process new data"""

    processed_data_dir = os.path.join(DATA_DIR, "processed")

    # Check for existing processed data
    if os.path.exists(processed_data_dir):
        csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith(".csv")]

        if csv_files:
            # Use the most recent file
            latest_file = max(
                csv_files,
                key=lambda x: os.path.getctime(os.path.join(processed_data_dir, x)),
            )
            data_path = os.path.join(processed_data_dir, latest_file)

            print(f"📂 Loading processed data from: {latest_file}")
            data = pd.read_csv(data_path)

            # Convert Date column back to datetime
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"])

            return data, latest_file.split("_")[0]  # Extract symbol from filename

    # If no processed data, fetch and process new data
    print("📡 No processed data found. Fetching and processing new data...")

    # Initialize data manager (will work in demo mode without API keys)
    data_manager = DataManager()

    # Use first symbol from config
    symbol = TRADING_SYMBOLS[0]
    print(f"🔍 Fetching data for {symbol}...")

    # Fetch historical data
    raw_data = data_manager.get_historical_data(
        symbol=symbol,
        period="2y",  # Get more data for better training
        interval="1d",
    )

    if raw_data.empty:
        raise ValueError(f"Could not fetch data for {symbol}")

    # Process data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_pipeline(raw_data)

    # Save processed data
    os.makedirs(processed_data_dir, exist_ok=True)
    output_file = os.path.join(
        processed_data_dir,
        f"{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.csv",
    )
    processed_data.to_csv(output_file, index=False)

    print(f"✅ Data processed and saved to: {output_file}")

    return processed_data, symbol


def main():
    """Main training pipeline with hyperparameter optimization"""

    print("🤖 AI Day Trading Bot - Enhanced Model Training")
    print("=" * 60)

    try:
        # Load or fetch data
        data, symbol = load_or_fetch_data()

        print(f"✅ Data loaded successfully for {symbol}")
        print(f"   Shape: {data.shape}")
        if "Date" in data.columns:
            print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")

        # Prepare data for training
        print("\n🎯 Preparing data for machine learning...")
        preprocessor = DataPreprocessor()

        # Use preprocessor to prepare training data
        training_data = preprocessor.prepare_for_training(
            data,
            target_horizon=1,  # Predict 1 day ahead returns
            test_size=0.2,
            validation_size=0.1,
        )

        print(f"✅ Training data prepared:")
        print(f"   Training: {len(training_data['X_train'])} samples")
        print(f"   Validation: {len(training_data['X_val'])} samples")
        print(f"   Test: {len(training_data['X_test'])} samples")
        print(f"   Features: {len(training_data['feature_names'])}")

        # Check if we want to run hyperparameter optimization
        print("\n🔧 Hyperparameter Optimization Options:")
        print("1. Quick training with default parameters")
        print("2. Full hyperparameter optimization (recommended, takes longer)")

        choice = input("\nEnter your choice (1 or 2, default=1): ").strip() or "1"

        if choice == "2":
            # Run hyperparameter optimization
            print("\n🚀 Starting hyperparameter optimization...")

            optimizer = HyperparameterOptimizer(
                n_trials=20, timeout=300
            )  # 5 min per model

            optimization_results = optimizer.optimize_all_models(
                training_data["X_train"],
                training_data["y_train"],
                training_data["X_val"],
                training_data["y_val"],
            )

            print("\n✅ Hyperparameter optimization complete!")

            # Show optimization results
            for model_type, results in optimization_results.items():
                if model_type != "metadata":
                    print(f"\n🏆 Best {model_type.upper()} parameters:")
                    print(f"   RMSE: {results['best_value']:.4f}")
                    print(f"   Trials: {results['n_trials']}")
                    for param, value in results["best_params"].items():
                        print(f"   {param}: {value}")

            # Get optimized model configurations
            model_configs = optimizer.get_optimized_model_configs()

        else:
            # Use default configurations for quick training
            trainer = ModelTrainer(models_dir="models")
            model_configs = trainer.get_default_model_configs()

        # Initialize model trainer
        print(f"\n🏋️  Training {len(model_configs)} models...")
        trainer = ModelTrainer(models_dir="models")

        # Train models
        training_results = trainer.train_multiple_models(
            model_configs,
            training_data["X_train"],
            training_data["y_train"],
            training_data["X_val"],
            training_data["y_val"],
        )

        print(f"\n✅ Training completed for {len(training_results)} models!")

        # Compare model performance
        print("\n📊 Comparing model performance...")
        comparison = trainer.compare_models(
            training_data["X_test"], training_data["y_test"]
        )

        # Display results
        print("\n🏆 Model Performance Comparison:")
        print(comparison.round(4))

        best_model = comparison.loc[comparison["rmse"].idxmin(), "model_name"]
        print(f"\n🥇 Best Model: {best_model}")

        # Feature importance
        print(f"\n📈 Top 10 Most Important Features:")
        feature_importance = trainer.get_feature_importance()
        if not feature_importance.empty:
            print(feature_importance.head(10))

        # Sample predictions
        print(f"\n🔮 Sample Predictions (last 5 test samples):")
        trainer.show_sample_predictions(
            training_data["X_test"].tail(), training_data["y_test"].tail()
        )

        # Create ensemble model
        print(f"\n🤝 Creating ensemble model...")
        ensemble_name = f"ensemble_{symbol}_{datetime.now().strftime('%Y%m%d')}"
        ensemble_performance = trainer.create_ensemble_model(
            training_data["X_test"], training_data["y_test"], ensemble_name
        )

        print(f"\n🎭 Ensemble Performance:")
        print(f"   RMSE: {ensemble_performance['rmse']:.4f}")
        print(
            f"   Direction Accuracy: {ensemble_performance['direction_accuracy']:.4f}"
        )
        print(f"   Ensemble saved to: models/{ensemble_name}.joblib")

        # Save comprehensive training summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "data_shape": list(data.shape),
            "training_samples": len(training_data["X_train"]),
            "validation_samples": len(training_data["X_val"]),
            "test_samples": len(training_data["X_test"]),
            "num_features": len(training_data["feature_names"]),
            "feature_names": training_data["feature_names"],
            "best_model": best_model,
            "model_performance": comparison.to_dict("records"),
            "ensemble_performance": ensemble_performance,
            "hyperparameter_optimization": choice == "2",
        }

        summary_path = os.path.join(
            "models",
            f"training_summary_enhanced_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n📋 Training summary saved to: {summary_path}")

        print("\n🎉 Enhanced model training completed successfully!")

        print(f"\n💡 Next steps:")
        print("1. Check training summary for detailed results")
        print("2. Run the dashboard to see live predictions: streamlit run app.py")
        print("3. Use the trained models for live trading signals")
        print("4. Monitor model performance and retrain periodically")

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
