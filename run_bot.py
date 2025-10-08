#!/usr/bin/env python3
"""Main script to run the AI Day Trading Bot"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_sources.data_manager import DataManager
from src.utils.data_preprocessor import DataPreprocessor
from src.ml_models.model_trainer import ModelTrainer
from config.config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'trading_bot_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_data(symbols, period="1y"):
    """Fetch market data for symbols"""
    logger.info(f"Fetching data for {len(symbols)} symbols")
    
    data_manager = DataManager(
        alpha_vantage_key=ALPHA_VANTAGE_API_KEY,
        polygon_key=POLYGON_API_KEY
    )
    
    all_data = {}
    for symbol in symbols:
        try:
            data = data_manager.get_historical_data(symbol, period=period)
            if not data.empty:
                all_data[symbol] = data
                logger.info(f"Fetched {len(data)} records for {symbol}")
            else:
                logger.warning(f"No data for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    
    return all_data

def process_data(data, symbol):
    """Process raw market data"""
    logger.info(f"Processing data for {symbol}")
    
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.process_pipeline(data)
    
    # Save processed data
    output_file = os.path.join(DATA_DIR, 'processed', f"{symbol}_processed_{datetime.now().strftime('%Y%m%d')}.csv")
    processed_data.to_csv(output_file, index=False)
    logger.info(f"Processed data saved to {output_file}")
    
    return processed_data

def train_models(data, symbol):
    """Train ML models on processed data"""
    logger.info(f"Training models for {symbol}")
    
    preprocessor = DataPreprocessor()
    training_data = preprocessor.prepare_for_training(data)
    
    trainer = ModelTrainer()
    model_configs = trainer.get_default_model_configs()
    
    # Train models
    results = trainer.train_multiple_models(
        model_configs,
        training_data['X_train'],
        training_data['y_train'],
        training_data['X_val'],
        training_data['y_val']
    )
    
    # Get best model
    best_name, best_model = trainer.get_best_model(
        X_test=training_data['X_test'],
        y_test=training_data['y_test']
    )
    
    logger.info(f"Best model for {symbol}: {best_name}")
    return trainer, best_model

def generate_signals(model, data, symbol):
    """Generate trading signals using trained model"""
    logger.info(f"Generating signals for {symbol}")
    
    try:
        # Get recent data for prediction
        recent_data = data.tail(100)  # Last 100 days
        
        # Make predictions
        predictions = model.predict(recent_data)
        latest_prediction = predictions[-1] if len(predictions) > 0 else 0
        
        # Simple signal logic
        if latest_prediction > 0.02:  # Expect >2% return
            signal = "STRONG_BUY"
        elif latest_prediction > 0.005:  # Expect >0.5% return
            signal = "BUY"
        elif latest_prediction < -0.02:  # Expect <-2% return
            signal = "STRONG_SELL"
        elif latest_prediction < -0.005:  # Expect <-0.5% return
            signal = "SELL"
        else:
            signal = "HOLD"
        
        signal_data = {
            'symbol': symbol,
            'signal': signal,
            'prediction': latest_prediction,
            'confidence': abs(latest_prediction),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Signal for {symbol}: {signal} (prediction: {latest_prediction:.4f})")
        return signal_data
        
    except Exception as e:
        logger.error(f"Error generating signals for {symbol}: {e}")
        return None

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI Day Trading Bot")
    parser.add_argument('--mode', choices=['fetch', 'train', 'signal', 'full'], 
                       default='full', help='Mode to run')
    parser.add_argument('--symbols', nargs='+', default=TRADING_SYMBOLS[:3],
                       help='Symbols to process')
    parser.add_argument('--period', default='1y', help='Data period to fetch')
    
    args = parser.parse_args()
    
    logger.info(f"Starting AI Trading Bot in {args.mode} mode")
    logger.info(f"Symbols: {args.symbols}")
    
    try:
        if args.mode in ['fetch', 'full']:
            # Fetch data
            all_data = fetch_data(args.symbols, args.period)
            
            if not all_data:
                logger.error("No data fetched. Check API keys and connectivity.")
                return
        
        if args.mode in ['train', 'full']:
            # Process and train for each symbol
            trained_models = {}
            
            for symbol in args.symbols:
                if args.mode == 'full' and symbol in all_data:
                    data = all_data[symbol]
                else:
                    # Load processed data
                    processed_files = [f for f in os.listdir(os.path.join(DATA_DIR, 'processed')) 
                                     if f.startswith(symbol) and f.endswith('.csv')]
                    if not processed_files:
                        logger.warning(f"No processed data found for {symbol}")
                        continue
                    
                    latest_file = max(processed_files)
                    data_path = os.path.join(DATA_DIR, 'processed', latest_file)
                    data = pd.read_csv(data_path)
                    data['Date'] = pd.to_datetime(data['Date'])
                
                # Process data
                processed_data = process_data(data, symbol)
                
                # Train models
                trainer, best_model = train_models(processed_data, symbol)
                trained_models[symbol] = (trainer, best_model)
        
        if args.mode in ['signal', 'full']:
            # Generate signals
            signals = []
            
            for symbol in args.symbols:
                if args.mode == 'full':
                    # Use trained model
                    if symbol in trained_models:
                        _, model = trained_models[symbol]
                        processed_data = process_data(all_data[symbol], symbol)
                        signal = generate_signals(model, processed_data, symbol)
                        if signal:
                            signals.append(signal)
                else:
                    # Load trained model
                    model_files = [f for f in os.listdir(MODELS_DIR) 
                                 if f.startswith(symbol) and f.endswith('.joblib')]
                    if not model_files:
                        logger.warning(f"No trained model found for {symbol}")
                        continue
                    
                    # Load the most recent model
                    latest_model_file = max(model_files)
                    model_path = os.path.join(MODELS_DIR, latest_model_file)
                    
                    # This would require loading the specific model type
                    # For now, skip if in signal-only mode without full pipeline
                    logger.info(f"Signal generation requires full pipeline for {symbol}")
            
            # Display signals
            if signals:
                print("\n📊 Trading Signals:")
                print("=" * 50)
                for signal in signals:
                    print(f"{signal['symbol']}: {signal['signal']} (confidence: {signal['confidence']:.3f})")
                
                # Save signals
                import json
                signals_file = os.path.join(DATA_DIR, f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(signals_file, 'w') as f:
                    json.dump(signals, f, indent=2)
                logger.info(f"Signals saved to {signals_file}")
        
        logger.info("Trading bot execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Import pandas here to avoid import errors during argument parsing
    import pandas as pd
    main()