#!/usr/bin/env python3
"""Example script for fetching market data and basic preprocessing"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_sources.data_manager import DataManager
from src.utils.data_preprocessor import DataPreprocessor
from config.config import ALPHA_VANTAGE_API_KEY, POLYGON_API_KEY, TRADING_SYMBOLS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate data fetching and preprocessing"""
    
    print("🤖 AI Day Trading Bot - Data Fetching Example")
    print("=" * 50)
    
    # Initialize data manager
    print("\n📡 Initializing data manager...")
    data_manager = DataManager(
        alpha_vantage_key=ALPHA_VANTAGE_API_KEY,
        polygon_key=POLYGON_API_KEY
    )
    
    print(f"Available data providers: {data_manager.get_available_providers()}")
    
    # Test symbols (start with a few popular ones)
    test_symbols = TRADING_SYMBOLS[:3]  # First 3 symbols
    print(f"\n📈 Testing with symbols: {test_symbols}")
    
    # Fetch historical data for each symbol
    all_data = {}
    
    for symbol in test_symbols:
        print(f"\n🔍 Fetching data for {symbol}...")
        
        try:
            # Get 1 year of daily data
            data = data_manager.get_historical_data(
                symbol=symbol,
                period="1y",
                interval="1d"
            )
            
            if not data.empty:
                print(f"✅ Successfully fetched {len(data)} records for {symbol}")
                print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
                print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
                
                all_data[symbol] = data
            else:
                print(f"❌ No data retrieved for {symbol}")
                
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
    
    if not all_data:
        print("\n❌ No data was successfully fetched. Check your API keys and internet connection.")
        return
    
    # Demonstrate preprocessing on one symbol
    symbol_to_process = list(all_data.keys())[0]
    raw_data = all_data[symbol_to_process]
    
    print(f"\n🔧 Preprocessing data for {symbol_to_process}...")
    
    try:
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Run complete preprocessing pipeline
        processed_data = preprocessor.process_pipeline(raw_data)
        
        print(f"✅ Preprocessing complete!")
        print(f"   Original shape: {raw_data.shape}")
        print(f"   Processed shape: {processed_data.shape}")
        
        # Show feature summary
        feature_summary = preprocessor.get_feature_summary(processed_data)
        print(f"\n📊 Feature Summary:")
        print(f"   Total features: {feature_summary['total_features']}")
        print(f"   Feature types:")
        for feature_type, count in feature_summary['feature_types'].items():
            if count > 0:
                print(f"     - {feature_type}: {count}")
        
        # Show sample of processed data
        print(f"\n📋 Sample of processed data (last 5 rows):")
        print(processed_data[['Date', 'Close', 'rsi', 'macd', 'bb_position', 'volume_ratio']].tail())
        
        # Prepare data for ML training
        print(f"\n🎯 Preparing data for machine learning...")
        training_data = preprocessor.prepare_for_training(
            processed_data,
            target_horizon=1,  # Predict 1 day ahead
            test_size=0.2,
            validation_size=0.1
        )
        
        print(f"✅ Data preparation complete!")
        print(f"   Training samples: {training_data['split_info']['train_samples']}")
        print(f"   Validation samples: {training_data['split_info']['val_samples']}")
        print(f"   Test samples: {training_data['split_info']['test_samples']}")
        print(f"   Features: {len(training_data['feature_names'])}")
        
        # Save processed data
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"{symbol_to_process}_processed_{datetime.now().strftime('%Y%m%d')}.csv")
        processed_data.to_csv(output_file, index=False)
        print(f"\n💾 Processed data saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return
    
    # Test current price fetching
    print(f"\n💰 Current Prices:")
    for symbol in test_symbols:
        try:
            current_price = data_manager.get_current_price(symbol)
            if current_price:
                print(f"   {symbol}: ${current_price:.2f}")
            else:
                print(f"   {symbol}: Price not available")
        except Exception as e:
            print(f"   {symbol}: Error - {e}")
    
    # Market status
    market_open = data_manager.is_market_open()
    print(f"\n🏛️  Market Status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
    
    print("\n🎉 Data fetching example completed successfully!")
    print("\nNext steps:")
    print("1. Set up your API keys in the .env file")
    print("2. Run the model training example: python scripts/train_models_example.py")
    print("3. Start the dashboard: streamlit run dashboard.py")

if __name__ == "__main__":
    main()