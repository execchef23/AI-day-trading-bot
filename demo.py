#!/usr/bin/env python3
"""Quick demo of AI Day Trading Bot capabilities"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def main():
    print("🤖 AI Day Trading Bot - Quick Demo")
    print("=" * 40)

    try:
        # Initialize components
        print("\n📡 Initializing data manager...")
        from src.data_sources.data_manager import DataManager

        dm = DataManager()

        print("🔍 Fetching sample data...")
        symbol = "AAPL"
        data = dm.get_historical_data(symbol, period="1mo")

        if data is not None and len(data) > 0:
            print(f"✅ Successfully fetched {len(data)} records for {symbol}")
            print(
                f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}"
            )

            # Show basic technical indicators work
            print("\n🔧 Testing technical analysis...")
            from src.utils.technical_indicators import TechnicalIndicators

            ti = TechnicalIndicators()

            # Add some basic indicators
            data_with_indicators = ti.add_all_indicators(data.copy())
            indicators_added = data_with_indicators.shape[1] - data.shape[1]
            print(f"✅ Added {indicators_added} technical indicators")

            # Show current market data
            print("\n🎯 Current market data:")
            latest = data.iloc[-1]
            print(f"   {symbol}: ${latest['Close']:.2f}")
            print(f"   Volume: {latest['Volume']:,}")
            print(
                f"   Change: {((latest['Close'] - latest['Open']) / latest['Open'] * 100):+.2f}%"
            )

            # Show that AI models are available
            print("\n🧠 Available AI models:")
            from src.ml_models import LightGBMModel, XGBoostModel

            print("   ✅ XGBoost - Fast gradient boosting")
            print("   ✅ LightGBM - Memory efficient boosting")
            if LSTMModel is not None:
                print("   ✅ LSTM - Deep learning time series")
            else:
                print("   ⚠️  LSTM - TensorFlow not installed")

        else:
            print("❌ Could not fetch data")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return

    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Visit http://localhost:8501 for the dashboard")
    print("2. Add your API keys to .env file for full functionality")
    print("3. Run: python run_bot.py --help for all options")


if __name__ == "__main__":
    # Import with error handling
    try:
        from src.ml_models import LSTMModel
    except ImportError:
        LSTMModel = None

    main()
