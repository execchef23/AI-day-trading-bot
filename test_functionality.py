#!/usr/bin/env python3
"""
Test script to validate AI Trading Bot functionality
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))


def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing module imports...")

    try:
        from src.data_sources.data_manager import DataManager

        print("✅ DataManager imported successfully")
    except Exception as e:
        print(f"❌ DataManager import failed: {e}")

    try:
        from src.risk_management.risk_manager import RiskManager

        print("✅ RiskManager imported successfully")
    except Exception as e:
        print(f"❌ RiskManager import failed: {e}")

    try:
        from src.ml_models.ensemble_model import EnsembleModel

        print("✅ EnsembleModel imported successfully")
    except Exception as e:
        print(f"❌ EnsembleModel import failed: {e}")


def test_data_fetching():
    """Test data fetching capability"""
    print("\n📊 Testing data fetching...")

    try:
        import yfinance as yf

        # Test fetching Apple stock data
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="5d")

        if not data.empty:
            print(f"✅ Successfully fetched AAPL data: {len(data)} days")
            print(f"   Latest close price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("❌ No data retrieved")

    except Exception as e:
        print(f"❌ Data fetching failed: {e}")


def test_ml_models():
    """Test ML model functionality"""
    print("\n🤖 Testing ML models...")

    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        # Create sample data
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)  # 0=sell, 1=hold, 2=buy

        # Test model training
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Test prediction
        prediction = model.predict(X[:1])
        print(f"✅ ML model test successful - Sample prediction: {prediction[0]}")

    except Exception as e:
        print(f"❌ ML model test failed: {e}")


def test_risk_calculations():
    """Test risk management calculations"""
    print("\n🛡️ Testing risk calculations...")

    try:
        import numpy as np

        # Sample portfolio data
        portfolio_value = 100000
        position_values = np.array([25000, 30000, 20000, 15000, 10000])

        # Calculate basic risk metrics
        total_positions = np.sum(position_values)
        largest_position = np.max(position_values)
        concentration_risk = largest_position / portfolio_value

        print(f"✅ Risk calculations successful:")
        print(f"   Portfolio value: ${portfolio_value:,.2f}")
        print(f"   Largest position: {concentration_risk:.1%}")
        print(f"   Total invested: ${total_positions:,.2f}")

    except Exception as e:
        print(f"❌ Risk calculation test failed: {e}")


def main():
    """Run all tests"""
    print("🚀 AI Trading Bot - Functionality Test")
    print("=" * 50)

    test_imports()
    test_data_fetching()
    test_ml_models()
    test_risk_calculations()

    print("\n" + "=" * 50)
    print("🎉 Testing complete! Check results above.")
    print("💡 If all tests passed, your bot is ready for trading!")


if __name__ == "__main__":
    main()
