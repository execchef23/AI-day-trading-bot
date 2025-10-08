#!/usr/bin/env python3
"""Test the backtesting system - simplified version"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    print("🔍 AI Day Trading Bot - Backtesting Test")
    print("=" * 50)
    
    try:
        # Import components
        from src.data_sources.data_manager import DataManager
        from src.backtesting.strategy_backtester import SignalBasedBacktester
        
        print("\n📡 Initializing components...")
        dm = DataManager()
        
        # Configure backtester
        backtester = SignalBasedBacktester(
            initial_capital=100000,  # $100k starting capital
            commission=1.0,          # $1 per trade
            position_size=0.2,       # 20% of portfolio per position
            max_positions=3,         # Max 3 positions at once
            stop_loss_pct=0.03,      # 3% stop loss
            take_profit_pct=0.08     # 8% take profit
        )
        
        print("✅ Components initialized successfully")
        
        # Fetch test data
        test_symbols = ["AAPL", "MSFT"]
        print(f"\n📊 Fetching data for {test_symbols}...")
        
        data = {}
        for symbol in test_symbols:
            symbol_data = dm.get_historical_data(symbol, period="2mo")
            if symbol_data is not None and len(symbol_data) > 0:
                data[symbol] = symbol_data
                print(f"   ✅ {symbol}: {len(symbol_data)} data points")
            else:
                print(f"   ❌ Failed to fetch data for {symbol}")
        
        if not data:
            print("❌ No data available for backtesting")
            return
        
        # Run backtest
        print(f"\n🚀 Running backtest on {len(data)} symbols...")
        
        # Run the backtest (will use all available data)
        results = backtester.run_backtest(data=data)
        
        # Display results
        print("\n📈 Backtest completed!")
        results.print_summary()
        
        # Show strategy configuration
        strategy_config = backtester.get_strategy_summary()
        print("\n⚙️ Strategy Configuration:")
        for key, value in strategy_config.items():
            if isinstance(value, float):
                if 'pct' in key:
                    print(f"   {key}: {value:.1%}")
                else:
                    print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n🎉 Backtesting test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during backtesting test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()