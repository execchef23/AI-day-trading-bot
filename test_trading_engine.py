#!/usr/bin/env python3
"""
Test Live Trading Engine

Simple test script to verify the trading engine functionality.
"""

import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from trading_engine.live_trading_engine import LiveTradingEngine, TradingConfig
    print("✅ Trading engine imported successfully")
except ImportError as e:
    print(f"❌ Failed to import trading engine: {e}")
    sys.exit(1)

def test_trading_engine():
    """Test basic trading engine functionality"""

    print("\n🤖 Testing Live Trading Engine")
    print("=" * 50)

    # Create configuration
    config = TradingConfig(
        symbols=["AAPL", "MSFT"],
        initial_capital=50000,
        max_positions=2,
        position_size_pct=0.25,
        dry_run=True,
        paper_trading=True,
        update_interval_seconds=5
    )

    # Create engine
    engine = LiveTradingEngine(config)
    print("✅ Trading engine created")

    # Test status
    status = engine.get_status()
    print(f"Initial status: {status['state']}")

    # Test start
    print("\n🚀 Starting trading engine...")
    result = engine.start_trading()
    print(f"Start result: {result}")

    if result["success"]:
        # Let it run briefly
        print("\n⏰ Running engine for 10 seconds...")
        time.sleep(10)

        # Check status
        status = engine.get_status()
        print(f"\nEngine status after 10s:")
        print(f"  State: {status['state']}")
        print(f"  Uptime: {status['uptime_hours']:.2f}h")
        print(f"  Positions: {status['positions_count']}")
        print(f"  Trades: {status['total_trades']}")

        # Check positions
        positions = engine.get_positions()
        print(f"  Active positions: {len(positions)}")

        # Check signals
        signals = engine.get_recent_signals()
        print(f"  Recent signals: {len(signals)}")

        # Stop engine
        print("\n🛑 Stopping trading engine...")
        result = engine.stop_trading()
        print(f"Stop result: {result}")

    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_trading_engine()
