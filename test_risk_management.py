"""Test risk management system with portfolio integration"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from src.data_sources import DataManager
    from src.risk_management import (
        PortfolioManager,
        PositionSizer,
        PositionSizingMethod,
        RiskManager,
        RiskParameters,
    )

    print("✓ Successfully imported risk management modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    exit(1)


def create_sample_market_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """Create sample market data for testing"""

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
    )

    # Generate realistic price data with some volatility
    np.random.seed(42)  # For reproducible results

    base_price = 100.0
    returns = np.random.normal(
        0.001, 0.02, len(dates)
    )  # 0.1% daily return, 2% volatility
    prices = [base_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data["Close"] = prices
    data["Open"] = data["Close"].shift(1).fillna(base_price)
    highs = []
    lows = []
    for _, row in data.iterrows():
        high_val = max(row["Open"], row["Close"]) * (1 + np.random.uniform(0, 0.01))
        low_val = min(row["Open"], row["Close"]) * (1 - np.random.uniform(0, 0.01))
        highs.append(high_val)
        lows.append(low_val)

    data["High"] = highs
    data["Low"] = lows
    data["Volume"] = np.random.randint(100000, 1000000, len(data))

    return data


def test_position_sizer():
    """Test position sizing functionality"""
    print("\n=== Testing Position Sizer ===")

    # Initialize with custom risk parameters
    risk_params = RiskParameters(
        max_portfolio_risk=0.02,  # 2% max risk per trade
        max_position_size=0.1,  # 10% max position size
        volatility_lookback=20,
    )

    sizer = PositionSizer(risk_params)

    # Test different position sizing methods
    portfolio_value = 100000.0
    current_price = 150.0
    volatility = 0.25  # 25% annualized volatility

    methods = [
        (PositionSizingMethod.FIXED_PERCENTAGE, {"percentage": 0.05}),
        (PositionSizingMethod.VOLATILITY_BASED, {}),
        (
            PositionSizingMethod.KELLY_CRITERION,
            {"win_probability": 0.6, "avg_win": 0.05, "avg_loss": 0.02},
        ),
        (PositionSizingMethod.FIXED_AMOUNT, {"amount": 5000}),
    ]

    for method, kwargs in methods:
        position_size = sizer.calculate_position_size(
            symbol="AAPL",
            current_price=current_price,
            portfolio_value=portfolio_value,
            volatility=volatility,
            method=method,
            **kwargs,
        )

        position_value = position_size * current_price
        position_pct = position_value / portfolio_value

        print(f"Method: {method.value}")
        print(f"  Position Size: {position_size} shares")
        print(f"  Position Value: ${position_value:,.2f}")
        print(f"  Portfolio %: {position_pct:.2%}")
        print()


def test_risk_manager():
    """Test comprehensive risk management"""
    print("\n=== Testing Risk Manager ===")

    risk_manager = RiskManager()

    # Create sample market data for correlation testing
    symbols = ["AAPL", "MSFT", "GOOGL"]
    market_data = {}
    current_prices = {}

    for symbol in symbols:
        data = create_sample_market_data(symbol)
        market_data[symbol] = data
        current_prices[symbol] = data["Close"].iloc[-1]

    # Test trade risk assessment
    symbol = "AAPL"
    proposed_position = 50.0
    current_price = current_prices[symbol]
    portfolio_value = 100000.0
    existing_positions = {"MSFT": 40.0, "GOOGL": 30.0}

    risk_assessment = risk_manager.assess_trade_risk(
        symbol=symbol,
        proposed_position=proposed_position,
        current_price=current_price,
        portfolio_value=portfolio_value,
        existing_positions=existing_positions,
        market_data=market_data,
    )

    print(f"Risk Assessment for {symbol}:")
    print(f"  Recommendation: {risk_assessment['recommendation']}")
    print(f"  Adjusted Position: {risk_assessment['adjusted_position']} shares")
    print(f"  Warnings: {len(risk_assessment['warnings'])}")

    for warning in risk_assessment["warnings"]:
        print(f"    - {warning}")

    print("\n  Risk Checks:")
    for check_name, check_result in risk_assessment["risk_checks"].items():
        status = "✓" if check_result.get("passed", True) else "✗"
        print(f"    {status} {check_name}: {check_result}")

    # Test stop-loss calculation
    entry_price = current_price
    volatility = 0.02  # 2% daily volatility

    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=entry_price, volatility=volatility, method="volatility"
    )

    take_profit = risk_manager.calculate_take_profit(
        entry_price=entry_price, stop_loss=stop_loss, risk_reward_ratio=2.0
    )

    print(f"\n  Entry Price: ${entry_price:.2f}")
    print(
        f"  Stop Loss: ${stop_loss:.2f} ({((stop_loss / entry_price - 1) * 100):+.2f}%)"
    )
    print(
        f"  Take Profit: ${take_profit:.2f} ({((take_profit / entry_price - 1) * 100):+.2f}%)"
    )

    # Test portfolio risk summary
    positions = {symbol: 50.0 for symbol in symbols}

    portfolio_summary = risk_manager.get_portfolio_risk_summary(
        positions=positions,
        current_prices=current_prices,
        portfolio_value=portfolio_value,
    )

    print(f"\n  Portfolio Summary:")
    print(f"    Total Positions: {portfolio_summary['total_positions']}")
    print(f"    Equity Exposure: {portfolio_summary['total_equity_exposure']:.2%}")
    print(
        f"    Diversification Score: {portfolio_summary['risk_metrics']['diversification_score']:.2f}"
    )
    print(f"    Overall Risk Level: {portfolio_summary['risk_metrics']['risk_level']}")


def test_portfolio_manager():
    """Test integrated portfolio management"""
    print("\n=== Testing Portfolio Manager ===")

    # Initialize data manager for realistic testing
    try:
        data_manager = DataManager()
        print("✓ Data manager initialized")
    except Exception as e:
        print(f"⚠ Could not initialize data manager: {e}")
        print("  Using mock data for testing")
        data_manager = None

    # Initialize portfolio manager
    portfolio_manager = PortfolioManager(
        initial_cash=100000.0, data_manager=data_manager
    )

    print(f"Initial Portfolio Value: ${portfolio_manager.get_current_value():,.2f}")

    # Test trade execution (dry run)
    test_trades = [
        ("AAPL", "BUY", 50, 150.0),
        ("MSFT", "BUY", 30, 300.0),
        ("GOOGL", "BUY", 20, 2500.0),
    ]

    for symbol, action, quantity, price in test_trades:
        result = portfolio_manager.execute_trade(
            symbol=symbol, action=action, quantity=quantity, price=price, dry_run=True
        )

        print(f"\n{result['message']}")
        if result["success"]:
            print(f"  Status: ✓ Success")
        else:
            print(f"  Status: ✗ Failed")

    # Test with actual trade execution (for demo)
    print("\n--- Executing Sample Trades ---")
    for symbol, action, quantity, price in test_trades[:2]:  # Execute first 2 trades
        result = portfolio_manager.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            stop_loss=price * 0.95,  # 5% stop loss
            take_profit=price * 1.1,  # 10% take profit
            dry_run=False,
        )

        print(f"{result['message']}")

        if result["success"] and result.get("new_position"):
            pos = result["new_position"]
            print(f"  New Position: {pos.quantity} shares @ ${pos.entry_price:.2f}")
            print(f"  Stop Loss: ${pos.stop_loss:.2f}")
            print(f"  Take Profit: ${pos.take_profit:.2f}")

    # Test portfolio summary
    current_prices = {symbol: price for symbol, _, _, price in test_trades}

    summary = portfolio_manager.get_portfolio_summary(current_prices)

    print(f"\n--- Portfolio Summary ---")
    print(f"Cash Balance: ${summary['cash_balance']:,.2f}")
    print(f"Positions Value: ${summary['positions_value']:,.2f}")
    print(f"Total Value: ${summary['total_value']:,.2f}")
    print(f"Total Return: {summary['total_return']:.2%}")
    print(f"Unrealized P&L: ${summary['unrealized_pnl']:,.2f}")
    print(f"Current Positions: {summary['current_positions']}")

    print(f"\nPosition Details:")
    for pos in summary["positions"]:
        print(
            f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['current_price']:.2f}"
        )
        print(f"    Value: ${pos['position_value']:,.2f}")
        print(f"    P&L: ${pos['unrealized_pnl']:,.2f}")

    # Test stop-loss and take-profit checking
    # Simulate price movement
    stress_prices = {
        "AAPL": 140.0,  # Price drop - might trigger stop loss
        "MSFT": 330.0,  # Price rise - might trigger take profit
    }

    triggered_orders = portfolio_manager.check_stop_losses_and_take_profits(
        stress_prices
    )

    if triggered_orders:
        print(f"\n--- Triggered Orders ---")
        for order in triggered_orders:
            print(
                f"{order['symbol']}: {order['trigger_type']} at ${order['current_price']:.2f}"
            )
            print(f"  Trigger Price: ${order['trigger_price']:.2f}")
            print(
                f"  Recommended: {order['recommended_action']} {order['quantity']} shares"
            )
    else:
        print(f"\nNo stop-loss or take-profit orders triggered.")


def main():
    """Run all risk management tests"""
    print("🚀 Testing AI Trading Bot Risk Management System")
    print("=" * 60)

    try:
        test_position_sizer()
        test_risk_manager()
        test_portfolio_manager()

        print("\n" + "=" * 60)
        print("✅ All risk management tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
