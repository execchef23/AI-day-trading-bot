"""
Test Small Account Trading System

Validate all components of the small account trading system.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

try:
    from src.small_account import (
        AccountTier,
        GrowthCalculator,
        GrowthScenario,
        GrowthStrategy,
        SmallAccountPositionSizer,
        SmallAccountStrategies,
        StrategyType,
    )

    SMALL_ACCOUNT_LOADED = True
except ImportError as e:
    print(f"❌ Failed to import small account modules: {e}")
    SMALL_ACCOUNT_LOADED = False


def test_position_sizer():
    """Test the position sizer functionality"""
    print("🧪 Testing Position Sizer...")

    if not SMALL_ACCOUNT_LOADED:
        raise ImportError("Small account modules not available")

    sizer = SmallAccountPositionSizer()

    # Test account tier classification
    assert sizer.get_account_tier(150) == AccountTier.MICRO
    assert sizer.get_account_tier(350) == AccountTier.SMALL
    assert sizer.get_account_tier(750) == AccountTier.MEDIUM
    assert sizer.get_account_tier(2500) == AccountTier.LARGE

    print("✅ Account tier classification works")

    # Test position sizing
    result = sizer.calculate_position_size(
        account_balance=500, stock_price=100, stop_loss_price=95, confidence=0.7
    )

    assert result.tier == AccountTier.SMALL
    assert result.can_afford == True
    assert result.dollar_amount > 0
    assert result.shares > 0

    print(
        f"✅ Position sizing: ${result.dollar_amount:.0f} for {result.shares:.2f} shares"
    )

    # Test insufficient funds
    result_large = sizer.calculate_position_size(
        account_balance=200,
        stock_price=1000,  # Very expensive stock
        stop_loss_price=950,
        confidence=0.8,
    )

    print(f"✅ Handles expensive stocks: {result_large.reason}")


def test_growth_strategies():
    """Test growth strategies"""
    print("\n🧪 Testing Growth Strategies...")

    if not SMALL_ACCOUNT_LOADED:
        raise ImportError("Small account modules not available")

    strategies = SmallAccountStrategies.get_all_strategies()

    assert len(strategies) == 4
    assert StrategyType.MOMENTUM_SCALP in strategies
    assert StrategyType.SWING_TRADING in strategies

    print("✅ All 4 strategies loaded")

    # Test strategy recommendation
    rec_conservative = SmallAccountStrategies.recommend_strategy(300, "conservative")
    rec_aggressive = SmallAccountStrategies.recommend_strategy(600, "aggressive")

    print(f"✅ Conservative recommendation: {rec_conservative.value}")
    print(f"✅ Aggressive recommendation: {rec_aggressive.value}")

    # Test strategy metrics
    metrics = SmallAccountStrategies.calculate_strategy_metrics(
        StrategyType.SWING_TRADING, 20
    )

    assert "monthly_return_rate" in metrics
    assert "win_rate" in metrics

    print(f"✅ Strategy metrics: {metrics['monthly_return_rate']:.2%} monthly return")


def test_growth_calculator():
    """Test growth calculator"""
    print("\n🧪 Testing Growth Calculator...")

    if not SMALL_ACCOUNT_LOADED:
        raise ImportError("Small account modules not available")

    calculator = GrowthCalculator()

    # Test projections
    projections = calculator.calculate_projections(500, 12)

    assert len(projections) == 4  # All scenarios
    assert GrowthScenario.MODERATE in projections

    moderate_proj = projections[GrowthScenario.MODERATE]
    assert moderate_proj.final_balance > 500  # Should grow

    print(f"✅ 12-month projection: ${moderate_proj.final_balance:,.0f}")

    # Test milestone analysis
    milestones = calculator.get_milestone_analysis(500, GrowthScenario.AGGRESSIVE)

    assert len(milestones) > 0
    assert milestones[0].target == 1000  # First milestone

    print(f"✅ Time to $1K: {milestones[0].months} months")

    # Test time to target
    months_to_2k = calculator.calculate_time_to_target(500, 2000, 0.12)
    assert months_to_2k > 0

    print(f"✅ Time to $2K at 12%/month: {months_to_2k} months")


def test_integration():
    """Test system integration"""
    print("\n🧪 Testing System Integration...")

    if not SMALL_ACCOUNT_LOADED:
        raise ImportError("Small account modules not available")

    # Simulate complete workflow
    account_balance = 400

    # 1. Get account tier
    sizer = SmallAccountPositionSizer()
    tier = sizer.get_account_tier(account_balance)

    # 2. Get strategy recommendation
    strategy_type = SmallAccountStrategies.recommend_strategy(
        account_balance, "moderate"
    )
    strategy = SmallAccountStrategies.get_strategy(strategy_type)

    # 3. Calculate position size
    position = sizer.calculate_position_size(
        account_balance=account_balance,
        stock_price=150,
        stop_loss_price=145,
        confidence=0.8,
    )

    # 4. Get growth projection
    calculator = GrowthCalculator()
    projections = calculator.calculate_projections(
        account_balance, 6, [GrowthScenario.MODERATE]
    )
    projection = projections[GrowthScenario.MODERATE]

    print(f"✅ Complete workflow for ${account_balance} account:")
    print(f"   • Tier: {tier.value}")
    print(f"   • Strategy: {strategy.name}")
    print(f"   • Position: ${position.dollar_amount:.0f}")
    print(f"   • 6-month projection: ${projection.final_balance:,.0f}")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Small Account Trading System Tests\n")

    if not SMALL_ACCOUNT_LOADED:
        print("❌ Cannot run tests - small account modules not loaded")
        print(
            "Make sure the src/small_account directory exists with all required files:"
        )
        print("  • __init__.py")
        print("  • position_sizer.py")
        print("  • growth_strategies.py")
        print("  • growth_calculator.py")
        return False

    try:
        test_position_sizer()
        test_growth_strategies()
        test_growth_calculator()
        test_integration()

        print(f"\n🎉 All tests passed! Small Account Trading System is ready.")
        print(f"\n📋 Quick Summary:")
        print(f"   • Position Sizer: Optimizes sizing for $200-$2000 accounts")
        print(f"   • Growth Strategies: 4 specialized strategies available")
        print(f"   • Growth Calculator: Projects milestones and timeframes")
        print(f"   • Integration: Complete workflow from setup to projections")

        print(f"\n🚀 Next Steps:")
        print(f"   1. Run: streamlit run app.py")
        print(f"   2. Navigate to '💰 Small Account' in sidebar")
        print(f"   3. Set your starting capital and explore!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
