"""
Growth Strategies for Small Accounts

Four specialized strategies designed for aggressive but safe account growth.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class StrategyType(Enum):
    """Strategy types optimized for small accounts"""

    MOMENTUM_SCALP = "momentum_scalp"
    SWING_TRADING = "swing_trading"
    BREAKOUT_TRADING = "breakout_trading"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class GrowthStrategy:
    """Strategy configuration and parameters"""

    name: str
    strategy_type: StrategyType
    target_return: float  # Target % return per trade
    hold_period: Tuple[int, int]  # Min, max days to hold
    win_rate: float  # Expected win rate
    risk_reward_ratio: float  # Risk:reward ratio
    max_positions: int  # Max simultaneous positions
    ideal_account_size: Tuple[int, int]  # Min, max account size
    description: str
    entry_criteria: List[str]
    exit_criteria: List[str]


class SmallAccountStrategies:
    """Collection of strategies optimized for small accounts"""

    STRATEGIES = {
        StrategyType.MOMENTUM_SCALP: GrowthStrategy(
            name="Momentum Scalping",
            strategy_type=StrategyType.MOMENTUM_SCALP,
            target_return=0.03,  # 3% target
            hold_period=(1, 3),  # 1-3 days
            win_rate=0.65,  # 65% win rate
            risk_reward_ratio=3.0,  # 3:1 risk/reward
            max_positions=2,
            ideal_account_size=(200, 800),
            description="Quick momentum plays with tight stops",
            entry_criteria=[
                "Strong volume surge (2x+ average)",
                "Price above key moving averages",
                "RSI between 60-80 (momentum zone)",
                "Breaking resistance with volume",
            ],
            exit_criteria=[
                "3% profit target hit",
                "1% stop loss triggered",
                "End of day if no momentum",
                "Volume dries up significantly",
            ],
        ),
        StrategyType.SWING_TRADING: GrowthStrategy(
            name="Swing Trading",
            strategy_type=StrategyType.SWING_TRADING,
            target_return=0.08,  # 8% target
            hold_period=(3, 10),  # 3-10 days
            win_rate=0.58,  # 58% win rate
            risk_reward_ratio=4.0,  # 4:1 risk/reward
            max_positions=3,
            ideal_account_size=(300, 1500),
            description="Medium-term swings capturing trend moves",
            entry_criteria=[
                "Pullback to key support levels",
                "Bullish divergence on RSI",
                "Volume confirmation on bounce",
                "Above 20-day moving average",
            ],
            exit_criteria=[
                "8% profit target reached",
                "2% stop loss hit",
                "Bearish reversal pattern",
                "10 days maximum hold",
            ],
        ),
        StrategyType.BREAKOUT_TRADING: GrowthStrategy(
            name="Breakout Trading",
            strategy_type=StrategyType.BREAKOUT_TRADING,
            target_return=0.12,  # 12% target
            hold_period=(2, 8),  # 2-8 days
            win_rate=0.52,  # 52% win rate
            risk_reward_ratio=4.5,  # 4.5:1 risk/reward
            max_positions=2,
            ideal_account_size=(400, 2000),
            description="High-reward breakout and gap plays",
            entry_criteria=[
                "Break above key resistance",
                "Gap up with high volume",
                "Earnings momentum plays",
                "News catalyst confirmation",
            ],
            exit_criteria=[
                "12% profit target hit",
                "2.5% stop loss triggered",
                "Failed breakout (back below resistance)",
                "Momentum exhaustion signals",
            ],
        ),
        StrategyType.MEAN_REVERSION: GrowthStrategy(
            name="Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            target_return=0.06,  # 6% target
            hold_period=(2, 7),  # 2-7 days
            win_rate=0.62,  # 62% win rate
            risk_reward_ratio=3.5,  # 3.5:1 risk/reward
            max_positions=3,
            ideal_account_size=(250, 1000),
            description="Oversold bounces in strong stocks",
            entry_criteria=[
                "RSI below 30 (oversold)",
                "Stock down 5%+ in 2 days",
                "Still above major support",
                "High-quality stock (low debt)",
            ],
            exit_criteria=[
                "6% profit target reached",
                "1.5% stop loss hit",
                "RSI back above 70",
                "7 days maximum hold",
            ],
        ),
    }

    @classmethod
    def get_strategy(cls, strategy_type: StrategyType) -> GrowthStrategy:
        """Get strategy by type"""
        return cls.STRATEGIES[strategy_type]

    @classmethod
    def get_all_strategies(cls) -> Dict[StrategyType, GrowthStrategy]:
        """Get all available strategies"""
        return cls.STRATEGIES.copy()

    @classmethod
    def recommend_strategy(
        cls, account_balance: float, risk_tolerance: str
    ) -> StrategyType:
        """Recommend best strategy based on account size and risk tolerance"""

        if risk_tolerance.lower() == "conservative":
            if account_balance < 400:
                return StrategyType.MEAN_REVERSION
            else:
                return StrategyType.SWING_TRADING

        elif risk_tolerance.lower() == "moderate":
            if account_balance < 300:
                return StrategyType.MOMENTUM_SCALP
            else:
                return StrategyType.SWING_TRADING

        elif risk_tolerance.lower() == "aggressive":
            if account_balance < 500:
                return StrategyType.MOMENTUM_SCALP
            else:
                return StrategyType.BREAKOUT_TRADING

        else:  # Default to swing trading
            return StrategyType.SWING_TRADING

    @classmethod
    def calculate_strategy_metrics(
        cls, strategy_type: StrategyType, trades_per_month: int
    ) -> Dict:
        """Calculate expected monthly metrics for a strategy"""

        strategy = cls.STRATEGIES[strategy_type]

        # Calculate expected returns
        avg_win = strategy.target_return
        avg_loss = -strategy.target_return / strategy.risk_reward_ratio

        expected_return_per_trade = (strategy.win_rate * avg_win) + (
            (1 - strategy.win_rate) * avg_loss
        )

        monthly_return = expected_return_per_trade * trades_per_month

        return {
            "expected_return_per_trade": expected_return_per_trade,
            "monthly_return_rate": monthly_return,
            "win_rate": strategy.win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades_per_month": trades_per_month,
            "risk_reward_ratio": strategy.risk_reward_ratio,
        }
        return {
            "expected_return_per_trade": expected_return_per_trade,
            "monthly_return_rate": monthly_return,
            "win_rate": strategy.win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades_per_month": trades_per_month,
            "risk_reward_ratio": strategy.risk_reward_ratio,
        }
