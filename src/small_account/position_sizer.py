"""
Position Sizer for Small Accounts

Optimized position sizing for accounts under $1,000 with fractional shares support.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AccountTier(Enum):
    """Account size tiers with different risk parameters"""

    MICRO = "micro"  # $50-$200
    SMALL = "small"  # $200-$500
    MEDIUM = "medium"  # $500-$2000
    LARGE = "large"  # $2000+


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation"""

    dollar_amount: float
    shares: float
    position_pct: float
    risk_amount: float
    tier: AccountTier
    can_afford: bool
    reason: str


class SmallAccountPositionSizer:
    """Position sizer optimized for small trading accounts"""

    # Account tier configurations
    TIER_CONFIG = {
        AccountTier.MICRO: {
            "max_position_pct": 0.40,
            "risk_per_trade": 0.03,
            "min_trade_size": 25,
            "confidence_multiplier": 1.5,
        },
        AccountTier.SMALL: {
            "max_position_pct": 0.25,
            "risk_per_trade": 0.02,
            "min_trade_size": 50,
            "confidence_multiplier": 1.2,
        },
        AccountTier.MEDIUM: {
            "max_position_pct": 0.15,
            "risk_per_trade": 0.015,
            "min_trade_size": 100,
            "confidence_multiplier": 1.0,
        },
        AccountTier.LARGE: {
            "max_position_pct": 0.10,
            "risk_per_trade": 0.01,
            "min_trade_size": 200,
            "confidence_multiplier": 0.8,
        },
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_account_tier(self, account_balance: float) -> AccountTier:
        """Determine account tier based on balance"""
        if account_balance < 200:
            return AccountTier.MICRO
        elif account_balance <= 500:
            return AccountTier.SMALL
        elif account_balance <= 2000:
            return AccountTier.MEDIUM
        else:
            return AccountTier.LARGE

    def calculate_position_size(
        self,
        account_balance: float,
        stock_price: float,
        stop_loss_price: float,
        confidence: float = 0.7,
        strategy_type: str = "swing",
    ) -> PositionSizeResult:
        """Calculate optimal position size for small accounts"""

        tier = self.get_account_tier(account_balance)
        config = self.TIER_CONFIG[tier]

        # Calculate risk per share
        risk_per_share = abs(stock_price - stop_loss_price)

        if risk_per_share <= 0:
            return PositionSizeResult(
                dollar_amount=0,
                shares=0,
                position_pct=0,
                risk_amount=0,
                tier=tier,
                can_afford=False,
                reason="Invalid stop loss price",
            )

        # Base risk amount
        base_risk = account_balance * config["risk_per_trade"]

        # Adjust for confidence
        confidence_adj = confidence * config["confidence_multiplier"]
        adjusted_risk = base_risk * min(confidence_adj, 1.5)

        # Calculate shares based on risk
        shares_by_risk = adjusted_risk / risk_per_share

        # Calculate dollar amount
        dollar_amount_by_risk = shares_by_risk * stock_price

        # Apply position size limits
        max_dollar_amount = account_balance * config["max_position_pct"]

        # Use the smaller of risk-based or position-limit based sizing
        final_dollar_amount = min(dollar_amount_by_risk, max_dollar_amount)
        final_shares = final_dollar_amount / stock_price

        # Check minimums
        if final_dollar_amount < config["min_trade_size"]:
            return PositionSizeResult(
                dollar_amount=0,
                shares=0,
                position_pct=0,
                risk_amount=0,
                tier=tier,
                can_afford=False,
                reason=f"Trade size ${final_dollar_amount:.0f} below minimum ${config['min_trade_size']}",
            )

        # Check if we can afford it
        if final_dollar_amount > account_balance * 0.95:
            return PositionSizeResult(
                dollar_amount=0,
                shares=0,
                position_pct=0,
                risk_amount=0,
                tier=tier,
                can_afford=False,
                reason="Insufficient funds (need 5% cash buffer)",
            )

        position_pct = final_dollar_amount / account_balance
        actual_risk = final_shares * risk_per_share

        return PositionSizeResult(
            dollar_amount=final_dollar_amount,
            shares=final_shares,
            position_pct=position_pct,
            risk_amount=actual_risk,
            tier=tier,
            can_afford=True,
            reason=f"Optimal size for {tier.value} account",
        )

    def get_recommended_symbols(self, account_balance: float) -> Dict[str, list]:
        """Get stock recommendations based on account size"""

        tier = self.get_account_tier(account_balance)

        recommendations = {
            AccountTier.MICRO: {
                "affordable": ["AMD", "F", "BAC", "SNAP", "NOK", "SIRI"],
                "fractional": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "avoid": ["BRK.A", "NVR"],
            },
            AccountTier.SMALL: {
                "affordable": ["AAPL", "MSFT", "AMD", "NVDA", "META", "NFLX"],
                "fractional": ["GOOGL", "AMZN", "TSLA"],
                "avoid": ["BRK.A", "NVR"],
            },
            AccountTier.MEDIUM: {
                "affordable": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
                "fractional": ["AMZN", "TSLA"],
                "avoid": ["BRK.A"],
            },
            AccountTier.LARGE: {
                "affordable": ["ALL_SYMBOLS"],
                "fractional": [],
                "avoid": [],
            },
        }

        return recommendations.get(tier, recommendations[AccountTier.SMALL])
