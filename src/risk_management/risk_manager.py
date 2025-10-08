"""Position sizing and risk management utilities"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Position sizing methods"""

    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_BASED = "volatility_based"
    RISK_PARITY = "risk_parity"


class RiskLevel(Enum):
    """Risk tolerance levels"""

    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3
    VERY_AGGRESSIVE = 4


@dataclass
class RiskParameters:
    """Risk management parameters"""

    max_portfolio_risk: float = 0.02  # 2% max portfolio risk per trade
    max_position_size: float = 0.1  # 10% max position size
    max_correlation: float = 0.7  # Max correlation between positions
    max_sector_exposure: float = 0.3  # Max 30% exposure to any sector
    stop_loss_multiplier: float = 2.0  # Stop loss as multiple of position size
    max_drawdown_limit: float = 0.15  # 15% max drawdown before reducing size
    volatility_lookback: int = 20  # Days for volatility calculation
    correlation_lookback: int = 60  # Days for correlation calculation


class PositionSizer:
    """Calculate position sizes based on risk parameters"""

    def __init__(self, risk_params: Optional[RiskParameters] = None):
        self.risk_params = risk_params or RiskParameters()
        self.logger = logging.getLogger(f"{__name__}.PositionSizer")

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        portfolio_value: float,
        volatility: float,
        method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_BASED,
        **kwargs,
    ) -> float:
        """Calculate appropriate position size"""

        if portfolio_value <= 0 or current_price <= 0:
            return 0.0

        try:
            if method == PositionSizingMethod.FIXED_PERCENTAGE:
                return self._fixed_percentage_size(
                    portfolio_value, current_price, **kwargs
                )

            elif method == PositionSizingMethod.FIXED_AMOUNT:
                return self._fixed_amount_size(current_price, **kwargs)

            elif method == PositionSizingMethod.VOLATILITY_BASED:
                return self._volatility_based_size(
                    portfolio_value, current_price, volatility, **kwargs
                )

            elif method == PositionSizingMethod.KELLY_CRITERION:
                return self._kelly_criterion_size(
                    portfolio_value, current_price, **kwargs
                )

            elif method == PositionSizingMethod.RISK_PARITY:
                return self._risk_parity_size(
                    portfolio_value, current_price, volatility, **kwargs
                )

            else:
                self.logger.warning(f"Unknown position sizing method: {method}")
                return self._fixed_percentage_size(portfolio_value, current_price)

        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def _fixed_percentage_size(
        self,
        portfolio_value: float,
        current_price: float,
        percentage: float = 0.1,
        **kwargs,
    ) -> float:
        """Calculate position size as fixed percentage of portfolio"""
        percentage = min(percentage, self.risk_params.max_position_size)
        position_value = portfolio_value * percentage
        return int(position_value / current_price)

    def _fixed_amount_size(
        self, current_price: float, amount: float = 10000, **kwargs
    ) -> float:
        """Calculate position size for fixed dollar amount"""
        return int(amount / current_price)

    def _volatility_based_size(
        self,
        portfolio_value: float,
        current_price: float,
        volatility: float,
        target_risk: Optional[float] = None,
        **kwargs,
    ) -> float:
        """Calculate position size based on volatility (risk parity approach)"""

        if volatility <= 0:
            volatility = 0.2  # Default 20% volatility

        target_risk = target_risk or self.risk_params.max_portfolio_risk

        # Position size = (Target Risk * Portfolio Value) / (Price * Volatility)
        position_value = (target_risk * portfolio_value) / volatility
        position_size = position_value / current_price

        # Apply maximum position size limit
        max_position_value = portfolio_value * self.risk_params.max_position_size
        max_position_size = max_position_value / current_price

        return int(min(position_size, max_position_size))

    def _kelly_criterion_size(
        self,
        portfolio_value: float,
        current_price: float,
        win_probability: float = 0.55,
        avg_win: float = 0.06,
        avg_loss: float = 0.03,
        **kwargs,
    ) -> float:
        """Calculate position size using Kelly Criterion"""

        # Kelly % = (bp - q) / b
        # where: b = odds (avg_win/avg_loss), p = win probability, q = loss probability

        if avg_loss <= 0:
            return 0.0

        b = avg_win / avg_loss  # Odds
        p = win_probability  # Win probability
        q = 1 - p  # Loss probability

        kelly_pct = (b * p - q) / b

        # Apply safety factor (typically use 25-50% of Kelly)
        safety_factor = 0.25
        kelly_pct *= safety_factor

        # Ensure within risk limits
        kelly_pct = max(0, min(kelly_pct, self.risk_params.max_position_size))

        position_value = portfolio_value * kelly_pct
        return int(position_value / current_price)

    def _risk_parity_size(
        self,
        portfolio_value: float,
        current_price: float,
        volatility: float,
        target_volatility: float = 0.15,
        **kwargs,
    ) -> float:
        """Calculate position size for risk parity"""

        if volatility <= 0:
            return 0.0

        # Scale position size inversely to volatility
        volatility_ratio = target_volatility / volatility
        base_percentage = self.risk_params.max_position_size

        adjusted_percentage = base_percentage * volatility_ratio
        adjusted_percentage = min(
            adjusted_percentage, self.risk_params.max_position_size
        )

        position_value = portfolio_value * adjusted_percentage
        return int(position_value / current_price)


class RiskManager:
    """Comprehensive risk management system"""

    def __init__(self, risk_params: Optional[RiskParameters] = None):
        self.risk_params = risk_params or RiskParameters()
        self.position_sizer = PositionSizer(self.risk_params)
        self.logger = logging.getLogger(f"{__name__}.RiskManager")

        # Track portfolio state
        self.current_positions = {}
        self.position_correlations = {}
        self.sector_exposures = {}
        self.portfolio_metrics = {}

    def assess_trade_risk(
        self,
        symbol: str,
        proposed_position: float,
        current_price: float,
        portfolio_value: float,
        existing_positions: Optional[Dict[str, float]] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Assess risk of a proposed trade"""

        existing_positions = existing_positions or {}

        risk_assessment = {
            "symbol": symbol,
            "proposed_position": proposed_position,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "risk_checks": {},
            "warnings": [],
            "recommendation": "PROCEED",
            "adjusted_position": proposed_position,
        }

        try:
            # 1. Position size check
            position_value = proposed_position * current_price
            position_pct = position_value / portfolio_value

            risk_assessment["risk_checks"]["position_size"] = {
                "current_pct": position_pct,
                "max_allowed": self.risk_params.max_position_size,
                "passed": position_pct <= self.risk_params.max_position_size,
            }

            if position_pct > self.risk_params.max_position_size:
                risk_assessment["warnings"].append(
                    f"Position size {position_pct:.1%} exceeds limit {self.risk_params.max_position_size:.1%}"
                )
                # Adjust position size
                max_position_value = (
                    portfolio_value * self.risk_params.max_position_size
                )
                risk_assessment["adjusted_position"] = int(
                    max_position_value / current_price
                )

            # 2. Portfolio concentration check
            total_positions_value = (
                sum(pos * current_price for pos in existing_positions.values())
                + position_value
            )

            concentration_ratio = total_positions_value / portfolio_value

            risk_assessment["risk_checks"]["concentration"] = {
                "total_equity_exposure": concentration_ratio,
                "max_recommended": 0.8,  # 80% max equity exposure
                "passed": concentration_ratio <= 0.8,
            }

            if concentration_ratio > 0.8:
                risk_assessment["warnings"].append(
                    f"Total equity exposure {concentration_ratio:.1%} is high"
                )

            # 3. Correlation check (if market data available)
            if market_data and len(existing_positions) > 0:
                correlation_risk = self._assess_correlation_risk(
                    symbol, list(existing_positions.keys()), market_data
                )
                risk_assessment["risk_checks"]["correlation"] = correlation_risk

            # 4. Volatility assessment
            if market_data and symbol in market_data:
                volatility_risk = self._assess_volatility_risk(
                    symbol, proposed_position, current_price, market_data[symbol]
                )
                risk_assessment["risk_checks"]["volatility"] = volatility_risk

            # 5. Overall risk recommendation
            failed_checks = sum(
                1
                for check in risk_assessment["risk_checks"].values()
                if not check.get("passed", True)
            )

            if failed_checks >= 2:
                risk_assessment["recommendation"] = "REJECT"
            elif failed_checks == 1 or len(risk_assessment["warnings"]) > 0:
                risk_assessment["recommendation"] = "CAUTION"

        except Exception as e:
            self.logger.error(f"Error assessing trade risk: {e}")
            risk_assessment["recommendation"] = "ERROR"
            risk_assessment["warnings"].append(f"Risk assessment error: {e}")

        return risk_assessment

    def _assess_correlation_risk(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Assess correlation risk with existing positions"""

        correlation_assessment = {
            "max_correlation": 0.0,
            "correlated_symbols": [],
            "passed": True,
        }

        try:
            if new_symbol not in market_data:
                return correlation_assessment

            new_returns = market_data[new_symbol]["Close"].pct_change().dropna()

            for existing_symbol in existing_symbols:
                if existing_symbol in market_data:
                    existing_returns = (
                        market_data[existing_symbol]["Close"].pct_change().dropna()
                    )

                    # Align data
                    common_dates = new_returns.index.intersection(
                        existing_returns.index
                    )
                    if len(common_dates) < self.risk_params.correlation_lookback:
                        continue

                    # Use recent data for correlation
                    recent_dates = common_dates[
                        -self.risk_params.correlation_lookback :
                    ]
                    correlation = new_returns[recent_dates].corr(
                        existing_returns[recent_dates]
                    )

                    if abs(correlation) > abs(
                        correlation_assessment["max_correlation"]
                    ):
                        correlation_assessment["max_correlation"] = correlation

                    if abs(correlation) > self.risk_params.max_correlation:
                        correlation_assessment["correlated_symbols"].append(
                            {"symbol": existing_symbol, "correlation": correlation}
                        )

            correlation_assessment["passed"] = (
                len(correlation_assessment["correlated_symbols"]) == 0
            )

        except Exception as e:
            self.logger.warning(f"Error calculating correlation risk: {e}")

        return correlation_assessment

    def _assess_volatility_risk(
        self,
        symbol: str,
        position_size: float,
        current_price: float,
        market_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Assess volatility risk of position"""

        volatility_assessment = {
            "annualized_volatility": 0.0,
            "position_risk": 0.0,
            "risk_level": "LOW",
            "passed": True,
        }

        try:
            returns = market_data["Close"].pct_change().dropna()

            if len(returns) < self.risk_params.volatility_lookback:
                return volatility_assessment

            # Calculate recent volatility
            recent_returns = returns[-self.risk_params.volatility_lookback :]
            daily_vol = recent_returns.std()
            annualized_vol = daily_vol * np.sqrt(252)

            # Calculate position risk (1-day VaR at 95% confidence)
            position_value = position_size * current_price
            daily_var = position_value * daily_vol * 1.65  # 95% VaR

            volatility_assessment["annualized_volatility"] = annualized_vol
            volatility_assessment["position_risk"] = daily_var

            # Classify risk level
            if annualized_vol > 0.5:  # >50%
                volatility_assessment["risk_level"] = "VERY_HIGH"
                volatility_assessment["passed"] = False
            elif annualized_vol > 0.3:  # >30%
                volatility_assessment["risk_level"] = "HIGH"
            elif annualized_vol > 0.2:  # >20%
                volatility_assessment["risk_level"] = "MODERATE"
            else:
                volatility_assessment["risk_level"] = "LOW"

        except Exception as e:
            self.logger.warning(f"Error calculating volatility risk: {e}")

        return volatility_assessment

    def calculate_stop_loss(
        self,
        entry_price: float,
        volatility: Optional[float] = None,
        method: str = "percentage",
    ) -> float:
        """Calculate appropriate stop-loss level"""

        if method == "percentage":
            # Simple percentage-based stop loss
            stop_distance = entry_price * 0.02  # 2% default
            return entry_price - stop_distance

        elif method == "volatility" and volatility:
            # Volatility-based stop loss (2x daily volatility)
            stop_distance = entry_price * volatility * 2
            return entry_price - stop_distance

        elif method == "atr":
            # Would use Average True Range if available
            # For now, fall back to percentage
            stop_distance = entry_price * 0.025  # 2.5%
            return entry_price - stop_distance

        else:
            # Default fallback
            stop_distance = entry_price * 0.02
            return entry_price - stop_distance

    def calculate_take_profit(
        self, entry_price: float, stop_loss: float, risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take-profit level based on risk-reward ratio"""

        risk_amount = entry_price - stop_loss
        reward_amount = risk_amount * risk_reward_ratio

        return entry_price + reward_amount

    def get_portfolio_risk_summary(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
    ) -> Dict[str, Any]:
        """Get comprehensive portfolio risk summary"""

        summary = {
            "total_positions": len(positions),
            "total_equity_exposure": 0.0,
            "position_breakdown": [],
            "risk_metrics": {},
            "warnings": [],
        }

        try:
            total_position_value = 0

            for symbol, quantity in positions.items():
                if symbol in current_prices and quantity > 0:
                    position_value = quantity * current_prices[symbol]
                    position_pct = position_value / portfolio_value

                    total_position_value += position_value

                    summary["position_breakdown"].append(
                        {
                            "symbol": symbol,
                            "quantity": quantity,
                            "current_price": current_prices[symbol],
                            "position_value": position_value,
                            "portfolio_pct": position_pct,
                        }
                    )

                    # Check individual position limits
                    if position_pct > self.risk_params.max_position_size:
                        summary["warnings"].append(
                            f"{symbol} position {position_pct:.1%} exceeds limit"
                        )

            summary["total_equity_exposure"] = total_position_value / portfolio_value

            # Overall risk metrics
            summary["risk_metrics"] = {
                "equity_exposure": summary["total_equity_exposure"],
                "cash_allocation": 1 - summary["total_equity_exposure"],
                "diversification_score": self._calculate_diversification_score(
                    summary["position_breakdown"]
                ),
                "risk_level": self._assess_overall_risk_level(summary),
            }

        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk summary: {e}")
            summary["warnings"].append(f"Risk calculation error: {e}")

        return summary

    def _calculate_diversification_score(self, positions: List[Dict]) -> float:
        """Calculate portfolio diversification score (0-1, higher is better)"""

        if not positions:
            return 1.0

        # Simple diversification based on position concentration
        position_weights = [pos["portfolio_pct"] for pos in positions]

        # Herfindahl-Hirschman Index (HHI) - lower is more diversified
        hhi = sum(weight**2 for weight in position_weights)

        # Convert to diversification score (invert and normalize)
        max_hhi = 1.0  # All money in one position
        diversification_score = 1 - (hhi / max_hhi)

        return diversification_score

    def _assess_overall_risk_level(self, summary: Dict) -> str:
        """Assess overall portfolio risk level"""

        equity_exposure = summary["total_equity_exposure"]
        num_warnings = len(summary["warnings"])
        num_positions = summary["total_positions"]

        if equity_exposure > 0.9 or num_warnings >= 3:
            return "HIGH"
        elif equity_exposure > 0.7 or num_warnings >= 2 or num_positions < 3:
            return "MODERATE"
        else:
            return "LOW"
