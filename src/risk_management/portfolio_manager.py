"""Portfolio management system integrating ML signals with risk management"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..data_sources import DataManager
from ..signals import SignalManager
from .risk_manager import PositionSizingMethod, RiskManager, RiskParameters

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a portfolio position"""

    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Represents a completed trade"""

    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    trade_type: str = "MARKET"  # 'MARKET', 'LIMIT', 'STOP'


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""

    timestamp: datetime
    cash_balance: float
    positions: List[Position]
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_return: float = 0.0
    total_return: float = 0.0


class PortfolioManager:
    """Manages portfolio positions and integrates signals with risk management"""

    def __init__(
        self,
        initial_cash: float = 100000.0,
        risk_params: Optional[RiskParameters] = None,
        data_manager: Optional[DataManager] = None,
        signal_manager: Optional[SignalManager] = None,
    ):
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions = {}  # symbol -> Position
        self.trade_history = []
        self.portfolio_history = []

        # Initialize managers
        self.risk_manager = RiskManager(risk_params)
        self.data_manager = data_manager
        self.signal_manager = signal_manager

        self.logger = logging.getLogger(f"{__name__}.PortfolioManager")

        # Performance tracking
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_fees": 0.0,
            "max_drawdown": 0.0,
            "peak_value": initial_cash,
        }

    def get_current_value(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate current portfolio value"""

        if current_prices is None and self.data_manager:
            # Get current prices from data manager
            current_prices = {}
            for symbol in self.positions.keys():
                try:
                    data = self.data_manager.get_market_data(
                        symbol, period="1d", limit=1
                    )
                    if not data.empty:
                        current_prices[symbol] = data["Close"].iloc[-1]
                except Exception as e:
                    self.logger.warning(
                        f"Could not get current price for {symbol}: {e}"
                    )
                    continue

        total_value = self.cash_balance

        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                position.current_price = current_prices[symbol]
                position_value = position.quantity * position.current_price
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.quantity
                total_value += position_value

        return total_value

    def process_signals(
        self, symbols: List[str], lookback_period: int = 30
    ) -> Dict[str, Any]:
        """Process trading signals and generate trade recommendations"""

        if not self.signal_manager or not self.data_manager:
            self.logger.warning("Signal manager or data manager not configured")
            return {}

        recommendations = {}
        current_prices = {}

        try:
            # Get market data and current prices
            market_data = {}
            for symbol in symbols:
                try:
                    data = self.data_manager.get_market_data(
                        symbol,
                        period="1y",
                        limit=252,  # 1 year of data
                    )
                    if not data.empty:
                        market_data[symbol] = data
                        current_prices[symbol] = data["Close"].iloc[-1]
                except Exception as e:
                    self.logger.warning(f"Could not get data for {symbol}: {e}")
                    continue

            # Generate signals for each symbol
            for symbol in symbols:
                if symbol not in market_data:
                    continue

                try:
                    # Get signals
                    signals = self.signal_manager.generate_signals(
                        symbol, market_data[symbol]
                    )

                    if not signals:
                        continue

                    latest_signal = signals[-1]
                    signal_strength = abs(latest_signal["strength"])
                    signal_direction = (
                        "BUY" if latest_signal["strength"] > 0 else "SELL"
                    )

                    # Skip weak signals
                    if signal_strength < 0.3:  # Minimum signal strength
                        continue

                    # Calculate position size
                    portfolio_value = self.get_current_value(current_prices)

                    if signal_direction == "BUY" and self.cash_balance > 1000:
                        # Calculate volatility for position sizing
                        returns = market_data[symbol]["Close"].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(252)  # Annualized

                        # Calculate suggested position size
                        suggested_position = (
                            self.risk_manager.position_sizer.calculate_position_size(
                                symbol=symbol,
                                current_price=current_prices[symbol],
                                portfolio_value=portfolio_value,
                                volatility=volatility,
                                method=PositionSizingMethod.VOLATILITY_BASED,
                            )
                        )

                        # Assess trade risk
                        existing_positions = {
                            s: p.quantity for s, p in self.positions.items()
                        }
                        risk_assessment = self.risk_manager.assess_trade_risk(
                            symbol=symbol,
                            proposed_position=suggested_position,
                            current_price=current_prices[symbol],
                            portfolio_value=portfolio_value,
                            existing_positions=existing_positions,
                            market_data=market_data,
                        )

                        # Calculate stop loss and take profit
                        stop_loss = self.risk_manager.calculate_stop_loss(
                            entry_price=current_prices[symbol],
                            volatility=volatility / np.sqrt(252),  # Daily volatility
                            method="volatility",
                        )

                        take_profit = self.risk_manager.calculate_take_profit(
                            entry_price=current_prices[symbol],
                            stop_loss=stop_loss,
                            risk_reward_ratio=2.0,
                        )

                        recommendations[symbol] = {
                            "action": signal_direction,
                            "signal_strength": signal_strength,
                            "current_price": current_prices[symbol],
                            "suggested_quantity": risk_assessment["adjusted_position"],
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "risk_assessment": risk_assessment,
                            "signal_details": latest_signal,
                        }

                    elif signal_direction == "SELL" and symbol in self.positions:
                        # Sell signal for existing position
                        current_position = self.positions[symbol]

                        recommendations[symbol] = {
                            "action": "SELL",
                            "signal_strength": signal_strength,
                            "current_price": current_prices[symbol],
                            "suggested_quantity": current_position.quantity,
                            "current_position": current_position,
                            "unrealized_pnl": (
                                current_prices[symbol] - current_position.entry_price
                            )
                            * current_position.quantity,
                            "signal_details": latest_signal,
                        }

                except Exception as e:
                    self.logger.error(f"Error processing signals for {symbol}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error in process_signals: {e}")

        return recommendations

    def execute_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Execute a trade (buy or sell)"""

        result = {"success": False, "message": "", "trade": None, "new_position": None}

        try:
            # Get current price if not provided
            if price <= 0.0:
                if self.data_manager:
                    data = self.data_manager.get_market_data(
                        symbol, period="1d", limit=1
                    )
                    if not data.empty:
                        price = data["Close"].iloc[-1]
                    else:
                        result["message"] = f"Could not get current price for {symbol}"
                        return result
                else:
                    result["message"] = "Price required when data manager not available"
                    return result

            trade_value = quantity * price
            fees = trade_value * 0.001  # 0.1% fee assumption

            if action.upper() == "BUY":
                # Check if we have enough cash
                total_cost = trade_value + fees
                if total_cost > self.cash_balance:
                    result["message"] = (
                        f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash_balance:.2f}"
                    )
                    return result

                if not dry_run:
                    # Execute buy order
                    self.cash_balance -= total_cost

                    # Update or create position
                    if symbol in self.positions:
                        # Average into existing position
                        existing_pos = self.positions[symbol]
                        total_quantity = existing_pos.quantity + quantity
                        avg_price = (
                            (existing_pos.quantity * existing_pos.entry_price)
                            + (quantity * price)
                        ) / total_quantity

                        existing_pos.quantity = total_quantity
                        existing_pos.entry_price = avg_price

                        if stop_loss:
                            existing_pos.stop_loss = stop_loss
                        if take_profit:
                            existing_pos.take_profit = take_profit

                        result["new_position"] = existing_pos
                    else:
                        # Create new position
                        new_position = Position(
                            symbol=symbol,
                            quantity=quantity,
                            entry_price=price,
                            entry_date=datetime.now(),
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            current_price=price,
                        )
                        self.positions[symbol] = new_position
                        result["new_position"] = new_position

                    # Record trade
                    trade = Trade(
                        symbol=symbol,
                        side="BUY",
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.now(),
                        fees=fees,
                    )
                    self.trade_history.append(trade)
                    result["trade"] = trade

                    self.metrics["total_trades"] += 1
                    self.metrics["total_fees"] += fees

                result["success"] = True
                result["message"] = (
                    f"{'[DRY RUN] ' if dry_run else ''}Bought {quantity} shares of {symbol} at ${price:.2f}"
                )

            elif action.upper() == "SELL":
                # Check if we have the position
                if symbol not in self.positions:
                    result["message"] = f"No position in {symbol} to sell"
                    return result

                position = self.positions[symbol]
                if quantity > position.quantity:
                    result["message"] = (
                        f"Cannot sell {quantity} shares, only have {position.quantity}"
                    )
                    return result

                if not dry_run:
                    # Execute sell order
                    proceeds = trade_value - fees
                    self.cash_balance += proceeds

                    # Calculate realized P&L
                    realized_pnl = (price - position.entry_price) * quantity - fees
                    position.realized_pnl += realized_pnl

                    # Update position
                    position.quantity -= quantity

                    if position.quantity <= 0:
                        # Close position completely
                        del self.positions[symbol]

                    # Record trade
                    trade = Trade(
                        symbol=symbol,
                        side="SELL",
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.now(),
                        fees=fees,
                    )
                    self.trade_history.append(trade)
                    result["trade"] = trade

                    self.metrics["total_trades"] += 1
                    self.metrics["total_fees"] += fees

                    # Update win/loss stats
                    if realized_pnl > 0:
                        self.metrics["winning_trades"] += 1
                    else:
                        self.metrics["losing_trades"] += 1

                result["success"] = True
                result["message"] = (
                    f"{'[DRY RUN] ' if dry_run else ''}Sold {quantity} shares of {symbol} at ${price:.2f}"
                )

            else:
                result["message"] = f"Invalid action: {action}. Use 'BUY' or 'SELL'"

        except Exception as e:
            result["message"] = f"Error executing trade: {e}"
            self.logger.error(f"Trade execution error: {e}")

        return result

    def check_stop_losses_and_take_profits(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """Check all positions for stop-loss and take-profit triggers"""

        triggered_orders = []

        if current_prices is None and self.data_manager:
            current_prices = {}
            for symbol in self.positions.keys():
                try:
                    data = self.data_manager.get_market_data(
                        symbol, period="1d", limit=1
                    )
                    if not data.empty:
                        current_prices[symbol] = data["Close"].iloc[-1]
                except Exception as e:
                    self.logger.warning(
                        f"Could not get current price for {symbol}: {e}"
                    )

        for symbol, position in list(self.positions.items()):
            if not current_prices or symbol not in current_prices:
                continue

            current_price = current_prices[symbol]

            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                triggered_orders.append(
                    {
                        "symbol": symbol,
                        "trigger_type": "STOP_LOSS",
                        "trigger_price": position.stop_loss,
                        "current_price": current_price,
                        "quantity": position.quantity,
                        "recommended_action": "SELL",
                    }
                )

            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                triggered_orders.append(
                    {
                        "symbol": symbol,
                        "trigger_type": "TAKE_PROFIT",
                        "trigger_price": position.take_profit,
                        "current_price": current_price,
                        "quantity": position.quantity,
                        "recommended_action": "SELL",
                    }
                )

        return triggered_orders

    def get_portfolio_summary(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""

        current_value = self.get_current_value(current_prices)
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        # Calculate returns
        total_return = (current_value - self.initial_cash) / self.initial_cash

        # Update drawdown tracking
        if current_value > self.metrics["peak_value"]:
            self.metrics["peak_value"] = current_value

        current_drawdown = (self.metrics["peak_value"] - current_value) / self.metrics[
            "peak_value"
        ]
        if current_drawdown > self.metrics["max_drawdown"]:
            self.metrics["max_drawdown"] = current_drawdown

        # Win rate calculation
        win_rate = 0.0
        completed_trades = (
            self.metrics["winning_trades"] + self.metrics["losing_trades"]
        )
        if completed_trades > 0:
            win_rate = self.metrics["winning_trades"] / completed_trades

        return {
            "timestamp": datetime.now(),
            "cash_balance": self.cash_balance,
            "positions_value": current_value - self.cash_balance,
            "total_value": current_value,
            "initial_value": self.initial_cash,
            "total_return": total_return,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_pnl": total_unrealized_pnl + total_realized_pnl,
            "current_positions": len(self.positions),
            "max_drawdown": self.metrics["max_drawdown"],
            "current_drawdown": current_drawdown,
            "total_trades": self.metrics["total_trades"],
            "win_rate": win_rate,
            "total_fees": self.metrics["total_fees"],
            "positions": [
                {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "position_value": pos.quantity * pos.current_price,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                }
                for pos in self.positions.values()
            ],
        }

    def save_portfolio_snapshot(
        self, current_prices: Optional[Dict[str, float]] = None
    ):
        """Save current portfolio state to history"""

        current_value = self.get_current_value(current_prices)
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        # Calculate daily return
        daily_return = 0.0
        if self.portfolio_history:
            previous_value = self.portfolio_history[-1].total_value
            daily_return = (current_value - previous_value) / previous_value

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            cash_balance=self.cash_balance,
            positions=list(self.positions.values()),
            total_value=current_value,
            unrealized_pnl=total_unrealized_pnl,
            realized_pnl=total_realized_pnl,
            daily_return=daily_return,
            total_return=(current_value - self.initial_cash) / self.initial_cash,
        )

        self.portfolio_history.append(snapshot)

        # Keep only last 252 snapshots (1 year of daily data)
        if len(self.portfolio_history) > 252:
            self.portfolio_history.pop(0)
