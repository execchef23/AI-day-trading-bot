"""
Portfolio Manager - Manages portfolio risk and positions
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Portfolio position data"""

    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    entry_time: str
    stop_loss: float
    take_profit: float


class PortfolioManager:
    """Manages portfolio positions and risk"""

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history = []
        logger.info(f"Portfolio Manager initialized with ${initial_capital:,.2f}")

    def add_position(self, symbol: str, quantity: int, price: float) -> bool:
        """Add a new position"""
        try:
            cost = quantity * price
            if cost > self.available_capital:
                logger.warning(f"Insufficient capital for {symbol}")
                return False

            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                pnl=0.0,
                pnl_pct=0.0,
                entry_time=datetime.now().isoformat(),
                stop_loss=price * 0.95,  # 5% stop loss
                take_profit=price * 1.15,  # 15% take profit
            )

            self.positions[symbol] = position
            self.available_capital -= cost
            logger.info(f"Added position: {quantity} shares of {symbol} @ ${price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False

    def remove_position(self, symbol: str) -> bool:
        """Remove a position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            proceeds = position.quantity * position.current_price
            self.available_capital += proceeds
            del self.positions[symbol]
            logger.info(f"Closed position: {symbol}")
            return True
        return False

    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for positions"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                position.current_price = price
                position.pnl = (price - position.entry_price) * position.quantity
                position.pnl_pct = ((price / position.entry_price) - 1) * 100

    def get_portfolio_value(self) -> float:
        """Get total portfolio value"""
        positions_value = sum(
            pos.current_price * pos.quantity for pos in self.positions.values()
        )
        return self.available_capital + positions_value

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        return [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "entry_time": pos.entry_time,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }
            for pos in self.positions.values()
        ]

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_value = self.get_portfolio_value()
        total_pnl = sum(pos.pnl for pos in self.positions.values())

        return {
            "total_value": total_value,
            "available_capital": self.available_capital,
            "positions_count": len(self.positions),
            "total_pnl": total_pnl,
            "total_return_pct": ((total_value / self.initial_capital) - 1) * 100,
        }
