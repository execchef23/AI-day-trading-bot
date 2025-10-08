"""Base backtesting framework for strategy validation"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0.0
    trade_type: str = "BUY"  # BUY or SELL
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    commission: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_time and self.entry_time:
            return self.exit_time - self.entry_time
        return None
    
    @property
    def pnl(self) -> float:
        """Calculate profit/loss for the trade"""
        if not self.exit_price:
            return 0.0
        
        if self.trade_type == "BUY":
            return (self.exit_price - self.entry_price) * self.quantity - self.commission
        else:  # SELL (short)
            return (self.entry_price - self.exit_price) * self.quantity - self.commission
    
    @property
    def pnl_percent(self) -> float:
        """Calculate percentage profit/loss"""
        if self.entry_price == 0:
            return 0.0
        return (self.pnl / (self.entry_price * self.quantity)) * 100


@dataclass
class Portfolio:
    """Portfolio state during backtesting"""
    initial_capital: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    trades: List[Trade]
    
    def __post_init__(self):
        if not hasattr(self, 'cash') or self.cash is None:
            self.cash = self.initial_capital
        if not hasattr(self, 'positions') or self.positions is None:
            self.positions = {}
        if not hasattr(self, 'trades') or self.trades is None:
            self.trades = []
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        # For simplicity, we'll calculate this when needed with current prices
        return self.cash + sum(self.positions.values())
    
    @property
    def total_return(self) -> float:
        """Calculate total return percentage"""
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100
    
    def get_position(self, symbol: str) -> float:
        """Get current position size for a symbol"""
        return self.positions.get(symbol, 0.0)


class BacktestResults:
    """Container for backtest results and performance metrics"""
    
    def __init__(self, portfolio: Portfolio, price_data: Dict[str, pd.DataFrame]):
        self.portfolio = portfolio
        self.price_data = price_data
        self.trades = portfolio.trades
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        metrics = {}
        
        # Basic metrics
        closed_trades = [t for t in self.trades if t.status == "CLOSED"]
        
        if not closed_trades:
            return {"error": "No closed trades to analyze"}
        
        # Trade statistics
        total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        metrics['total_trades'] = total_trades
        metrics['winning_trades'] = len(winning_trades)
        metrics['losing_trades'] = len(losing_trades)
        metrics['win_rate'] = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL statistics
        pnls = [t.pnl for t in closed_trades]
        metrics['total_pnl'] = sum(pnls)
        metrics['average_pnl'] = np.mean(pnls) if pnls else 0
        metrics['max_win'] = max(pnls) if pnls else 0
        metrics['max_loss'] = min(pnls) if pnls else 0
        
        # Return metrics
        metrics['total_return_pct'] = self.portfolio.total_return
        
        if winning_trades:
            metrics['average_win'] = np.mean([t.pnl for t in winning_trades])
        else:
            metrics['average_win'] = 0
            
        if losing_trades:
            metrics['average_loss'] = np.mean([t.pnl for t in losing_trades])
        else:
            metrics['average_loss'] = 0
        
        # Risk metrics
        if losing_trades and winning_trades:
            metrics['profit_factor'] = abs(metrics['average_win']) / abs(metrics['average_loss'])
        else:
            metrics['profit_factor'] = float('inf') if winning_trades else 0
        
        # Duration metrics
        durations = [t.duration.total_seconds() / 3600 for t in closed_trades if t.duration]  # hours
        if durations:
            metrics['average_trade_duration_hours'] = np.mean(durations)
            metrics['max_trade_duration_hours'] = max(durations)
            metrics['min_trade_duration_hours'] = min(durations)
        
        # Drawdown calculation (simplified)
        metrics['max_drawdown_pct'] = self._calculate_max_drawdown()
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown percentage"""
        # Simplified drawdown calculation
        # In a full implementation, this would track portfolio value over time
        
        cumulative_returns = []
        running_pnl = 0
        
        for trade in self.trades:
            if trade.status == "CLOSED":
                running_pnl += trade.pnl
                return_pct = (running_pnl / self.portfolio.initial_capital) * 100
                cumulative_returns.append(return_pct)
        
        if not cumulative_returns:
            return 0.0
        
        # Calculate drawdown
        peak = cumulative_returns[0]
        max_drawdown = 0
        
        for return_val in cumulative_returns:
            if return_val > peak:
                peak = return_val
            
            drawdown = peak - return_val
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of backtest results"""
        return {
            'portfolio': {
                'initial_capital': self.portfolio.initial_capital,
                'final_value': self.portfolio.total_value,
                'total_return_pct': self.portfolio.total_return,
                'cash_remaining': self.portfolio.cash
            },
            'trading': {
                'total_trades': len(self.trades),
                'open_trades': len([t for t in self.trades if t.is_open]),
                'closed_trades': len([t for t in self.trades if t.status == "CLOSED"])
            },
            'performance': self.metrics
        }
    
    def print_summary(self):
        """Print a formatted summary of results"""
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("="*60)
        
        summary = self.get_summary()
        
        # Portfolio Performance
        portfolio = summary['portfolio']
        print(f"\n💰 Portfolio Performance:")
        print(f"   Initial Capital: ${portfolio['initial_capital']:,.2f}")
        print(f"   Final Value: ${portfolio['final_value']:,.2f}")
        print(f"   Total Return: {portfolio['total_return_pct']:+.2f}%")
        print(f"   Cash Remaining: ${portfolio['cash_remaining']:,.2f}")
        
        # Trading Statistics
        trading = summary['trading']
        print(f"\n📈 Trading Statistics:")
        print(f"   Total Trades: {trading['total_trades']}")
        print(f"   Closed Trades: {trading['closed_trades']}")
        print(f"   Open Trades: {trading['open_trades']}")
        
        # Performance Metrics
        if 'error' not in self.metrics:
            metrics = self.metrics
            print(f"\n🎯 Performance Metrics:")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Total PnL: ${metrics['total_pnl']:,.2f}")
            print(f"   Average PnL: ${metrics['average_pnl']:,.2f}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            
            if metrics.get('average_trade_duration_hours'):
                print(f"   Avg Trade Duration: {metrics['average_trade_duration_hours']:.1f} hours")
        
        print("\n" + "="*60)


class BaseBacktester(ABC):
    """Base class for backtesting engines"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 1.0,
        **params
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> BacktestResults:
        """Run the backtest and return results"""
        pass
    
    def _create_initial_portfolio(self) -> Portfolio:
        """Create initial portfolio state"""
        return Portfolio(
            initial_capital=self.initial_capital,
            cash=self.initial_capital,
            positions={},
            trades=[]
        )
    
    def _execute_trade(
        self,
        portfolio: Portfolio,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_type: str = "BUY",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Trade]:
        """Execute a trade and update portfolio"""
        
        cost = abs(quantity * price) + self.commission
        
        # Check if we have enough cash (for buys) or shares (for sells)
        if trade_type == "BUY":
            if portfolio.cash < cost:
                self.logger.warning(f"Insufficient cash for {symbol} trade: need ${cost:.2f}, have ${portfolio.cash:.2f}")
                return None
        
        # Create trade
        trade = Trade(
            symbol=symbol,
            entry_time=timestamp,
            entry_price=price,
            quantity=abs(quantity),
            trade_type=trade_type,
            commission=self.commission,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Update portfolio
        if trade_type == "BUY":
            portfolio.cash -= cost
            portfolio.positions[symbol] = portfolio.positions.get(symbol, 0) + quantity
        else:  # SELL
            portfolio.cash += (quantity * price) - self.commission
            portfolio.positions[symbol] = portfolio.positions.get(symbol, 0) - quantity
        
        portfolio.trades.append(trade)
        
        self.logger.debug(f"Executed {trade_type} trade: {quantity} shares of {symbol} at ${price:.2f}")
        return trade
    
    def _close_trade(
        self,
        portfolio: Portfolio,
        trade: Trade,
        exit_price: float,
        exit_time: datetime,
        reason: str = "SIGNAL"
    ) -> None:
        """Close an open trade"""
        
        if not trade.is_open:
            return
        
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = "CLOSED"
        
        # Update portfolio cash
        if trade.trade_type == "BUY":
            # Sell the shares
            proceeds = (trade.quantity * exit_price) - self.commission
            portfolio.cash += proceeds
            portfolio.positions[trade.symbol] -= trade.quantity
        else:
            # Cover short position
            cost = (trade.quantity * exit_price) + self.commission
            portfolio.cash -= cost
            portfolio.positions[trade.symbol] += trade.quantity
        
        # Clean up zero positions
        if abs(portfolio.positions.get(trade.symbol, 0)) < 1e-6:
            portfolio.positions[trade.symbol] = 0
        
        self.logger.debug(f"Closed trade: {trade.symbol} PnL=${trade.pnl:.2f} ({reason})")