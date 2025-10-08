"""Strategy backtester using signal generation system"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from .base_backtester import BaseBacktester, BacktestResults, Portfolio, Trade
from ..signals.signal_manager import SignalManager
from ..signals import SignalType, TradingSignal

logger = logging.getLogger(__name__)


class SignalBasedBacktester(BaseBacktester):
    """Backtester that uses signal generation for trading decisions"""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 1.0,
        position_size: float = 0.1,  # Fraction of portfolio per position
        max_positions: int = 5,
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.06,  # 6% take profit
        **params
    ):
        super().__init__(initial_capital, commission, **params)
        
        self.position_size = position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Initialize signal manager
        self.signal_manager = SignalManager(**params)
        
        # Track active trades per symbol
        self.active_trades = {}
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **kwargs
    ) -> BacktestResults:
        """Run backtest using signal-based trading strategy"""
        
        self.logger.info(f"Starting backtest with {len(data)} symbols")
        
        # Initialize portfolio
        portfolio = self._create_initial_portfolio()
        
        # Determine date range
        all_dates = set()
        for symbol_data in data.values():
            all_dates.update(symbol_data.index)
        
        all_dates = sorted(all_dates)
        
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        if not all_dates:
            raise ValueError("No valid dates found for backtesting")
        
        self.logger.info(f"Backtesting from {all_dates[0]} to {all_dates[-1]} ({len(all_dates)} days)")
        
        # Main backtesting loop
        for current_date in all_dates:
            self._process_trading_day(portfolio, data, current_date)
        
        # Close any remaining open positions
        self._close_all_positions(portfolio, data, all_dates[-1])
        
        self.logger.info(f"Backtest completed. Portfolio value: ${portfolio.total_value:.2f}")
        
        return BacktestResults(portfolio, data)
    
    def _process_trading_day(
        self,
        portfolio: Portfolio,
        data: Dict[str, pd.DataFrame],
        current_date: datetime
    ):
        """Process trading decisions for a single day"""
        
        # First, check existing positions for stop-loss/take-profit
        self._check_existing_positions(portfolio, data, current_date)
        
        # Then, look for new trading opportunities
        self._find_new_opportunities(portfolio, data, current_date)
    
    def _check_existing_positions(
        self,
        portfolio: Portfolio,
        data: Dict[str, pd.DataFrame],
        current_date: datetime
    ):
        """Check existing positions for exit conditions"""
        
        # Get all open trades
        open_trades = [t for t in portfolio.trades if t.is_open]
        
        for trade in open_trades:
            symbol = trade.symbol
            
            if symbol not in data:
                continue
            
            symbol_data = data[symbol]
            
            # Check if we have data for this date
            if current_date not in symbol_data.index:
                continue
            
            current_price = symbol_data.loc[current_date, 'Close']
            
            # Check stop-loss and take-profit conditions
            should_close, reason = self._should_close_trade(trade, current_price)
            
            if should_close:
                self._close_trade(portfolio, trade, current_price, current_date, reason)
                if symbol in self.active_trades:
                    del self.active_trades[symbol]
    
    def _should_close_trade(self, trade: Trade, current_price: float) -> tuple[bool, str]:
        """Determine if a trade should be closed"""
        
        if trade.trade_type == "BUY":
            # Check stop-loss
            if trade.stop_loss and current_price <= trade.stop_loss:
                return True, "STOP_LOSS"
            
            # Check take-profit
            if trade.take_profit and current_price >= trade.take_profit:
                return True, "TAKE_PROFIT"
            
            # Check percentage-based exits
            pct_change = (current_price - trade.entry_price) / trade.entry_price
            
            if pct_change <= -self.stop_loss_pct:
                return True, "STOP_LOSS_PCT"
            
            if pct_change >= self.take_profit_pct:
                return True, "TAKE_PROFIT_PCT"
        
        else:  # SELL (short position)
            # For short positions, logic is inverted
            if trade.stop_loss and current_price >= trade.stop_loss:
                return True, "STOP_LOSS"
            
            if trade.take_profit and current_price <= trade.take_profit:
                return True, "TAKE_PROFIT"
            
            pct_change = (trade.entry_price - current_price) / trade.entry_price
            
            if pct_change <= -self.stop_loss_pct:
                return True, "STOP_LOSS_PCT"
            
            if pct_change >= self.take_profit_pct:
                return True, "TAKE_PROFIT_PCT"
        
        return False, ""
    
    def _find_new_opportunities(
        self,
        portfolio: Portfolio,
        data: Dict[str, pd.DataFrame],
        current_date: datetime
    ):
        """Look for new trading opportunities"""
        
        # Don't open new positions if we're at max capacity
        open_positions = len([t for t in portfolio.trades if t.is_open])
        if open_positions >= self.max_positions:
            return
        
        # Check each symbol for signals
        for symbol, symbol_data in data.items():
            
            # Skip if we already have a position in this symbol
            if symbol in self.active_trades:
                continue
            
            # Check if we have enough data and current date exists
            if current_date not in symbol_data.index:
                continue
            
            # Get historical data up to current date for signal generation
            historical_data = symbol_data[symbol_data.index <= current_date]
            
            if len(historical_data) < 20:  # Need minimum data for signals
                continue
            
            # Generate signal
            try:
                signal = self.signal_manager.get_consensus_signal(historical_data, symbol)
                
                if signal and self._should_take_signal(signal, portfolio):
                    self._execute_signal_trade(portfolio, signal, current_date, historical_data)
                    
            except Exception as e:
                self.logger.warning(f"Error generating signal for {symbol}: {e}")
    
    def _should_take_signal(self, signal: TradingSignal, portfolio: Portfolio) -> bool:
        """Determine if we should act on a signal"""
        
        # Only take strong signals
        if signal.confidence < 0.6:
            return False
        
        # Only take buy/sell signals (not hold)
        if signal.signal_type not in [SignalType.BUY, SignalType.STRONG_BUY, 
                                     SignalType.SELL, SignalType.STRONG_SELL]:
            return False
        
        # Check if we have enough cash for buy signals
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            position_value = portfolio.cash * self.position_size
            required_cash = position_value + self.commission
            
            if portfolio.cash < required_cash:
                return False
        
        return True
    
    def _execute_signal_trade(
        self,
        portfolio: Portfolio,
        signal: TradingSignal,
        current_date: datetime,
        historical_data: pd.DataFrame
    ):
        """Execute a trade based on a signal"""
        
        symbol = signal.symbol
        current_price = signal.price
        
        # Calculate position size
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            # Buy signal
            position_value = portfolio.cash * self.position_size
            quantity = int(position_value / current_price)
            
            if quantity <= 0:
                return
            
            # Calculate stop-loss and take-profit prices
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
            
            trade = self._execute_trade(
                portfolio=portfolio,
                symbol=symbol,
                quantity=quantity,
                price=current_price,
                timestamp=current_date,
                trade_type="BUY",
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if trade:
                self.active_trades[symbol] = trade
                self.logger.info(f"BUY signal executed: {quantity} shares of {symbol} at ${current_price:.2f}")
        
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            # For now, we'll only handle selling existing positions
            # In a full implementation, you might add short selling
            
            current_position = portfolio.get_position(symbol)
            if current_position > 0:
                # Close existing long position
                existing_trade = self.active_trades.get(symbol)
                if existing_trade and existing_trade.is_open:
                    self._close_trade(portfolio, existing_trade, current_price, current_date, "SELL_SIGNAL")
                    if symbol in self.active_trades:
                        del self.active_trades[symbol]
                    
                    self.logger.info(f"SELL signal executed: closed position in {symbol} at ${current_price:.2f}")
    
    def _close_all_positions(
        self,
        portfolio: Portfolio,
        data: Dict[str, pd.DataFrame],
        final_date: datetime
    ):
        """Close all remaining open positions at the end of backtest"""
        
        open_trades = [t for t in portfolio.trades if t.is_open]
        
        for trade in open_trades:
            symbol = trade.symbol
            
            if symbol in data and final_date in data[symbol].index:
                final_price = data[symbol].loc[final_date, 'Close']
                self._close_trade(portfolio, trade, final_price, final_date, "END_OF_BACKTEST")
            else:
                # If we don't have price data, close at entry price (no gain/loss)
                trade.exit_price = trade.entry_price
                trade.exit_time = final_date
                trade.status = "CLOSED"
        
        self.active_trades.clear()
        self.logger.info(f"Closed {len(open_trades)} remaining positions at end of backtest")
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of strategy parameters"""
        return {
            'initial_capital': self.initial_capital,
            'position_size': self.position_size,
            'max_positions': self.max_positions,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'commission': self.commission,
            'signal_generator': 'MLTechnicalSignalGenerator'
        }