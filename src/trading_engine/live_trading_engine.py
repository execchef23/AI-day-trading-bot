"""
Live Trading Engine - Real-time automated trading system

This module coordinates ML models, signal generation, and portfolio management
into an automated trading system with configurable parameters.
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingState(Enum):
    """Trading engine states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class TradingConfig:
    """Trading engine configuration"""

    # Core settings
    enabled: bool = False
    state: TradingState = TradingState.STOPPED

    # Trading parameters
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"])
    initial_capital: float = 100000.0
    max_positions: int = 5
    position_size_pct: float = 0.20  # 20% of portfolio per position

    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_portfolio_risk: float = 0.10  # 10% max portfolio at risk

    # Signal generation
    signal_threshold: float = 0.6  # Minimum confidence for signals
    signal_cooldown_minutes: int = 15  # Minutes between signal generation
    use_ml_models: bool = True
    use_technical_analysis: bool = True

    # Execution settings
    execution_delay_seconds: float = 1.0  # Delay between trades
    paper_trading: bool = True  # Start with paper trading
    dry_run: bool = True

    # Monitoring
    update_interval_seconds: int = 30  # Portfolio update frequency
    log_trades: bool = True
    save_state: bool = True


@dataclass
class TradingMetrics:
    """Real-time trading metrics"""

    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    portfolio_value: float = 100000.0
    cash_balance: float = 100000.0

    # Activity metrics
    signals_generated: int = 0
    signals_executed: int = 0
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    # System metrics
    uptime_hours: float = 0.0
    errors_count: int = 0
    last_error: Optional[str] = None


class LiveTradingEngine:
    """
    Real-time trading engine that coordinates all trading components
    """

    def __init__(self, config: Optional[TradingConfig] = None):
        """Initialize the trading engine"""

        self.config = config or TradingConfig()
        self.metrics = TradingMetrics()
        self.state = TradingState.STOPPED

        # Component initialization
        self.data_manager = None
        self.signal_manager = None
        self.portfolio_manager = None
        self.ml_models = {}

        # Threading and async
        self.engine_thread = None
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Data storage
        self.positions = {}
        self.signals_history = []
        self.trades_history = []
        self.price_data = {}

        # State persistence
        self.state_file = "data/trading_engine_state.json"
        self.logs_dir = "logs"

        # Initialize components
        self._initialize_components()

        logger.info("Live Trading Engine initialized")

    def _initialize_components(self):
        """Initialize trading components with error handling"""

        try:
            # Import with fallbacks
            sys_path = os.path.join(os.path.dirname(__file__), "..")
            if sys_path not in sys.path:
                sys.path.append(sys_path)

            # Data Manager
            try:
                from data_sources.data_manager import DataManager
                self.data_manager = DataManager()
                logger.info("✅ Data Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Data Manager failed: {e}")

            # Signal Manager
            try:
                from signals.signal_manager import SignalManager
                self.signal_manager = SignalManager()
                logger.info("✅ Signal Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Signal Manager failed: {e}")

            # Portfolio Manager
            try:
                from risk_management.portfolio_manager import PortfolioManager
                self.portfolio_manager = PortfolioManager(
                    initial_cash=self.config.initial_capital
                )
                logger.info("✅ Portfolio Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Portfolio Manager failed: {e}")

        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")

    def start_trading(self) -> Dict[str, Any]:
        """Start the trading engine"""

        if self.state == TradingState.RUNNING:
            return {"success": False, "message": "Trading engine already running"}

        try:
            self.state = TradingState.STARTING
            logger.info("🚀 Starting Live Trading Engine...")

            # Reset stop event and metrics
            self.stop_event.clear()
            self.metrics.start_time = datetime.now()

            # Load previous state if available
            self._load_state()

            # Validate configuration
            validation_result = self._validate_config()
            if not validation_result["valid"]:
                self.state = TradingState.ERROR
                return {
                    "success": False,
                    "message": f"Configuration invalid: {validation_result['errors']}"
                }

            # Start main trading loop in separate thread
            self.engine_thread = threading.Thread(target=self._trading_loop, daemon=True)
            self.engine_thread.start()

            self.state = TradingState.RUNNING
            self.config.enabled = True

            logger.info("✅ Live Trading Engine started successfully")
            return {
                "success": True,
                "message": "Trading engine started",
                "state": self.state.value,
                "config": self._get_config_summary()
            }

        except Exception as e:
            self.state = TradingState.ERROR
            error_msg = f"Failed to start trading engine: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def stop_trading(self) -> Dict[str, Any]:
        """Stop the trading engine"""

        try:
            logger.info("🛑 Stopping Live Trading Engine...")

            self.state = TradingState.STOPPED
            self.config.enabled = False
            self.stop_event.set()

            # Wait for thread to finish (with timeout)
            if self.engine_thread and self.engine_thread.is_alive():
                self.engine_thread.join(timeout=5.0)

            # Save current state
            self._save_state()

            logger.info("✅ Live Trading Engine stopped")
            return {
                "success": True,
                "message": "Trading engine stopped",
                "final_metrics": self._get_metrics_summary()
            }

        except Exception as e:
            error_msg = f"Error stopping trading engine: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def pause_trading(self) -> Dict[str, Any]:
        """Pause trading without stopping the engine"""

        if self.state != TradingState.RUNNING:
            return {"success": False, "message": "Engine not running"}

        self.state = TradingState.PAUSED
        logger.info("⏸️ Trading engine paused")
        return {"success": True, "message": "Trading paused"}

    def resume_trading(self) -> Dict[str, Any]:
        """Resume paused trading"""

        if self.state != TradingState.PAUSED:
            return {"success": False, "message": "Engine not paused"}

        self.state = TradingState.RUNNING
        logger.info("▶️ Trading engine resumed")
        return {"success": True, "message": "Trading resumed"}

    def _trading_loop(self):
        """Main trading loop - runs in separate thread"""

        logger.info("🔄 Trading loop started")

        try:
            while not self.stop_event.is_set():
                loop_start = time.time()

                # Only process if running (not paused)
                if self.state == TradingState.RUNNING:
                    try:
                        # Main trading cycle
                        self._update_market_data()
                        self._generate_signals()
                        self._process_signals()
                        self._update_positions()
                        self._check_risk_limits()
                        self._update_metrics()

                    except Exception as e:
                        logger.error(f"❌ Error in trading loop: {e}")
                        self.metrics.errors_count += 1
                        self.metrics.last_error = str(e)

                        # Pause on critical errors
                        if "critical" in str(e).lower():
                            self.state = TradingState.ERROR
                            break

                # Calculate sleep time to maintain update interval
                loop_duration = time.time() - loop_start
                sleep_time = max(0, self.config.update_interval_seconds - loop_duration)

                if not self.stop_event.wait(sleep_time):
                    continue
                else:
                    break

        except Exception as e:
            logger.error(f"❌ Critical error in trading loop: {e}")
            self.state = TradingState.ERROR

        logger.info("🔄 Trading loop ended")

    def _update_market_data(self):
        """Update market data for all symbols"""

        if not self.data_manager:
            return

        try:
            for symbol in self.config.symbols:
                # Get latest price data
                data = self.data_manager.get_market_data(symbol, period="1d", limit=100)
                if not data.empty:
                    self.price_data[symbol] = data

        except Exception as e:
            logger.warning(f"⚠️ Failed to update market data: {e}")

    def _generate_signals(self):
        """Generate trading signals using ML and technical analysis"""

        # Check cooldown period
        if (self.metrics.last_signal_time and
            datetime.now() - self.metrics.last_signal_time <
            timedelta(minutes=self.config.signal_cooldown_minutes)):
            return

        if not self.signal_manager:
            return

        try:
            new_signals = []

            for symbol in self.config.symbols:
                if symbol not in self.price_data:
                    continue

                # Generate signal for this symbol
                signal = self.signal_manager.generate_signal(
                    symbol=symbol,
                    data=self.price_data[symbol],
                    use_ml=self.config.use_ml_models,
                    use_technical=self.config.use_technical_analysis
                )

                if signal and signal.confidence >= self.config.signal_threshold:
                    new_signals.append(signal)
                    self.signals_history.append({
                        "timestamp": datetime.now(),
                        "symbol": symbol,
                        "signal": signal.signal_type.value,
                        "confidence": signal.confidence,
                        "price": signal.price
                    })

            if new_signals:
                logger.info(f"📡 Generated {len(new_signals)} signals")
                self.metrics.signals_generated += len(new_signals)
                self.metrics.last_signal_time = datetime.now()

        except Exception as e:
            logger.warning(f"⚠️ Signal generation failed: {e}")

    def _process_signals(self):
        """Process signals and execute trades"""

        if not self.portfolio_manager:
            return

        try:
            # Get recent signals to process
            recent_signals = [
                s for s in self.signals_history
                if datetime.now() - s["timestamp"] < timedelta(minutes=5)
            ]

            for signal_data in recent_signals:
                symbol = signal_data["symbol"]
                signal_type = signal_data["signal"]

                # Check if we should execute this signal
                if self._should_execute_signal(symbol, signal_type):
                    result = self._execute_trade(symbol, signal_type, signal_data)

                    if result["success"]:
                        self.metrics.signals_executed += 1
                        self.metrics.last_trade_time = datetime.now()
                        logger.info(f"✅ Executed trade: {symbol} {signal_type}")

                    # Add execution delay
                    time.sleep(self.config.execution_delay_seconds)

        except Exception as e:
            logger.warning(f"⚠️ Signal processing failed: {e}")

    def _should_execute_signal(self, symbol: str, signal_type: str) -> bool:
        """Determine if we should execute a signal"""

        # Check position limits
        if len(self.positions) >= self.config.max_positions and signal_type in ["BUY", "STRONG_BUY"]:
            return False

        # Check if we already have a position
        if symbol in self.positions and signal_type in ["BUY", "STRONG_BUY"]:
            return False  # Don't double up

        # Check cash availability for buys
        if signal_type in ["BUY", "STRONG_BUY"]:
            position_value = self.metrics.portfolio_value * self.config.position_size_pct
            if position_value > self.metrics.cash_balance:
                return False

        return True

    def _execute_trade(self, symbol: str, signal_type: str, signal_data: Dict) -> Dict[str, Any]:
        """Execute a trade based on signal"""

        if not self.portfolio_manager:
            return {"success": False, "message": "Portfolio manager not available"}

        try:
            current_price = signal_data.get("price", 0)

            if signal_type in ["BUY", "STRONG_BUY"]:
                # Calculate position size
                position_value = self.metrics.portfolio_value * self.config.position_size_pct
                quantity = int(position_value / current_price)

                if quantity > 0:
                    result = self.portfolio_manager.execute_trade(
                        symbol=symbol,
                        action="BUY",
                        quantity=quantity,
                        price=current_price,
                        stop_loss=current_price * (1 - self.config.stop_loss_pct),
                        take_profit=current_price * (1 + self.config.take_profit_pct),
                        dry_run=self.config.dry_run
                    )

                    if result["success"]:
                        self.positions[symbol] = {
                            "quantity": quantity,
                            "entry_price": current_price,
                            "entry_time": datetime.now(),
                            "stop_loss": current_price * (1 - self.config.stop_loss_pct),
                            "take_profit": current_price * (1 + self.config.take_profit_pct)
                        }

                    return result

            elif signal_type in ["SELL", "STRONG_SELL"] and symbol in self.positions:
                # Sell existing position
                position = self.positions[symbol]
                result = self.portfolio_manager.execute_trade(
                    symbol=symbol,
                    action="SELL",
                    quantity=position["quantity"],
                    price=current_price,
                    dry_run=self.config.dry_run
                )

                if result["success"]:
                    # Calculate P&L
                    pnl = (current_price - position["entry_price"]) * position["quantity"]
                    self.metrics.total_pnl += pnl

                    if pnl > 0:
                        self.metrics.winning_trades += 1
                    else:
                        self.metrics.losing_trades += 1

                    # Remove position
                    del self.positions[symbol]

                return result

            return {"success": False, "message": "No valid trade action"}

        except Exception as e:
            error_msg = f"Trade execution failed: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def _update_positions(self):
        """Update position values and check exit conditions"""

        try:
            for symbol, position in list(self.positions.items()):
                if symbol not in self.price_data:
                    continue

                current_price = self.price_data[symbol]["Close"].iloc[-1]

                # Check stop loss
                if current_price <= position["stop_loss"]:
                    logger.info(f"🛑 Stop loss triggered for {symbol}")
                    self._execute_trade(symbol, "SELL", {"price": current_price})

                # Check take profit
                elif current_price >= position["take_profit"]:
                    logger.info(f"🎯 Take profit triggered for {symbol}")
                    self._execute_trade(symbol, "SELL", {"price": current_price})

        except Exception as e:
            logger.warning(f"⚠️ Position update failed: {e}")

    def _check_risk_limits(self):
        """Check and enforce risk limits"""

        try:
            # Calculate current portfolio metrics
            total_value = self.metrics.cash_balance
            total_risk = 0.0

            for symbol, position in self.positions.items():
                if symbol in self.price_data:
                    current_price = self.price_data[symbol]["Close"].iloc[-1]
                    position_value = current_price * position["quantity"]
                    total_value += position_value

                    # Calculate risk (potential loss to stop loss)
                    risk_per_share = position["entry_price"] - position["stop_loss"]
                    position_risk = risk_per_share * position["quantity"]
                    total_risk += position_risk

            self.metrics.portfolio_value = total_value

            # Check daily loss limit
            daily_loss_pct = abs(self.metrics.daily_pnl) / self.config.initial_capital
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                logger.warning(f"🚨 Daily loss limit reached: {daily_loss_pct:.2%}")
                self.state = TradingState.PAUSED

            # Check portfolio risk limit
            portfolio_risk_pct = total_risk / total_value if total_value > 0 else 0
            if portfolio_risk_pct >= self.config.max_portfolio_risk:
                logger.warning(f"🚨 Portfolio risk limit exceeded: {portfolio_risk_pct:.2%}")
                # Could trigger position reduction here

        except Exception as e:
            logger.warning(f"⚠️ Risk check failed: {e}")

    def _update_metrics(self):
        """Update trading metrics"""

        try:
            now = datetime.now()
            self.metrics.last_update = now

            # Calculate uptime
            uptime_delta = now - self.metrics.start_time
            self.metrics.uptime_hours = uptime_delta.total_seconds() / 3600

            # Update trade counts
            self.metrics.total_trades = self.metrics.winning_trades + self.metrics.losing_trades

            # Calculate unrealized P&L
            unrealized_pnl = 0.0
            for symbol, position in self.positions.items():
                if symbol in self.price_data:
                    current_price = self.price_data[symbol]["Close"].iloc[-1]
                    unrealized_pnl += (current_price - position["entry_price"]) * position["quantity"]

            self.metrics.unrealized_pnl = unrealized_pnl

        except Exception as e:
            logger.warning(f"⚠️ Metrics update failed: {e}")

    def _validate_config(self) -> Dict[str, Any]:
        """Validate trading configuration"""

        errors = []

        if self.config.initial_capital <= 0:
            errors.append("Initial capital must be positive")

        if not self.config.symbols:
            errors.append("No trading symbols specified")

        if self.config.position_size_pct <= 0 or self.config.position_size_pct > 1:
            errors.append("Position size percentage must be between 0 and 1")

        if self.config.signal_threshold < 0 or self.config.signal_threshold > 1:
            errors.append("Signal threshold must be between 0 and 1")

        return {"valid": len(errors) == 0, "errors": errors}

    def _save_state(self):
        """Save current engine state to file"""

        if not self.config.save_state:
            return

        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

            state_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "symbols": self.config.symbols,
                    "initial_capital": self.config.initial_capital,
                    "position_size_pct": self.config.position_size_pct,
                    # Add other important config items
                },
                "metrics": {
                    "total_trades": self.metrics.total_trades,
                    "winning_trades": self.metrics.winning_trades,
                    "losing_trades": self.metrics.losing_trades,
                    "total_pnl": self.metrics.total_pnl,
                    "portfolio_value": self.metrics.portfolio_value,
                    # Add other metrics
                },
                "positions": self.positions,
                "recent_signals": self.signals_history[-50:],  # Keep last 50 signals
                "recent_trades": self.trades_history[-100:]  # Keep last 100 trades
            }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)

            logger.debug("💾 Engine state saved")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save state: {e}")

    def _load_state(self):
        """Load previous engine state from file"""

        if not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # Restore positions
            self.positions = state_data.get("positions", {})

            # Restore signal history
            self.signals_history = state_data.get("recent_signals", [])

            # Restore trade history
            self.trades_history = state_data.get("recent_trades", [])

            logger.info("📂 Previous engine state loaded")

        except Exception as e:
            logger.warning(f"⚠️ Failed to load state: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current trading engine status"""

        return {
            "state": self.state.value,
            "enabled": self.config.enabled,
            "uptime_hours": self.metrics.uptime_hours,
            "last_update": self.metrics.last_update.isoformat() if self.metrics.last_update else None,
            "positions_count": len(self.positions),
            "total_trades": self.metrics.total_trades,
            "win_rate": (self.metrics.winning_trades / max(self.metrics.total_trades, 1)) * 100,
            "total_pnl": self.metrics.total_pnl,
            "portfolio_value": self.metrics.portfolio_value,
            "errors_count": self.metrics.errors_count,
            "last_error": self.metrics.last_error
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions with live P&L"""

        positions_list = []

        for symbol, position in self.positions.items():
            current_price = 0.0
            if symbol in self.price_data and not self.price_data[symbol].empty:
                current_price = self.price_data[symbol]["Close"].iloc[-1]

            pnl = (current_price - position["entry_price"]) * position["quantity"]
            pnl_pct = (pnl / (position["entry_price"] * position["quantity"])) * 100

            positions_list.append({
                "symbol": symbol,
                "quantity": position["quantity"],
                "entry_price": position["entry_price"],
                "current_price": current_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "entry_time": position["entry_time"].isoformat(),
                "stop_loss": position["stop_loss"],
                "take_profit": position["take_profit"]
            })

        return positions_list

    def get_recent_signals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent trading signals"""

        return self.signals_history[-limit:] if self.signals_history else []

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update trading configuration"""

        try:
            # Update config attributes
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            logger.info("⚙️ Trading configuration updated")
            return {"success": True, "message": "Configuration updated"}

        except Exception as e:
            error_msg = f"Failed to update configuration: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def _get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""

        return {
            "symbols": self.config.symbols,
            "max_positions": self.config.max_positions,
            "position_size_pct": self.config.position_size_pct,
            "stop_loss_pct": self.config.stop_loss_pct,
            "take_profit_pct": self.config.take_profit_pct,
            "signal_threshold": self.config.signal_threshold,
            "paper_trading": self.config.paper_trading,
            "dry_run": self.config.dry_run
        }

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""

        return {
            "uptime_hours": round(self.metrics.uptime_hours, 2),
            "total_trades": self.metrics.total_trades,
            "winning_trades": self.metrics.winning_trades,
            "win_rate": round((self.metrics.winning_trades / max(self.metrics.total_trades, 1)) * 100, 1),
            "total_pnl": round(self.metrics.total_pnl, 2),
            "portfolio_value": round(self.metrics.portfolio_value, 2),
            "signals_generated": self.metrics.signals_generated,
            "signals_executed": self.metrics.signals_executed
        }


# Singleton instance for global access
_trading_engine_instance = None


def get_trading_engine() -> LiveTradingEngine:
    """Get or create the global trading engine instance"""
    global _trading_engine_instance

    if _trading_engine_instance is None:
        _trading_engine_instance = LiveTradingEngine()

    return _trading_engine_instance


if __name__ == "__main__":
    # Test the trading engine
    engine = LiveTradingEngine()

    print("🤖 Testing Live Trading Engine")
    print("=" * 50)

    # Start engine
    result = engine.start_trading()
    print(f"Start result: {result}")

    # Let it run for a bit
    time.sleep(10)

    # Check status
    status = engine.get_status()
    print(f"Status: {status}")

    # Stop engine
    result = engine.stop_trading()
    print(f"Stop result: {result}")
