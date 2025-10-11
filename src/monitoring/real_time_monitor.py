"""
Real-Time Monitoring System

Advanced monitoring, alerting, and notification system for the AI trading bot.
Provides live position tracking, P&L updates, risk metrics, and intelligent alerts.
"""

import json
import logging
import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts"""
    POSITION_CHANGE = "position_change"
    PNL_THRESHOLD = "pnl_threshold"
    RISK_LIMIT = "risk_limit"
    SYSTEM_ERROR = "system_error"
    MARKET_CONDITION = "market_condition"
    TRADE_EXECUTION = "trade_execution"
    ENGINE_STATUS = "engine_status"
    PERFORMANCE_MILESTONE = "performance_milestone"


@dataclass
class Alert:
    """Alert data structure"""

    id: str
    timestamp: datetime
    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "alert_type": self.alert_type.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system"""

    # Alert thresholds
    pnl_warning_threshold: float = 0.05  # 5% loss warning
    pnl_critical_threshold: float = 0.10  # 10% loss critical
    position_size_warning: float = 0.25   # 25% of portfolio warning

    # Risk monitoring
    max_drawdown_warning: float = 0.08    # 8% drawdown warning
    correlation_warning: float = 0.75     # High correlation warning
    volatility_warning: float = 0.30      # High volatility warning

    # System monitoring
    engine_health_check_interval: int = 30  # seconds
    performance_update_interval: int = 60   # seconds
    alert_cooldown_seconds: int = 300       # 5 minutes between similar alerts

    # Persistence
    store_alerts: bool = True
    max_stored_alerts: int = 1000
    database_path: str = "data/monitoring.db"


@dataclass
class PerformanceMetrics:
    """Real-time performance tracking"""

    timestamp: datetime = field(default_factory=datetime.now)

    # Portfolio metrics
    total_value: float = 0.0
    cash_balance: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0

    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_trade_duration: float = 0.0  # in hours

    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    volatility: float = 0.0
    beta: float = 1.0

    # Position metrics
    position_count: int = 0
    largest_position_pct: float = 0.0
    portfolio_correlation: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_value": self.total_value,
            "cash_balance": self.cash_balance,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "daily_pnl": self.daily_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_trade_duration": self.avg_trade_duration,
            "current_drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "volatility": self.volatility,
            "beta": self.beta,
            "position_count": self.position_count,
            "largest_position_pct": self.largest_position_pct,
            "portfolio_correlation": self.portfolio_correlation
        }


class RealTimeMonitor:
    """Main real-time monitoring system"""

    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.alerts = deque(maxlen=self.config.max_stored_alerts)
        self.performance_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.current_metrics = PerformanceMetrics()

        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        self.stop_event = threading.Event()

        # Component references
        self.trading_engine = None
        self.portfolio_manager = None

        # Alert management
        self.alert_cooldowns = defaultdict(float)

        logger.info("Real-Time Monitor initialized")

    def set_trading_components(self, trading_engine=None, portfolio_manager=None):
        """Set references to trading components"""
        self.trading_engine = trading_engine
        self.portfolio_manager = portfolio_manager

    def start_monitoring(self) -> Dict[str, Any]:
        """Start the monitoring system"""

        if self.is_running:
            return {"success": False, "message": "Monitor already running"}

        try:
            self.is_running = True
            self.stop_event.clear()

            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()

            # Create startup alert
            self.create_alert(
                AlertLevel.INFO,
                AlertType.ENGINE_STATUS,
                "Monitoring Started",
                "Real-time monitoring system has been activated"
            )

            logger.info("Real-time monitoring started")
            return {"success": True, "message": "Monitoring started"}

        except Exception as e:
            self.is_running = False
            error_msg = f"Failed to start monitoring: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop the monitoring system"""

        try:
            self.is_running = False
            self.stop_event.set()

            # Wait for thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)

            # Create shutdown alert
            self.create_alert(
                AlertLevel.INFO,
                AlertType.ENGINE_STATUS,
                "Monitoring Stopped",
                "Real-time monitoring system has been deactivated"
            )

            logger.info("Real-time monitoring stopped")
            return {"success": True, "message": "Monitoring stopped"}

        except Exception as e:
            error_msg = f"Failed to stop monitoring: {e}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}

    def create_alert(self, level: AlertLevel, alert_type: AlertType,
                    title: str, message: str, data: Optional[Dict[str, Any]] = None) -> Alert:
        """Create a new alert"""

        alert_id = f"{alert_type.value}_{int(datetime.now().timestamp())}"

        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            alert_type=alert_type,
            title=title,
            message=message,
            data=data or {}
        )

        # Check cooldown
        cooldown_key = f"{alert_type.value}_{level.value}"
        if time.time() - self.alert_cooldowns[cooldown_key] < self.config.alert_cooldown_seconds:
            logger.debug(f"Alert {cooldown_key} in cooldown, skipping")
            return alert

        self.alert_cooldowns[cooldown_key] = time.time()

        # Store alert
        self.alerts.append(alert)

        logger.info(f"Created alert: {level.value} - {title}")
        return alert

    def _monitoring_loop(self):
        """Main monitoring loop"""

        logger.info("Monitoring loop started")

        try:
            while not self.stop_event.is_set():
                try:
                    # Update performance metrics
                    self._update_performance_metrics()

                    # Check for alerts
                    self._check_performance_alerts()
                    self._check_position_alerts()
                    self._check_risk_alerts()

                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")

                # Sleep between checks
                if not self.stop_event.wait(10.0):  # Check every 10 seconds
                    continue
                else:
                    break

        except Exception as e:
            logger.error(f"Critical error in monitoring loop: {e}")

        logger.info("Monitoring loop ended")

    def _update_performance_metrics(self):
        """Update current performance metrics"""

        try:
            now = datetime.now()

            # Get data from trading components
            if self.trading_engine:
                status = self.trading_engine.get_status()
                positions = self.trading_engine.get_positions()

                self.current_metrics.total_trades = status.get("total_trades", 0)
                self.current_metrics.win_rate = status.get("win_rate", 0.0)
                self.current_metrics.total_value = status.get("portfolio_value", 0.0)
                self.current_metrics.position_count = len(positions)

                # Calculate P&L metrics
                total_pnl = sum(pos.get("pnl", 0) for pos in positions)
                self.current_metrics.unrealized_pnl = total_pnl

                # Calculate largest position
                if positions:
                    largest_pnl = max(abs(pos.get("pnl", 0)) for pos in positions)
                    if self.current_metrics.total_value > 0:
                        self.current_metrics.largest_position_pct = (
                            largest_pnl / self.current_metrics.total_value
                        )

            self.current_metrics.timestamp = now

            # Store in history
            self.performance_history.append(self.current_metrics)

        except Exception as e:
            logger.error(f"Failed to update performance metrics: {e}")

    def _check_performance_alerts(self):
        """Check for performance-based alerts"""

        try:
            metrics = self.current_metrics

            # P&L alerts
            if metrics.total_value > 0:
                pnl_pct = metrics.unrealized_pnl / metrics.total_value

                if pnl_pct <= -self.config.pnl_critical_threshold:
                    self.create_alert(
                        AlertLevel.CRITICAL,
                        AlertType.PNL_THRESHOLD,
                        "Critical Loss Alert",
                        f"Portfolio has lost {pnl_pct:.1%} - exceeding critical threshold",
                        {"pnl_pct": pnl_pct, "threshold": self.config.pnl_critical_threshold}
                    )
                elif pnl_pct <= -self.config.pnl_warning_threshold:
                    self.create_alert(
                        AlertLevel.WARNING,
                        AlertType.PNL_THRESHOLD,
                        "Loss Warning Alert",
                        f"Portfolio has lost {pnl_pct:.1%} - approaching warning threshold",
                        {"pnl_pct": pnl_pct, "threshold": self.config.pnl_warning_threshold}
                    )

        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")

    def _check_position_alerts(self):
        """Check for position-related alerts"""

        try:
            if not self.trading_engine:
                return

            positions = self.trading_engine.get_positions()

            for position in positions:
                symbol = position.get("symbol")
                pnl_pct = position.get("pnl_pct", 0)

                # Large loss in individual position
                if pnl_pct <= -10.0:  # 10% loss
                    self.create_alert(
                        AlertLevel.WARNING,
                        AlertType.POSITION_CHANGE,
                        f"{symbol} Large Loss",
                        f"{symbol} position down {pnl_pct:.1f}%",
                        {"symbol": symbol, "pnl_pct": pnl_pct}
                    )

                # Large gain in individual position
                elif pnl_pct >= 15.0:  # 15% gain
                    self.create_alert(
                        AlertLevel.INFO,
                        AlertType.POSITION_CHANGE,
                        f"{symbol} Large Gain",
                        f"{symbol} position up {pnl_pct:.1f}% - consider profit taking",
                        {"symbol": symbol, "pnl_pct": pnl_pct}
                    )

        except Exception as e:
            logger.error(f"Failed to check position alerts: {e}")

    def _check_risk_alerts(self):
        """Check for risk-related alerts"""

        try:
            metrics = self.current_metrics

            # Position concentration alert
            if metrics.largest_position_pct >= self.config.position_size_warning:
                self.create_alert(
                    AlertLevel.WARNING,
                    AlertType.RISK_LIMIT,
                    "High Position Concentration",
                    f"Largest position represents {metrics.largest_position_pct:.1%} of portfolio",
                    {"concentration": metrics.largest_position_pct}
                )

        except Exception as e:
            logger.error(f"Failed to check risk alerts: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get unresolved alerts"""
        return [alert for alert in self.alerts if not alert.resolved]

    def get_recent_alerts(self, minutes: int = 60) -> List[Alert]:
        """Get alerts from last N minutes"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff]

    def acknowledge_alert(self, alert_id: str):
        """Mark alert as acknowledged"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                break

    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                break

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring system status"""
        return {
            "is_running": self.is_running,
            "active_alerts_count": len(self.get_active_alerts()),
            "total_alerts_count": len(self.alerts),
            "current_metrics": self.current_metrics.to_dict()
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "status": self.get_monitoring_status(),
            "active_alerts": [alert.to_dict() for alert in self.get_active_alerts()],
            "recent_alerts": [alert.to_dict() for alert in self.get_recent_alerts(60)],
            "performance_metrics": self.current_metrics.to_dict(),
            "performance_trends": self._get_performance_trends()
        }

    def _get_performance_trends(self) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        if not self.performance_history:
            return {}

        recent_metrics = list(self.performance_history)[-60:]  # Last 60 data points

        return {
            "timestamps": [m.timestamp.isoformat() for m in recent_metrics],
            "portfolio_values": [m.total_value for m in recent_metrics],
            "pnl_values": [m.unrealized_pnl for m in recent_metrics],
            "position_counts": [m.position_count for m in recent_metrics]
        }


# Singleton instance for global access
_monitor_instance = None


def get_monitor() -> RealTimeMonitor:
    """Get or create the global monitoring instance"""
    global _monitor_instance

    if _monitor_instance is None:
        _monitor_instance = RealTimeMonitor()

    return _monitor_instance
