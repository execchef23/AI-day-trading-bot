"""
Signal Manager - Generates trading signals
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Trading signal data"""

    symbol: str
    signal: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    timestamp: str
    technical_score: float = 0.0
    ml_score: float = 0.0
    reason: str = ""


class SignalManager:
    """Manages trading signal generation"""

    def __init__(self):
        self.data_manager = None
        self.signals_history = []
        logger.info("Signal Manager initialized")

    def set_data_manager(self, data_manager):
        """Connect data manager"""
        self.data_manager = data_manager
        logger.info("Data manager connected to Signal Manager")

    def generate_signals(self, symbols: List[str]) -> List[TradingSignal]:
        """Generate trading signals for symbols"""
        logger.info(f"Generating signals for {len(symbols)} symbols")

        # Demo implementation
        signals = []
        return signals

    def get_signal_for_symbol(self, symbol: str) -> Optional[TradingSignal]:
        """Get latest signal for a symbol"""
        return None

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals"""
        return self.signals_history[-limit:] if self.signals_history else []
