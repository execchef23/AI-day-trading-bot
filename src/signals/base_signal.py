"""Base signal generator interface"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading decisions"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalStrength(Enum):
    """Signal strength levels"""

    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


class TradingSignal:
    """Trading signal with metadata"""

    def __init__(
        self,
        symbol: str,
        signal_type: SignalType,
        strength: SignalStrength,
        confidence: float,
        price: float,
        timestamp: pd.Timestamp,
        reasoning: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength
        self.confidence = confidence  # 0.0 to 1.0
        self.price = price
        self.timestamp = timestamp
        self.reasoning = reasoning
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "price": self.price,
            "timestamp": self.timestamp,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"TradingSignal({self.symbol}, {self.signal_type.value}, "
            f"strength={self.strength.value}, confidence={self.confidence:.2f})"
        )


class BaseSignalGenerator(ABC):
    """Base class for all signal generators"""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def generate_signal(
        self, data: pd.DataFrame, symbol: str, **kwargs
    ) -> Optional[TradingSignal]:
        """Generate trading signal from market data"""
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data format"""
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

        if data.empty:
            self.logger.warning("Empty data provided")
            return False

        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.warning(f"Missing required columns: {missing_columns}")
            return False

        return True

    def calculate_confidence(
        self, technical_score: float, ml_score: float, volume_score: float
    ) -> float:
        """Calculate overall confidence score"""
        weights = {
            "technical": self.params.get("technical_weight", 0.4),
            "ml": self.params.get("ml_weight", 0.4),
            "volume": self.params.get("volume_weight", 0.2),
        }

        confidence = (
            technical_score * weights["technical"]
            + ml_score * weights["ml"]
            + volume_score * weights["volume"]
        )

        return np.clip(confidence, 0.0, 1.0)
