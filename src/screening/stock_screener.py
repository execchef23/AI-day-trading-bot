"""
Stock Growth Screener - Identifies high-growth opportunities
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GrowthCategory(Enum):
    """Growth categories for stocks"""

    EXPLOSIVE_GROWTH = "explosive_growth"
    HIGH_GROWTH = "high_growth"
    MODERATE_GROWTH = "moderate_growth"
    STABLE_GROWTH = "stable_growth"
    LOW_GROWTH = "low_growth"


class ScreeningCriteria(Enum):
    """Screening criteria flags"""

    MOMENTUM_BREAKOUT = "momentum_breakout"
    VOLUME_SURGE = "volume_surge"
    TECHNICAL_SETUP = "technical_setup"
    RSI_RECOVERY = "rsi_recovery"
    MACD_CROSS = "macd_cross"


@dataclass
class ScreeningResult:
    """Result from stock screening"""

    symbol: str
    score: float
    growth_category: GrowthCategory
    current_price: float
    target_price: Optional[float] = None
    technical_score: float = 0.0
    momentum_score: float = 0.0
    volume_score: float = 0.0
    ml_prediction: float = 0.0
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    volume_ratio: Optional[float] = None
    price_change_1w: Optional[float] = None
    criteria_met: List[ScreeningCriteria] = None
    risk_factors: List[str] = None


class StockGrowthScreener:
    """Identifies high-growth stock opportunities"""

    def __init__(self):
        self.data_manager = None
        logger.info("Stock Growth Screener initialized")

        # ✅ ADD: Beginner-friendly settings
        self.beginner_mode = True
        self.price_max = 50.0
        self.price_min = 2.0
        self.volume_min = 500000

    def set_components(self, data_manager=None, signal_manager=None):
        """Connect external components"""
        self.data_manager = data_manager
        logger.info("Screener components connected")

    def run_full_scan(
        self, symbols: Optional[List[str]] = None
    ) -> Dict[str, ScreeningResult]:
        """Run full screening scan"""
        logger.info("Running full stock screening scan")

        # Demo implementation - return empty results
        # In production, this would analyze real market data
        results = {}

        logger.info(f"Scan complete: {len(results)} opportunities found")
        return results

    def run_beginner_scan(self, max_price: float = 50.0) -> Dict[str, ScreeningResult]:
        """
        Scan for beginner-friendly affordable stocks under $50

        Args:
            max_price: Maximum stock price to include

        Returns:
            Dictionary of screening results
        """
        logger.info("🔍 Running beginner-friendly stock scan")

        # Affordable stock universe (under $50)
        affordable_stocks = [
            # Tech/Growth (Under $50)
            "SOFI",
            "PLTR",
            "NIO",
            "RIVN",
            "LCID",
            "PLUG",
            # Established (Under $50)
            "F",
            "BAC",
            "WFC",
            "PFE",
            "T",
            "VZ",
            "INTC",
            # ETFs
            "SQQQ",
            "TQQQ",
            "SPXL",
            "UVXY",
        ]

        results = {}

        for symbol in affordable_stocks:
            try:
                # Use existing screening logic
                result = self._screen_single_stock(symbol)

                # Only include if under max price
                if result and result.current_price <= max_price:
                    results[symbol] = result

            except Exception as e:
                logger.debug(f"Could not screen {symbol}: {e}")
                continue

        logger.info(
            f"✅ Beginner scan complete: {len(results)} affordable stocks found"
        )
        return results

    def _get_affordable_stocks(self) -> List[str]:
        """
        Get list of affordable, liquid stocks for beginners
        """

        # ✅ Popular affordable stocks (constantly updated)
        affordable_stocks = [
            # Fintech & Banking (Under $20)
            "SOFI",
            "NU",
            "UPST",
            "AFRM",
            "SQ",
            "HOOD",
            # EV & Clean Energy (Under $30)
            "NIO",
            "LCID",
            "RIVN",
            "FSR",
            "BLNK",
            "CHPT",
            # Tech & Software (Under $50)
            "PLTR",
            "SNAP",
            "PINS",
            "U",
            "DDOG",
            "NET",
            # Retail & Consumer (Under $25)
            "WISH",
            "BBBY",
            "GME",
            "AMC",
            "TLRY",
            "SNDL",
            # Healthcare (Under $40)
            "TELADOC",
            "TDOC",
            "MRNA",
            "BNTX",
            # Telecom (Under $20)
            "T",
            "VZ",
            "TMUS",
            # Industrials (Under $30)
            "F",
            "GM",
            "AAL",
            "UAL",
            "CCL",
            # Banks (Under $50)
            "BAC",
            "WFC",
            "C",
            "JPM",
            # REITs (Under $30)
            "O",
            "AGNC",
            "NLY",
            # Crypto Exposure (Variable)
            "COIN",
            "MARA",
            "RIOT",
            "MSTR",
            # ETFs (Fractional shares)
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "VOO",
        ]

        # ✅ Add stocks from config
        from config.config import TRADING_SYMBOLS

        affordable_stocks.extend(TRADING_SYMBOLS)

        # Remove duplicates
        return list(set(affordable_stocks))

    def get_screening_status(self) -> Dict:
        """Get current screening status"""
        return {
            "is_active": False,
            "last_scan": None,
            "opportunities_found": 0,
            "scan_count": 0,
        }


# Singleton instance
_screener_instance = None


def get_screener() -> StockGrowthScreener:
    """Get or create screener instance"""
    global _screener_instance
    if _screener_instance is None:
        _screener_instance = StockGrowthScreener()
    return _screener_instance
