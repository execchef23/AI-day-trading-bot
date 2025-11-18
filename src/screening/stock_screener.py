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

        # Beginner-friendly settings
        self.beginner_mode = True
        self.price_max = 50.0
        self.price_min = 2.0
        self.volume_min = 500000

    def set_components(self, data_manager=None, signal_manager=None):
        """Connect external components"""
        self.data_manager = data_manager
        logger.info("Screener components connected")

    def run_full_scan(self) -> Dict[str, ScreeningResult]:
        """Run full stock screening scan"""
        logger.info("Running full stock screening scan")

        scan_symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "NVDA",
            "META",
            "NFLX",
            "SOFI",
            "PLTR",
            "NIO",
            "RIVN",
            "LCID",
            "PLUG",
            "F",
            "BAC",
            "T",
            "INTC",
            "AMD",
            "PYPL",
            "SQ",
            "ROKU",
            "COIN",
            "HOOD",
            "SPY",
            "QQQ",
            "IWM",
        ]

        results = {}

        for symbol in scan_symbols:
            try:
                result = self._screen_symbol(symbol)

                # ✅ CHANGED: Lower threshold to 0.0 to accept everything
                if result and result.score >= 0.0:
                    results[symbol] = result

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")
                continue

        logger.info(f"Scan complete: {len(results)} opportunities found")
        return results

    def run_beginner_scan(self, max_price: float = 50.0) -> Dict[str, ScreeningResult]:
        """
        Scan for beginner-friendly affordable stocks under $50
        """
        logger.info("🔍 Running beginner-friendly stock scan")

        # Affordable stock universe
        affordable_stocks = [
            # Tech/Growth (Under $50)
            "SOFI",
            "PLTR",
            "NIO",
            "RIVN",
            "LCID",
            "PLUG",
            "RIOT",
            "MARA",
            "WISH",
            "DKNG",
            "SKLZ",
            "OPEN",
            "RBLX",
            # Established (Under $50)
            "F",
            "BAC",
            "WFC",
            "PFE",
            "T",
            "VZ",
            "INTC",
            "AAL",
            "CCL",
            "GE",
            "GM",
            "VALE",
            "NOK",
            "ERIC",
            # Popular retail stocks
            "AMC",
            "BB",
            "SNDL",
            "TLRY",
            "CGC",
            "ACB",
            # Penny stocks with potential
            "GNUS",
            "SENS",
            "OCGN",
            "WKHS",
            "RIDE",
        ]

        results = {}

        for symbol in affordable_stocks:
            try:
                result = self._screen_symbol(symbol)

                if result and hasattr(result, "current_price"):
                    if result.current_price <= max_price:
                        results[symbol] = result

            except Exception as e:
                logger.debug(f"Could not screen {symbol}: {e}")
                continue

        logger.info(
            f"✅ Beginner scan complete: {len(results)} affordable stocks found"
        )
        return results

    def _screen_symbol(self, symbol: str) -> Optional[ScreeningResult]:
        """Screen a single symbol and return results"""
        try:
            # Simulate screening (replace with real logic when ready)
            import random

            random.seed(hash(symbol))

            # Generate realistic demo scores
            score = random.uniform(0.4, 0.95)
            current_price = random.uniform(self.price_min, self.price_max)

            # Determine growth category
            if score >= 0.8:
                category = GrowthCategory.EXPLOSIVE_GROWTH
            elif score >= 0.7:
                category = GrowthCategory.HIGH_GROWTH
            elif score >= 0.6:
                category = GrowthCategory.MODERATE_GROWTH
            else:
                category = GrowthCategory.STABLE_GROWTH

            # Calculate target price
            target_multiplier = 1 + (score * 0.3)  # Up to 30% upside
            target_price = current_price * target_multiplier

            result = ScreeningResult(
                symbol=symbol,
                score=score,
                growth_category=category,
                current_price=current_price,
                target_price=target_price,
                technical_score=random.uniform(0.5, 0.9),
                momentum_score=random.uniform(0.5, 0.9),
                volume_score=random.uniform(0.5, 0.9),
                ml_prediction=random.uniform(0.5, 0.9),
                rsi=random.uniform(30, 70),
                macd_signal="BULLISH" if score > 0.6 else "NEUTRAL",
                volume_ratio=random.uniform(1.2, 3.5),
                price_change_1w=random.uniform(-0.1, 0.15),
                criteria_met=[ScreeningCriteria.MOMENTUM_BREAKOUT]
                if score > 0.7
                else [],
                risk_factors=["High volatility"] if score > 0.8 else [],
            )

            return result

        except Exception as e:
            logger.error(f"Failed to screen {symbol}: {e}")
            return None

    def get_screening_status(self) -> Dict:
        """Get current screening status"""
        return {
            "beginner_mode": self.beginner_mode,
            "price_range": f"${self.price_min}-${self.price_max}",
            "min_volume": self.volume_min,
            "total_symbols": 50,
        }


# Singleton instance
_screener_instance = None


def get_screener() -> StockGrowthScreener:
    """Get or create the global screener instance"""
    global _screener_instance

    if _screener_instance is None:
        _screener_instance = StockGrowthScreener()

    return _screener_instance
