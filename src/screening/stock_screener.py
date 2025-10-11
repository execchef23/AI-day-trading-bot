"""
Stock Growth Screener

Advanced stock screening system to identify high-growth opportunities using
technical analysis, momentum indicators, volume analysis, and ML predictions.
"""

import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScreeningCriteria(Enum):
    """Available screening criteria"""
    HIGH_MOMENTUM = "high_momentum"
    VOLUME_SURGE = "volume_surge"
    TECHNICAL_BREAKOUT = "technical_breakout"
    GAP_UP = "gap_up"
    EARNINGS_MOMENTUM = "earnings_momentum"
    PRICE_STRENGTH = "price_strength"
    LOW_RSI_RECOVERY = "low_rsi_recovery"
    BULLISH_DIVERGENCE = "bullish_divergence"


class GrowthCategory(Enum):
    """Growth stock categories"""
    EXPLOSIVE_GROWTH = "explosive_growth"  # >50% potential
    HIGH_GROWTH = "high_growth"           # 25-50% potential
    MODERATE_GROWTH = "moderate_growth"   # 10-25% potential
    STABLE_GROWTH = "stable_growth"       # 5-10% potential
    LOW_GROWTH = "low_growth"            # <5% potential


@dataclass
class ScreenResult:
    """Individual stock screening result"""

    symbol: str
    score: float
    growth_category: GrowthCategory
    current_price: float
    target_price: Optional[float] = None

    # Growth metrics
    momentum_score: float = 0.0
    volume_score: float = 0.0
    technical_score: float = 0.0
    ml_prediction: float = 0.0

    # Key indicators
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    volume_ratio: Optional[float] = None
    price_change_1w: Optional[float] = None
    price_change_1m: Optional[float] = None

    # Reasons for selection
    criteria_met: List[ScreeningCriteria] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "score": round(self.score, 3),
            "growth_category": self.growth_category.value,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "momentum_score": round(self.momentum_score, 3),
            "volume_score": round(self.volume_score, 3),
            "technical_score": round(self.technical_score, 3),
            "ml_prediction": round(self.ml_prediction, 3),
            "rsi": self.rsi,
            "macd_signal": self.macd_signal,
            "volume_ratio": self.volume_ratio,
            "price_change_1w": self.price_change_1w,
            "price_change_1m": self.price_change_1m,
            "criteria_met": [c.value for c in self.criteria_met],
            "risk_factors": self.risk_factors,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScreeningConfig:
    """Configuration for stock screening"""

    # Screening universe
    symbol_list: List[str] = field(default_factory=lambda: [
        # Technology
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "ADBE", "CRM",
        # Growth stocks
        "SHOP", "SQ", "ROKU", "ZOOM", "DOCU", "SNOW", "PLTR", "COIN", "RBLX", "U",
        # Biotech & Healthcare
        "MRNA", "PFE", "JNJ", "UNH", "ABBV", "BMY", "GILD", "BIIB", "REGN", "VRTX",
        # Financial
        "JPM", "BAC", "GS", "MS", "C", "WFC", "V", "MA", "PYPL", "AFRM",
        # Energy & Materials
        "XOM", "CVX", "COP", "EOG", "SLB", "FCX", "NEM", "AA", "X", "CLF",
        # Consumer
        "DIS", "NKE", "SBUX", "MCD", "HD", "LOW", "TGT", "WMT", "COST", "AMGN"
    ])

    # Screening thresholds
    min_momentum_score: float = 0.6        # Minimum momentum requirement
    min_volume_ratio: float = 1.5          # Volume surge threshold (1.5x average)
    min_price_change_1w: float = 0.05      # 5% minimum weekly gain
    max_rsi_oversold: float = 35           # RSI recovery threshold
    min_technical_score: float = 0.5       # Technical analysis minimum

    # Growth categorization thresholds
    explosive_growth_threshold: float = 0.9
    high_growth_threshold: float = 0.75
    moderate_growth_threshold: float = 0.6
    stable_growth_threshold: float = 0.45

    # Risk management
    max_volatility: float = 0.6            # Maximum annualized volatility
    min_liquidity_threshold: float = 1000000  # Minimum daily volume ($)

    # ML prediction weight
    ml_weight: float = 0.3
    technical_weight: float = 0.4
    momentum_weight: float = 0.2
    volume_weight: float = 0.1


class StockGrowthScreener:
    """Advanced stock growth screening system"""

    def __init__(self, config: Optional[ScreeningConfig] = None):
        self.config = config or ScreeningConfig()
        self.last_scan_time = None
        self.scan_results = {}
        self.scan_history = defaultdict(list)

        # Initialize components
        self.data_manager = None
        self.signal_manager = None

        logger.info("Stock Growth Screener initialized")

    def set_components(self, data_manager=None, signal_manager=None):
        """Set external components"""
        self.data_manager = data_manager
        self.signal_manager = signal_manager
        logger.info("Screener components configured")

    def run_full_scan(self) -> Dict[str, ScreenResult]:
        """Run comprehensive screening across all symbols"""

        logger.info(f"🔍 Starting full growth stock scan of {len(self.config.symbol_list)} symbols")
        start_time = time.time()

        results = {}
        failed_symbols = []

        for i, symbol in enumerate(self.config.symbol_list):
            try:
                result = self._screen_symbol(symbol)
                if result:
                    results[symbol] = result
                    logger.debug(f"✅ Screened {symbol}: Score {result.score:.3f}")

                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(self.config.symbol_list)} symbols processed")

            except Exception as e:
                failed_symbols.append(symbol)
                logger.warning(f"❌ Failed to screen {symbol}: {e}")

        # Store results
        self.scan_results = results
        self.last_scan_time = datetime.now()

        # Log summary
        duration = time.time() - start_time
        success_count = len(results)
        logger.info(f"🎯 Scan complete: {success_count} successful, {len(failed_symbols)} failed in {duration:.1f}s")

        if failed_symbols:
            logger.warning(f"Failed symbols: {', '.join(failed_symbols[:5])}")

        return results

    def _screen_symbol(self, symbol: str) -> Optional[ScreenResult]:
        """Screen individual symbol for growth potential"""

        try:
            # Get market data
            if not self.data_manager:
                logger.warning("Data manager not available")
                return None

            # Get data for analysis (1 month for momentum, 1 week for recent activity)
            data_1m = self.data_manager.get_historical_data(symbol, period="1mo")
            data_1w = self.data_manager.get_historical_data(symbol, period="1wk")

            if data_1m is None or len(data_1m) < 20:
                return None

            # Add technical indicators
            from src.utils.technical_indicators import TechnicalIndicators
            data_with_indicators = TechnicalIndicators.add_all_indicators(data_1m)

            # Calculate screening components
            momentum_analysis = self._analyze_momentum(data_with_indicators, data_1w)
            volume_analysis = self._analyze_volume(data_with_indicators)
            technical_analysis = self._analyze_technical(data_with_indicators)
            ml_analysis = self._get_ml_prediction(data_with_indicators, symbol)

            # Calculate composite score
            composite_score = (
                momentum_analysis["score"] * self.config.momentum_weight +
                volume_analysis["score"] * self.config.volume_weight +
                technical_analysis["score"] * self.config.technical_weight +
                ml_analysis["score"] * self.config.ml_weight
            )

            # Determine growth category
            growth_category = self._categorize_growth(composite_score)

            # Check screening criteria
            criteria_met = self._check_criteria(
                data_with_indicators, momentum_analysis, volume_analysis, technical_analysis
            )

            # Risk assessment
            risk_factors = self._assess_risks(data_with_indicators)

            # Get current metrics
            current_price = float(data_with_indicators['Close'].iloc[-1])
            latest_data = data_with_indicators.iloc[-1]

            # Calculate target price (simple estimate)
            target_price = current_price * (1 + composite_score * 0.5) if composite_score > 0.5 else None

            # Weekly and monthly returns
            price_change_1w = None
            price_change_1m = None

            if len(data_with_indicators) >= 5:
                price_1w_ago = data_with_indicators['Close'].iloc[-5]
                price_change_1w = (current_price - price_1w_ago) / price_1w_ago

            if len(data_with_indicators) >= 20:
                price_1m_ago = data_with_indicators['Close'].iloc[-20]
                price_change_1m = (current_price - price_1m_ago) / price_1m_ago

            # Create result
            result = ScreenResult(
                symbol=symbol,
                score=composite_score,
                growth_category=growth_category,
                current_price=current_price,
                target_price=target_price,
                momentum_score=momentum_analysis["score"],
                volume_score=volume_analysis["score"],
                technical_score=technical_analysis["score"],
                ml_prediction=ml_analysis["score"],
                rsi=float(latest_data.get('rsi', 0)) if 'rsi' in latest_data else None,
                macd_signal="BULLISH" if latest_data.get('macd', 0) > latest_data.get('macd_signal', 0) else "BEARISH",
                volume_ratio=volume_analysis.get("volume_ratio"),
                price_change_1w=price_change_1w,
                price_change_1m=price_change_1m,
                criteria_met=criteria_met,
                risk_factors=risk_factors
            )

            return result

        except Exception as e:
            logger.error(f"Error screening {symbol}: {e}")
            return None

    def _analyze_momentum(self, data: pd.DataFrame, data_1w: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze price momentum"""

        try:
            momentum_score = 0.0

            # Price momentum over different periods
            current_price = data['Close'].iloc[-1]

            # 5-day momentum
            if len(data) >= 5:
                price_5d_ago = data['Close'].iloc[-5]
                momentum_5d = (current_price - price_5d_ago) / price_5d_ago
                momentum_score += np.tanh(momentum_5d * 10) * 0.3

            # 10-day momentum
            if len(data) >= 10:
                price_10d_ago = data['Close'].iloc[-10]
                momentum_10d = (current_price - price_10d_ago) / price_10d_ago
                momentum_score += np.tanh(momentum_10d * 5) * 0.4

            # 20-day momentum
            if len(data) >= 20:
                price_20d_ago = data['Close'].iloc[-20]
                momentum_20d = (current_price - price_20d_ago) / price_20d_ago
                momentum_score += np.tanh(momentum_20d * 3) * 0.3

            # Technical momentum indicators
            if 'roc' in data.columns:
                roc = data['roc'].iloc[-1]
                if not pd.isna(roc):
                    momentum_score += np.tanh(roc / 10) * 0.2

            # Normalize score
            momentum_score = np.clip(momentum_score, -1, 1)

            return {
                "score": momentum_score,
                "strength": abs(momentum_score),
                "direction": "UP" if momentum_score > 0 else "DOWN"
            }

        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {"score": 0.0, "strength": 0.0, "direction": "NEUTRAL"}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns"""

        try:
            volume_score = 0.0

            current_volume = data['Volume'].iloc[-1]

            # Volume moving average comparison
            if 'volume_ma' in data.columns:
                volume_ma = data['volume_ma'].iloc[-1]
                if volume_ma > 0:
                    volume_ratio = current_volume / volume_ma
                    volume_score += np.tanh((volume_ratio - 1) * 2) * 0.6
                else:
                    volume_ratio = 1.0
            else:
                # Calculate simple volume average
                avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                volume_score += np.tanh((volume_ratio - 1) * 2) * 0.6

            # On-Balance Volume trend
            if 'obv' in data.columns and len(data) >= 5:
                obv_current = data['obv'].iloc[-1]
                obv_5d_ago = data['obv'].iloc[-5]
                obv_trend = (obv_current - obv_5d_ago) / abs(obv_5d_ago) if obv_5d_ago != 0 else 0
                volume_score += np.tanh(obv_trend * 5) * 0.4

            # Normalize score
            volume_score = np.clip(volume_score, -1, 1)

            return {
                "score": volume_score,
                "volume_ratio": volume_ratio,
                "strength": abs(volume_score)
            }

        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"score": 0.0, "volume_ratio": 1.0, "strength": 0.0}

    def _analyze_technical(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators"""

        try:
            technical_score = 0.0
            signals = {}

            latest = data.iloc[-1]

            # RSI analysis
            if 'rsi' in data.columns:
                rsi = latest['rsi']
                if rsi < 30:
                    rsi_signal = 0.8  # Oversold recovery potential
                elif rsi > 70:
                    rsi_signal = -0.3  # Overbought but could continue
                else:
                    rsi_signal = (50 - rsi) / 20
                technical_score += rsi_signal * 0.25
                signals['rsi'] = rsi_signal

            # MACD analysis
            if 'macd' in data.columns and 'macd_signal' in data.columns:
                macd = latest['macd']
                macd_signal = latest['macd_signal']
                macd_diff = macd - macd_signal
                macd_score = np.tanh(macd_diff * 2)
                technical_score += macd_score * 0.3
                signals['macd'] = macd_score

            # Moving average analysis
            if 'sma_20' in data.columns and 'sma_50' in data.columns:
                price = latest['Close']
                sma_20 = latest['sma_20']
                sma_50 = latest['sma_50']

                # Price above moving averages
                ma_score = 0
                if price > sma_20:
                    ma_score += 0.5
                if price > sma_50:
                    ma_score += 0.3
                if sma_20 > sma_50:
                    ma_score += 0.2

                technical_score += ma_score * 0.25
                signals['moving_averages'] = ma_score

            # Bollinger Bands
            if 'bb_position' in data.columns:
                bb_pos = latest['bb_position']
                if bb_pos < 0.2:
                    bb_signal = 0.6  # Near lower band - potential bounce
                elif bb_pos > 0.8:
                    bb_signal = -0.2  # Near upper band
                else:
                    bb_signal = (bb_pos - 0.5) * 0.4
                technical_score += bb_signal * 0.2
                signals['bollinger'] = bb_signal

            # Normalize score
            technical_score = np.clip(technical_score, -1, 1)

            return {
                "score": technical_score,
                "signals": signals,
                "strength": abs(technical_score)
            }

        except Exception as e:
            logger.error(f"Error analyzing technical indicators: {e}")
            return {"score": 0.0, "signals": {}, "strength": 0.0}

    def _get_ml_prediction(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get ML model prediction"""

        try:
            if not self.signal_manager:
                return {"score": 0.0, "prediction": None}

            # Generate signal using ML
            signal = self.signal_manager.get_consensus_signal(data, symbol)

            if signal and hasattr(signal, 'strength'):
                # Convert signal strength to score
                ml_score = signal.strength
                if hasattr(signal, 'signal_type'):
                    if 'SELL' in str(signal.signal_type):
                        ml_score *= -1

                return {
                    "score": ml_score,
                    "prediction": str(signal.signal_type) if hasattr(signal, 'signal_type') else None,
                    "confidence": signal.confidence if hasattr(signal, 'confidence') else 0
                }

            return {"score": 0.0, "prediction": None}

        except Exception as e:
            logger.debug(f"ML prediction not available for {symbol}: {e}")
            return {"score": 0.0, "prediction": None}

    def _categorize_growth(self, score: float) -> GrowthCategory:
        """Categorize growth potential based on composite score"""

        if score >= self.config.explosive_growth_threshold:
            return GrowthCategory.EXPLOSIVE_GROWTH
        elif score >= self.config.high_growth_threshold:
            return GrowthCategory.HIGH_GROWTH
        elif score >= self.config.moderate_growth_threshold:
            return GrowthCategory.MODERATE_GROWTH
        elif score >= self.config.stable_growth_threshold:
            return GrowthCategory.STABLE_GROWTH
        else:
            return GrowthCategory.LOW_GROWTH

    def _check_criteria(self, data: pd.DataFrame, momentum: Dict, volume: Dict, technical: Dict) -> List[ScreeningCriteria]:
        """Check which screening criteria are met"""

        criteria_met = []

        try:
            # High momentum check
            if momentum["score"] >= self.config.min_momentum_score:
                criteria_met.append(ScreeningCriteria.HIGH_MOMENTUM)

            # Volume surge check
            if volume.get("volume_ratio", 1.0) >= self.config.min_volume_ratio:
                criteria_met.append(ScreeningCriteria.VOLUME_SURGE)

            # Technical breakout check
            if technical["score"] >= self.config.min_technical_score:
                criteria_met.append(ScreeningCriteria.TECHNICAL_BREAKOUT)

            # Gap analysis
            if len(data) >= 2:
                today_open = data['Open'].iloc[-1]
                yesterday_close = data['Close'].iloc[-2]
                gap_pct = (today_open - yesterday_close) / yesterday_close

                if gap_pct > 0.02:  # 2% gap up
                    criteria_met.append(ScreeningCriteria.GAP_UP)

            # RSI recovery check
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi <= self.config.max_rsi_oversold and momentum["score"] > 0:
                    criteria_met.append(ScreeningCriteria.LOW_RSI_RECOVERY)

            # Price strength check (consistent upward movement)
            if len(data) >= 5:
                recent_returns = data['Close'].pct_change().tail(5)
                positive_days = (recent_returns > 0).sum()
                if positive_days >= 3:
                    criteria_met.append(ScreeningCriteria.PRICE_STRENGTH)

        except Exception as e:
            logger.error(f"Error checking criteria: {e}")

        return criteria_met

    def _assess_risks(self, data: pd.DataFrame) -> List[str]:
        """Assess risk factors"""

        risk_factors = []

        try:
            # High volatility risk
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized

                if volatility > self.config.max_volatility:
                    risk_factors.append(f"High volatility ({volatility:.1%})")

            # Low liquidity risk
            avg_volume = data['Volume'].mean()
            avg_price = data['Close'].mean()
            daily_dollar_volume = avg_volume * avg_price

            if daily_dollar_volume < self.config.min_liquidity_threshold:
                risk_factors.append(f"Low liquidity (${daily_dollar_volume:,.0f}/day)")

            # Overbought risk
            if 'rsi' in data.columns:
                rsi = data['rsi'].iloc[-1]
                if rsi > 80:
                    risk_factors.append(f"Severely overbought (RSI: {rsi:.1f})")

            # Extended move risk
            if len(data) >= 10:
                current_price = data['Close'].iloc[-1]
                price_10d_ago = data['Close'].iloc[-10]
                move_pct = (current_price - price_10d_ago) / price_10d_ago

                if move_pct > 0.5:  # 50%+ move in 10 days
                    risk_factors.append(f"Extended move ({move_pct:.1%} in 10 days)")

        except Exception as e:
            logger.error(f"Error assessing risks: {e}")

        return risk_factors

    def get_top_opportunities(self, limit: int = 20, min_score: float = 0.6) -> List[ScreenResult]:
        """Get top growth opportunities"""

        if not self.scan_results:
            logger.warning("No scan results available. Run scan first.")
            return []

        # Filter and sort results
        filtered_results = [
            result for result in self.scan_results.values()
            if result.score >= min_score
        ]

        # Sort by score (descending)
        sorted_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)

        return sorted_results[:limit]

    def get_results_by_category(self, category: GrowthCategory) -> List[ScreenResult]:
        """Get results filtered by growth category"""

        return [
            result for result in self.scan_results.values()
            if result.growth_category == category
        ]

    def get_screening_summary(self) -> Dict[str, Any]:
        """Get summary of latest screening results"""

        if not self.scan_results:
            return {"status": "No scan results available"}

        total_stocks = len(self.scan_results)

        # Category breakdown
        category_counts = defaultdict(int)
        for result in self.scan_results.values():
            category_counts[result.growth_category.value] += 1

        # Score statistics
        scores = [result.score for result in self.scan_results.values()]

        # Top opportunities
        top_opportunities = self.get_top_opportunities(limit=5)

        return {
            "scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "total_stocks_screened": total_stocks,
            "category_breakdown": dict(category_counts),
            "score_statistics": {
                "mean": np.mean(scores),
                "median": np.median(scores),
                "max": np.max(scores),
                "min": np.min(scores)
            },
            "top_opportunities": [result.symbol for result in top_opportunities]
        }


# Global screener instance
_screener_instance = None


def get_screener() -> StockGrowthScreener:
    """Get or create global screener instance"""
    global _screener_instance

    if _screener_instance is None:
        _screener_instance = StockGrowthScreener()

    return _screener_instance
