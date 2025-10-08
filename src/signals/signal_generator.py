"""Advanced signal generator combining ML predictions with technical analysis"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..ml_models import LightGBMModel, XGBoostModel
from ..utils.technical_indicators import TechnicalIndicators
from .base_signal import BaseSignalGenerator, SignalStrength, SignalType, TradingSignal

logger = logging.getLogger(__name__)


class MLTechnicalSignalGenerator(BaseSignalGenerator):
    """Advanced signal generator combining ML predictions with technical analysis"""

    def __init__(self, **params):
        super().__init__("MLTechnicalSignalGenerator", **params)
        self.technical_indicators = TechnicalIndicators()

        # Initialize models
        self.models = {}
        self._initialize_models()

        # Signal thresholds
        self.thresholds = {
            "strong_buy": params.get("strong_buy_threshold", 0.8),
            "buy": params.get("buy_threshold", 0.6),
            "sell": params.get("sell_threshold", -0.6),
            "strong_sell": params.get("strong_sell_threshold", -0.8),
        }

    def _initialize_models(self):
        """Initialize ML models for prediction"""
        try:
            self.models["xgboost"] = XGBoostModel()
            self.models["lightgbm"] = LightGBMModel()
            self.logger.info("ML models initialized successfully")
        except Exception as e:
            self.logger.warning(f"Error initializing ML models: {e}")

    def generate_signal(
        self, data: pd.DataFrame, symbol: str, **kwargs
    ) -> Optional[TradingSignal]:
        """Generate comprehensive trading signal"""

        if not self.validate_data(data):
            return None

        try:
            # Add technical indicators
            data_with_indicators = self._add_technical_indicators(data)

            # Calculate individual signal components
            technical_signal = self._calculate_technical_signal(data_with_indicators)
            ml_signal = self._calculate_ml_signal(data_with_indicators, symbol)
            volume_signal = self._calculate_volume_signal(data_with_indicators)
            momentum_signal = self._calculate_momentum_signal(data_with_indicators)

            # Combine signals
            combined_score = self._combine_signals(
                technical_signal, ml_signal, volume_signal, momentum_signal
            )

            # Generate final signal
            signal_type, strength = self._determine_signal_type(combined_score)

            if signal_type == SignalType.HOLD:
                return None

            # Calculate confidence
            confidence = self._calculate_signal_confidence(
                technical_signal, ml_signal, volume_signal, momentum_signal
            )

            # Create reasoning
            reasoning = {
                "technical_score": technical_signal["score"],
                "ml_score": ml_signal["score"],
                "volume_score": volume_signal["score"],
                "momentum_score": momentum_signal["score"],
                "combined_score": combined_score,
                "key_indicators": self._get_key_indicators(data_with_indicators),
                "market_conditions": self._assess_market_conditions(
                    data_with_indicators
                ),
            }

            # Create signal
            current_price = float(data_with_indicators["Close"].iloc[-1])
            timestamp = data_with_indicators.index[-1]

            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                price=current_price,
                timestamp=timestamp,
                reasoning=reasoning,
                metadata={
                    "generator": self.name,
                    "data_points": len(data),
                    "indicators_used": list(technical_signal["indicators"].keys()),
                },
            )

            self.logger.info(f"Generated {signal} for {symbol}")
            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to the data"""
        try:
            return self.technical_indicators.add_all_indicators(data.copy())
        except Exception as e:
            self.logger.warning(f"Error adding technical indicators: {e}")
            return data.copy()

    def _calculate_technical_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical analysis signal"""
        try:
            latest = data.iloc[-1]
            signals = {}
            scores = []

            # RSI Signal (30-70 range)
            if "rsi" in data.columns:
                rsi = latest["rsi"]
                if rsi < 30:
                    rsi_signal = 0.8  # Oversold - Buy signal
                elif rsi > 70:
                    rsi_signal = -0.8  # Overbought - Sell signal
                else:
                    rsi_signal = (50 - rsi) / 20  # Neutral zone
                signals["rsi"] = rsi_signal
                scores.append(rsi_signal)

            # MACD Signal
            if "macd" in data.columns and "macd_signal" in data.columns:
                macd_diff = latest["macd"] - latest["macd_signal"]
                macd_signal = np.tanh(macd_diff * 2)  # Normalize between -1 and 1
                signals["macd"] = macd_signal
                scores.append(macd_signal)

            # Bollinger Bands Signal
            if "bb_position" in data.columns:
                bb_pos = latest["bb_position"]
                if bb_pos > 1:
                    bb_signal = -0.6  # Above upper band - sell
                elif bb_pos < 0:
                    bb_signal = 0.6  # Below lower band - buy
                else:
                    bb_signal = (0.5 - bb_pos) * 0.8
                signals["bollinger"] = bb_signal
                scores.append(bb_signal)

            # Moving Average Crossover
            if "sma_20" in data.columns and "sma_50" in data.columns:
                price = latest["Close"]
                sma_20 = latest["sma_20"]
                sma_50 = latest["sma_50"]

                if price > sma_20 > sma_50:
                    ma_signal = 0.7  # Strong uptrend
                elif price < sma_20 < sma_50:
                    ma_signal = -0.7  # Strong downtrend
                else:
                    ma_signal = (sma_20 - sma_50) / price * 10

                signals["moving_averages"] = ma_signal
                scores.append(ma_signal)

            # Stochastic Oscillator
            if "stoch_k" in data.columns:
                stoch = latest["stoch_k"]
                if stoch < 20:
                    stoch_signal = 0.6  # Oversold
                elif stoch > 80:
                    stoch_signal = -0.6  # Overbought
                else:
                    stoch_signal = (50 - stoch) / 30
                signals["stochastic"] = stoch_signal
                scores.append(stoch_signal)

            # Calculate overall technical score
            overall_score = np.mean(scores) if scores else 0.0

            return {
                "score": overall_score,
                "indicators": signals,
                "strength": abs(overall_score),
            }

        except Exception as e:
            self.logger.warning(f"Error in technical signal calculation: {e}")
            return {"score": 0.0, "indicators": {}, "strength": 0.0}

    def _calculate_ml_signal(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Calculate ML prediction signal"""
        try:
            # This is a placeholder for now - in full implementation,
            # we would use trained models to predict price direction

            # For now, let's use a simple trend-based prediction
            if len(data) < 10:
                return {"score": 0.0, "predictions": {}, "confidence": 0.0}

            # Calculate short-term trend
            recent_prices = data["Close"].tail(5).values
            price_changes = np.diff(recent_prices)
            trend_score = np.mean(price_changes) / recent_prices[-1] * 100

            # Normalize trend score
            ml_score = np.tanh(trend_score)

            predictions = {
                "trend_direction": "up" if ml_score > 0 else "down",
                "trend_strength": abs(ml_score),
                "raw_trend_score": trend_score,
            }

            return {
                "score": ml_score,
                "predictions": predictions,
                "confidence": abs(ml_score),
            }

        except Exception as e:
            self.logger.warning(f"Error in ML signal calculation: {e}")
            return {"score": 0.0, "predictions": {}, "confidence": 0.0}

    def _calculate_volume_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume-based signal"""
        try:
            if "volume_ratio" not in data.columns:
                return {"score": 0.0, "analysis": {}, "strength": 0.0}

            latest = data.iloc[-1]
            volume_ratio = latest["volume_ratio"]

            # High volume supports the move
            if volume_ratio > 1.5:
                volume_strength = 0.8
            elif volume_ratio > 1.2:
                volume_strength = 0.4
            elif volume_ratio < 0.8:
                volume_strength = -0.3  # Low volume - weak signal
            else:
                volume_strength = 0.1

            # Check for volume trend
            if len(data) >= 5:
                recent_volume = data["Volume"].tail(5)
                volume_trend = (
                    recent_volume.iloc[-1] - recent_volume.iloc[0]
                ) / recent_volume.iloc[0]
                volume_trend_score = np.tanh(volume_trend)
            else:
                volume_trend_score = 0.0

            # Combine volume signals
            volume_score = (volume_strength + volume_trend_score) / 2

            analysis = {
                "volume_ratio": volume_ratio,
                "volume_strength": volume_strength,
                "volume_trend": volume_trend_score,
                "current_volume": latest["Volume"],
            }

            return {
                "score": volume_score,
                "analysis": analysis,
                "strength": abs(volume_score),
            }

        except Exception as e:
            self.logger.warning(f"Error in volume signal calculation: {e}")
            return {"score": 0.0, "analysis": {}, "strength": 0.0}

    def _calculate_momentum_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum-based signal"""
        try:
            momentum_score = 0.0
            momentum_indicators = {}

            # Price momentum (rate of change)
            if len(data) >= 10:
                current_price = data["Close"].iloc[-1]
                past_price = data["Close"].iloc[-10]
                price_momentum = (current_price - past_price) / past_price
                momentum_score += np.tanh(price_momentum * 5)
                momentum_indicators["price_momentum"] = price_momentum

            # Williams %R momentum
            if "williams_r" in data.columns:
                williams = data["williams_r"].iloc[-1]
                if williams < -80:
                    williams_signal = 0.6  # Oversold momentum
                elif williams > -20:
                    williams_signal = -0.6  # Overbought momentum
                else:
                    williams_signal = (williams + 50) / 30
                momentum_score += williams_signal * 0.3
                momentum_indicators["williams_r"] = williams_signal

            # Average momentum score
            momentum_score = momentum_score / 2 if momentum_indicators else 0.0

            return {
                "score": momentum_score,
                "indicators": momentum_indicators,
                "strength": abs(momentum_score),
            }

        except Exception as e:
            self.logger.warning(f"Error in momentum signal calculation: {e}")
            return {"score": 0.0, "indicators": {}, "strength": 0.0}

    def _combine_signals(
        self,
        technical: Dict[str, Any],
        ml: Dict[str, Any],
        volume: Dict[str, Any],
        momentum: Dict[str, Any],
    ) -> float:
        """Combine all signal components into final score"""

        # Weights for different signal types
        weights = {
            "technical": self.params.get("technical_weight", 0.4),
            "ml": self.params.get("ml_weight", 0.3),
            "volume": self.params.get("volume_weight", 0.2),
            "momentum": self.params.get("momentum_weight", 0.1),
        }

        # Calculate weighted score
        combined_score = (
            technical["score"] * weights["technical"]
            + ml["score"] * weights["ml"]
            + volume["score"] * weights["volume"]
            + momentum["score"] * weights["momentum"]
        )

        # Apply volume confirmation
        if volume["strength"] < 0.2:
            combined_score *= 0.7  # Reduce signal strength for low volume

        return combined_score

    def _determine_signal_type(
        self, combined_score: float
    ) -> tuple[SignalType, SignalStrength]:
        """Determine signal type and strength from combined score"""

        if combined_score >= self.thresholds["strong_buy"]:
            return SignalType.STRONG_BUY, SignalStrength.VERY_STRONG
        elif combined_score >= self.thresholds["buy"]:
            return (
                SignalType.BUY,
                SignalStrength.STRONG
                if combined_score > 0.7
                else SignalStrength.MODERATE,
            )
        elif combined_score <= self.thresholds["strong_sell"]:
            return SignalType.STRONG_SELL, SignalStrength.VERY_STRONG
        elif combined_score <= self.thresholds["sell"]:
            return (
                SignalType.SELL,
                SignalStrength.STRONG
                if combined_score < -0.7
                else SignalStrength.MODERATE,
            )
        else:
            return SignalType.HOLD, SignalStrength.WEAK

    def _calculate_signal_confidence(
        self,
        technical: Dict[str, Any],
        ml: Dict[str, Any],
        volume: Dict[str, Any],
        momentum: Dict[str, Any],
    ) -> float:
        """Calculate overall confidence in the signal"""

        # Base confidence on signal agreement
        scores = [technical["score"], ml["score"], volume["score"], momentum["score"]]

        # Check if signals agree (same direction)
        positive_signals = sum(1 for score in scores if score > 0.1)
        negative_signals = sum(1 for score in scores if score < -0.1)

        agreement_ratio = max(positive_signals, negative_signals) / len(scores)

        # Base confidence on average strength
        avg_strength = np.mean(
            [
                technical["strength"],
                ml.get("confidence", 0.0),
                volume["strength"],
                momentum["strength"],
            ]
        )

        # Combine agreement and strength
        confidence = agreement_ratio * 0.6 + avg_strength * 0.4

        return np.clip(confidence, 0.0, 1.0)

    def _get_key_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get key indicator values for reasoning"""
        latest = data.iloc[-1]
        indicators = {}

        key_cols = [
            "rsi",
            "macd",
            "bb_position",
            "volume_ratio",
            "stoch_k",
            "williams_r",
        ]
        for col in key_cols:
            if col in data.columns:
                indicators[col] = float(latest[col])

        return indicators

    def _assess_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current market conditions"""
        conditions = {}

        try:
            # Volatility assessment
            if len(data) >= 20:
                recent_returns = data["Close"].pct_change().tail(20)
                volatility = recent_returns.std() * np.sqrt(252)  # Annualized

                if volatility > 0.4:
                    conditions["volatility"] = "high"
                elif volatility > 0.2:
                    conditions["volatility"] = "moderate"
                else:
                    conditions["volatility"] = "low"

            # Trend assessment
            if "sma_20" in data.columns and "sma_50" in data.columns:
                latest = data.iloc[-1]
                if latest["Close"] > latest["sma_20"] > latest["sma_50"]:
                    conditions["trend"] = "strong_uptrend"
                elif latest["Close"] < latest["sma_20"] < latest["sma_50"]:
                    conditions["trend"] = "strong_downtrend"
                elif latest["Close"] > latest["sma_20"]:
                    conditions["trend"] = "uptrend"
                elif latest["Close"] < latest["sma_20"]:
                    conditions["trend"] = "downtrend"
                else:
                    conditions["trend"] = "sideways"

            # Market phase
            if "rsi" in data.columns:
                rsi = data["rsi"].iloc[-1]
                if rsi > 70:
                    conditions["market_phase"] = "overbought"
                elif rsi < 30:
                    conditions["market_phase"] = "oversold"
                else:
                    conditions["market_phase"] = "normal"

        except Exception as e:
            self.logger.warning(f"Error assessing market conditions: {e}")

        return conditions
