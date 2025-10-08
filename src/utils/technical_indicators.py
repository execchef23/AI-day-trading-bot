"""Technical indicators calculation using pandas and numpy"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate various technical indicators for trading analysis"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close_prev = np.abs(high - close.shift())
        low_close_prev = np.abs(low - close.shift())
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    @staticmethod
    def volume_indicators(volume: pd.Series, close: pd.Series, window: int = 20) -> Dict[str, pd.Series]:
        """Volume-based indicators"""
        # Volume Moving Average
        volume_ma = volume.rolling(window=window).mean()
        
        # On-Balance Volume
        obv = (volume * np.sign(close.diff())).cumsum()
        
        # Volume Price Trend
        vpt = (volume * (close.pct_change())).cumsum()
        
        return {
            'volume_ma': volume_ma,
            'obv': obv,
            'vpt': vpt
        }
    
    @staticmethod
    def momentum_indicators(close: pd.Series, window: int = 10) -> Dict[str, pd.Series]:
        """Price momentum indicators"""
        # Rate of Change
        roc = ((close - close.shift(window)) / close.shift(window)) * 100
        
        # Momentum
        momentum = close - close.shift(window)
        
        # Price Rate of Change
        proc = close.pct_change(periods=window) * 100
        
        return {
            'roc': roc,
            'momentum': momentum,
            'proc': proc
        }
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to a dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        df = df.copy()
        
        try:
            # Moving Averages
            df['sma_10'] = TechnicalIndicators.sma(df['Close'], 10)
            df['sma_20'] = TechnicalIndicators.sma(df['Close'], 20)
            df['sma_50'] = TechnicalIndicators.sma(df['Close'], 50)
            df['ema_12'] = TechnicalIndicators.ema(df['Close'], 12)
            df['ema_26'] = TechnicalIndicators.ema(df['Close'], 26)
            
            # RSI
            df['rsi'] = TechnicalIndicators.rsi(df['Close'])
            
            # MACD
            macd_data = TechnicalIndicators.macd(df['Close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = TechnicalIndicators.bollinger_bands(df['Close'])
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
            df['bb_position'] = (df['Close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
            
            # Stochastic Oscillator
            stoch_data = TechnicalIndicators.stochastic_oscillator(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = stoch_data['k_percent']
            df['stoch_d'] = stoch_data['d_percent']
            
            # Williams %R
            df['williams_r'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
            
            # ATR
            df['atr'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
            
            # Volume Indicators
            volume_data = TechnicalIndicators.volume_indicators(df['Volume'], df['Close'])
            df['volume_ma'] = volume_data['volume_ma']
            df['obv'] = volume_data['obv']
            df['vpt'] = volume_data['vpt']
            
            # Momentum Indicators
            momentum_data = TechnicalIndicators.momentum_indicators(df['Close'])
            df['roc'] = momentum_data['roc']
            df['momentum'] = momentum_data['momentum']
            df['proc'] = momentum_data['proc']
            
            # Price-based features
            df['high_low_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
            df['close_open_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            
            # Volatility
            df['volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252) * 100
            
            logger.info(f"Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Symbol']])} technical indicators")
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
        
        return df
    
    @staticmethod
    def get_signal_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate overall signal strength based on multiple indicators
        
        Returns:
            Series with signal strength (-1 to 1, where -1 is strong sell, 1 is strong buy)
        """
        signals = pd.Series(0.0, index=df.index)
        
        try:
            # RSI signals
            signals += np.where(df['rsi'] > 70, -0.2, 0)  # Overbought
            signals += np.where(df['rsi'] < 30, 0.2, 0)   # Oversold
            
            # MACD signals
            signals += np.where(df['macd'] > df['macd_signal'], 0.15, -0.15)
            
            # Moving Average signals
            signals += np.where(df['Close'] > df['sma_20'], 0.1, -0.1)
            signals += np.where(df['sma_10'] > df['sma_20'], 0.1, -0.1)
            
            # Bollinger Bands signals
            signals += np.where(df['bb_position'] > 0.8, -0.1, 0)  # Near upper band
            signals += np.where(df['bb_position'] < 0.2, 0.1, 0)   # Near lower band
            
            # Volume confirmation
            volume_signal = np.where(df['Volume'] > df['volume_ma'], 0.05, -0.05)
            signals += volume_signal
            
            # Clip signals to [-1, 1]
            signals = np.clip(signals, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            signals = pd.Series(0.0, index=df.index)
        
        return signals