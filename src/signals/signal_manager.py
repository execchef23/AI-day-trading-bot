"""Signal manager for coordinating multiple signal generators"""

from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from datetime import datetime

from .base_signal import TradingSignal, SignalType, SignalStrength
from .signal_generator import MLTechnicalSignalGenerator

logger = logging.getLogger(__name__)


class SignalManager:
    """Manages multiple signal generators and provides consolidated signals"""
    
    def __init__(self, **params):
        self.params = params
        self.generators = {}
        self.signal_history = {}
        
        # Initialize default signal generators
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize signal generators"""
        try:
            # Main ML + Technical signal generator
            self.generators['ml_technical'] = MLTechnicalSignalGenerator(**self.params)
            
            logger.info(f"Initialized {len(self.generators)} signal generators")
        except Exception as e:
            logger.error(f"Error initializing signal generators: {e}")
    
    def add_generator(self, name: str, generator) -> None:
        """Add a custom signal generator"""
        self.generators[name] = generator
        logger.info(f"Added signal generator: {name}")
    
    def generate_signals(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        **kwargs
    ) -> Dict[str, Optional[TradingSignal]]:
        """Generate signals from all active generators"""
        signals = {}
        
        for name, generator in self.generators.items():
            try:
                signal = generator.generate_signal(data, symbol, **kwargs)
                signals[name] = signal
                
                if signal:
                    logger.debug(f"Generated signal from {name}: {signal}")
                
            except Exception as e:
                logger.error(f"Error generating signal from {name}: {e}")
                signals[name] = None
        
        # Store in history
        self._store_signal_history(symbol, signals)
        
        return signals
    
    def get_consensus_signal(
        self, 
        data: pd.DataFrame, 
        symbol: str,
        **kwargs
    ) -> Optional[TradingSignal]:
        """Generate a consensus signal from all generators"""
        
        signals = self.generate_signals(data, symbol, **kwargs)
        
        # Filter out None signals
        valid_signals = {name: signal for name, signal in signals.items() if signal is not None}
        
        if not valid_signals:
            return None
        
        # If only one signal, return it
        if len(valid_signals) == 1:
            return list(valid_signals.values())[0]
        
        # Calculate consensus
        return self._calculate_consensus(valid_signals, data, symbol)
    
    def _calculate_consensus(
        self, 
        signals: Dict[str, TradingSignal], 
        data: pd.DataFrame, 
        symbol: str
    ) -> Optional[TradingSignal]:
        """Calculate consensus signal from multiple generators"""
        
        if not signals:
            return None
        
        # Analyze signal agreement
        signal_types = [signal.signal_type for signal in signals.values()]
        confidences = [signal.confidence for signal in signals.values()]
        
        # Count signal types
        type_counts = {}
        for signal_type in signal_types:
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
        
        # Find most common signal type
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # If no consensus (all different), return None
        if type_counts[most_common_type] == 1 and len(signals) > 1:
            return None
        
        # Calculate consensus confidence
        same_type_signals = [s for s in signals.values() if s.signal_type == most_common_type]
        avg_confidence = sum(s.confidence for s in same_type_signals) / len(same_type_signals)
        
        # Reduce confidence if there's disagreement
        agreement_ratio = len(same_type_signals) / len(signals)
        consensus_confidence = avg_confidence * agreement_ratio
        
        # Determine strength based on consensus
        if consensus_confidence > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif consensus_confidence > 0.6:
            strength = SignalStrength.STRONG
        elif consensus_confidence > 0.4:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Create consensus reasoning
        reasoning = {
            'consensus_type': most_common_type.value,
            'agreement_ratio': agreement_ratio,
            'individual_signals': {
                name: {
                    'type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'strength': signal.strength.value
                }
                for name, signal in signals.items()
            },
            'consensus_method': 'majority_vote_with_confidence_weighting'
        }
        
        # Use the most confident signal's metadata as base
        best_signal = max(signals.values(), key=lambda x: x.confidence)
        
        # Create consensus signal
        consensus_signal = TradingSignal(
            symbol=symbol,
            signal_type=most_common_type,
            strength=strength,
            confidence=consensus_confidence,
            price=best_signal.price,
            timestamp=best_signal.timestamp,
            reasoning=reasoning,
            metadata={
                'generator': 'SignalManager_Consensus',
                'generators_used': list(signals.keys()),
                'num_signals': len(signals),
                'agreement_ratio': agreement_ratio
            }
        )
        
        logger.info(f"Generated consensus signal: {consensus_signal}")
        return consensus_signal
    
    def _store_signal_history(self, symbol: str, signals: Dict[str, Optional[TradingSignal]]):
        """Store signal history for analysis"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        # Store with timestamp
        entry = {
            'timestamp': datetime.now(),
            'signals': signals
        }
        
        self.signal_history[symbol].append(entry)
        
        # Keep only last 100 signals per symbol
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol] = self.signal_history[symbol][-100:]
    
    def get_signal_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get recent signal history for a symbol"""
        if symbol not in self.signal_history:
            return []
        
        return self.signal_history[symbol][-limit:]
    
    def get_performance_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get performance metrics for signal generators"""
        history = self.signal_history.get(symbol, [])
        
        if not history:
            return {}
        
        metrics = {
            'total_signals': len(history),
            'generator_activity': {},
            'signal_type_distribution': {},
            'average_confidence': {}
        }
        
        # Analyze generator activity
        for entry in history:
            for gen_name, signal in entry['signals'].items():
                if gen_name not in metrics['generator_activity']:
                    metrics['generator_activity'][gen_name] = 0
                
                if signal is not None:
                    metrics['generator_activity'][gen_name] += 1
                    
                    # Signal type distribution
                    signal_type = signal.signal_type.value
                    if signal_type not in metrics['signal_type_distribution']:
                        metrics['signal_type_distribution'][signal_type] = 0
                    metrics['signal_type_distribution'][signal_type] += 1
                    
                    # Average confidence
                    if gen_name not in metrics['average_confidence']:
                        metrics['average_confidence'][gen_name] = []
                    metrics['average_confidence'][gen_name].append(signal.confidence)
        
        # Calculate average confidences
        for gen_name, confidences in metrics['average_confidence'].items():
            if confidences:
                metrics['average_confidence'][gen_name] = sum(confidences) / len(confidences)
            else:
                metrics['average_confidence'][gen_name] = 0.0
        
        return metrics
    
    def reset_history(self, symbol: Optional[str] = None):
        """Reset signal history"""
        if symbol:
            if symbol in self.signal_history:
                del self.signal_history[symbol]
        else:
            self.signal_history.clear()
        
        logger.info(f"Reset signal history for {'all symbols' if not symbol else symbol}")