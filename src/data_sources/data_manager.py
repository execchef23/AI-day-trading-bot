"""Data manager to coordinate multiple data sources and provide unified interface"""

import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from .yahoo_finance_provider import YahooFinanceProvider
from .alpha_vantage_provider import AlphaVantageProvider
from .polygon_provider import PolygonProvider

logger = logging.getLogger(__name__)

class DataManager:
    """Manages multiple data providers and provides a unified interface"""
    
    def __init__(self, 
                 alpha_vantage_key: Optional[str] = None,
                 polygon_key: Optional[str] = None):
        """Initialize data manager with available providers"""
        
        self.providers = {}
        
        # Always include Yahoo Finance (free)
        self.providers['yahoo'] = YahooFinanceProvider()
        
        # Add Alpha Vantage if API key provided
        if alpha_vantage_key:
            self.providers['alpha_vantage'] = AlphaVantageProvider(alpha_vantage_key)
        
        # Add Polygon if API key provided
        if polygon_key:
            self.providers['polygon'] = PolygonProvider(polygon_key)
        
        # Set primary and fallback providers
        self.primary_provider = 'yahoo'  # Most reliable for free
        self.fallback_providers = ['alpha_vantage', 'polygon'] if len(self.providers) > 1 else []
        
        logger.info(f"DataManager initialized with providers: {list(self.providers.keys())}")
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y", 
                          interval: str = "1d",
                          provider: Optional[str] = None) -> pd.DataFrame:
        """Get historical data with fallback providers"""
        
        providers_to_try = [provider] if provider else [self.primary_provider] + self.fallback_providers
        
        for prov_name in providers_to_try:
            if prov_name not in self.providers:
                continue
                
            try:
                logger.info(f"Fetching historical data for {symbol} from {prov_name}")
                data = self.providers[prov_name].get_historical_data(symbol, period, interval)
                
                if not data.empty:
                    logger.info(f"Successfully fetched {len(data)} records for {symbol} from {prov_name}")
                    return data
                else:
                    logger.warning(f"No data returned from {prov_name} for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error fetching from {prov_name}: {e}")
        
        logger.error(f"Failed to fetch historical data for {symbol} from all providers")
        return pd.DataFrame()
    
    def get_current_price(self, 
                         symbol: str, 
                         provider: Optional[str] = None) -> Optional[float]:
        """Get current price with fallback providers"""
        
        providers_to_try = [provider] if provider else [self.primary_provider] + self.fallback_providers
        
        for prov_name in providers_to_try:
            if prov_name not in self.providers:
                continue
                
            try:
                price = self.providers[prov_name].get_current_price(symbol)
                
                if price is not None and price > 0:
                    logger.debug(f"Got current price for {symbol} from {prov_name}: ${price}")
                    return price
                    
            except Exception as e:
                logger.error(f"Error fetching current price from {prov_name}: {e}")
        
        logger.error(f"Failed to fetch current price for {symbol} from all providers")
        return None
    
    def get_multiple_quotes(self, 
                          symbols: List[str], 
                          provider: Optional[str] = None) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get multiple quotes with fallback providers"""
        
        providers_to_try = [provider] if provider else [self.primary_provider] + self.fallback_providers
        
        for prov_name in providers_to_try:
            if prov_name not in self.providers:
                continue
                
            try:
                quotes = self.providers[prov_name].get_multiple_quotes(symbols)
                
                # Check if we got valid data for most symbols
                valid_quotes = sum(1 for quote in quotes.values() if quote is not None)
                if valid_quotes >= len(symbols) * 0.5:  # At least 50% success rate
                    logger.info(f"Got quotes for {valid_quotes}/{len(symbols)} symbols from {prov_name}")
                    return quotes
                    
            except Exception as e:
                logger.error(f"Error fetching multiple quotes from {prov_name}: {e}")
        
        logger.error(f"Failed to fetch quotes for symbols from all providers")
        return {symbol: None for symbol in symbols}
    
    def get_company_info(self, 
                        symbol: str, 
                        provider: Optional[str] = None) -> Dict[str, Union[str, float, int]]:
        """Get company information with fallback providers"""
        
        providers_to_try = [provider] if provider else [self.primary_provider] + self.fallback_providers
        
        for prov_name in providers_to_try:
            if prov_name not in self.providers:
                continue
                
            try:
                info = self.providers[prov_name].get_company_info(symbol)
                
                if info and 'error' not in info:
                    logger.info(f"Got company info for {symbol} from {prov_name}")
                    return info
                    
            except Exception as e:
                logger.error(f"Error fetching company info from {prov_name}: {e}")
        
        logger.error(f"Failed to fetch company info for {symbol} from all providers")
        return {'symbol': symbol, 'error': 'No data available'}
    
    def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """Validate multiple symbols"""
        results = {}
        
        for symbol in symbols:
            price = self.get_current_price(symbol)
            results[symbol] = price is not None and price > 0
        
        return results
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def is_market_open(self) -> bool:
        """Check if market is open using primary provider"""
        return self.providers[self.primary_provider].is_market_open()