"""Base data provider interface for all market data sources"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta

class BaseDataProvider(ABC):
    """Abstract base class for all data providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._rate_limit_delay = 1.0  # seconds between requests
    
    @abstractmethod
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y", 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical price data
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Date
        """
        pass
    
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current/latest price for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price as float
        """
        pass
    
    @abstractmethod
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get current quotes for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary with symbol as key and quote data as value
        """
        pass
    
    @abstractmethod
    def get_company_info(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """Get company information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company information
        """
        pass
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (basic implementation)
        
        Returns:
            True if market is open, False otherwise
        """
        now = datetime.now()
        # Basic check for NYSE hours (9:30 AM - 4:00 PM ET, Monday-Friday)
        # Note: This doesn't account for holidays
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            price = self.get_current_price(symbol)
            return price is not None and price > 0
        except Exception:
            return False