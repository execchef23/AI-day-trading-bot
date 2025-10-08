"""Yahoo Finance data provider using yfinance library"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Union
import time
from .base_provider import BaseDataProvider
import logging

logger = logging.getLogger(__name__)

class YahooFinanceProvider(BaseDataProvider):
    """Yahoo Finance data provider - free and reliable"""
    
    def __init__(self):
        super().__init__()
        self._rate_limit_delay = 0.5  # Yahoo Finance is more lenient
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y", 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Standardize column names
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
            data = data.drop(['Dividends', 'Stock Splits'], axis=1, errors='ignore')
            data.reset_index(inplace=True)
            data['Symbol'] = symbol
            
            time.sleep(self._rate_limit_delay)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if data.empty:
                logger.warning(f"No current price data for {symbol}")
                return None
            
            current_price = data['Close'].iloc[-1]
            time.sleep(self._rate_limit_delay)
            return float(current_price)
            
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get quotes for multiple symbols"""
        quotes = {}
        
        try:
            # Use yfinance's download function for multiple symbols
            data = yf.download(symbols, period="1d", interval="1m", group_by="ticker")
            
            for symbol in symbols:
                try:
                    if len(symbols) == 1:
                        symbol_data = data
                    else:
                        symbol_data = data[symbol]
                    
                    if not symbol_data.empty:
                        latest = symbol_data.iloc[-1]
                        quotes[symbol] = {
                            'price': float(latest['Close']),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'volume': int(latest['Volume']),
                            'timestamp': str(symbol_data.index[-1])
                        }
                    else:
                        logger.warning(f"No data available for {symbol}")
                        quotes[symbol] = None
                        
                except Exception as symbol_error:
                    logger.error(f"Error processing {symbol}: {symbol_error}")
                    quotes[symbol] = None
            
            time.sleep(self._rate_limit_delay)
            return quotes
            
        except Exception as e:
            logger.error(f"Error fetching multiple quotes: {e}")
            return {symbol: None for symbol in symbols}
    
    def get_company_info(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """Get company information from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            company_info = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                '52_week_high': info.get('fiftyTwoWeekHigh', 0),
                '52_week_low': info.get('fiftyTwoWeekLow', 0),
                'average_volume': info.get('averageVolume', 0),
                'description': info.get('longBusinessSummary', 'N/A')
            }
            
            time.sleep(self._rate_limit_delay)
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}