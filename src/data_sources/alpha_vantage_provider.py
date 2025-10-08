"""Alpha Vantage data provider for premium market data"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Union
import time
from .base_provider import BaseDataProvider
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AlphaVantageProvider(BaseDataProvider):
    """Alpha Vantage data provider - requires API key"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self._rate_limit_delay = 12.0  # Alpha Vantage free tier: 5 calls per minute
    
    def get_historical_data(self, 
                          symbol: str, 
                          period: str = "1y", 
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical data from Alpha Vantage"""
        try:
            # Map periods to Alpha Vantage functions
            if interval in ['1m', '5m', '15m', '30m', '60m']:
                function = 'TIME_SERIES_INTRADAY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': interval,
                    'apikey': self.api_key
                }
            else:
                function = 'TIME_SERIES_DAILY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'outputsize': 'full',  # Get full historical data
                    'apikey': self.api_key
                }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return pd.DataFrame()
            
            # Extract time series data
            if interval in ['1m', '5m', '15m', '30m', '60m']:
                time_series_key = f'Time Series ({interval})'
            else:
                time_series_key = 'Time Series (Daily)'
            
            if time_series_key not in data:
                logger.error(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Standardize column names
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df = df.astype(float)
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)
            df['Symbol'] = symbol
            
            time.sleep(self._rate_limit_delay)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price from Alpha Vantage"""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Global Quote' in data:
                price = float(data['Global Quote']['05. price'])
                time.sleep(self._rate_limit_delay)
                return price
            else:
                logger.error(f"No quote data found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get quotes for multiple symbols (sequential due to rate limits)"""
        quotes = {}
        
        for symbol in symbols:
            try:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': symbol,
                    'apikey': self.api_key
                }
                
                response = requests.get(self.base_url, params=params)
                data = response.json()
                
                if 'Global Quote' in data:
                    quote_data = data['Global Quote']
                    quotes[symbol] = {
                        'price': float(quote_data['05. price']),
                        'open': float(quote_data['02. open']),
                        'high': float(quote_data['03. high']),
                        'low': float(quote_data['04. low']),
                        'volume': int(quote_data['06. volume']),
                        'change': float(quote_data['09. change']),
                        'change_percent': quote_data['10. change percent'],
                        'timestamp': quote_data['07. latest trading day']
                    }
                else:
                    logger.warning(f"No quote data for {symbol}")
                    quotes[symbol] = None
                
                time.sleep(self._rate_limit_delay)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")
                quotes[symbol] = None
        
        return quotes
    
    def get_company_info(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """Get company overview from Alpha Vantage"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Symbol' in data:
                company_info = {
                    'symbol': data.get('Symbol', symbol),
                    'company_name': data.get('Name', 'N/A'),
                    'sector': data.get('Sector', 'N/A'),
                    'industry': data.get('Industry', 'N/A'),
                    'market_cap': int(data.get('MarketCapitalization', 0)) if data.get('MarketCapitalization', '0').isdigit() else 0,
                    'pe_ratio': float(data.get('PERatio', 0)) if data.get('PERatio', 'None') != 'None' else 0,
                    'dividend_yield': float(data.get('DividendYield', 0)) if data.get('DividendYield', 'None') != 'None' else 0,
                    'beta': float(data.get('Beta', 0)) if data.get('Beta', 'None') != 'None' else 0,
                    '52_week_high': float(data.get('52WeekHigh', 0)),
                    '52_week_low': float(data.get('52WeekLow', 0)),
                    'description': data.get('Description', 'N/A')
                }
                
                time.sleep(self._rate_limit_delay)
                return company_info
            else:
                logger.error(f"No company data found for {symbol}")
                return {'symbol': symbol, 'error': 'No data found'}
                
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}