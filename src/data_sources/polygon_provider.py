"""Polygon.io data provider for real-time market data"""

try:
    from polygon import RESTClient

    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False
    RESTClient = None
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd

from .base_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class PolygonProvider(BaseDataProvider):
    """Polygon.io data provider - requires API key"""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        if not POLYGON_AVAILABLE:
            raise ImportError(
                "Polygon.io client not available. Install with: pip install polygon-api-client"
            )
        self.client = RESTClient(api_key)
        self._rate_limit_delay = 12.0  # Free tier: 5 calls per minute

    def get_historical_data(
        self, symbol: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Get historical data from Polygon.io"""
        try:
            # Calculate date range based on period
            end_date = datetime.now().date()

            if period == "1d":
                start_date = end_date - timedelta(days=1)
            elif period == "5d":
                start_date = end_date - timedelta(days=5)
            elif period == "1mo":
                start_date = end_date - timedelta(days=30)
            elif period == "3mo":
                start_date = end_date - timedelta(days=90)
            elif period == "6mo":
                start_date = end_date - timedelta(days=180)
            elif period == "1y":
                start_date = end_date - timedelta(days=365)
            elif period == "2y":
                start_date = end_date - timedelta(days=730)
            elif period == "5y":
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=365)

            # Map interval to Polygon timespan
            if interval in ["1m", "5m", "15m", "30m", "60m"]:
                timespan = "minute"
                multiplier = int(interval.replace("m", ""))
            elif interval == "1h":
                timespan = "hour"
                multiplier = 1
            elif interval == "1d":
                timespan = "day"
                multiplier = 1
            else:
                timespan = "day"
                multiplier = 1

            # Get data from Polygon
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=start_date,
                to=end_date,
            )

            if not aggs:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            data_list = []
            for agg in aggs:
                data_list.append(
                    {
                        "Date": datetime.fromtimestamp(agg.timestamp / 1000),
                        "Open": agg.open,
                        "High": agg.high,
                        "Low": agg.low,
                        "Close": agg.close,
                        "Volume": agg.volume,
                    }
                )

            df = pd.DataFrame(data_list)
            df["Symbol"] = symbol
            df = df.sort_values("Date").reset_index(drop=True)

            time.sleep(self._rate_limit_delay)
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Polygon.io"""
        try:
            # Get last trade
            last_trade = self.client.get_last_trade(symbol)

            if last_trade:
                time.sleep(self._rate_limit_delay)
                return float(last_trade.price)
            else:
                logger.warning(f"No current price data for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def get_multiple_quotes(
        self, symbols: List[str]
    ) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get quotes for multiple symbols"""
        quotes = {}

        for symbol in symbols:
            try:
                # Get last quote
                last_quote = self.client.get_last_quote(symbol)

                if last_quote:
                    quotes[symbol] = {
                        "price": float(last_quote.last.price),
                        "bid": float(last_quote.last.bid),
                        "ask": float(last_quote.last.ask),
                        "bid_size": int(last_quote.last.bid_size),
                        "ask_size": int(last_quote.last.ask_size),
                        "timestamp": str(
                            datetime.fromtimestamp(last_quote.last.timestamp / 1000)
                        ),
                    }
                else:
                    logger.warning(f"No quote data for {symbol}")
                    quotes[symbol] = None

                time.sleep(self._rate_limit_delay)

            except Exception as e:
                logger.error(f"Error fetching quote for {symbol}: {e}")
                quotes[symbol] = None

        return quotes

    def get_company_info(self, symbol: str) -> Dict[str, Union[str, float, int]]:
        """Get company information from Polygon.io"""
        try:
            # Get ticker details
            ticker_details = self.client.get_ticker_details(symbol)

            if ticker_details:
                company_info = {
                    "symbol": symbol,
                    "company_name": ticker_details.name or "N/A",
                    "description": ticker_details.description or "N/A",
                    "market_cap": ticker_details.market_cap or 0,
                    "total_employees": ticker_details.total_employees or 0,
                    "list_date": str(ticker_details.list_date)
                    if ticker_details.list_date
                    else "N/A",
                    "locale": ticker_details.locale or "N/A",
                    "currency_name": ticker_details.currency_name or "USD",
                    "primary_exchange": ticker_details.primary_exchange or "N/A",
                }

                time.sleep(self._rate_limit_delay)
                return company_info
            else:
                logger.error(f"No company data found for {symbol}")
                return {"symbol": symbol, "error": "No data found"}

        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}
