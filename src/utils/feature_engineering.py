"""Feature engineering for machine learning models"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create features for machine learning models"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features"""
        df = df.copy()
        
        try:
            # Returns
            df['return_1d'] = df['Close'].pct_change()
            df['return_2d'] = df['Close'].pct_change(2)
            df['return_5d'] = df['Close'].pct_change(5)
            df['return_10d'] = df['Close'].pct_change(10)
            
            # Log returns
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Price ratios
            df['high_close_ratio'] = df['High'] / df['Close']
            df['low_close_ratio'] = df['Low'] / df['Close']
            df['open_close_ratio'] = df['Open'] / df['Close']
            
            # Price position within day's range
            df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
            
            # Gap indicators
            df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
            df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
            df['gap_size'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
            
            logger.debug("Created price-based features")
            
        except Exception as e:
            logger.error(f"Error creating price features: {e}")
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volume-based features"""
        df = df.copy()
        
        try:
            # Volume ratios
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            df['volume_price_trend'] = df['Volume'] * df['return_1d']
            
            # Volume-weighted average price
            df['vwap'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
            
            # Money flow
            df['money_flow'] = df['Volume'] * df['Close']
            df['money_flow_ratio'] = df['money_flow'] / df['money_flow'].rolling(20).mean()
            
            logger.debug("Created volume-based features")
            
        except Exception as e:
            logger.error(f"Error creating volume features: {e}")
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility-based features"""
        df = df.copy()
        
        try:
            # Rolling volatility (different windows)
            for window in [5, 10, 20, 30]:
                df[f'volatility_{window}d'] = df['return_1d'].rolling(window).std() * np.sqrt(252)
                df[f'volatility_ratio_{window}d'] = df[f'volatility_{window}d'] / df[f'volatility_{window}d'].rolling(60).mean()
            
            # Realized volatility
            df['realized_vol'] = np.sqrt(252) * df['return_1d'].rolling(20).std()
            
            # True Range based volatility
            high_low = df['High'] - df['Low']
            high_close_prev = np.abs(df['High'] - df['Close'].shift())
            low_close_prev = np.abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df['avg_true_range'] = true_range.rolling(14).mean()
            df['atr_ratio'] = df['avg_true_range'] / df['Close']
            
            logger.debug("Created volatility-based features")
            
        except Exception as e:
            logger.error(f"Error creating volatility features: {e}")
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        try:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Extract time components
                df['day_of_week'] = df['Date'].dt.dayofweek
                df['day_of_month'] = df['Date'].dt.day
                df['month'] = df['Date'].dt.month
                df['quarter'] = df['Date'].dt.quarter
                df['year'] = df['Date'].dt.year
                
                # Cyclical encoding for time features
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                
                # Market regime indicators
                df['is_monday'] = (df['day_of_week'] == 0).astype(int)
                df['is_friday'] = (df['day_of_week'] == 4).astype(int)
                df['is_month_end'] = (df['Date'].dt.is_month_end).astype(int)
                df['is_quarter_end'] = (df['Date'].dt.is_quarter_end).astype(int)
                
            logger.debug("Created time-based features")
            
        except Exception as e:
            logger.error(f"Error creating time features: {e}")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lagged features"""
        df = df.copy()
        
        try:
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            logger.debug(f"Created lag features for {len(columns)} columns with lags {lags}")
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        
        try:
            for col in columns:
                if col in df.columns:
                    for window in windows:
                        df[f'{col}_mean_{window}d'] = df[col].rolling(window).mean()
                        df[f'{col}_std_{window}d'] = df[col].rolling(window).std()
                        df[f'{col}_min_{window}d'] = df[col].rolling(window).min()
                        df[f'{col}_max_{window}d'] = df[col].rolling(window).max()
                        df[f'{col}_median_{window}d'] = df[col].rolling(window).median()
                        
                        # Percentile position within rolling window
                        df[f'{col}_percentile_{window}d'] = df[col].rolling(window).rank(pct=True)
            
            logger.debug(f"Created rolling features for {len(columns)} columns with windows {windows}")
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {e}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame, horizons: List[int] = [1, 3, 5]) -> pd.DataFrame:
        """Create target variables for prediction"""
        df = df.copy()
        
        try:
            for horizon in horizons:
                # Future returns
                df[f'target_return_{horizon}d'] = df['Close'].pct_change(horizon).shift(-horizon)
                
                # Binary classification targets
                df[f'target_up_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)
                df[f'target_down_{horizon}d'] = (df[f'target_return_{horizon}d'] < 0).astype(int)
                
                # Categorical targets (strong up, up, neutral, down, strong down)
                conditions = [
                    df[f'target_return_{horizon}d'] > 0.02,  # Strong up
                    df[f'target_return_{horizon}d'] > 0.005,  # Up
                    (df[f'target_return_{horizon}d'] >= -0.005) & (df[f'target_return_{horizon}d'] <= 0.005),  # Neutral
                    df[f'target_return_{horizon}d'] < -0.005,  # Down
                    df[f'target_return_{horizon}d'] < -0.02   # Strong down
                ]
                choices = [4, 3, 2, 1, 0]  # 4=Strong Up, 0=Strong Down
                df[f'target_category_{horizon}d'] = np.select(conditions, choices, default=2)
            
            logger.debug(f"Created target variables for horizons {horizons}")
            
        except Exception as e:
            logger.error(f"Error creating target variables: {e}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        logger.info("Starting feature engineering pipeline")
        
        # Create all feature types
        df = self.create_price_features(df)
        df = self.create_volume_features(df)
        df = self.create_volatility_features(df)
        df = self.create_time_features(df)
        
        # Create lag features for key indicators
        key_columns = ['Close', 'Volume', 'return_1d', 'rsi', 'macd']
        df = self.create_lag_features(df, key_columns, [1, 2, 3, 5])
        
        # Create rolling features
        rolling_columns = ['return_1d', 'volume_ratio', 'rsi']
        df = self.create_rolling_features(df, rolling_columns, [5, 10, 20])
        
        # Create target variables
        df = self.create_target_variables(df)
        
        # Store feature columns (excluding original OHLCV and targets)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Symbol'] + \
                      [col for col in df.columns if col.startswith('target_')]
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Feature engineering complete. Created {len(self.feature_columns)} features")
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale features for machine learning
        
        Args:
            df: DataFrame with features
            method: 'standard' or 'minmax'
        """
        df = df.copy()
        
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
            
            # Scale only numeric feature columns
            numeric_features = df[self.feature_columns].select_dtypes(include=[np.number]).columns
            
            if len(numeric_features) > 0:
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
                self.scalers[method] = scaler
                logger.info(f"Scaled {len(numeric_features)} features using {method} scaling")
            
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
        
        return df
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict:
        """Prepare data for feature importance analysis"""
        # Remove rows with NaN values in features or targets
        feature_data = df[self.feature_columns + ['target_return_1d']].dropna()
        
        return {
            'features': feature_data[self.feature_columns],
            'target': feature_data['target_return_1d'],
            'feature_names': self.feature_columns
        }