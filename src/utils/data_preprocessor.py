"""Data preprocessing utilities for the trading bot"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .feature_engineering import FeatureEngineer
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Complete data preprocessing pipeline for trading data"""

    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.is_fitted = False
        self.statistics = {}

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data format and quality"""
        errors = []

        # Check required columns
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Check data types
        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in numeric_columns:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} must be numeric")

        # Check for empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")

        # Check for minimum data points
        if len(df) < 50:
            errors.append(f"Insufficient data points: {len(df)}. Need at least 50.")

        # Check price relationships
        if not errors and len(df) > 0:
            invalid_prices = (
                (df["High"] < df["Low"])
                | (df["High"] < df["Close"])
                | (df["Low"] > df["Close"])
            )
            if invalid_prices.any():
                errors.append(
                    f"Invalid price relationships found in {invalid_prices.sum()} rows"
                )

        # Check for negative values
        if not errors:
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns and (df[col] < 0).any():
                    errors.append(f"Negative values found in {col}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare raw market data"""
        logger.info(f"Cleaning data with {len(df)} rows")
        df = df.copy()

        try:
            # Ensure Date column is datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date").reset_index(drop=True)

            # Remove duplicates
            initial_len = len(df)
            if "Date" in df.columns:
                df = df.drop_duplicates(subset=["Date"], keep="last")
            else:
                df = df.drop_duplicates()

            if len(df) < initial_len:
                logger.info(f"Removed {initial_len - len(df)} duplicate rows")

            # Handle missing values
            numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
            missing_counts = df[numeric_columns].isnull().sum()

            if missing_counts.any():
                logger.warning(
                    f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}"
                )

                # Forward fill then backward fill
                df[numeric_columns] = (
                    df[numeric_columns].fillna(method="ffill").fillna(method="bfill")
                )

            # Remove outliers (basic approach)
            df = self._remove_outliers(df)

            # Ensure positive volumes
            df["Volume"] = df["Volume"].abs()

            # Fix zero volumes
            zero_volume_mask = df["Volume"] == 0
            if zero_volume_mask.any():
                logger.warning(f"Found {zero_volume_mask.sum()} rows with zero volume")
                # Replace with average volume
                avg_volume = df["Volume"][df["Volume"] > 0].mean()
                df.loc[zero_volume_mask, "Volume"] = avg_volume

            logger.info(f"Data cleaning complete. Final shape: {df.shape}")

        except Exception as e:
            logger.error(f"Error during data cleaning: {e}")

        return df

    def _remove_outliers(self, df: pd.DataFrame, method: str = "iqr") -> pd.DataFrame:
        """Remove outliers from price and volume data"""
        df = df.copy()

        try:
            price_columns = ["Open", "High", "Low", "Close"]

            for col in price_columns + ["Volume"]:
                if col in df.columns:
                    if method == "iqr":
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)

                    elif method == "zscore":
                        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                        outliers = z_scores > 3

                    if outliers.any():
                        logger.info(f"Found {outliers.sum()} outliers in {col}")
                        # Replace outliers with median
                        df.loc[outliers, col] = df[col].median()

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        logger.info("Adding technical indicators")
        return TechnicalIndicators.add_all_indicators(df)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        logger.info("Engineering features")
        return self.feature_engineer.prepare_features(df)

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        target_horizon: int = 1,
        test_size: float = 0.2,
        validation_size: float = 0.1,
    ) -> Dict[str, Union[pd.DataFrame, List[str]]]:
        """Prepare data for machine learning training

        Args:
            df: Preprocessed DataFrame with features
            target_horizon: Days ahead to predict
            test_size: Fraction for test set
            validation_size: Fraction for validation set

        Returns:
            Dictionary with train/val/test splits and metadata
        """
        logger.info("Preparing data for training")

        try:
            # Select target variable
            target_col = f"target_return_{target_horizon}d"
            if target_col not in df.columns:
                raise ValueError(f"Target column {target_col} not found")

            # Automatically detect feature columns if not already set
            if not self.feature_engineer.feature_columns:
                logger.info("Feature columns not set, auto-detecting from DataFrame")
                exclude_cols = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "Date",
                    "Symbol",
                ] + [col for col in df.columns if col.startswith("target_")]
                self.feature_engineer.feature_columns = [
                    col for col in df.columns if col not in exclude_cols
                ]
                logger.info(
                    f"Auto-detected {len(self.feature_engineer.feature_columns)} feature columns"
                )

            # Remove rows with NaN in features or target
            feature_cols = self.feature_engineer.feature_columns
            clean_data = df[feature_cols + [target_col]].dropna()

            if len(clean_data) == 0:
                raise ValueError("No valid data after removing NaN values")

            logger.info(
                f"Using {len(clean_data)} samples with {len(feature_cols)} features"
            )

            # Time-based split (important for time series data)
            total_samples = len(clean_data)
            train_end = int(total_samples * (1 - test_size - validation_size))
            val_end = int(total_samples * (1 - test_size))

            # Split data
            train_data = clean_data.iloc[:train_end]
            val_data = clean_data.iloc[train_end:val_end]
            test_data = clean_data.iloc[val_end:]

            # Prepare features and targets
            result = {
                "X_train": train_data[feature_cols],
                "y_train": train_data[target_col],
                "X_val": val_data[feature_cols],
                "y_val": val_data[target_col],
                "X_test": test_data[feature_cols],
                "y_test": test_data[target_col],
                "feature_names": feature_cols,
                "target_name": target_col,
                "split_info": {
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "test_samples": len(test_data),
                    "total_samples": total_samples,
                },
            }

            # Store statistics
            self.statistics["feature_stats"] = {
                "mean": train_data[feature_cols].mean().to_dict(),
                "std": train_data[feature_cols].std().to_dict(),
                "min": train_data[feature_cols].min().to_dict(),
                "max": train_data[feature_cols].max().to_dict(),
            }

            self.is_fitted = True
            logger.info(
                f"Data preparation complete. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
            )

            return result

        except Exception as e:
            logger.error(f"Error preparing data for training: {e}")
            raise

    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info("Starting complete preprocessing pipeline")

        # Validate data
        is_valid, errors = self.validate_data(df)
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")

        # Clean data
        df = self.clean_data(df)

        # Add technical indicators
        df = self.add_technical_indicators(df)

        # Engineer features
        df = self.engineer_features(df)

        logger.info(f"Preprocessing pipeline complete. Final shape: {df.shape}")
        return df

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of features"""
        if not self.feature_engineer.feature_columns:
            return {}

        feature_cols = [
            col for col in self.feature_engineer.feature_columns if col in df.columns
        ]

        return {
            "total_features": len(feature_cols),
            "numeric_features": len(
                df[feature_cols].select_dtypes(include=[np.number]).columns
            ),
            "missing_values": df[feature_cols].isnull().sum().to_dict(),
            "feature_types": {
                "price_features": len(
                    [
                        col
                        for col in feature_cols
                        if any(x in col for x in ["return", "ratio", "gap"])
                    ]
                ),
                "technical_indicators": len(
                    [
                        col
                        for col in feature_cols
                        if any(x in col for x in ["rsi", "macd", "sma", "ema", "bb_"])
                    ]
                ),
                "volume_features": len(
                    [
                        col
                        for col in feature_cols
                        if "volume" in col or "obv" in col or "vpt" in col
                    ]
                ),
                "volatility_features": len(
                    [col for col in feature_cols if "volatility" in col or "atr" in col]
                ),
                "time_features": len(
                    [
                        col
                        for col in feature_cols
                        if any(x in col for x in ["day_", "month", "quarter", "year"])
                    ]
                ),
                "lag_features": len([col for col in feature_cols if "_lag_" in col]),
                "rolling_features": len(
                    [
                        col
                        for col in feature_cols
                        if any(
                            x in col
                            for x in ["_mean_", "_std_", "_min_", "_max_", "_median_"]
                        )
                    ]
                ),
            },
        }
