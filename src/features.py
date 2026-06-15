"""
Feature engineering module for expense categorization and forecasting.
Handles data preprocessing and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict


class FeatureEngineer:
    """Feature extraction and engineering for expense tracking."""
    
    @staticmethod
    def extract_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from transaction data for categorization."""
        df_feat = df.copy()
        
        # Amount features
        df_feat['amount_log'] = np.log1p(df_feat['amount'])
        df_feat['amount_norm'] = (df_feat['amount'] - df_feat['amount'].mean()) / (df_feat['amount'].std() + 1e-8)
        
        # Description features
        df_feat['description_length'] = df_feat['description'].fillna('').str.len()
        df_feat['word_count'] = df_feat['description'].fillna('').str.split().str.len()
        
        # Vendor features
        df_feat['vendor_length'] = df_feat['vendor'].fillna('').str.len()
        
        # Keyword features (simple heuristics)
        df_feat['has_supermarket_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'walmart|trader|whole|kroger|safeway|store', regex=True
        ).astype(int)
        
        df_feat['has_transport_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'uber|lyft|taxi|fuel|gas|transit|metro|subway', regex=True
        ).astype(int)
        
        df_feat['has_dining_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'restaurant|cafe|pizza|burger|diner|grill|bar', regex=True
        ).astype(int)
        
        df_feat['has_entertainment_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'movie|cinema|netflix|spotify|game|theater|concert', regex=True
        ).astype(int)
        
        df_feat['has_utility_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'electric|water|gas|internet|phone|utility|bill', regex=True
        ).astype(int)
        
        df_feat['has_healthcare_keywords'] = df_feat['description'].fillna('').str.lower().str.contains(
            r'pharmacy|doctor|hospital|medical|clinic|dental', regex=True
        ).astype(int)
        
        return df_feat
    
    @staticmethod
    def prepare_categorization_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature matrix for categorization model."""
        df_feat = FeatureEngineer.extract_transaction_features(df)
        
        # Select features for model
        feature_cols = [
            'amount_log', 'amount_norm', 'description_length', 'word_count',
            'vendor_length', 'has_supermarket_keywords', 'has_transport_keywords',
            'has_dining_keywords', 'has_entertainment_keywords',
            'has_utility_keywords', 'has_healthcare_keywords'
        ]
        
        return df_feat[feature_cols], feature_cols
    
    @staticmethod
    def prepare_forecasting_data(df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for spending forecasting (time-series features)."""
        df_monthly = df.copy()
        
        # Ensure date column is datetime
        df_monthly['date'] = pd.to_datetime(df_monthly['date'])
        df_monthly = df_monthly.sort_values('date')
        
        # Extract time features
        df_monthly['month'] = df_monthly['date'].dt.month
        df_monthly['year'] = df_monthly['date'].dt.year
        df_monthly['quarter'] = df_monthly['date'].dt.quarter
        
        # Create lag features
        df_monthly['lag_1'] = df_monthly['amount'].shift(1)
        df_monthly['lag_2'] = df_monthly['amount'].shift(2)
        df_monthly['lag_3'] = df_monthly['amount'].shift(3)
        
        # Rolling statistics
        df_monthly['rolling_mean_3'] = df_monthly['amount'].rolling(window=3, min_periods=1).mean()
        df_monthly['rolling_std_3'] = df_monthly['amount'].rolling(window=3, min_periods=1).std().fillna(0)
        df_monthly['rolling_mean_6'] = df_monthly['amount'].rolling(window=6, min_periods=1).mean()
        
        # Seasonality encoding
        df_monthly['month_sin'] = np.sin(2 * np.pi * df_monthly['month'] / 12)
        df_monthly['month_cos'] = np.cos(2 * np.pi * df_monthly['month'] / 12)
        
        # Year-over-year growth
        if len(df_monthly) > 12:
            df_monthly['yoy_growth'] = (
                (df_monthly['amount'] / df_monthly['amount'].shift(12) - 1) * 100
            ).fillna(0)
        else:
            df_monthly['yoy_growth'] = 0
        
        return df_monthly
    
    @staticmethod
    def aggregate_transactions_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate transaction-level data to monthly totals."""
        df_agg = df.copy()
        df_agg['date'] = pd.to_datetime(df_agg['date'])
        
        monthly = (
            df_agg.set_index('date')
            .resample('MS')['amount']
            .sum()
            .reset_index()
        )
        monthly['date'] = monthly['date'].dt.to_period('M').dt.end_time
        monthly.columns = ['date', 'amount']
        
        return monthly
    
    @staticmethod
    def get_category_rolling_stats(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Compute per-category rolling mean and std for anomaly detection."""
        stats = {}
        
        for category in df['category'].unique():
            if pd.isna(category):
                continue
            
            cat_data = df[df['category'] == category]['amount']
            
            if len(cat_data) > 0:
                mean = cat_data.mean()
                std = cat_data.std()
                stats[category] = (mean, std if std > 0 else 1.0)
            else:
                stats[category] = (0, 1.0)
        
        return stats
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies in transactions using statistical method."""
        df_anom = df.copy()
        df_anom['is_anomaly'] = 0
        df_anom['anomaly_score'] = 0.0
        
        # Per-category anomaly detection
        for category in df_anom['category'].unique():
            if pd.isna(category):
                continue
            
            mask = df_anom['category'] == category
            cat_data = df_anom.loc[mask]
            
            if len(cat_data) > 0:
                mean = cat_data['amount'].mean()
                std = cat_data['amount'].std()
                
                if std > 0:
                    anomaly_scores = np.abs((cat_data['amount'] - mean) / std)
                    is_anomaly = anomaly_scores > threshold
                    
                    df_anom.loc[mask, 'anomaly_score'] = anomaly_scores
                    df_anom.loc[mask, 'is_anomaly'] = is_anomaly.astype(int)
        
        return df_anom
    
    @staticmethod
    def prepare_forecast_features(monthly_df: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for XGBoost forecasting."""
        df_feat = FeatureEngineer.prepare_forecasting_data(monthly_df)
        
        # Drop NaN values from lag features
        df_feat = df_feat.dropna(subset=['lag_2'])
        
        if feature_cols is None:
            feature_cols = [
                'month', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3',
                'rolling_std_3', 'rolling_mean_6', 'month_sin', 'month_cos', 'yoy_growth'
            ]
        
        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df_feat.columns]
        
        return df_feat[feature_cols].values, feature_cols
