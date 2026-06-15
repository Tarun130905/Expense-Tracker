"""
Machine learning models for categorization and forecasting.
Handles model training, inference, and persistence.
"""

import pickle
import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


class CategoryClassifier:
    """RandomForest-based expense categorizer."""
    
    def __init__(self, model_path: str = "models/categorizer.pkl"):
        """Initialize categorizer."""
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.load_or_init()
    
    def load_or_init(self) -> None:
        """Load model or initialize new one."""
        if os.path.exists(self.model_path):
            self.load()
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.label_encoder = LabelEncoder()
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Train categorizer on labeled data."""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train model
        self.model.fit(X, y_encoded)
        self.feature_names = feature_names
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save()
        
        # Return metrics
        train_accuracy = self.model.score(X, y_encoded)
        
        return {
            'accuracy': train_accuracy,
            'classes': list(self.label_encoder.classes_),
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict categories and confidence scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions and probabilities
        y_pred_encoded = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        confidence = np.max(y_proba, axis=1)
        
        # Decode predictions
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, confidence
    
    def predict_single(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict category for single transaction."""
        y_pred, confidence = self.predict(features.reshape(1, -1))
        return y_pred[0], float(confidence[0])
    
    def save(self) -> None:
        """Save model to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names
            }, f)
    
    def load(self) -> None:
        """Load model from disk."""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.feature_names = data['feature_names']
    
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None and self.label_encoder is not None


class SpendingForecaster:
    """XGBoost-based spending forecaster."""
    
    def __init__(self, model_path: str = "models/forecaster.pkl"):
        """Initialize forecaster."""
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        self.load_or_init()
    
    def load_or_init(self) -> None:
        """Load model or initialize new one."""
        if os.path.exists(self.model_path):
            self.load()
        else:
            self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], 
              n_rounds: int = 100) -> Dict[str, Any]:
        """Train forecaster on historical data."""
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train
        self.model = xgb.train(params, dtrain, num_boost_round=n_rounds, verbose_eval=False)
        self.feature_names = feature_names
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.save()
        
        # Compute training metrics
        train_pred = self.model.predict(dtrain)
        rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        mae = np.mean(np.abs(y - train_pred))
        mape = np.mean(np.abs((y - train_pred) / (y + 1e-8))) * 100
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict spending amount."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)
    
    def predict_single(self, features: np.ndarray) -> float:
        """Predict single spending amount."""
        pred = self.predict(features.reshape(1, -1))
        return float(pred[0])
    
    def save(self) -> None:
        """Save model to disk."""
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names
            }, f)
    
    def load(self) -> None:
        """Load model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_names = data['feature_names']
    
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self.model is not None


class AnomalyDetector:
    """Statistical anomaly detector for transactions."""
    
    def __init__(self):
        """Initialize detector."""
        self.category_stats = {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """Fit detector on transaction data."""
        for category in df['category'].unique():
            if pd.isna(category):
                continue
            
            cat_data = df[df['category'] == category]['amount']
            self.category_stats[category] = {
                'mean': cat_data.mean(),
                'std': max(cat_data.std(), 1.0)  # Avoid division by zero
            }
    
    def detect(self, df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
        """Detect anomalies using statistical threshold."""
        df_anom = df.copy()
        df_anom['is_anomaly'] = 0
        df_anom['anomaly_score'] = 0.0
        
        for category in df_anom['category'].unique():
            if pd.isna(category):
                continue
            
            if category not in self.category_stats:
                continue
            
            mask = df_anom['category'] == category
            stats = self.category_stats[category]
            
            mean = stats['mean']
            std = stats['std']
            
            # Compute z-scores
            anomaly_scores = np.abs((df_anom.loc[mask, 'amount'] - mean) / std)
            is_anomaly = anomaly_scores > threshold
            
            df_anom.loc[mask, 'anomaly_score'] = anomaly_scores
            df_anom.loc[mask, 'is_anomaly'] = is_anomaly.astype(int)
        
        return df_anom
    
    def detect_single(self, category: str, amount: float, threshold: float = 2.0) -> Tuple[bool, float]:
        """Detect if single transaction is anomalous."""
        if category not in self.category_stats:
            return False, 0.0
        
        stats = self.category_stats[category]
        mean = stats['mean']
        std = stats['std']
        
        score = np.abs((amount - mean) / std)
        is_anom = score > threshold
        
        return is_anom, float(score)
