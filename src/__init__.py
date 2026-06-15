"""
Expense Tracker - AI-Enhanced Personal Finance Management
"""

__version__ = "1.0.0"
__author__ = "AI Development Team"

from .database import ExpenseDatabase
from .models import CategoryClassifier, SpendingForecaster, AnomalyDetector
from .features import FeatureEngineer

__all__ = [
    'ExpenseDatabase',
    'CategoryClassifier',
    'SpendingForecaster',
    'AnomalyDetector',
    'FeatureEngineer'
]
