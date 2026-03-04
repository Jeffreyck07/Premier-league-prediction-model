"""Alternate public exports for EPL predictor package."""

from .backtester import BacktestEngine, ModelEvaluator
from .config import AppConfig
from .data_loader import DataLoader
from .features import FeatureEngineer
from .inference import PredictionService
from .models import BettingOptimizer, DixonColesModel, ModelTrainer

__all__ = [
    "AppConfig",
    "DataLoader",
    "FeatureEngineer",
    "DixonColesModel",
    "BettingOptimizer",
    "ModelTrainer",
    "PredictionService",
    "ModelEvaluator",
    "BacktestEngine",
]
