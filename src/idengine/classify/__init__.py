"""Trainable local document classification."""

from idengine.classify.model import Prediction, TextClassifier
from idengine.classify.service import ClassifierService, TrainingResult

__all__ = ["ClassifierService", "Prediction", "TextClassifier", "TrainingResult"]
