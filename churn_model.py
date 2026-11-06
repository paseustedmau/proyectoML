import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ChurnModel(BaseEstimator, ClassifierMixin):
    """Pipeline entrenado + umbral de decisión optimizado."""
    def __init__(self, pipeline=None, threshold=0.5):
        self.pipeline = pipeline
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
