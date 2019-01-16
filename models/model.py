__author__ = "Koren Gast"
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import KFold
import numpy as np


class GenModel(object):
    def __init__(self):
        self.model = None
        self.name = "Generic"

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds), recall_score(y, preds), precision_score(y, preds)

    def predict(self, X):
        return self.model.predict(X)
