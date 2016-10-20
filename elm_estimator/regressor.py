# coding:utf-8
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from .function import sigmoid


class ELMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_hidden=2000):
        self.n_hidden = n_hidden
        self.a = None
        self.b = None
        self.beta = None

    def fit(self, X, y):
        n, d = X.shape
        self.a = np.random.randn(self.n_hidden, d)
        self.b = np.random.randn(self.n_hidden)
        h = sigmoid(X, self.a, self.b)
        self.beta = np.dot(np.linalg.pinv(h), y)
        return self

    def predict(self, X):
        n, d = X.shape
        h = sigmoid(X, self.a, self.b)
        return np.dot(h, self.beta).reshape(n)
