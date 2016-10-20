# coding:utf-8
import numpy as np
from scipy.spatial.distance import cdist


def sigmoid(X, a, b):
    t = np.dot(a, X.T).T + b
    return 1.0/(1.0 + np.exp(-t))


def gaussian(X, a, b):
    return np.exp(-b * cdist(X, a))
