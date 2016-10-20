# coding:utf-8
import numpy as np


def sigmoid(X, a, b):
    t = np.dot(a, X.T).T + b
    return 1.0/(1.0 + np.exp(-t))
