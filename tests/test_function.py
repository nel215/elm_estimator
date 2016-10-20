# coding:utf-8
import numpy as np
from elm_estimator.function import *


def test_sigmoid():
    np.random.seed(0)
    X = np.random.randn(10, 5)
    a = np.random.randn(100, 5)
    b = np.random.randn(100)
    h = sigmoid(X, a, b)
    assert (10, 100) == h.shape
