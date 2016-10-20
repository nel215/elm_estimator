# coding:utf-8
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from elm_estimator import ELMRegressor


def test_hidden_types():
    assert ELMRegressor.hidden_types == ['sigmoid', 'gaussian']


def test_boston():
    np.random.seed(0)
    data, target = load_boston(True)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    cv = ShuffleSplit(n_splits=20, test_size=0.05)
    regressor = ELMRegressor(100)
    score = cross_val_score(regressor, data, target, cv=cv, n_jobs=-1)
    assert score.mean() > 0.80
