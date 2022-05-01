from sktime.transformations.base import BaseTransformer
from sktime.utils.datetime import _get_duration, _get_freq
from statsmodels.tsa.filters.cf_filter import cffilter
from scipy.signal import savgol_filter
import numpy as np


class CustomDeseasonalizer(BaseTransformer):
    def __init__(self):
        self._y_index = None
        self.seasonal_ = None
        super(CustomDeseasonalizer, self).__init__()

    def _set_y_index(self, y):
        self._y_index = y.index

    def _fit(self, X, y=None):
        self._set_y_index(X)

        filtered = savgol_filter(X, 101, 5, 0, 1.0, -1, "interp")
        #cycle, trend = cffilter(X, low=1200, high=1500, drift=False)
        self.seasonal_ = X - filtered

        return self

    def _transform(self, X, y=None):
        Xt = X - self.seasonal_
        return Xt

    def _inverse_transform(self, X, y=None):
        Xt = X + self.seasonal_
        return Xt

    def _update(self, X, y=None, update_params=False):
        self._set_y_index(X)
        return self
