import darts
import numpy as np

from darts.models import FourTheta, NaiveMean
from darts.utils.utils import SeasonalityMode


class MainPredictor:
    def __init__(self, window, horizon):
        self.window = window
        self.horizon = horizon

    def __predict_naive__(self, history):
        model = NaiveMean()
        model.fit(history)
        return model.predict(self.horizon), "Using naive mean (history set is too small)"

    def predict(self, history: darts.TimeSeries):
        if len(history) < 10:
            return self.__predict_naive__(history)
        else:
            window = history[-self.window:]

            if np.isclose(window.pd_series().mean(), 0):
                return self.__predict_naive__(window)
            else:
                model = FourTheta(theta=3, season_mode=SeasonalityMode.ADDITIVE)
                model.fit(window)
                return model.predict(self.horizon), "Using 4Theta method"
