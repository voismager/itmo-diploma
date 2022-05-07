import datetime

import darts
import numpy as np
import pandas as pd

from darts.models import FourTheta, NaiveSeasonal, KalmanForecaster
from darts.utils.utils import SeasonalityMode


class MainPredictor:
    def __init__(self, window, horizon):
        self.window = window
        self.horizon = horizon
        self.quantile = 0.5

    def __predict_naive__(self, history):
        model = NaiveSeasonal()
        model.fit(history)
        return model.predict(self.horizon), "Using naive mean (history set is too small)"

    def predict(self, history: darts.TimeSeries):
        if len(history) < self.horizon:
            return self.__predict_naive__(history)
        else:
            window = history[-self.window:]

            if np.isclose(window.pd_series().mean(), 0):
                return self.__predict_naive__(window)
            else:
                model = KalmanForecaster()
                model.fit(window)
                predictions = model.predict(self.horizon, num_samples=100)
                return predictions.quantile_timeseries(self.quantile), "Using stochastic Kalman filter"




def load_data(path):
    import csv
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))

        return rows


if __name__ == '__main__':
    tasks_numbers = load_data("../simulation/data.csv")[:5000]
    predictor = MainPredictor(512, 100)
    history = []
    index = []

    current = datetime.datetime.now()
    pred = None

    for i in range(len(tasks_numbers)):
        current += datetime.timedelta(seconds=1)

        history.append(tasks_numbers[i])
        index.append(current)

        if i % 100 == 0 and i > 500:
            pd_history = darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index)))
            prediction = predictor.predict(pd_history)[0]

            if pred is None:
                pred = prediction
            else:
                pred = pred.append(prediction)

    darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index), name="Actual")).plot()
    pred.plot()
    show()

    pred.plot()
    show()
