import datetime

import darts
import numpy as np
import pandas as pd
from darts.models import NaiveSeasonal, ARIMA
from matplotlib.pyplot import show
from sklearn.metrics import r2_score, mean_absolute_error
from sktime.transformations.series.detrend import ConditionalDeseasonalizer


class MainPredictor:
    def __init__(self, horizon, quantile, measurement_frequency_ms, seasonality):
        self.horizon = horizon
        self.quantile = quantile
        self.__setup__(seasonality, measurement_frequency_ms)

    def __setup__(self, seasonality_name, measurement_frequency_ms):
        if seasonality_name == 'daily':
            self.seasonality = 86400000 // measurement_frequency_ms
            self.window = self.seasonality * 4
        else:
            self.seasonality = None
            self.window = 512

        print(self.window)
        print(self.seasonality)

    def __predict_naive__(self, history):
        model = NaiveSeasonal(K=1)
        model.fit(history)
        return model.predict(self.horizon), "Using last value"

    def predict(self, history: darts.TimeSeries):
        if len(history) < self.seasonality:
            return self.__predict_naive__(history)
        else:
            window = history[-self.window:]
            pd_window = window.pd_series()
            mean = pd_window.mean()
            var = pd_window.var()

            if np.isclose(mean, 0) or var < 8:
                return self.__predict_naive__(window)
            else:
                deseasonalizer = ConditionalDeseasonalizer(sp=self.seasonality)
                pd_transformed_window = darts.TimeSeries.from_series(deseasonalizer.fit_transform(pd_window))

                model = ARIMA(4, 0, 1)
                model.fit(pd_transformed_window[-min(self.window - self.seasonality, 512):])
                predictions = model.predict(self.horizon, num_samples=50)

                quantile = predictions.quantile_timeseries(self.quantile)
                transformed = darts.TimeSeries.from_series(deseasonalizer.inverse_transform(quantile.pd_series()))

                return transformed, "Using ARIMA"




def load_data(path):
    import csv
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))

        return rows


if __name__ == '__main__':
    tasks_numbers = load_data("../simulation/validation_data.csv")
    predictor = MainPredictor(100, 0.5, 30000, 'daily')
    history = []
    index = []

    current = datetime.datetime.now()
    pred = None

    for i in range(len(tasks_numbers)):
        current += datetime.timedelta(seconds=1)

        history.append(tasks_numbers[i])
        index.append(current)

        if i % 100 == 0 and i > 1024:
            pd_history = darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index)))
            print("Start")
            prediction = predictor.predict(pd_history)[0]
            print("Stop")

            if pred is None:
                pred = prediction
            else:
                pred = pred.append(prediction)

    actual = darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index), name="Actual"))

    actual.plot()
    pred.plot()
    show()

    actual.plot()
    show()

    pred.plot()
    show()

    print(r2_score(actual.pd_series()[-(12500 - 5760):], pred.pd_series()[5760:]))
    print(mean_absolute_error(actual.pd_series()[-(12500 - 5760):], pred.pd_series()[5760:]))