import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.compose import ForecastingPipeline, RecursiveTimeSeriesRegressionForecaster
from sktime.transformations.series.detrend import Detrender, Deseasonalizer
from sktime.transformations.series.difference import Differencer

from python.deseasonalizer import CustomDeseasonalizer
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from scipy import fftpack

from sktime_dl.deeplearning import SimpleRNNRegressor, MLPRegressor
from nbeats_keras.model import NBeatsNet as NBeatsKeras

from python.predictor import Predictor


class CustomPredictor(Predictor):
    def __init__(self, updates_every):
        self.model = None
        self.offset = 0
        self.updates_every = updates_every

    def short_name(self):
        return "Custom"

    def name(self):
        return "Custom"

    def get_prediction(self, history, t=1):
        if self.model is None:
            index = pd.Index(list(range(len(history))))
            data = pd.Series(history, index=index)
            self.offset += len(history)

            self.model = ForecastingPipeline([
                ("transf", CustomDeseasonalizer()),
                ("forecaster", ARIMA(order=(2, 0, 3)))
            ])

            self.model.fit(data)
            prediction = self.model.predict(np.arange(1, t + 1))
        else:
            index = pd.Index(list(range(self.offset, self.offset + self.updates_every)))
            data = pd.Series(history[-self.updates_every:], index=index)
            self.offset += self.updates_every
            self.model.update(data)
            prediction = self.model.predict(np.arange(1, t + 1))

        return prediction.tolist()


class CustomPredictor1(Predictor):
    def __init__(self, updates_every, train):
        self.offset = len(train)
        self.updates_every = updates_every

        estimator = SimpleRNNRegressor(nb_epochs=100, verbose=True)

        forecaster = RecursiveTimeSeriesRegressionForecaster(
            estimator=estimator,
            window_length=updates_every
        )

        data = pd.Series(train)
        forecaster.fit(data)
        self.model = forecaster

    def short_name(self):
        return "Custom1"

    def name(self):
        return "Custom1"

    def get_prediction(self, history, t=1):
        index = pd.Index(list(range(self.offset, self.offset + self.updates_every)))
        data = pd.Series(history[-self.updates_every:], index=index)

        prediction = self.model.update_predict_single(data, np.arange(1, t + 1), update_params=False)
        self.offset += t
        print(prediction)
        return prediction.tolist()


class CustomPredictor2(Predictor):
    def __init__(self, updates_every, train):
        self.offset = len(train)
        self.updates_every = updates_every
        self.model = ForecastingPipeline([
            ("differencer", Differencer(lags=1)),
            (
                "forecaster",
                RecursiveTimeSeriesRegressionForecaster(
                    estimator=SimpleRNNRegressor(nb_epochs=6, verbose=True),
                    window_length=updates_every
                )
            )
        ])
        self.model.fit(pd.Series(train))

    def short_name(self):
        return "Custom2"

    def name(self):
        return "Custom2"

    def get_prediction(self, history, t=1):
        index = pd.Index(list(range(self.offset, self.offset + self.updates_every)))
        data = pd.Series(history[-self.updates_every:], index=index)

        prediction = self.model.update_predict_single(data, np.arange(1, t + 1), update_params=False)
        self.offset += t
        print(prediction)
        return prediction[0].tolist()