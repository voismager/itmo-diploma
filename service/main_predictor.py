import darts
import numpy as np
import pandas as pd
from darts.models import NaiveSeasonal, ARIMA, FFT


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

    def __predict_naive__(self, history):
        model = NaiveSeasonal(K=1)
        model.fit(history)
        return model.predict(self.horizon), "Using last value"

    def predict(self, history: darts.TimeSeries):
        if len(history) < self.window // 2:
            return self.__predict_naive__(history)
        else:
            window = history[-self.window:]
            pd_window = window.pd_series()
            mean = pd_window.mean()
            var = pd_window.var()

            if np.isclose(mean, 0) or var < 8:
                return self.__predict_naive__(window)
            else:
                if self.seasonality is None:
                    model = ARIMA(3, 1, 1)
                    model.fit(window)
                    predictions = model.predict(self.horizon, num_samples=50)
                    quantile = predictions.quantile_timeseries(self.quantile)
                    return quantile, "Using ARIMA"
                else:
                    fft_model = FFT(nr_freqs_to_keep=4)
                    fft_model.fit(window)

                    seasonal_ts = np.array([fft_model.predicted_values[i % len(fft_model.predicted_values)] for i in range(-len(window), 0)])
                    seasonal_ts = darts.TimeSeries.from_series(pd.Series(data=seasonal_ts, index=window.time_index))
                    seasonal_pred_ts = fft_model.predict(self.horizon)

                    arima_window_size = min(self.window - self.seasonality, 512)
                    model = ARIMA(3, 1, 1)
                    model.fit(window[-arima_window_size:], future_covariates=seasonal_ts[-arima_window_size:])
                    return model.predict(self.horizon, future_covariates=seasonal_pred_ts), "Using ARIMA with seasonality"