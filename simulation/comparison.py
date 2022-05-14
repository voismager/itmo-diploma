import csv
import darts
import datetime
import pandas as pd
import numpy as np
from darts.utils.utils import ModelMode, SeasonalityMode
from darts.utils.statistics import remove_seasonality, check_seasonality
from sktime.forecasting.trend import PolynomialTrendForecaster
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
from scipy import fftpack
from scipy import optimize

from symfit import parameters, variables, sin, cos, Fit
from matplotlib.pyplot import show, plot, ylabel, title, xlabel
from sklearn.metrics import r2_score, mean_absolute_error, max_error
from darts.models import NaiveMean, NaiveSeasonal, ARIMA, AutoARIMA, FFT, ExponentialSmoothing, NBEATSModel, KalmanForecaster, Theta, FourTheta, MovingAverage, \
    KalmanFilter
from sktime.transformations.series.detrend import Deseasonalizer, Detrender


def load_data(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))

        return rows


class NaivePredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        model = NaiveMean()
        model.fit(history[-200:])
        return model.predict(self.horizon)


class ESPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        # 2880

        # if len(history) < 5760:
        #     model = NaiveSeasonal()
        #     model.fit(history)
        #     return model.predict(self.horizon)

        model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.NONE)
        model.fit(history[-200:])
        return model.predict(self.horizon)


class KalmanPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        model = KalmanForecaster()
        model.fit(history[-200:])
        return model.predict(self.horizon)


class ThetaPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        if len(history) < 1420 * 2:
            model = NaiveSeasonal()
            model.fit(history)
            return model.predict(self.horizon)

        model = Theta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=1420)
        model.fit(history[-1420 * 2:])
        return model.predict(self.horizon)


class FourThetaPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        if len(history) < 5760:
            model = NaiveSeasonal()
            model.fit(history)
            return model.predict(self.horizon)

        model = FourTheta(season_mode=SeasonalityMode.ADDITIVE, seasonality_period=1420)
        model.fit(history[-1420 * 2:])
        return model.predict(self.horizon)


class NbeatsPredictor:
    def __init__(self, horizon, train):
        self.horizon = horizon
        self.model = NBEATSModel(400, horizon)
        self.model.fit(train, verbose=True, epochs=10)

    def predict(self, history):
        if len(history) < 400:
            model = NaiveSeasonal()
            model.fit(history)
            return model.predict(self.horizon)

        return self.model.predict(self.horizon, series=history[-400:])


class ArimaPredictor:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, history):
        if len(history) < 5760:
            model = NaiveSeasonal()
            model.fit(history)
            return model.predict(self.horizon)

        # window = history[-2880*2:]
        # seasoned_window = KalmanFilter().fit(window).filter(window)
        # seasoned_prediction = KalmanForecaster().fit(window).predict(self.horizon)
        #
        # model = ARIMA(3, 1, 1)
        # model.fit(window[-500:], future_covariates=seasoned_window[-500:])
        # prediction = model.predict(self.horizon, future_covariates=seasoned_prediction)
        #
        # return seasoned_prediction + prediction

        window = history[-1420*2:]

        fft_model = FFT(nr_freqs_to_keep=4)
        fft_model.fit(window)

        seasonal_ts = np.array([fft_model.predicted_values[i % len(fft_model.predicted_values)] for i in range(-len(window), 0)])
        seasonal_ts = darts.TimeSeries.from_series(pd.Series(data=seasonal_ts, index=window.time_index))
        seasonal_pred_ts = fft_model.predict(self.horizon)

        #seasonal = seasonal_decompose(window.pd_series(), period=2880, model="additive", filt=None, two_sided=True, extrapolate_trend=0).seasonal
        #seasonal_ts = darts.TimeSeries.from_series(seasonal)
        #seasonal_pred_ts = ARIMA(3, 1, 1).fit(seasonal_ts).predict(self.horizon)

        model = ARIMA(3, 1, 1)
        model.fit(window[-200:], future_covariates=seasonal_ts[-200:])
        return model.predict(self.horizon, future_covariates=seasonal_pred_ts)

        # pd_window = history[-1420*2:].pd_series()
        #
        # deseasonalizer = Deseasonalizer(sp=1420)
        #
        # pd_transformed = deseasonalizer.fit_transform(pd_window)
        #
        # model = ARIMA(3, 1, 1)
        # model.fit(darts.TimeSeries.from_series(pd_transformed)[-200:])
        # predictions = model.predict(self.horizon)
        #
        # pd_transformed = deseasonalizer.inverse_transform(predictions.pd_series())
        #
        # return darts.TimeSeries.from_series(pd_transformed)


def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


def test_filter():
    tasks_numbers = load_data("../validation_data/validation_data.csv")
    data = pd.Series(data=tasks_numbers, index=pd.RangeIndex(0, len(tasks_numbers)))
    ts = darts.TimeSeries.from_series(data)

    fft_model = FFT(nr_freqs_to_keep=4)
    fft_model.fit(ts)

    seasonal_ts = np.array([fft_model.predicted_values[i % len(fft_model.predicted_values)] for i in range(-len(ts), 0)])
    seasonal_ts = darts.TimeSeries.from_series(pd.Series(data=seasonal_ts, index=ts.time_index))

    ts.plot(label="Measurements")
    seasonal_ts.plot(label="FFT filtered")

    xlabel("Time")
    ylabel("Number of tasks")
    title("Online Store")

    # x, y = variables('x, y')
    # w, = parameters('w', value=1/2880)
    # model_dict = {y: fourier_series(x, f=w, n=4)}
    #
    # x_arr = np.linspace(0, len(data), num=len(data))
    # fit = Fit(model_dict, x=x_arr, y=data.to_numpy())
    # fit_result = fit.execute()
    #
    # print(fit_result)
    # plot(x_arr, fit.model(x=x_arr, **fit_result.params).y, color='green', ls=':')

    # deseasonalizer = Deseasonalizer(sp=2880)
    # ts.plot()
    # darts.TimeSeries.from_series(deseasonalizer.fit_transform(ts.pd_series())).plot()

    ##result = STL(ts.pd_series(), period=2880)
    # result.plot()

    # ts.plot()
    # (ts-res).plot()
    show()


if __name__ == '__main__':
    #test_filter()

    tasks_numbers = load_data("../validation_data/test_data.csv")

    test_size = int(len(tasks_numbers) * 0.3)
    first_time = datetime.datetime.now()

    history = []
    index = []

    for i in range(len(tasks_numbers) - test_size):
        time = first_time + datetime.timedelta(seconds=i + 1)
        history.append(tasks_numbers[i])
        index.append(time)

    train_data = darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index)))
    predictor = ArimaPredictor(100)

    history = []
    index = []
    current = first_time
    pred = None

    for i in range(len(tasks_numbers)):
        current += datetime.timedelta(seconds=1)
        history.append(tasks_numbers[i])
        index.append(current)

        if i > 0 and i % 100 == 0:
            print(f"{i}")
            prediction = predictor.predict(darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index))))

            if pred is None:
                pred = prediction
            else:
                pred = pred.append(prediction)

    actual = darts.TimeSeries.from_series(pd.Series(data=history, index=pd.DatetimeIndex(index), name="Actual"))

    pred_slice = pred.slice(actual.start_time(), actual.end_time())
    actual_slice = actual.slice(pred_slice.start_time(), pred_slice.end_time())

    test_actual = actual_slice[-test_size:]
    test_pred = pred_slice[-test_size:]

    agg_pred_slice = []
    agg_actual_slice = []
    times = []
    i = 0

    while True:
        if 100 * i >= len(test_pred):
            agg_pred_slice = darts.TimeSeries.from_series(pd.Series(data=agg_pred_slice, index=pd.DatetimeIndex(times), name="Agg Actual"))
            agg_actual_slice = darts.TimeSeries.from_series(pd.Series(data=agg_actual_slice, index=pd.DatetimeIndex(times), name="Agg Pred"))
            break

        pred = test_pred[100 * i:100 * (i + 1)]
        actual = test_actual[100 * i:100 * (i + 1)]
        agg_pred_slice.append(pred.sum(axis=0).pd_series()[0])
        agg_actual_slice.append(actual.sum(axis=0).pd_series()[0])
        times.append(actual.start_time())
        i += 1

    test_actual = darts.TimeSeries.from_times_and_values(pd.RangeIndex(0, len(test_actual)), test_actual.values())
    test_pred = darts.TimeSeries.from_times_and_values(pd.RangeIndex(0, len(test_pred)), test_pred.values())

    test_actual.plot(label="Measurements")
    test_pred.plot(label="Prediction")
    xlabel("Time")
    ylabel("Number of tasks")
    title("Online Store")
    show()

    agg_actual_slice = darts.TimeSeries.from_times_and_values(pd.RangeIndex(0, len(agg_actual_slice)), agg_actual_slice.values())
    agg_pred_slice = darts.TimeSeries.from_times_and_values(pd.RangeIndex(0, len(agg_pred_slice)), agg_pred_slice.values())

    agg_actual_slice.plot(label="Measurements")
    agg_pred_slice.plot(label="Prediction")
    ylabel("Number of tasks")
    xlabel("Time")
    title("Online Store")
    show()

    points_r2 = r2_score(test_actual.pd_series(), test_pred.pd_series())
    agg_r2 = r2_score(agg_actual_slice.pd_series(), agg_pred_slice.pd_series())
    points_mae = mean_absolute_error(test_actual.pd_series(), test_pred.pd_series())
    agg_mae = mean_absolute_error(agg_actual_slice.pd_series(), agg_pred_slice.pd_series())
    points_me = max_error(test_actual.pd_series(), test_pred.pd_series())
    agg_me = max_error(agg_actual_slice.pd_series(), agg_pred_slice.pd_series())

    print("Points r2", points_r2)
    print("Agg r2", agg_r2)
    print("Points MAE", points_mae)
    print("Agg MAE", agg_mae)
    print("Points ME", points_me)
    print("Agg ME", agg_me)

    print(f"{round(points_r2, 2)} & {round(agg_r2, 2)} & {round(points_mae, 2)} & {round(agg_mae)} & {round(points_me)} & {round(agg_me)}")
