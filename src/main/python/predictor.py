from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from tbats import BATS, TBATS


def weighed_average(weights, history):
    weighted = [weights[i] * element for i, element in enumerate(history)]
    return sum(weighted) / sum(weights)


def fibonacci(n):
    result = []
    a = 1
    b = 1
    result.append(a)
    for _ in range(n - 1):
        a, b = b, a + b
        result.append(a)
    return result


def uniform(n):
    return [1 for _ in range(n)]


class Predictor(object):
    def name(self):
        pass

    def short_name(self):
        pass

    def get_prediction(self, history, t=1):
        pass


class PrecisePredictor(Predictor):
    def __init__(self, data, offset, task_length):
        self.data = data
        self.offset = offset
        self.task_length = task_length

    def name(self):
        return "Precise"

    def short_name(self):
        return "P"

    def get_prediction(self, history, t=1):
        data = [x for x in self.data[self.offset: self.offset + t]]
        self.offset += t

        if len(data) == 0:
            return [0]
        else:
            return data


class LastValuePredictor(Predictor):
    def name(self):
        return "LastValue"

    def short_name(self):
        return "LV"

    def get_prediction(self, history, t=1):
        return [history[-1] for _ in range(t)]


class SimpleMovingAveragePredictor(Predictor):
    def __init__(self, window):
        self.window = window

    def name(self):
        return "SimpleMovingAverage"

    def short_name(self):
        return "SMA"

    def get_prediction(self, history, t=1):
        result = []
        window = history[-self.window:]

        for time in range(t):
            next_prediction = sum(window) / self.window
            window.append(next_prediction)
            window.pop(0)
            result.append(next_prediction)

        return result


class WeighedMovingAveragePredictor(Predictor):
    def __init__(self, weights_length, weights_name):
        if weights_name == "Uniform":
            self.weights = uniform(weights_length)
        elif weights_name == "Fibonacci":
            self.weights = fibonacci(weights_length)

        self.window = weights_length
        self.__name = f"WeighedMovingAverage(w={weights_name})"

    def name(self):
        return self.__name

    def short_name(self):
        return "WMA"

    def get_prediction(self, history, t=1):
        result = []
        window = history[-self.window:]

        for time in range(t):
            next_prediction = weighed_average(self.weights, window)
            window.append(next_prediction)
            window.pop(0)
            result.append(next_prediction)

        return result


class DoubleExponentialSmoothingPredictor(Predictor):
    def __init__(self, weights_length, weights_name, a, b):
        if weights_name == "Uniform":
            self.weights = uniform(weights_length)
        elif weights_name == "Fibonacci":
            self.weights = fibonacci(weights_length)

        self.window = weights_length
        self.a = a
        self.b = b
        self.current_trend = None
        self.current_smoothing = 0
        self.__name = f"DoubleExponentialSmoothing(a={self.a}, b={self.b}, w={weights_name})"

    def name(self):
        return self.__name

    def short_name(self):
        return "DES"

    def get_prediction(self, history, t=1):
        result = []
        window = history[-self.window:]

        if self.current_trend is None:
            self.current_trend = sum(window) / self.window

        for time in range(t):
            wma = weighed_average(self.weights, window)
            smoothing = (self.a * wma) + ((1 - self.a) * (self.current_smoothing + self.current_trend))
            trend = self.b * (smoothing - self.current_smoothing) + ((1 - self.b) * self.current_trend)
            self.current_smoothing = smoothing
            self.current_trend = trend
            next_prediction = max(smoothing + trend, 0)
            window.append(next_prediction)
            window.pop(0)
            result.append(next_prediction)

        return result


class ArimaPredictor(Predictor):
    def __init__(self, window):
        self.window = window

    def get_prediction(self, history, t=1):
        window = history[-self.window:]
        model = ARIMA(window, order=(5, 1, 0))
        model_fit = model.fit()
        return model_fit.forecast(n_periods=t)


class TbatsPredictor(Predictor):
    def get_prediction(self, history, t=1):
        estimator = TBATS(seasonal_periods=[1000])
        fitted_model = estimator.fit(history)
        forecast = fitted_model.forecast(steps=t)
        print(forecast)
        return forecast


