import pmdarima as pm

from simulation.predictor import Predictor


class ArimaPredictor(Predictor):
    def __init__(self):
        self.model = None

    def name(self):
        return "AutoArima"

    def short_name(self):
        return "AA"

    def get_prediction(self, history, t=1):
        if self.model is None:
            self.model = pm.auto_arima(history, test='adf', error_action='ignore', trace=True,
                                       suppress_warnings=True, maxiter=15,
                                       seasonal=True, m=60, stationary=True)
        else:
            self.model.update(history[-t:])

        prediction = self.model.predict(n_periods=t)
        print(prediction)
        return prediction
