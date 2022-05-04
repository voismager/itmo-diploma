from darts.models import FourTheta
from darts.utils.utils import SeasonalityMode


class ThetaPredictor:
    def __init__(self, window, horizon):
        self.window = window
        self.horizon = horizon

    def predict(self, history):
        model = FourTheta(theta=3, season_mode=SeasonalityMode.ADDITIVE)
        model.fit(history[-self.window:])
        return model.predict(self.horizon)
