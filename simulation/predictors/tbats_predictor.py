from sktime.forecasting.tbats import TBATS
import numpy as np

from simulation.predictor import Predictor


class TbatsPredictor(Predictor):
    def __init__(self, data):
        self.forecaster = TBATS(
            use_box_cox=False,
            use_trend=False,
            use_damped_trend=False,
            sp=[1000],
            use_arma_errors=False,
            n_jobs=1)

        self.forecaster.fit(np.asarray(data))
        self.offset = 1

    def short_name(self):
        return "TBATS"

    def name(self):
        return "TBATS"

    def get_prediction(self, history, t=1):
        fh = list(range(self.offset, t + self.offset))
        self.offset += t
        prediction = [max(0, p[0]) for p in self.forecaster.predict(fh=fh)]
        print(prediction)
        return prediction
