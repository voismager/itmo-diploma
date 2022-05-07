import numpy as np
from darts import TimeSeries
from darts.models import FourTheta
from darts.utils.utils import SeasonalityMode

from python.simulation.predictor import Predictor


def load_data():
    import csv
    with open("../data.csv", "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))
        return rows


class MyPredictor(Predictor):
    def __init__(self, data):
        pass

        # self.model = NBEATSModel(
        #     input_chunk_length=512,
        #     output_chunk_length=100,
        #     generic_architecture=True,
        #     num_stacks=10,
        #     num_blocks=1,
        #     num_layers=4,
        #     layer_widths=512,
        #     n_epochs=10,
        #     nr_epochs_val_period=1,
        #     batch_size=800,
        #     model_name="nbeats_run",
        # )

        # self.model = TCNModel(
        #     input_chunk_length=512,
        #     output_chunk_length=100,
        #     kernel_size=2,
        #     num_filters=4,
        #     dilation_base=2,
        #     dropout=0,
        #     random_state=0,
        #     likelihood=GaussianLikelihood(),
        # )
        #
        # data = np.array(data)
        # c = int(len(data) * 0.9)
        # train, val = data[:c], data[c:]
        #
        # train = TimeSeries.from_values(train)
        # val = TimeSeries.from_values(val)
        #
        # self.model.fit(train, val_series=val, verbose=True, epochs=10)

    def short_name(self):
        return "My"

    def name(self):
        return "My"

    def get_prediction(self, history, t=1):
        model = FourTheta(theta=3, season_mode=SeasonalityMode.ADDITIVE)
        model.fit(TimeSeries.from_values(np.array(history[-512:])))
        prediction = model.predict(t)
        return prediction.values().reshape(t)
