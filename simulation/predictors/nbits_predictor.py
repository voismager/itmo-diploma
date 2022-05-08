import numpy as np
from nbeats_keras.model import NBeatsNet as NBeatsKeras

from simulation.predictor import Predictor


def __unison_shuffled_copies__(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def __convert_to_matrix__(arr, look_back, splits):
    x, y = [], []

    for i in range(len(arr) - look_back):
        d = i + look_back
        x.append(arr[i:d])
        y.append(arr[d])

    x = np.array(x)
    y = np.array(y)
    x, y = __unison_shuffled_copies__(x, y)

    x = x.reshape((len(x), look_back, 1))
    y = y.reshape((len(y), 1, 1))

    x = np.array_split(x, splits)
    y = np.array_split(y, splits)

    return x, y


class Model(object):
    def __init__(self, x, y, horizon, backcast, loss):
        print(f"Creating model with {backcast} backcast...")

        c = len(y) // 10 * 9
        x_train, y_train, x_test, y_test = x[:c], y[:c], x[c:], y[c:]

        model = NBeatsKeras(
            backcast_length=backcast, forecast_length=horizon,
            stack_types=(NBeatsKeras.GENERIC_BLOCK, NBeatsKeras.GENERIC_BLOCK),
            nb_blocks_per_stack=3, thetas_dim=(4, 4), share_weights_in_stack=True,
            hidden_layer_units=128
        )

        model.compile(loss=loss, optimizer='adam')
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64, shuffle=True, verbose=True)

        self.horizon = horizon
        self.backcast = backcast
        self.model = model

    def predict(self, history):
        data = np.array(history[-self.backcast:])

        if len(data) < self.backcast:
            raise Exception(f"History should have at least {self.backcast} points!")

        data = data.reshape((1, -self.backcast, 1))
        prediction = self.model.predict(data).reshape(self.horizon)
        return prediction


class NbitsPredictor(Predictor):
    def __init__(self, horizon, train):
        self.offset = len(train)
        self.horizon = horizon
        self.__prepare_models__(train, horizon)

    def __prepare_models__(self, train, horizon):
        self.backcasts = [1024, 2048]
        self.models = []

        splits = 2

        for backcast in self.backcasts:
            for loss in ["mae"]:
                x, y = __convert_to_matrix__(train, backcast, splits)
                for split in range(splits):
                    self.models.append(Model(x[split], y[split], horizon, backcast, loss))

    def short_name(self):
        return "N-BITS"

    def name(self):
        return "N-BITS"

    def get_prediction(self, history, t=1):
        predictions = np.array([m.predict(history) for m in self.models])
        return np.median(predictions, axis=0)
