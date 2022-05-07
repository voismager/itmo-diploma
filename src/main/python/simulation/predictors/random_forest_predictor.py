from sklearn.ensemble import RandomForestRegressor
from keras import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler


from python.simulation.predictor import Predictor

import numpy as np


class NeuralPredictor(Predictor):
    def __init__(self):
        model = Sequential()
        model.add(LSTM(100, input_shape=(1, 100), activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
        self.model = model
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def __convert2matrix__(self, data_arr, look_back):
        X, Y = [], []
        for i in range(len(data_arr) - look_back):
            d = i + look_back
            X.append(data_arr[i:d])
            Y.append(data_arr[d])
        return np.array(X), np.array(Y)

    def short_name(self):
        return "N"

    def name(self):
        return "Neural"

    def get_prediction(self, history, t=1):
        arr = np.asarray(history)
        arr = np.reshape(arr, (-1, 1))
        arr = self.scaler.fit_transform(arr)

        X, Y = self.__convert2matrix__(arr, 100)
        print(X, Y)

        model = self.model
        model.fit(X, Y, epochs=100, batch_size=30)

        result = []
        window = arr[-100:]

        for time in range(t):
            prediction = model.predict([window])[0][0]
            window.append(prediction)
            window.pop(0)
            result.append(prediction)

        print(result)
        return result


class RandomForestPredictor(Predictor):
    def __init__(self):
        self.model = None

    def name(self):
        return "RandomForest"

    def short_name(self):
        return "RF"

    def get_prediction(self, history, t=1):
        model = RandomForestRegressor()
        model.fit(np.arange(len(history)).reshape(-1, 1), np.asarray(history))
        prediction = model.predict(np.arange(len(history), len(history) + t).reshape(-1, 1))
        return prediction
