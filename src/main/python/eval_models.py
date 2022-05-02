import numpy as np
from darts.models import ExponentialSmoothing, Prophet, AutoARIMA, Theta, FourTheta
from darts import TimeSeries
from darts.metrics import mape, mae, smape
from darts.utils.utils import ModelMode, SeasonalityMode, TrendMode


def load_data():
    import csv
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))
        return rows


def eval_model(train, val, model):
    model.fit(train)
    forecast = model.predict(len(val))
    print("Model {} obtains sMAPE: {:.2f}%".format(model, smape(val, forecast)))


if __name__ == '__main__':
    data = TimeSeries.from_values(np.array(load_data()))
    train = data[0:35000]
    val = data[35000:44000]

    eval_model(train, val, ExponentialSmoothing())
    eval_model(train, val, AutoARIMA())
    eval_model(train, val, Theta(season_mode=SeasonalityMode.ADDITIVE))
    eval_model(train, val, FourTheta(theta=4, season_mode=SeasonalityMode.ADDITIVE))


# if __name__ == '__main__':
#     data = load_data()
#     train = data[0:35000]
#     test = data[35000:44000]
#
#     model = ExponentialSmoothing()
#     #predictor = MyPredictor(train)
#
#     model.fit(TimeSeries.from_values(np.array(train)))
#     print(model.seasonal_periods)
#     model.predict(2000).plot()
#
#     #plot(list(range(len(data))), data)
#     #predictor.model.predict(2000).plot()
#     show()
    #prediction = predictor.get_prediction(test[0:512], t=100)
    #print(prediction)