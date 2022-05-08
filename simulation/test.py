import numpy as np

from matplotlib.pyplot import plot, show


def __convert_to_matrix__(arr, look_back):
    x, y = [], []

    for i in range(len(arr) - look_back):
        d = i + look_back
        x.append(arr[i:d])
        y.append(arr[d])

    return \
        np.array(x).reshape((len(x), look_back, 1)), \
        np.array(y).reshape((len(y), 1, 1))


def load_data():
    import csv

    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))
        return rows


if __name__ == '__main__':
    data = load_data()
    plot(data[4000:4500])
    show()

    # num_samples, time_steps, input_dim, output_dim = 10, 5, 1, 1
    # x = np.random.uniform(size=(num_samples, time_steps, 1))
    # y = np.mean(x, axis=1, keepdims=True)
    # print(x)
    # print()
    # print(y)
    #
    # print()
    # data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # x, y = __convert_to_matrix__(data, 5)
    # print(x)
    # print()
    # print(y)

    # transformer = Differencer(lags=1)
    #
    # data = load_data()[10000:]
    # X = list(range(len(data)))
    # differenced = transformer.fit_transform(pd.Series(data))
    # X_diff = list(range(len(differenced)))
    #
    # plot(data)
    #
    # p_coeff = np.polyfit(X, data, 31)
    # p = np.poly1d(p_coeff)
    # plot(X, p(X))
    # show()

    # windows = [5, 7, 9, 51, 101, 201, 501, 1001, 2001, 3001]
    #
    # for w in windows:
    #     filtered = savgol_filter(differenced, w, 4, 1, 1.0, -1, "interp")
    #     plot(filtered)
    #     show()
