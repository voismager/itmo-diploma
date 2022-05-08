import csv

import numpy as np
from matplotlib.pyplot import plot, show, bar, figure
from scipy.optimize import differential_evolution
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sktime.transformations.series.detrend import Detrender, Deseasonalizer

from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.hp_filter import hpfilter

from scipy import fftpack
from cluster import QueueCluster
from sim_statistics import Statistics
from predictor import \
    PrecisePredictor, \
    LastValuePredictor, \
    WeighedMovingAveragePredictor
from predictors.nbits_predictor import NbitsPredictor
from predictors.my_predictor import MyPredictor
from predictors.arima_predictor import ArimaPredictor
from predictors.composite_predictor import CompositePredictor
from predictors.random_forest_predictor import RandomForestPredictor, NeuralPredictor
from scaling_decision_maker import ScalingDecisionMaker
from task import TaskPool

SLA = 100


def load_data():
    with open("validation_data.csv", "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))
        return rows


def print_data(input_data):
    X = list(range(len(input_data)))

    plot(X, input_data)
    show()

    #from sktime.utils.seasonality import autocorrelation_seasonality_test
    #import pandas as pd

    #print(autocorrelation_seasonality_test(pd.Series(input_data), 1250))
    #result = seasonal_decompose(input_data, period=1250)

    #cycle, trend = cffilter(input_data, low=1200, high=1500, drift=False)

    #sig_fft = fftpack.fft(input_data)
    #power = np.abs(sig_fft)**2

    # sample_freq = fftpack.fftfreq(len(input_data), d=len(input_data))
    #
    # pos_mask = np.where(sample_freq > 0)
    # freqs = sample_freq[pos_mask]
    # peak_freq = freqs[power[pos_mask].argmax()]
    #
    # high_freq_fft = sig_fft.copy()
    # high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
    # filtered_sig = fftpack.ifft(high_freq_fft)
    #
    # plot(X, input_data)
    # plot(X, filtered_sig)
    #plot(X, input_data - filtered_sig)

    # The second value is the p-value.
    # If this p-value is smaller than 0.05
    # you can reject the null hypothesis (reject non-stationarity)
    # and accept the alternative hypothesis (stationarity).
    # adf, pval, usedlag, nobs, crit_vals, icbest = adfuller(input_data)
    # print('ADF test statistic:', adf)
    # print('ADF p-values:', pval)
    # print('ADF number of lags used:', usedlag)
    # print('ADF number of observations:', nobs)
    # print('ADF critical values:', crit_vals)
    # print('ADF best information criterion:', icbest)
    #
    #plot_acf(np.asarray(input_data), lags=100)
    #plot_pacf(np.asarray(input_data), lags=50)
    show()


def run(input_data, predictor, scaling_decision_maker, start_predictor_after, worker_setup_delay):
    number_of_tasks_iter = iter(input_data)
    tasks_number_stat = input_data

    scale_decision_each_t = worker_setup_delay
    task_length = 5
    task_pool = TaskPool()
    cluster = QueueCluster(1, worker_setup_delay, 200)
    tick = 0

    statistics = Statistics(cluster, task_pool, start_predictor_after, worker_setup_delay, SLA)

    while True:
        current_number_of_tasks = next(number_of_tasks_iter, -1)

        if current_number_of_tasks == -1:
            if task_pool.all_tasks_are_finished():
                break
            else:
                statistics.add_tasks_metric([])
                tasks_number_stat.append(0)
        else:
            tasks = [task_pool.create_task(task_length, tick) for _ in range(current_number_of_tasks)]
            cluster.submit(tasks)
            statistics.add_tasks_metric(tasks)

        cluster.on_tick(tick)

        if tick % scale_decision_each_t == 0 and tick >= start_predictor_after:
            predictions = predictor.get_prediction(statistics.tasks_number_history, t=scale_decision_each_t)
            statistics.add_prediction(tick, predictions)
            scaling_decision_maker.decide(statistics.tasks_number_history, statistics.tasks_length_histogram, predictions, scale_decision_each_t, cluster)

        task_pool.update()

        if tick % 1000 == 0:
            print(f"Tick {tick}")
            pass

        tick += 1

    statistics.set_other_stats(cluster.get_stats(), tick)
    return statistics


def optimize_scale_decision_maker(input_data):
    def run_based_on_data(x):
        c_1, c_2, c_3, dur_up, dur_down = x

        if c_1 > c_2 > c_3:
            print("Input:", x)
            result_stats = run(input_data, PrecisePredictor(input_data, 200, 5), ScalingDecisionMaker(c_1, c_2, c_3, dur_up, dur_down), 200, 100)
            return result_stats.get_cost_metric_value()
        else:
            return 1e16

    print(differential_evolution(
        run_based_on_data,
        [(0, 1), (0, 1), (0, 1), (1, 10), (1, 10)],
        maxiter=3
    ))


def run_visuals(train_data, test_data):
    predictors = [
        #ArimaPredictor(),
        #NeuralPredictor(),
        MyPredictor(train_data),
        #NbitsPredictor(100, train_data),
        #CustomPredictor(100),
        #TbatsPredictor(train_data),
        #RandomForestPredictor(),
        #LastValuePredictor(),
        # SimpleMovingAveragePredictor(500),
        WeighedMovingAveragePredictor(500, "Fibonacci"),
        #PrecisePredictor(input_data, 200, 5),
        # WeighedMovingAveragePredictor(500, "Fibonacci"),
        # DoubleExponentialSmoothingPredictor(500, "Fibonacci", 0.9, 0.6),
        # DoubleExponentialSmoothingPredictor(500, "Uniform", 0.2, 0.2)
    ]

    metrics = dict()

    for predictor in predictors:
        stats = run(test_data, predictor, ScalingDecisionMaker(1, 0.75, 1, 2, 1), 2048, 100)
        stats.print(predictor.name())
        results = stats.evaluate()

        for k, v in results.items():
            if k not in metrics:
                metrics[k] = dict()

            metrics[k][predictor.short_name()] = v

    for name, metric in metrics.items():
        fig = figure()
        bar(metric.keys(), metric.values())
        fig.suptitle(name)
        show()


if __name__ == '__main__':
    data = load_data()
    print_data(data)
    #train = data[0:35000]
    #test = data[35000:44000]
    #run_visuals(train, test)
