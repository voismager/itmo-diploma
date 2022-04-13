import csv

import numpy as np
from matplotlib.pyplot import plot, show, bar, legend, figure
from scipy.optimize import differential_evolution

from cluster import QueueCluster
from cost_metric import \
    SLAWaitingTimeCostMetric, \
    RentingCostMetric, \
    ScaleDownsCostMetric, \
    ScaleUpsCostMetric
from load_metric import \
    TotalRequiredLengthLoadMetric, \
    TasksNumberLoadMetric, \
    MedianWaitingTimeLoadMetric
from predictor import \
    PrecisePredictor, \
    LastValuePredictor, \
    SimpleMovingAveragePredictor, \
    WeighedMovingAveragePredictor, \
    DoubleExponentialSmoothingPredictor, \
    ArimaPredictor, \
    TbatsPredictor
from scaling_decision_maker import ScalingDecisionMaker
from task import TaskPool

SLA = 100


class Statistics(object):
    def __init__(self, cluster, task_pool):
        self.cluster = cluster
        self.task_pool = task_pool
        self.metric_history = []
        self.prediction_history = []
        self.stats = dict()
        self.ticks = None

    def add_metric(self, metric_value):
        self.metric_history.append(metric_value)

    def add_prediction(self, start_tick, predictions):
        for tick in range(start_tick, start_tick + len(predictions)):
            self.prediction_history.append((tick, predictions[tick - start_tick]))

    def set_other_stats(self, stats, ticks):
        self.stats = stats
        self.ticks = ticks

    def evaluate(self, start_predictor_after, worker_setup_delay):
        metric_prediction_mae = 0

        for tick, prediction_value in self.prediction_history:
            if tick < len(self.metric_history):
                metric_prediction_mae += abs(prediction_value - self.metric_history[tick])

        metric_prediction_mae = metric_prediction_mae / len(self.prediction_history)

        sla_metric, top_waiting_time_tasks = SLAWaitingTimeCostMetric(SLA, start_predictor_after + worker_setup_delay).calculate(self.cluster, self.task_pool)

        results = {
            "Mean absolute error": metric_prediction_mae,
            f"SLA violation total time ({SLA}s)": sla_metric,
            "Over-renting total time": RentingCostMetric().calculate(self.cluster, self.task_pool),
            "Total scale ups": ScaleUpsCostMetric().calculate(self.cluster, self.task_pool),
            "Total scale downs": ScaleDownsCostMetric().calculate(self.cluster, self.task_pool),
            "Top waiting time tasks": top_waiting_time_tasks
        }

        #for cost in results:
        #    print(f"{cost[0]}: {cost[1]}")

        #costs_arr = np.asarray([cost[1] for cost in results])
        #coefficients = np.asarray([1, 2, 100, 10])

        #total_cost = np.sum(costs_arr * coefficients)
        #print(f"Total cost: {total_cost}")
        #print()
        return results

    def print(self, name):
        ticks_list = list(range(self.ticks))

        # plot(ticks_list, self.stats['queue_size'], label='Queue Size')
        # plot(ticks_list, stats['number_of_tasks'], label='Number of Tasks')
        # legend()
        # show()

        fig = figure()
        plot(ticks_list, self.metric_history, label='Metric Value')
        plot([p[0] for p in self.prediction_history], [p[1] for p in self.prediction_history], label='Prediction Value')
        plot(ticks_list, self.stats['workers_number'], label='Workers Num.')
        fig.suptitle(name)
        legend()
        show()


def load_data():
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))
        return rows


def print_data(input_data):
    plot(list(range(len(input_data))), input_data, label="Data")
    show()


def run(input_data, predictor, start_predictor_after, worker_setup_delay):
    number_of_tasks_iter = iter(input_data)
    tasks_number_stat = input_data

    scale_decision_each_t = worker_setup_delay
    task_length = 5
    task_pool = TaskPool()
    cluster = QueueCluster(1, worker_setup_delay, 2)
    metric = TasksNumberLoadMetric()
    tick = 0
    scaling_decision_maker = ScalingDecisionMaker(500, 100, 60, 10, 4)

    statistics = Statistics(cluster, task_pool)

    while True:
        current_number_of_tasks = next(number_of_tasks_iter, -1)

        if current_number_of_tasks == -1:
            if task_pool.all_tasks_are_finished():
                break
            else:
                statistics.add_metric(metric.calculate(cluster, task_pool, []))
                tasks_number_stat.append(0)
        else:
            tasks = [task_pool.create_task(task_length, tick) for _ in range(current_number_of_tasks)]
            cluster.submit(tasks)
            statistics.add_metric(metric.calculate(cluster, task_pool, tasks))

        cluster.on_tick(tick)

        if tick % scale_decision_each_t == 0 and tick >= start_predictor_after:
            predictions = predictor.get_prediction(statistics.metric_history, t=scale_decision_each_t)
            statistics.add_prediction(tick, predictions)
            scaling_decision_maker.decide(statistics.metric_history, predictions, scale_decision_each_t, cluster)

        task_pool.update()

        if tick % 1000 == 0:
            print(f"Tick {tick}")
            pass

        tick += 1

    statistics.set_other_stats(cluster.get_stats(), tick)
    return statistics


def optimize_des_predictor(input_data):
    def run_based_on_data(x):
        result_stats = run(input_data, DoubleExponentialSmoothingPredictor([1, 1, 2, 3, 5, 8, 13], x[0], x[1]))
        return result_stats.evaluate()

    print(differential_evolution(run_based_on_data, [(0, 1), (0, 1)]))


def run_visuals(input_data):
    predictors = [
        PrecisePredictor(input_data, 200, 5),
        #LastValuePredictor(),
        #SimpleMovingAveragePredictor(500),
        #WeighedMovingAveragePredictor(500, "Fibonacci"),
        #DoubleExponentialSmoothingPredictor(500, "Fibonacci", 0.9, 0.6),
        #DoubleExponentialSmoothingPredictor(500, "Uniform", 0.2, 0.2)
    ]

    mean_absolute_error = dict()
    sla_violation_time = dict()
    over_renting_time = dict()
    scale_ups = dict()
    scale_downs = dict()

    for predictor in predictors:
        stats = run(input_data, predictor, 200, 100)
        stats.print(predictor.name())
        results = stats.evaluate(200, 100)

        print("Mean absolute error:", results["Mean absolute error"])
        print(f"SLA violation total time ({SLA}s):", results[f"SLA violation total time ({SLA}s)"])
        print("Over-renting total time", results["Over-renting total time"])
        print("Top waiting time tasks: ", [t.waiting_time for t in results["Top waiting time tasks"]])

        mean_absolute_error[predictor.short_name()] = results["Mean absolute error"]
        sla_violation_time[predictor.short_name()] = results[f"SLA violation total time ({SLA}s)"]
        over_renting_time[predictor.short_name()] = results["Over-renting total time"]
        scale_ups[predictor.short_name()] = results["Total scale ups"]
        scale_downs[predictor.short_name()] = results["Total scale downs"]

    fig = figure()
    bar(mean_absolute_error.keys(), mean_absolute_error.values())
    fig.suptitle("Mean absolute error")
    show()

    fig = figure()
    bar(sla_violation_time.keys(), sla_violation_time.values())
    fig.suptitle(f"SLA violation total time ({SLA}s)")
    show()

    fig = figure()
    bar(over_renting_time.keys(), over_renting_time.values())
    fig.suptitle("Over-renting total time")
    show()

    fig = figure()
    bar(scale_ups.keys(), scale_ups.values())
    fig.suptitle("Total scale ups")
    show()

    fig = figure()
    bar(scale_downs.keys(), scale_downs.values())
    fig.suptitle("Total scale downs")
    show()


if __name__ == '__main__':
    data = load_data()[10000:20000]
    # print_data(data)
    # optimize_des_predictor(data)
    run_visuals(data)
