import math

from matplotlib.pyplot import plot, show, legend, figure
from sklearn.metrics import r2_score, mean_absolute_error, max_error

from cost_metric import \
    SLAWaitingTimeCostMetric, \
    RentingCostMetric, \
    ScaleDownsCostMetric, \
    ScaleUpsCostMetric


def mae(actual, predicted):
    return mean_absolute_error(actual, predicted)


def max_err(actual, predicted):
    return max_error(actual, predicted)


def corr(actual, predicted):
    return r2_score(actual, predicted)


class Histogram(object):
    def __init__(self):
        self.bins = dict()

    def add(self, key):
        if key in self.bins:
            self.bins[key] += 1
        else:
            self.bins[key] = 1

    def mean(self):
        result = 0
        total = 0
        for k, v in self.bins.items():
            result += (k * v)
            total += v
        return result / total

    def median(self):
        values_sum = sum(self.bins.values())
        total = 0

        if values_sum % 2 == 0:
            median_index = values_sum / 2 - 1
            sorted_keys = sorted(self.bins.keys())
            keys_iter = iter(sorted_keys)
            for k in keys_iter:
                total += self.bins[k]
                if total > median_index:
                    if total > median_index + 1:
                        return k
                    else:
                        return (k + next(keys_iter)) / 2
        else:
            median_index = math.floor(values_sum / 2)
            for k in sorted(self.bins.keys()):
                total += self.bins[k]
                if total >= median_index:
                    return k

    def as_distr(self) -> (list, list):
        values_sum = sum(self.bins.values())
        keys = []
        weights = []

        for k, v in self.bins.items():
            keys.append(k)
            weights.append(v / values_sum)

        return keys, weights


class Statistics(object):
    def __init__(self, cluster, task_pool, start_predictor_after, worker_setup_delay, sla):
        self.cluster = cluster
        self.sla = sla
        self.task_pool = task_pool
        self.start_predictor_after = start_predictor_after
        self.worker_setup_delay = worker_setup_delay
        self.tasks_number_history = []
        self.tasks_length_histogram = Histogram()
        self.prediction_history = []
        self.stats = dict()
        self.ticks = None

    def add_tasks_metric(self, new_tasks):
        self.tasks_number_history.append(len(new_tasks))
        for task in new_tasks:
            self.tasks_length_histogram.add(task.total_length)

    def add_prediction(self, start_tick, predictions):
        for tick in range(start_tick, start_tick + len(predictions)):
            self.prediction_history.append((tick, predictions[tick - start_tick]))

    def set_other_stats(self, stats, ticks):
        self.stats = stats
        self.ticks = ticks

    def get_cost_metric_value(self):
        sla_cost, _ = SLAWaitingTimeCostMetric(self.sla, self.start_predictor_after + self.worker_setup_delay) \
            .calculate(self.cluster, self.task_pool)

        renting_cost = RentingCostMetric().calculate(self.cluster, self.task_pool)

        cost = sla_cost * 0.3 + renting_cost * 0.7
        print(f"Cost: {cost}")
        return cost

    def evaluate(self):
        start = self.prediction_history[0][0]
        actual = self.tasks_number_history[start:start + len(self.prediction_history)]
        prediction = [p[1] for p in self.prediction_history[:len(actual)]]

        metric_corr = corr(actual, prediction)
        metric_mae = mae(actual, prediction)
        metric_max_err = max_err(actual, prediction)

        metric_sla, top_waiting_time_tasks = SLAWaitingTimeCostMetric(self.sla, self.start_predictor_after + self.worker_setup_delay).calculate(self.cluster,                                                                                                                      self.task_pool)
        metric_rent = RentingCostMetric().calculate(self.cluster, self.task_pool)

        results = {
            "Mean absolute error": metric_mae,
            "Coefficient of determination": metric_corr,
            "Max error": metric_max_err,
            f"SLA violation total time ({self.sla}s)": metric_sla,
            "Over-renting total time": metric_rent,
            #"Total scale ups": ScaleUpsCostMetric().calculate(self.cluster, self.task_pool),
            #"Total scale downs": ScaleDownsCostMetric().calculate(self.cluster, self.task_pool),
            # "Top waiting time tasks": top_waiting_time_tasks,
            "Final cost": metric_rent * 0.7 + metric_sla * 0.3
        }

        return results

    def print(self, name):
        ticks_list = list(range(self.ticks))

        fig = figure()
        plot(ticks_list, self.tasks_number_history, label='Metric Value')
        plot([p[0] for p in self.prediction_history], [p[1] for p in self.prediction_history], label='Prediction Value')
        #plot(ticks_list, self.stats['workers_number'], label='Workers Num.')
        fig.suptitle(name)
        legend()
        show()
