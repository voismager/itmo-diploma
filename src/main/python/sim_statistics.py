import math

from matplotlib.pyplot import plot, show, bar, legend, figure

from cost_metric import \
    SLAWaitingTimeCostMetric, \
    RentingCostMetric, \
    ScaleDownsCostMetric, \
    ScaleUpsCostMetric


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
        metric_prediction_mae = 0

        for tick, prediction_value in self.prediction_history:
            if tick < len(self.tasks_number_history):
                metric_prediction_mae += abs(prediction_value - self.tasks_number_history[tick])

        metric_prediction_mae = metric_prediction_mae / len(self.prediction_history)

        sla_metric, top_waiting_time_tasks = SLAWaitingTimeCostMetric(self.sla, self.start_predictor_after + self.worker_setup_delay).calculate(self.cluster,
                                                                                                                                                self.task_pool)
        renting_metric = RentingCostMetric().calculate(self.cluster, self.task_pool)

        results = {
            "Mean absolute error": metric_prediction_mae,
            f"SLA violation total time ({self.sla}s)": sla_metric,
            "Over-renting total time": renting_metric,
            "Total scale ups": ScaleUpsCostMetric().calculate(self.cluster, self.task_pool),
            "Total scale downs": ScaleDownsCostMetric().calculate(self.cluster, self.task_pool),
            "Top waiting time tasks": top_waiting_time_tasks,
            "Final cost": renting_metric * 0.7 + sla_metric * 0.3
        }

        # for cost in results:
        #    print(f"{cost[0]}: {cost[1]}")

        # costs_arr = np.asarray([cost[1] for cost in results])
        # coefficients = np.asarray([1, 2, 100, 10])

        # total_cost = np.sum(costs_arr * coefficients)
        # print(f"Total cost: {total_cost}")
        # print()
        return results

    def print(self, name):
        ticks_list = list(range(self.ticks))

        # plot(ticks_list, self.stats['queue_size'], label='Queue Size')
        # plot(ticks_list, stats['number_of_tasks'], label='Number of Tasks')
        # legend()
        # show()

        fig = figure()
        plot(ticks_list, self.tasks_number_history, label='Metric Value')
        plot([p[0] for p in self.prediction_history], [p[1] for p in self.prediction_history], label='Prediction Value')
        plot(ticks_list, self.stats['workers_number'], label='Workers Num.')
        fig.suptitle(name)
        legend()
        show()
