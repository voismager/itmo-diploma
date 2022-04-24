import math
import statistics as st

from cluster import QueueCluster
from potential_tasks_counter import number_of_potential_tasks


def median_absolute_deviation(input_data):
    median = st.median(input_data)
    abs_dif = [abs(median - x) for x in input_data]
    return st.median(abs_dif)


def provisioned_capacity_at(t, cluster: QueueCluster):
    result = 0

    for worker in cluster.active_workers:
        if worker.is_available_for_new_task(t=t):
            result += worker.capacity

    return result


class ScalingDecisionMaker(object):
    def __init__(self, c_1, c_2, c_3, dur_up, dur_down):
        self.c_1 = 1
        self.c_2 = 1
        self.c_3 = 1
        self.tick_up_timer = 5
        self.tick_down_timer = 5
        self.dur_up = int(dur_up)
        self.dur_down = int(dur_down)

    def decide(self, tasks_number_history, tasks_length_histogram, predictions, prediction_delay, cluster):
        distr = tasks_length_histogram.as_distr()
        provisioned = number_of_potential_tasks(prediction_delay, cluster, distr)
        prediction = sum(predictions)

        if prediction > provisioned:
            self.tick_up_timer = 0
            self.tick_down_timer = 0
            workers = math.ceil((prediction - provisioned) * (tasks_length_histogram.median() / prediction_delay))
            cluster.scale_up(workers)

        elif prediction < provisioned:
            self.tick_up_timer = 0
            self.tick_down_timer += 1
            if self.tick_down_timer > self.dur_down:
                workers = math.ceil((provisioned - prediction) * (tasks_length_histogram.median() / prediction_delay))
                cluster.scale_down(min(workers, len(cluster.truly_active_workers()) - 1))

        # elif prediction > provisioned * self.c_2:
        #     self.tick_down_timer = 0
        #     self.tick_up_timer += 1
        #     if self.tick_up_timer > self.dur_up:
        #         cluster.scale_up(prediction - provisioned)

        else:
            self.tick_up_timer = 0
            self.tick_down_timer = 0
