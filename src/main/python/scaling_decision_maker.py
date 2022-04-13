import math
import statistics as st

from cluster import QueueCluster


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


class SLABasedScalingDecisionMaker(object):
    def __init__(self, sla, scaling_delay):
        self.sla = sla
        self.scaling_delay = scaling_delay
        self.tick_up_timer = 0
        self.tick_down_timer = 0
        self.c_1 = 0.75
        self.c_2 = 0.5
        self.c_3 = 0.2

    def decide(self, predicted_median_sla, cluster: QueueCluster):
        if predicted_median_sla > self.sla * self.c_1:
            self.tick_up_timer = 0
            self.tick_down_timer = 0
            workers = int(predicted_median_sla / (self.sla * self.c_1))
            cluster.scale_up(workers)

        elif predicted_median_sla > self.sla * self.c_2:
            self.tick_down_timer = 0
            self.tick_up_timer += 1
            if self.tick_up_timer > self.scaling_delay:
                cluster.scale_up(1)

        elif predicted_median_sla < self.sla * self.c_3:
            self.tick_up_timer = 0
            self.tick_down_timer += 1
            if self.tick_down_timer > self.scaling_delay:
                cluster.scale_down(1)

        else:
            self.tick_up_timer = 0
            self.tick_down_timer = 0


class ScalingDecisionMaker(object):
    def __init__(self, window_size, c_1, c_2, c_3, scaling_delay):
        self.window_size = window_size
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.dur_u = 0
        self.timer_0 = 0

        self.tick_up_timer = 0
        self.tick_down_timer = 0
        self.scaling_delay = scaling_delay

    def decide(self, history, predictions, prediction_delay, cluster):
        provisioned = provisioned_capacity_at(prediction_delay, cluster)
        diff = math.ceil(sum(predictions) / 16) - provisioned

        if diff > 0:
            cluster.scale_up(diff)
        elif diff < 0:
            diff = min(-diff, len(cluster.truly_active_workers()) - 1)
            cluster.scale_down(diff)

        # prediction = sum(predictions)
        #
        # if prediction > provisioned:
        #     self.timer_0 += 0
        #     if self.timer_0 > self.dur_u:
        #         cluster
        #
        #
        # if diff > self.c_1:
        #     self.tick_up_timer = 0
        #     self.tick_down_timer = 0
        #     cluster.scale_up(math.ceil(diff / self.c_1))
        # elif diff > self.c_2:
        #     self.tick_down_timer = 0
        #     self.tick_up_timer += 1
        #     if self.tick_up_timer > self.scaling_delay:
        #         cluster.scale_up(math.ceil(diff / self.c_2))
        # elif diff < self.c_3:
        #     self.tick_up_timer = 0
        #     self.tick_down_timer += 1
        #     if self.tick_down_timer > self.scaling_delay:
        #         if diff > 0:
        #             cluster.scale_down(math.ceil(self.c_3 / diff))
        #         elif diff == 0:
        #             cluster.scale_down(math.ceil(self.c_3))
        #         else:
        #             cluster.scale_down(min(math.ceil(self.c_3 - diff / self.c_3), self.c_1))
        # else:
        #     self.tick_up_timer = 0
        #     self.tick_down_timer = 0
