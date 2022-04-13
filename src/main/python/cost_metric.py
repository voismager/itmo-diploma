from cluster import QueueCluster
from task import TaskPool


class CostMetric(object):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool):
        pass


class SLAWaitingTimeCostMetric(CostMetric):
    def __init__(self, threshold, delay):
        self.threshold = threshold
        self.delay = delay

    def calculate(self, cluster: QueueCluster, task_pool: TaskPool):
        top_waiting_time = sorted([t for t in task_pool.finished_tasks if t.created_at > self.delay], key=lambda t: t.waiting_time)
        waiting_time = [max(t.waiting_time - self.threshold, 0) for t in task_pool.finished_tasks if t.created_at > self.delay]
        return sum(waiting_time), top_waiting_time[-100:]


class RentingCostMetric(CostMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool):
        total_needed_time = sum([t.total_length for t in task_pool.finished_tasks])
        total_worker_time = sum([worker.running_time for worker in cluster.all_workers()])
        return total_worker_time - total_needed_time


class ScaleUpsCostMetric(CostMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool):
        return cluster.scale_ups


class ScaleDownsCostMetric(CostMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool):
        return cluster.scale_downs
