import statistics as st

from cluster import QueueCluster
from task import TaskPool


class LoadMetric(object):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool, new_tasks):
        pass


class MedianWaitingTimeLoadMetric(LoadMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool, new_tasks):
        queue = cluster.tasks_queue
        if len(queue) == 0:
            return 0
        else:
            return st.median([t.waiting_time for t in queue])


class TotalRequiredLengthLoadMetric(LoadMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool, new_tasks):
        return sum([t.total_length for t in new_tasks])


class TasksNumberLoadMetric(LoadMetric):
    def calculate(self, cluster: QueueCluster, task_pool: TaskPool, new_tasks):
        return len(new_tasks)


class QueueSizeLoadMetric(LoadMetric):
    def __init__(self, norm_coefficient):
        self.max_size = norm_coefficient

    def calculate(self, cluster, task_pool: TaskPool, new_tasks):
        queue = cluster.tasks_queue
        return len(queue) / self.max_size
