from random import choices

from cluster import QueueCluster
from task import Task


# Indicates how many tasks can be started in period from 0 to t offset
def number_of_potential_tasks(t, cluster: QueueCluster, task_lengths_distr: (list, list)):
    result = 0
    simulated_workers = []

    for worker in cluster.active_workers:
        simulated_workers.append(worker.get_copy())

    for time in range(t):
        for worker in simulated_workers:
            worker.on_tick(time)

            if worker.is_available_for_new_task():
                result += 1
                task_length = choices(task_lengths_distr[0], task_lengths_distr[1])[0]
                worker.submit(Task(task_length, time))

    result -= len(cluster.tasks_queue)
    return result


if __name__ == '__main__':
    cluster = QueueCluster(1, 100, 10)
    task_lengths_distr = ([5], [1.0])
    result = number_of_potential_tasks(100, cluster, task_lengths_distr)
    print(result)
