from worker import Worker


class Cluster(object):
    def submit(self, tasks):
        pass

    def scale_up(self, n):
        pass

    def on_tick(self, tick):
        pass

    def get_stats(self):
        pass


class QueueCluster(Cluster):
    def __init__(self, worker_capacity, worker_setup_delay, workers_number):
        self.worker_capacity = worker_capacity
        self.worker_setup_delay = worker_setup_delay
        self.tasks_queue = []
        self.active_workers = {Worker(self.worker_capacity, 0) for _ in range(workers_number)}
        self.stopped_workers = set()

        self.workers_number_stat = []
        self.queue_size_stat = []

        self.scale_ups = 0
        self.scale_downs = 0

    def __find_most_available_workers(self, n):
        def key_fun(worker: Worker):
            if worker.current_task is None:
                return 0

            return worker.current_task.left_length

        truly_active_workers = filter(lambda w: not w.marked_to_stop, self.active_workers)
        return sorted(list(truly_active_workers), key=key_fun)[0:n]

    def all_workers(self):
        return self.active_workers.union(self.stopped_workers)

    def truly_active_workers(self):
        return list(filter(lambda w: not w.marked_to_stop, self.active_workers))

    def get_stats(self):
        return {
            "workers_number": self.workers_number_stat,
            "queue_size": self.queue_size_stat
        }

    def submit(self, tasks):
        self.tasks_queue.extend(tasks)

    def scale_up(self, n):
        if n <= 0:
            return

        for i in range(n):
            self.scale_ups += 1
            self.active_workers.add(Worker(self.worker_capacity, self.worker_setup_delay))

    def scale_down(self, n):
        if n <= 0:
            return

        if len(self.active_workers) == 0:
            return

        stopped_workers = set()

        most_available_workers = self.__find_most_available_workers(n)

        for worker in most_available_workers:
            if worker.is_available_for_new_task():
                worker.stop()
                stopped_workers.add(worker)
                self.scale_downs += 1
            else:
                worker.marked_to_stop = True

        self.active_workers = self.active_workers - stopped_workers
        self.stopped_workers = self.stopped_workers.union(stopped_workers)

    def on_tick(self, tick):
        stopped_workers = set()

        for worker in self.active_workers:
            worker.on_tick(tick)

            if worker.is_available_for_new_task() and len(self.tasks_queue) > 0:
                worker.submit(self.tasks_queue.pop(0))

            if worker.is_stopped():
                stopped_workers.add(worker)
                self.scale_downs += 1

        self.active_workers = self.active_workers - stopped_workers
        self.stopped_workers = self.stopped_workers.union(stopped_workers)

        self.workers_number_stat.append(len(self.active_workers))
        self.queue_size_stat.append(len(self.tasks_queue))

        for task in self.tasks_queue:
            task.increase_waiting_time()
