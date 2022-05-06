class SimulationCache:
    def __init__(self):
        self.active_threads = None
        self.task_queue = None

    def empty(self):
        return self.active_threads is None

    def get(self):
        if self.active_threads is None:
            return None
        else:
            return self.active_threads, self.task_queue

    def put(self, active_threads, task_queue):
        self.active_threads = [dict(thr) for thr in active_threads]
        self.task_queue = [dict(task) for task in task_queue]

