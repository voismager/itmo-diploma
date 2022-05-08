class Task(object):
    def __init__(self, length, created_at):
        self.total_length = length
        self.left_length = length
        self.created_at = created_at
        self.waiting_time = 0

    def increase_waiting_time(self):
        self.waiting_time += 1

    def is_finished(self):
        return self.left_length == 0

    def decrease_length(self, by):
        self.left_length = max(0, self.left_length - by)

    def get_copy(self):
        task = Task(self.total_length, self.created_at)
        task.left_length = self.left_length
        task.waiting_time = self.waiting_time
        return task


class TaskPool(object):
    def __init__(self):
        self.active_tasks = []
        self.finished_tasks = []

    def total_required_capacity(self):
        return sum(map(lambda t: t.left_length, self.active_tasks))

    def update(self):
        finished_tasks = [t for t in self.active_tasks if t.is_finished()]
        active_tasks = [t for t in self.active_tasks if not t.is_finished()]
        self.finished_tasks.extend(finished_tasks)
        self.active_tasks = active_tasks
        return len(active_tasks)

    def all_tasks_are_finished(self) -> bool:
        return len(self.active_tasks) == 0

    def create_task(self, length, tick):
        task = Task(length, tick)
        self.active_tasks.append(task)
        return task
