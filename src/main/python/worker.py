import math

from task import Task

STARTING = 0
RUNNING = 1
STOPPED = 2

ID_COUNTER = 0


class Worker(object):
    def __init__(self, capacity, delay):
        global ID_COUNTER
        ID_COUNTER += 1
        self.id = ID_COUNTER

        self.running_time = 0
        self.capacity = capacity
        self.current_task = None
        self.delay = delay
        self.marked_to_stop = False

        if delay == 0:
            self.state = RUNNING
        else:
            self.state = STARTING

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def submit(self, task: Task):
        if self.state != RUNNING:
            raise Exception("Worker is not active!")

        if self.current_task is not None:
            raise Exception("Current task is in progress!")

        self.current_task = task

    def on_tick(self, tick: int):
        if self.state == RUNNING:
            self.running_time += 1

            if self.current_task is None:
                return

            self.current_task.decrease_length(self.capacity)

            if self.current_task.is_finished():
                self.current_task = None

                if self.marked_to_stop:
                    self.stop()

        elif self.state == STARTING:
            self.delay -= 1
            if self.delay == 0:
                self.state = RUNNING

    def is_stopped(self):
        return self.state == STOPPED

    def is_available_for_new_task(self, t=0) -> bool:
        if t <= 0:
            return self.state == RUNNING and self.current_task is None
        else:
            if self.state == STARTING:
                return self.delay <= t
            elif self.state == RUNNING:
                if self.current_task is None:
                    return True
                else:
                    if self.marked_to_stop:
                        return False

                    return math.ceil(self.current_task.left_length / self.capacity) <= t
            else:
                return False

    def stop(self):
        self.state = STOPPED
