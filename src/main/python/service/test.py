import csv
import datetime
import random
import uuid
import ciso8601

import pandas as pd
import requests
import numpy as np
import darts

from decision_history import plot_history_dict
from matplotlib.pyplot import plot, show, bar, figure, hist

base_url = "http://127.0.0.1:5000"


class RentCounter:
    def __init__(self, cost_per_ms):
        self.cost_per_ms = cost_per_ms
        self.threads = {}

    def on_thread_started(self, thread, time):
        self.threads[thread["id"]] = {"started": time}

    def on_thread_stopped(self, thread, time):
        self.threads[thread["id"]]["stopped"] = time

    def do_print(self):
        total_rent_ms = 0
        for thr in self.threads.values():
            started = thr["started"]
            stopped = thr["stopped"]
            delta_ms = (stopped - started).total_seconds() * 1000
            total_rent_ms += delta_ms

        print(f"Total rent (s): {total_rent_ms / 1000}")
        print(f"Total rent cost: {total_rent_ms * self.cost_per_ms}")


class SLACounter:
    def __init__(self, threshold_ms, cost_per_ms):
        self.threshold_ms = threshold_ms
        self.cost_per_ms = cost_per_ms
        self.sum = 0
        self.min = 1e10
        self.max = 0
        self.count = 0

    def add(self, task):
        arrived_at = ciso8601.parse_datetime(task["arrived_at"])
        started_at = ciso8601.parse_datetime(task["started_at"])
        delta_ms = (started_at - arrived_at).total_seconds() * 1000

        if delta_ms > self.threshold_ms:
            self.sum += (delta_ms - self.threshold_ms)
            self.max = max((delta_ms - self.threshold_ms), self.max)
            self.min = min((delta_ms - self.threshold_ms), self.min)

        self.count += 1

    def do_print(self):
        print(f"Max SLA violation time (s): {self.max / 1000}")
        print(f"Min SLA violation time (s): {self.min / 1000}")
        print(f"Average SLA violation time (s): {self.sum / self.count / 1000}")
        print(f"Total SLA cost: {self.sum * self.cost_per_ms}")


class Plot:
    def __init__(self, name):
        self.name = name
        self.index = []
        self.values = []

    def add(self, index, value):
        self.index.append(index)
        self.values.append(value)

    def to_series(self):
        index = pd.DatetimeIndex(self.index)
        return pd.Series(data=self.values, index=index, name=self.name)

    def to_time_series(self):
        return darts.TimeSeries.from_series(self.to_series())


def load_data(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        rows = []
        for line in reader:
            rows.append(int(line[0]))

        return rows


def create_engine(delay, delta, freq, sla):
    response = requests.post(base_url + "/engines", json={
        "worker_setup_delay_ms": delay,
        "delta_ms": delta,
        "measurement_frequency_ms": freq,
        "task_length_order": "S",
        "params": {
            "sla_threshold_ms": sla,
            "sla_cost_per_ms": 1000,
            "rent_cost_per_ms": 1,
            "max_threads": 3000
        }
    })

    return response.json()["id"]


def get_stats(engine_id):
    response = requests.get(base_url + "/engines/" + engine_id + "/stats")
    body = response.json()

    predicted_tasks_plot = Plot("Predicted Tasks")

    for prediction in body["prediction_history"]:
        date = ciso8601.parse_datetime(prediction["date"])
        value = prediction["value"]
        predicted_tasks_plot.add(date, value)

    return predicted_tasks_plot, body["task_distribution"], body["decision_history"]


def get_scaling_decision(engine_id, threads, history, current_time):
    # Prepare data to send
    history_to_send = {}
    for task_id, task in history.items():
        arrived_at = task["arrived_at"]

        if arrived_at in history_to_send:
            history_to_send[arrived_at].append(task)
        else:
            history_to_send[arrived_at] = [task]
    history_to_send = [{"arrived_at": arrived_at, "tasks": tasks} for arrived_at, tasks in history_to_send.items()]

    threads_to_send = []
    for thread in threads:
        if thread["active"]:
            if thread["time_left_ms"] == 0:
                threads_to_send.append({"state": "idle"})
            else:
                task = history[thread["task_id"]]
                processing_time_passed_ms = task["length_ms"] - thread["time_left_ms"]
                threads_to_send.append({"state": "occupied", "processing_time_passed_ms": processing_time_passed_ms})

    response = requests.post(base_url + "/engines/" + engine_id + "/scaling", json={
        "last_timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
        "active_threads": threads_to_send,
        "tasks_history": history_to_send
    })

    # Cleanup history
    history = {task_id: task for (task_id, task) in history.items() if task.get("finished_at") is None}
    body = response.json()
    return history, body["scaling_decision"], body["predicted_tasks_number"]["values"]


def random_task_length():
    x = np.random.normal(loc=4000, scale=1500)
    if x < 0:
        return random_task_length()
    else:
        return x


def random_id():
    return str(uuid.uuid4())


def run_test(tasks_numbers, setup_delay_ms, delta_ms, freq_ms, sla_threshold_ms):
    ms_from_last_decision = 0

    decision_period_ms = setup_delay_ms + delta_ms
    threads = []
    queue = []
    history = {}
    current_time = datetime.datetime.now()
    tick = 0

    tasks_plot = Plot("Tasks")
    all_threads_plot = Plot("All Threads")
    active_threads_plot = Plot("Active Threads")
    sla_counter = SLACounter(sla_threshold_ms, 60000)
    rent_counter = RentCounter(1)

    for _ in range(100):
        thread = {"time_left_ms": 0, "task_id": None, "active": True, "id": random_id()}
        threads.append(thread)
        rent_counter.on_thread_started(thread, current_time)

    while True:
        if tick % 1000 == 0:
            print(f"tick={tick}, time={current_time}, queue={len(queue)}, threads={len(threads)}")

        for thread in threads:
            time_left_ms = thread["time_left_ms"]
            time_left_ms = max(0, time_left_ms - freq_ms)
            thread["time_left_ms"] = time_left_ms

            if thread["active"]:
                if time_left_ms == 0:
                    if thread.get("task_id") is not None:
                        finished_task_id = thread["task_id"]
                        history[finished_task_id]["finished_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        thread["task_id"] = None

                    if len(queue) > 0:
                        task = queue.pop(0)
                        task_id = task["id"]
                        thread["time_left_ms"] = task["length_ms"]
                        thread["task_id"] = task["id"]
                        history[task_id]["started_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        sla_counter.add(history[task_id])
            else:
                if time_left_ms == 0:
                    thread["active"] = True
                    thread["id"] = random_id()
                    rent_counter.on_thread_started(thread, current_time)

        if tick < len(tasks_numbers):
            tasks_number = tasks_numbers[tick]
            for _ in range(tasks_number):
                length_ms = random_task_length()
                task = {
                    "id": str(uuid.uuid4()),
                    "arrived_at": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "length_ms": length_ms
                }

                queue.append(task)
                history[task["id"]] = dict(task)

            tasks_plot.add(current_time, tasks_number)
        else:
            tasks_plot.add(current_time, 0)
            if len(queue) == 0:
                for thread in threads:
                    if thread["active"]:
                        rent_counter.on_thread_stopped(thread, current_time)
                break

        if ms_from_last_decision >= decision_period_ms:
            ms_from_last_decision = 0
            history, decision, predictions = get_scaling_decision(engine_id, threads, history, current_time)

            if decision["action"] == "scale_up":
                for _ in range(decision["threads"]):
                    threads.append({"time_left_ms": setup_delay_ms, "task_id": None, "active": False})
            elif decision["action"] == "scale_down":
                number_of_threads_to_shut_down = min(decision["threads"], len(threads))
                threads.sort(key=lambda thr: thr["time_left_ms"] * int(thr["active"]))
                threads_to_shut_down = threads[:number_of_threads_to_shut_down]
                for thread in threads_to_shut_down:
                    if thread["active"]:
                        rent_counter.on_thread_stopped(thread, current_time)
                        if thread.get("task_id") is not None:
                            finished_task_id = thread["task_id"]
                            history[finished_task_id]["finished_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")
                            thread["task_id"] = None

                threads = threads[number_of_threads_to_shut_down:]

        active_threads_plot.add(current_time, len([thr for thr in threads if thr["active"]]))
        all_threads_plot.add(current_time, len(threads))
        tick += 1
        current_time = current_time + datetime.timedelta(milliseconds=freq_ms)
        ms_from_last_decision += freq_ms

    return tasks_plot, active_threads_plot, all_threads_plot, sla_counter, rent_counter


if __name__ == '__main__':
    tasks_numbers = load_data("../data.csv")[:5000]
    setup_delay_ms = 100000
    delta_ms = 50000
    freq_ms = 2000
    sla_threshold_ms = 200000

    engine_id = create_engine(setup_delay_ms, delta_ms, freq_ms, sla_threshold_ms)
    print(f"Engine id: {engine_id}")

    tasks_plot, active_threads_plot, all_threads_plot, sla_counter, rent_counter = run_test(tasks_numbers, setup_delay_ms, delta_ms, freq_ms, sla_threshold_ms)
    predicted_tasks_plot, task_distribution, decision_history = get_stats(engine_id)

    sla_counter.do_print()
    rent_counter.do_print()

    plot_history_dict(decision_history, freq=f"{setup_delay_ms + delta_ms}ms")
    show()

    bar(task_distribution["values"], task_distribution["weights"], width=1000)
    show()

    tasks_plot.to_time_series().plot()
    predicted_tasks_plot.to_time_series().plot()
    show()

    active_threads_plot.to_time_series().plot()
    all_threads_plot.to_time_series().plot()
    show()
