import csv
import datetime
import random
import uuid
import ciso8601

import pandas as pd
import requests
import numpy as np
import darts

from matplotlib.pyplot import plot, show, bar, figure

base_url = "http://127.0.0.1:5000"


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


def create_engine(delay, freq):
    response = requests.post(base_url + "/engines", json={
        "worker_setup_delay_ms": delay,
        "measurement_frequency_ms": freq,
        "task_length_order": "S",
        "params": {"sla_ms": 10000}
    })

    return response.json()["id"]


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
    return response.json()["scaling_decision"], history


def run_test():
    tasks_numbers = load_data("../data.csv")

    setup_delay_ms = 100000
    ms_from_last_decision = 0
    freq_ms = 2000

    engine_id = create_engine(setup_delay_ms, freq_ms)

    threads = []
    queue = []
    history = {}
    now = datetime.datetime.now()
    tick = 0

    for _ in range(100):
        threads.append({"time_left_ms": 0, "task_id": None, "active": True})

    tasks_plot = Plot("Tasks")
    all_threads_plot = Plot("All Threads")
    active_threads_plot = Plot("Active Threads")

    while True:
        current_time = now + datetime.timedelta(milliseconds=tick * freq_ms)

        if tick % 1000 == 0:
            print(f"tick={tick}, time={current_time}, queue={len(queue)}, threads={len(threads)}")

        for thread in threads:
            time_left_ms = thread["time_left_ms"]
            time_left_ms = max(0, time_left_ms - freq_ms)
            thread["time_left_ms"] = time_left_ms

            if thread["active"]:
                if time_left_ms == 0 and len(queue) > 0:
                    task = queue.pop(0)
                    history[task["id"]]["started_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")

                    if thread.get("task_id") is not None:
                        old_task_id = thread.get("task_id")
                        history[old_task_id]["finished_at"] = current_time.strftime("%Y-%m-%d %H:%M:%S")

                    thread["time_left_ms"] = task["length_ms"]
                    thread["task_id"] = task["id"]
            else:
                if time_left_ms == 0:
                    thread["active"] = True

        if tick < len(tasks_numbers):
            tasks_number = tasks_numbers[tick]
            for _ in range(tasks_number):
                length_ms = np.random.normal(loc=4000, scale=100)
                task = {
                    "id": str(uuid.uuid4()),
                    "arrived_at": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "length_ms": length_ms
                }

                queue.append(task)
                history[task["id"]] = task

            tasks_plot.add(current_time, tasks_number)
        else:
            tasks_plot.add(current_time, 0)
            if len(queue) == 0:
                break

        if ms_from_last_decision > setup_delay_ms:
            ms_from_last_decision = 0
            decision, history = get_scaling_decision(engine_id, threads, history, current_time)

            if decision["action"] == "scale_up":
                for _ in range(decision["threads"]):
                    threads.append({"time_left_ms": setup_delay_ms, "task_id": None, "active": False})

        active_threads_plot.add(current_time, len([thr for thr in threads if thr["active"]]))
        all_threads_plot.add(current_time, len(threads))
        tick += 1
        ms_from_last_decision += freq_ms

    return tasks_plot, active_threads_plot, all_threads_plot


if __name__ == '__main__':
    tasks_plot, active_threads_plot, all_threads_plot = run_test()
    active_threads_plot.to_time_series().plot()
    all_threads_plot.to_time_series().plot()
    tasks_plot.to_time_series().plot()
    show()
