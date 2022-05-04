import math
from random import choices

import numpy as np
import pandas as pd
import ciso8601
from darts import TimeSeries
import datetime

from python.service.main_predictor import MainPredictor

request_example = {
    "active_threads": [
        {
            "state": "idle"
        },
        {
            "state": "occupied",
            "processing_time_passed_ms": 20
        }
    ],
    "last_timestamp": "2022-07-04 14:59:58",
    "tasks_history": [
        {
            "arrived_at": "2022-07-04 12:59:58",
            "tasks": [
                {
                    "id": "1",
                    "started_at": "2022-07-04 12:59:59",
                    "finished_at": "2022-07-04 13:06:54"
                }
            ]
        }
    ]
}


class Histogram:
    def __init__(self, interval):
        if interval == "MS":
            interval_length = 1
        elif interval == "S":
            interval_length = 1000
        elif interval == "M":
            interval_length = 30 * 1000
        elif interval == "H":
            interval_length = 60 * 30 * 500
        else:
            raise ValueError("Interval should be one of ['MS' (milliseconds), 'S' (seconds), 'M' (minutes), 'H' (hours)]")

        self.bins = {}
        self.interval_length = interval_length

    def median(self):
        pass

    def as_distribution(self) -> (list, list):
        total = sum([b["count"] for b in self.bins.values()])
        values = []
        weights = []

        for v in self.bins.values():
            values.append(v["avg_value"])
            weights.append(v["count"] / total)

        return values, weights

    def add(self, milliseconds):
        if milliseconds < 0:
            raise ValueError("Value should be non-negative")

        bin_index = math.floor(milliseconds / self.interval_length)

        if bin_index in self.bins:
            self.bins[bin_index]["count"] += 1
        else:
            self.bins[bin_index] = {
                "min_value": bin_index * self.interval_length,
                "max_value": (bin_index + 1) * self.interval_length,
                "avg_value": self.interval_length * (2 * bin_index + 1) / 2,
                "count": 1
            }


def __rand_task_time__(distribution, k=1):
    if k == 1:
        return choices(distribution[0], distribution[1])[0]
    else:
        return choices(distribution[0], distribution[1])


def __strip_distribution__(distribution, low_value):
    values = []
    weights = []

    max_value, max_value_index = 0, 0

    for i, value in enumerate(distribution[0]):
        if value > max_value:
            max_value = value
            max_value_index = i

        if value > low_value:
            values.append(value)
            weights.append(distribution[1][i])

    if len(values) == 0:
        return [distribution[0][max_value_index]], [1]
    else:
        return values, weights


class TasksHistory:
    def __init__(self, measurement_frequency_ms, task_length_order):
        self.measurement_frequency_ms = measurement_frequency_ms
        self.measurement_frequency_literal = f"{measurement_frequency_ms}ms"
        self.history = None
        self.current_queue = {}
        self.processed_task_ids_by_histogram = set()
        self.task_lengths_histogram = Histogram(task_length_order)

    def last_datetime(self):
        if self.history is None:
            return None
        else:
            return self.history.end_time().to_pydatetime()

    def update(self, tasks_history, last_timestamp):
        if len(tasks_history) == 0:
            self.__add__(last_timestamp, [])
            return

        tasks_history.sort(key=lambda group: ciso8601.parse_datetime(group["arrived_at"]))

        for task_group in tasks_history:
            arrived_at = task_group["arrived_at"]
            tasks = task_group["tasks"]
            self.__add__(arrived_at, tasks)

        last_arrived_at_ts = ciso8601.parse_datetime(tasks_history[-1]["arrived_at"])
        last_timestamp_ts = ciso8601.parse_datetime(last_timestamp)
        if last_timestamp_ts > last_arrived_at_ts:
            self.__add__(last_timestamp, [])

    def __add__(self, arrived_at, tasks):
        if self.history is None:
            index = pd.DatetimeIndex([arrived_at])
            history = TimeSeries.from_series(pd.Series([len(tasks)], index=index), freq=self.measurement_frequency_literal)
            self.history = history
        else:
            arrived_at_ts = ciso8601.parse_datetime(arrived_at)

            if arrived_at_ts > self.history.end_time().to_pydatetime():
                next_ts = self.history.end_time().to_pydatetime() + datetime.timedelta(milliseconds=self.measurement_frequency_ms)

                if next_ts == arrived_at_ts:
                    index = pd.DatetimeIndex([arrived_at])
                    history = TimeSeries.from_series(pd.Series([len(tasks)], index=index), freq=self.measurement_frequency_literal)
                    self.history = self.history.append(history)
                else:
                    index = pd.DatetimeIndex([next_ts, arrived_at])
                    history = TimeSeries.from_series(
                        pd.Series([0, len(tasks)], index=index),
                        freq=self.measurement_frequency_literal,
                        fill_missing_dates=True,
                        fillna_value=0
                    )
                    self.history = self.history.append(history)

        for task in tasks:
            task_id = task.get("id")
            started_at = task.get("started_at")
            finished_at = task.get("finished_at")

            if task_id in self.current_queue:
                if started_at is not None or finished_at is not None:
                    self.current_queue.pop(task_id)
            else:
                if started_at is None and finished_at is None:
                    self.current_queue[task_id] = {"arrived_at": arrived_at}

            if task_id not in self.processed_task_ids_by_histogram:
                if started_at is not None and finished_at is not None:
                    started_at_ts = ciso8601.parse_datetime(started_at)
                    finished_at_ts = ciso8601.parse_datetime(finished_at)
                    delta_ms = int((finished_at_ts - started_at_ts).total_seconds() * 1000)
                    self.task_lengths_histogram.add(delta_ms)
                    self.processed_task_ids_by_histogram.add(task_id)


def __get_horizon_steps__(worker_setup_delay_ms, measurement_frequency_ms):
    return math.ceil(worker_setup_delay_ms / measurement_frequency_ms)


class PredictiveScalingEngine:
    def __init__(self, engine_id, worker_setup_delay_ms, measurement_frequency_ms, task_length_order, params):
        self.id = engine_id
        self.worker_setup_delay_ms = worker_setup_delay_ms
        self.measurement_frequency_ms = measurement_frequency_ms
        self.horizon_steps = __get_horizon_steps__(worker_setup_delay_ms, measurement_frequency_ms)
        self.predictor = MainPredictor(512, self.horizon_steps)
        self.history = TasksHistory(measurement_frequency_ms, task_length_order)
        self.prediction_history = None
        self.params = params

    def __update_prediction_history__(self, predictions: TimeSeries):
        if self.prediction_history is None:
            self.prediction_history = predictions
        else:
            end_time = self.prediction_history.end_time().to_pydatetime()
            start_time = predictions.start_time().to_pydatetime()

            if start_time > end_time:
                self.prediction_history = self.prediction_history.append(predictions)

    def __get_number_of_needed_threads__(self, active_threads, predictions, now: datetime.datetime):
        # Init some params
        created_threads = 0
        min_idle_threads = 1e6

        sla_ms = self.params["sla_ms"]
        task_lengths_distribution = self.history.task_lengths_histogram.as_distribution()

        # Setup simulation
        simulated_threads = []
        for thread in active_threads:
            if thread["state"] == "idle":
                simulated_threads.append({"time_left_ms": 0})
            elif thread["state"] == "occupied":
                stripped_distribution = __strip_distribution__(task_lengths_distribution, thread["processing_time_passed_ms"])
                task_length = __rand_task_time__(stripped_distribution)
                simulated_threads.append({"time_left_ms": task_length - thread["processing_time_passed_ms"]})

        simulated_tasks_queue = [
            {"length_ms": __rand_task_time__(task_lengths_distribution), "arrived_at": ciso8601.parse_datetime(task["arrived_at"])}
            for task in self.history.current_queue.values()
        ]

        simulated_tasks_queue.sort(key=lambda task: task["arrived_at"])

        def update_task_in_queue(task):
            nonlocal created_threads, simulated_threads

            arrived_at_ts = task["arrived_at"]
            delta_ms = int((current_time - arrived_at_ts).total_seconds() * 1000)

            if delta_ms >= sla_ms:
                # If task already violates SLA, create a new thread
                idle_thread = next((thr for thr in simulated_threads if thr["time_left_ms"] == 0), None)
                if idle_thread is not None:
                    idle_thread["time_left_ms"] = task["length_ms"]
                    return True
                else:
                    simulated_threads.append({"time_left_ms": task["length_ms"]})
                    created_threads += 1
                    return True
            else:
                # If task doesn't violate SLA, run it on an idle thread if one exists
                idle_thread = next((thr for thr in simulated_threads if thr["time_left_ms"] == 0), None)
                if idle_thread is not None:
                    idle_thread["time_left_ms"] = task["length_ms"]
                    return True
                else:
                    return False

        # Run simulation
        for tick in range(self.horizon_steps + 1):
            current_time = now + datetime.timedelta(milliseconds=tick * self.measurement_frequency_ms)

            if tick > 0:
                for thr in simulated_threads:
                    time_left_ms = thr["time_left_ms"]
                    time_left_ms = max(0, time_left_ms - self.measurement_frequency_ms)
                    thr["time_left_ms"] = time_left_ms

                new_tasks = [
                    {"length_ms": __rand_task_time__(task_lengths_distribution), "arrived_at": current_time}
                    for _ in range(predictions[tick - 1])
                ]

                simulated_tasks_queue = simulated_tasks_queue + new_tasks

            simulated_tasks_queue = [task for task in simulated_tasks_queue if not update_task_in_queue(task)]
            min_idle_threads = min(min_idle_threads, len([thr for thr in simulated_threads if thr["time_left_ms"] == 0]))

        return created_threads - min_idle_threads

    def get_scaling_decision(self, active_threads, tasks_history, last_timestamp):
        self.history.update(tasks_history, last_timestamp)

        predictions, message = self.predictor.predict(self.history.history)
        predictions = predictions.map(lambda p: np.around(p).clip(min=0))
        predictions_values = predictions.values().reshape(-1).astype(int)

        self.__update_prediction_history__(predictions)

        threads = self.__get_number_of_needed_threads__(
            active_threads, predictions_values,
            self.history.last_datetime()
        )

        if threads > 0:
            decision = {
                "action": "scale_up",
                "threads": threads
            }

        elif threads < 0:
            decision = {
                "action": "scale_down",
                "threads": -threads
            }

        else:
            decision = {
                "action": "do_nothing"
            }

        predictions_series = predictions.pd_series()

        return {
            "predicted_tasks_number": {
                "message": message,
                "values": [
                    {"date": t[0], "value": t[1]}
                    for t in zip(predictions_series.index.strftime("%Y-%m-%d %H:%M:%S").tolist(), predictions_series.tolist())
                ]
            },
            "scaling_decision": decision
        }

    def get_stats(self):
        distribution = self.history.task_lengths_histogram.as_distribution()
        history = self.history.history.pd_series()
        prediction_history = self.prediction_history.pd_series()

        return {
            "task_distribution": {
                "values": distribution[0],
                "weights": distribution[1]
            },
            "prediction_history": [
                {"date": t[0], "value": t[1]}
                for t in zip(prediction_history.index.strftime("%Y-%m-%d %H:%M:%S").tolist(), prediction_history.tolist())
            ],
            "history": [
                {"date": t[0], "value": t[1]}
                for t in zip(history.index.tolist(), history.tolist())
            ],
            "queue": self.history.current_queue
        }

# def __get_number_of_potential_assigned_tasks__(self, queue_length, active_threads):
#     result = 0
#
#     queue_length_left = queue_length
#     horizon = self.worker_setup_delay_ms
#     task_lengths_distribution = self.history.task_lengths_histogram.as_distribution()
#
#     simulated_threads = []
#
#     for thread in active_threads:
#         if thread["state"] == "idle":
#             simulated_threads.append({"time_left_ms": 0})
#         elif thread["state"] == "occupied":
#             stripped_distribution = __strip_distribution__(task_lengths_distribution, thread["processing_time_passed_ms"])
#             task_length = __rand_task_time__(stripped_distribution)
#             simulated_threads.append({"time_left_ms": task_length - thread["processing_time_passed_ms"]})
#
#     tick = 0
#     while True:
#         idle_thread = next((thread for thread in simulated_threads if thread["time_left_ms"] == 0), None)
#
#         if idle_thread is not None:
#             task_length = __rand_task_time__(task_lengths_distribution)
#             idle_thread["time_left_ms"] = task_length
#
#             if queue_length_left > 0:
#                 result -= 1
#                 queue_length_left -= 1
#             else:
#                 result += 1
#         else:
#             min_time_left_ms = min(thread["time_left_ms"] for thread in simulated_threads)
#             tick += min_time_left_ms
#
#             if tick >= horizon:
#                 return result
#
#             for thread in simulated_threads:
#                 thread["time_left_ms"] -= min_time_left_ms
