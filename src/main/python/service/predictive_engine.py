import datetime
import math
import random
from random import choices

import ciso8601
import numpy as np
import pandas as pd
from darts import TimeSeries
from scipy.optimize import minimize_scalar

from main_predictor import MainPredictor
from decision_history import DecisionHistory

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


def __remove_idle_threads__(active_threads, limit):
    occupied_threads = [thr for thr in active_threads if thr["time_left_ms"] > 0]
    idle_threads_number = len(active_threads) - len(occupied_threads)
    if idle_threads_number > limit:
        diff = idle_threads_number - limit
        return occupied_threads + [{"time_left_ms": 0} for _ in range(diff)]
    else:
        return occupied_threads


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


class TaskLengthGenerator:
    def __init__(self, distribution, seed):
        self.distribution = distribution
        self.random = random.Random(seed)

    def next_task_length(self, low_value=0):
        if low_value > 0:
            stripped_distribution = __strip_distribution__(self.distribution, low_value)
            return choices(stripped_distribution[0], stripped_distribution[1])[0]
        else:
            return choices(self.distribution[0], self.distribution[1])[0]


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
    def __init__(self, engine_id, worker_setup_delay_ms, delta_ms, measurement_frequency_ms, task_length_order, params):
        self.id = engine_id
        self.measurement_frequency_ms = measurement_frequency_ms
        self.horizon_steps = __get_horizon_steps__(worker_setup_delay_ms, measurement_frequency_ms)
        self.horizon_steps_with_delta = self.horizon_steps * 2
        self.delta = self.horizon_steps
        self.predictor = MainPredictor(512, self.horizon_steps)
        self.predictor_with_delta = MainPredictor(512, self.horizon_steps * 2)
        self.history = TasksHistory(measurement_frequency_ms, task_length_order)
        self.decision_history = DecisionHistory(params["max_threads"])
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

    def __calculate_total_cost__(self, total_time_working_ms, task_queue, assigned_tasks, current_time: datetime.datetime):
        sla_threshold_ms = self.params["sla_threshold_ms"]
        sla_cost_per_ms = self.params["sla_cost_per_ms"]
        rent_cost_per_ms = self.params["rent_cost_per_ms"]

        total_cost = total_time_working_ms * rent_cost_per_ms

        for task in task_queue:
            arrived_at = task["arrived_at"]
            time_in_queue_ms = (current_time - arrived_at).total_seconds() * 1000
            if time_in_queue_ms > sla_threshold_ms:
                diff = time_in_queue_ms - sla_threshold_ms
                total_cost += (diff * sla_cost_per_ms)

        for task in assigned_tasks:
            arrived_at = task["arrived_at"]
            assigned_at = task["assigned_at"]
            time_in_queue_ms = (assigned_at - arrived_at).total_seconds() * 1000
            if time_in_queue_ms > sla_threshold_ms:
                diff = time_in_queue_ms - sla_threshold_ms
                total_cost += (diff * sla_cost_per_ms)

        return total_cost

    def __calculate_local_future_cost__(
            self, threads_number,
            predictions_with_delta, active_threads, task_queue,
            current_time: datetime.datetime, generator: TaskLengthGenerator) -> float:

        assigned_tasks = []

        def assign_task_if_possible(task, time):
            nonlocal active_threads
            # Run task on an idle thread if one exists
            idle_thread = next((thr for thr in active_threads if thr["time_left_ms"] == 0), None)
            if idle_thread is not None:
                idle_thread["time_left_ms"] = task["length_ms"]
                task["assigned_at"] = time
                assigned_tasks.append(task)
                return True
            else:
                return False

        # What happens if we do nothing
        if threads_number == 0:
            for tick in range(self.horizon_steps_with_delta):
                current_time += datetime.timedelta(milliseconds=self.measurement_frequency_ms)

                # Add predicted number of tasks
                task_queue = task_queue + [
                    {"length_ms": generator.next_task_length(), "arrived_at": current_time}
                    for _ in range(predictions_with_delta[tick])
                ]

                # Assign tasks if possible and update queue
                task_queue = [task for task in task_queue if not assign_task_if_possible(task, current_time)]

                # Update threads time
                for thr in active_threads:
                    time_left_ms = thr["time_left_ms"]
                    time_left_ms = max(0, time_left_ms - self.measurement_frequency_ms)
                    thr["time_left_ms"] = time_left_ms

            # Calculate total threads working time during this period
            total_time_working_ms = len(active_threads) * self.horizon_steps_with_delta * self.measurement_frequency_ms

            return self.__calculate_total_cost__(total_time_working_ms, task_queue, assigned_tasks, current_time)

        # What happens if we add some threads
        # New threads are created after delay and billed only after creation
        elif threads_number > 0:
            old_threads_number = len(active_threads)

            for tick in range(self.horizon_steps_with_delta):
                current_time += datetime.timedelta(milliseconds=self.measurement_frequency_ms)

                # When delay passes, add new threads
                if tick == (self.horizon_steps - 1):
                    active_threads = active_threads + [{"time_left_ms": 0} for _ in range(threads_number)]

                # Add predicted number of tasks
                task_queue = task_queue + [
                    {"length_ms": generator.next_task_length(), "arrived_at": current_time}
                    for _ in range(predictions_with_delta[tick])
                ]

                # Assign tasks if possible and update queue
                task_queue = [task for task in task_queue if not assign_task_if_possible(task, current_time)]

                # Update threads time
                for thr in active_threads:
                    time_left_ms = thr["time_left_ms"]
                    time_left_ms = max(0, time_left_ms - self.measurement_frequency_ms)
                    thr["time_left_ms"] = time_left_ms

            # Calculate total threads working time during this period
            old_threads_total_time_working_ms = old_threads_number * self.horizon_steps_with_delta * self.measurement_frequency_ms
            new_threads_total_time_working_ms = threads_number * self.delta * self.measurement_frequency_ms
            total_time_working_ms = old_threads_total_time_working_ms + new_threads_total_time_working_ms

            return self.__calculate_total_cost__(total_time_working_ms, task_queue, assigned_tasks, current_time)

        # What happens if we remove some threads
        # Threads are removed instantly if there are no tasks
        else:
            start_time = current_time

            threads_number = -threads_number
            target_threads_number = max(0, len(active_threads) - threads_number)
            active_threads = __remove_idle_threads__(active_threads, len(active_threads) - target_threads_number)
            if len(active_threads) > target_threads_number:
                active_threads.sort(key=lambda thr: thr["time_left_ms"])
                for i in range(len(active_threads) - target_threads_number):
                    active_threads[i]["delete"] = True

            # Hold refs to all threads
            all_threads = [thr for thr in active_threads]

            for tick in range(self.horizon_steps_with_delta):
                current_time += datetime.timedelta(milliseconds=self.measurement_frequency_ms)

                # Add predicted number of tasks
                task_queue = task_queue + [
                    {"length_ms": generator.next_task_length(), "arrived_at": current_time}
                    for _ in range(predictions_with_delta[tick])
                ]

                # Assign tasks if possible and update queue
                task_queue = [task for task in task_queue if not assign_task_if_possible(task, current_time)]

                # Update threads time
                for thr in active_threads:
                    time_left_ms = thr["time_left_ms"]
                    time_left_ms = max(0, time_left_ms - self.measurement_frequency_ms)
                    thr["time_left_ms"] = time_left_ms

                    if time_left_ms == 0 and thr.get("delete", False):
                        thr["destroyed_at"] = current_time

                # Delete marked threads if finished
                active_threads = [thr for thr in active_threads if not thr.get("delete", False) or not thr["time_left_ms"] == 0]

            total_time_working_ms = 0
            for thr in all_threads:
                if thr.get("delete", False):
                    destroyed_at: datetime.datetime = thr["destroyed_at"]
                    time_working_ms = (destroyed_at - start_time).total_seconds() * 1000
                    total_time_working_ms += time_working_ms
                else:
                    total_time_working_ms += (self.horizon_steps_with_delta * self.measurement_frequency_ms)

            return self.__calculate_total_cost__(total_time_working_ms, task_queue, assigned_tasks, current_time)

    def __get_threads_number__(self, active_threads, predictions_with_delta, current_time: datetime.datetime):
        task_lengths_distribution = self.history.task_lengths_histogram.as_distribution()
        length_generator = TaskLengthGenerator(task_lengths_distribution, 1)

        # Setup simulation input
        simulated_threads = []
        for thread in active_threads:
            if thread["state"] == "idle":
                simulated_threads.append({"time_left_ms": 0})
            elif thread["state"] == "occupied":
                task_length = length_generator.next_task_length(low_value=thread["processing_time_passed_ms"])
                simulated_threads.append({"time_left_ms": task_length - thread["processing_time_passed_ms"]})

        simulated_tasks_queue = sorted([
            {"length_ms": length_generator.next_task_length(), "arrived_at": ciso8601.parse_datetime(task["arrived_at"])}
            for task in self.history.current_queue.values()
        ], key=lambda task: task["arrived_at"])

        def run_simulation(threads):
            nonlocal simulated_threads, simulated_tasks_queue, predictions_with_delta, current_time, task_lengths_distribution
            simulated_threads_copy = [dict(thr) for thr in simulated_threads]
            simulated_tasks_queue_copy = [dict(task) for task in simulated_tasks_queue]
            return self.__calculate_local_future_cost__(
                threads_number=threads,
                predictions_with_delta=predictions_with_delta,
                active_threads=simulated_threads_copy,
                task_queue=simulated_tasks_queue_copy,
                current_time=current_time,
                generator=TaskLengthGenerator(task_lengths_distribution, 1)
            )

        optimal_threads = self.__optimize_cost_function__(run_simulation, len(active_threads))
        self.decision_history.add(optimal_threads, current_time)
        return optimal_threads

    def __optimize_cost_function__(self, run_simulation, current_threads):
        min_cost = run_simulation(0)

        if min_cost == 0:
            return 0
        else:
            bounds = self.decision_history.get_boundaries(current_threads)
            results = {0: min_cost}

            print(f"Bounds: {bounds}")

            def memory_fun(x):
                nonlocal results
                threads = int(x)
                if threads in results:
                    return results[threads]
                else:
                    cost = run_simulation(threads)
                    print(f"Threads={threads}, Cost={cost}")
                    results[threads] = cost
                    return cost

            result = int(minimize_scalar(memory_fun, bounds=(bounds[0] - 1, bounds[1] + 1), method='bounded').x)
            print(f"Optimal threads={result}")
            return result

    def get_scaling_decision(self, active_threads, tasks_history, last_timestamp):
        self.history.update(tasks_history, last_timestamp)

        # predictions, message = self.predictor.predict(self.history.history)
        # predictions = predictions.map(lambda p: np.around(p).clip(min=0))
        # predictions_values = predictions.values().reshape(-1).astype(int)
        #
        # self.__update_prediction_history__(predictions)
        #
        # threads = self.__get_number_of_needed_threads__(
        #     active_threads, predictions_values,
        #     self.history.last_datetime()
        # )

        predictions, message = self.predictor_with_delta.predict(self.history.history)
        predictions = predictions.map(lambda p: np.around(p).clip(min=0))
        predictions_values = predictions.values().reshape(-1).astype(int)
        self.__update_prediction_history__(predictions)

        threads = self.__get_threads_number__(active_threads, predictions_values, self.history.last_datetime())

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
            "decision_history": self.decision_history.to_dict(),
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


# def __get_number_of_needed_threads__(self, active_threads, predictions, now: datetime.datetime):
#     # Init some params
#     created_threads = 0
#     min_idle_threads = 1e6
#
#     sla_threshold_ms = self.params["sla_threshold_ms"]
#     max_threads = self.params["max_threads"]
#
#     task_lengths_distribution = self.history.task_lengths_histogram.as_distribution()
#
#     # Setup simulation
#     simulated_threads = []
#     for thread in active_threads:
#         if thread["state"] == "idle":
#             simulated_threads.append({"time_left_ms": 0})
#         elif thread["state"] == "occupied":
#             stripped_distribution = __strip_distribution__(task_lengths_distribution, thread["processing_time_passed_ms"])
#             task_length = __rand_task_time__(stripped_distribution)
#             simulated_threads.append({"time_left_ms": task_length - thread["processing_time_passed_ms"]})
#
#     simulated_tasks_queue = [
#         {"length_ms": __rand_task_time__(task_lengths_distribution), "arrived_at": ciso8601.parse_datetime(task["arrived_at"])}
#         for task in self.history.current_queue.values()
#     ]
#
#     simulated_tasks_queue.sort(key=lambda task: task["arrived_at"])
#
#     def update_task_in_queue(task):
#         nonlocal created_threads, simulated_threads
#
#         arrived_at_ts = task["arrived_at"]
#         delta_ms = int((current_time - arrived_at_ts).total_seconds() * 1000)
#
#         if delta_ms >= sla_threshold_ms:
#             # If task already violates SLA, create a new thread
#             idle_thread = next((thr for thr in simulated_threads if thr["time_left_ms"] == 0), None)
#             if idle_thread is not None:
#                 idle_thread["time_left_ms"] = task["length_ms"]
#                 return True
#             else:
#                 if len(simulated_threads) < max_threads:
#                     simulated_threads.append({"time_left_ms": task["length_ms"]})
#                     created_threads += 1
#                     return True
#                 else:
#                     return False
#         else:
#             # If task doesn't violate SLA, run it on an idle thread if one exists
#             idle_thread = next((thr for thr in simulated_threads if thr["time_left_ms"] == 0), None)
#             if idle_thread is not None:
#                 idle_thread["time_left_ms"] = task["length_ms"]
#                 return True
#             else:
#                 return False
#
#     # Run simulation
#     for tick in range(self.horizon_steps + 1):
#         current_time = now + datetime.timedelta(milliseconds=tick * self.measurement_frequency_ms)
#
#         if tick > 0:
#             for thr in simulated_threads:
#                 time_left_ms = thr["time_left_ms"]
#                 time_left_ms = max(0, time_left_ms - self.measurement_frequency_ms)
#                 thr["time_left_ms"] = time_left_ms
#
#             new_tasks = [
#                 {"length_ms": __rand_task_time__(task_lengths_distribution), "arrived_at": current_time}
#                 for _ in range(predictions[tick - 1])
#             ]
#
#             simulated_tasks_queue = simulated_tasks_queue + new_tasks
#
#         simulated_tasks_queue = [task for task in simulated_tasks_queue if not update_task_in_queue(task)]
#         min_idle_threads = min(min_idle_threads, len([thr for thr in simulated_threads if thr["time_left_ms"] == 0]))
#
#     return created_threads - min_idle_threads
