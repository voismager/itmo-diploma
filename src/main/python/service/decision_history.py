import datetime
import math
import random
import ciso8601

import darts
import pandas as pd
from matplotlib.pyplot import show


def plot_history(decision_history, freq):
    threads_history = pd.Series(
        data=decision_history.threads_history,
        index=pd.DatetimeIndex(decision_history.times, freq=freq),
        name="Threads Delta"
    )

    upper_boundary_history = pd.Series(
        data=[b[1] for b in decision_history.boundaries_history],
        index=pd.DatetimeIndex(decision_history.times, freq=freq),
        name="Upper Bound"
    )

    lower_boundary_history = pd.Series(
        data=[b[0] for b in decision_history.boundaries_history],
        index=pd.DatetimeIndex(decision_history.times, freq=freq),
        name="Lower Bound"
    )

    upper_extend_boundaries_history = pd.Series(
        data=[b[1] for b in decision_history.extend_boundaries_history],
        index=pd.DatetimeIndex(decision_history.times, freq=freq),
        name="Upper Extend Bound"
    )

    lower_extend_boundaries_history = pd.Series(
        data=[b[0] for b in decision_history.extend_boundaries_history],
        index=pd.DatetimeIndex(decision_history.times, freq=freq),
        name="Lower Extend Bound"
    )

    darts.TimeSeries.from_series(threads_history).plot()
    darts.TimeSeries.from_series(upper_boundary_history).plot()
    darts.TimeSeries.from_series(lower_boundary_history).plot()
    darts.TimeSeries.from_series(upper_extend_boundaries_history).plot()
    darts.TimeSeries.from_series(lower_extend_boundaries_history).plot()


def plot_history_dict(decision_history_dict, freq):
    decision_history = DecisionHistory(0)
    decision_history.times = [ciso8601.parse_datetime(t) for t in decision_history_dict["times"]]
    decision_history.threads_history = decision_history_dict["threads_delta_history"]
    decision_history.boundaries_history = [
        (b[0], b[1]) for b in
        zip(decision_history_dict["lower_boundary_history"], decision_history_dict["upper_boundary_history"])
    ]
    decision_history.extend_boundaries_history = [
        (b[0], b[1]) for b in
        zip(decision_history_dict["lower_extend_boundaries_history"], decision_history_dict["upper_extend_boundaries_history"])
    ]
    plot_history(decision_history, freq)


class DecisionHistory:
    def __init__(self, max_threads):
        self.max_threads = max_threads
        self.initial_boundaries = (-max(2, max_threads // (2 ** 3)), max(2, max_threads // (2 ** 3)))
        self.boundaries = self.initial_boundaries
        self.extend_coefficient = 0.9

        self.times = []
        self.threads_history = []
        self.boundaries_history = []
        self.extend_boundaries_history = []

    def __get_extend_boundaries__(self, boundaries):
        return (
            math.ceil(boundaries[0] * self.extend_coefficient),
            math.floor(boundaries[1] * self.extend_coefficient)
        )

    def add(self, decision_threads, time):
        extend_boundaries = self.__get_extend_boundaries__(self.boundaries)

        if decision_threads > 0:
            if decision_threads >= extend_boundaries[1]:
                self.boundaries = (self.initial_boundaries[0], self.boundaries[1] * 2)
                extend_boundaries = self.__get_extend_boundaries__(self.boundaries)

        elif decision_threads < 0:
            if -decision_threads >= -extend_boundaries[0]:
                self.boundaries = (self.boundaries[0] * 2, self.initial_boundaries[1])
                extend_boundaries = self.__get_extend_boundaries__(self.boundaries)

        self.times.append(time)
        self.threads_history.append(decision_threads)
        self.boundaries_history.append(self.boundaries)
        self.extend_boundaries_history.append(extend_boundaries)

    def get_boundaries(self, current_threads):
        lower_bound = max(-current_threads, self.boundaries[0])
        upper_bound = min(self.max_threads - current_threads, self.boundaries[1])
        return lower_bound, upper_bound

    def to_dict(self):
        times = [time.strftime("%Y-%m-%d %H:%M:%S") for time in self.times]
        threads_delta_history = self.threads_history
        lower_boundary_history = [b[0] for b in self.boundaries_history]
        upper_boundary_history = [b[1] for b in self.boundaries_history]
        lower_extend_boundaries_history = [b[0] for b in self.extend_boundaries_history]
        upper_extend_boundaries_history = [b[1] for b in self.extend_boundaries_history]

        return {
            "times": times,
            "threads_delta_history": threads_delta_history,
            "lower_boundary_history": lower_boundary_history,
            "upper_boundary_history": upper_boundary_history,
            "lower_extend_boundaries_history": lower_extend_boundaries_history,
            "upper_extend_boundaries_history": upper_extend_boundaries_history
        }


if __name__ == '__main__':
    current_time = datetime.datetime.now()
    current_threads = 0

    history = DecisionHistory(3000)

    for _ in range(20):
        current_time += datetime.timedelta(milliseconds=2000)
        boundaries = history.get_boundaries(current_threads)
        current_threads_delta = random.randint(boundaries[0], boundaries[1] + 1)
        current_threads += current_threads_delta
        history.add(current_threads_delta, current_time)

    plot_history(history, "2S")
    show()
