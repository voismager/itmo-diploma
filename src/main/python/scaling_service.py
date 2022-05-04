import uuid

from predictive_engine import PredictiveScalingEngine


class ScalingService:
    def __init__(self):
        self.engines = dict()
        self.__create_engine__(
            "1",
            worker_setup_delay_ms=100000,
            measurement_frequency_ms=2000,
            task_length_order="S",
            params={"sla_ms": 10000}
        )

    def create_engine(self, worker_setup_delay_ms, measurement_frequency_ms, task_length_order, params):
        engine_id = str(uuid.uuid4())
        return self.__create_engine__(
            engine_id,
            worker_setup_delay_ms, measurement_frequency_ms, task_length_order,
            params
        )

    def __create_engine__(self, engine_id, worker_setup_delay_ms, measurement_frequency_ms, task_length_order, params):
        engine = PredictiveScalingEngine(
            engine_id,
            worker_setup_delay_ms, measurement_frequency_ms, task_length_order,
            params
        )
        self.engines[engine_id] = engine
        return engine

    def get_scaling_decision(self, engine_id, active_threads, tasks_history):
        engine = self.engines.get(engine_id)

        if engine is None:
            raise KeyError("Engine with specified id is not found!")

        return engine.get_scaling_decision(active_threads, tasks_history)

    def get_task_length_distribution(self, engine_id):
        engine = self.engines.get(engine_id)

        if engine is None:
            raise KeyError("Engine with specified id is not found!")

        return engine.get_task_length_distribution()
