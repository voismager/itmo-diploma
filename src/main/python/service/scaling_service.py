import uuid

from predictive_engine import PredictiveScalingEngine


class ScalingService:
    def __init__(self):
        self.engines = dict()
        self.__create_engine__(
            "1",
            worker_setup_delay_ms=100000,
            delta_ms=100000,
            measurement_frequency_ms=2000,
            task_length_order="S",
            params={
                "sla_threshold_ms": 10000,
                "sla_cost_per_ms": 1,
                "rent_cost_per_ms": 1,
                "cost_per_scaling_decision": 1,
                "overestimation_coefficient": 0,
                "max_threads": 3000,
                "seasonality": "none"
            }
        )

    def create_engine(self, worker_setup_delay_ms, delta_ms, measurement_frequency_ms, task_length_order, params):
        engine_id = str(uuid.uuid4())
        return self.__create_engine__(
            engine_id,
            worker_setup_delay_ms, delta_ms, measurement_frequency_ms, task_length_order,
            params
        )

    def __create_engine__(self, engine_id, worker_setup_delay_ms, delta_ms, measurement_frequency_ms, task_length_order, params):
        engine = PredictiveScalingEngine(
            engine_id,
            worker_setup_delay_ms, delta_ms, measurement_frequency_ms, task_length_order,
            params
        )
        self.engines[engine_id] = engine
        return engine

    def delete_engine(self, engine_id):
        if engine_id not in self.engines:
            raise KeyError("Engine with specified id is not found!")

        engine = self.engines.pop(engine_id)
        del engine
        print(f"Deleted {engine_id} engine")
        return engine_id

    def get_scaling_decision(self, engine_id, active_threads, tasks_history, last_timestamp):
        engine = self.engines.get(engine_id)

        if engine is None:
            raise KeyError("Engine with specified id is not found!")

        return engine.get_scaling_decision(active_threads, tasks_history, last_timestamp)

    def get_stats(self, engine_id):
        engine = self.engines.get(engine_id)

        if engine is None:
            raise KeyError("Engine with specified id is not found!")

        return engine.get_stats()
