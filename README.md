# Auto Scaling Service 

## Setup
```
cd ./service && python3 service.py
```

## API Methods


### POST /engines

Creates and configures scaling engine.

Parameters:
* `worker_setup_delay_ms` - average worker setup time 
* `delta_ms` - additional time length used in prediction. Should be non-zero.
* `measurement_frequency_ms` - frequency in which global queue state is measured
* `task_length_order` - can be one of ["MS", "S", "M", "H"]. Used to construct task length histogram
* `params.sla_threshold_ms` - threshold above which task time in queue is considered to be a violation
* `params.sla_cost_per_ms` - cost per each ms a task is remained in queue above threshold
* `params.rent_cost_per_ms` - rent cost per each thread
* `params.cost_per_scaling_decision` - cost per each thread creation / deletion
* `params.overestimation_coefficient` - can be in a range of [-1, 1]. 0 is a default value. Higher value resolves in higher number of predicted tasks
* `params.max_threads` - max threads in cluster
* `params.initial_max_step_threads` - max value of threads to add / remove on each decision step. Can grow over time
* `params.seasonality` - can be one of ["daily", "none"]. Used to correct prediction by including seasonal component

Example request:
```
{
    "worker_setup_delay_ms": 100000,
    "delta_ms": 100000,
    "measurement_frequency_ms": 2000,
    "task_length_order": "S",
    "params": {
        "sla_threshold_ms": 10000,
        "sla_cost_per_ms": 1,
        "rent_cost_per_ms": 1,
        "cost_per_scaling_decision": 100000,
        "overestimation_coefficient": 0,
        "max_threads": 3000,
        "initial_max_step_threads": 100,
        "seasonality": "daily"
    }
}
```
Example response:
```
{
    "id": "d6202671-838c-4429-bf60-a6a9211f2ed6",
    "measurement_frequency_ms": 2000,
    "params": {
        "cost_per_scaling_decision": 100000,
        "initial_max_step_threads": 100,
        "max_threads": 3000,
        "overestimation_coefficient": 0,
        "rent_cost_per_ms": 1,
        "seasonality": "daily",
        "sla_cost_per_ms": 1,
        "sla_threshold_ms": 10000
    },
    "task_length_distribution_interval_ms": 1000
}
```


### DELETE /engines/{id}

Delete the specified engine.

### GET /engines/stats

Get some internal statistics for the specified engine.

Example response:
```
{
    "decision_history": {
        "lower_boundary_history": [-100],
        "lower_extend_boundaries_history": [-90],
        "threads_delta_history": [0],
        "times": ["2022-07-04 12:00:44"],
        "upper_boundary_history": [100],
        "upper_extend_boundaries_history": [90]
    },
    "history": [
        {"date": "Mon, 04 Jul 2022 12:00:38 GMT", "value": 0.0},
        {"date": "Mon, 04 Jul 2022 12:00:40 GMT", "value": 0.0},
        {"date": "Mon, 04 Jul 2022 12:00:42 GMT", "value": 0.0},
        {"date": "Mon, 04 Jul 2022 12:00:44 GMT", "value": 0.0}
    ],
    "prediction_history": [
        {"date": "2022-07-04 12:00:46", "value": 0.0},
        {"date": "2022-07-04 12:00:48", "value": 0.0},
        {"date": "2022-07-04 12:00:50", "value": 0.0},
        {"date": "2022-07-04 12:00:52", "value": 0.0},
        {"date": "2022-07-04 12:00:54", "value": 0.0},
        {"date": "2022-07-04 12:00:56", "value": 0.0},
        {"date": "2022-07-04 12:00:58", "value": 0.0},
        {"date": "2022-07-04 12:01:00", "value": 0.0}
    ],
    "queue": {
        "4": {"arrived_at": "2022-07-04 12:00:04"},
        "5": {"arrived_at": "2022-07-04 12:00:06"}
    },
    "task_distribution": {
        "values": [
            10500.0,
            20500.0
        ],
        "weights": [
            0.6666666666666666,
            0.3333333333333333
        ]
    }
}
```


### POST /engines/{id}/scaling

Make a scaling decision based on received data and history.

Parameters:
* `active_threads` - array of all active threads in the cluster. 
  * `state` - either 'idle' or 'occupied'
  * `processing_time_passed_ms` - should be specified only on 'occupied' state. Indicates number of ms current task has been processing
* `tasks_history` - array of tasks grouped by arrival time
  * `arrived_at` - time in which tasks arrived
  * `tasks` - group of tasks
    * `id` - unique id of task
    * `started_at` - should be specified if task has started on thread
    * `finished_at` - should be specified if task has finished
* `last_timestamp` - last time of measurements in cluster. Usually it equals to last arrival time in history

Note that `tasks_history.arrived_at` dates should strictly follow a pattern of 
`first_observation_time + measurement_frequency_ms * t`.

History can have gaps. In this case gaps are treated like there were no new tasks in this period. 

Task with the same id can be sent in multiple calls. It's needed to gradually update its status (arrived -> started -> finished). 

Example request:

```
{
    "active_threads" : [
        {"state" : "idle"},
        {"state" : "occupied", "processing_time_passed_ms": 20}
    ],
    "last_timestamp": "2022-07-04 12:00:10",
    "tasks_history": [
        {
            "arrived_at": "2022-07-04 12:00:00",
            "tasks": [
                {"id": "1", "started_at": "2022-07-04 12:00:20", "finished_at": "2022-07-04 12:00:30"}
            ]
        },
        {
            "arrived_at": "2022-07-04 12:00:02",
            "tasks": [
                {"id": "2", "started_at": "2022-07-04 12:00:20", "finished_at": "2022-07-04 12:00:30"},
                {"id": "3", "started_at": "2022-07-04 12:00:20"}
            ]
        },
        {
            "arrived_at": "2022-07-04 12:00:04",
            "tasks": [
                {"id": "4"}
            ]
        },
        {
            "arrived_at": "2022-07-04 12:00:06",
            "tasks": [
                {"id": "5"}
            ]
        },
        {
            "arrived_at": "2022-07-04 12:00:10",
            "tasks": [
                {"id": "6"}
            ]
        }
    ]
}
```
Example response:
```
{
    "predicted_tasks_number": {
        "message": "Using last value",
        "values": [
            {"date": "2022-07-04 12:00:46", "value": 0.0},
            {"date": "2022-07-04 12:00:48","value": 0.0},
            {"date": "2022-07-04 12:00:50","value": 0.0}
        ]
    },
    "scaling_decision": {
        "action": "scale_down",
        "threads": 10
    }
}
```
* `predicted_tasks_number` - predicted number of tasks for the next `worker_setup_delay_ms` + `delta_ms` period with frequency of `measurement_frequency_ms`.
* `scaling_decision` - can be one of ['scale_up', 'scale_down', 'do_nothing']. In case of scale up or scale down `threads` field is also specified. 