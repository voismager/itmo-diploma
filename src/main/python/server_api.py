from flask import Flask, request, jsonify, abort

from scalingservice import ScalingService

api = Flask(__name__)
service = ScalingService()


@api.route('/engines', methods=['POST'])
def create_scaling_engine():
    content = request.get_json()

    engine = service.create_engine(
        content["worker_setup_delay_ms"],
        content["measurement_frequency_ms"],
        content["task_length_order"],
        content["params"]
    )

    return jsonify({
        "id": engine.id,
        "worker_setup_delay_ms": engine.worker_setup_delay_ms,
        "measurement_frequency_ms": engine.measurement_frequency_ms,
        "task_length_distribution_interval_ms": engine.task_lengths_histogram.interval_length,
        "params": engine.params
    })


@api.route('/engines/<engine_id>/distribution', methods=['GET'])
def get_task_length_distribution(engine_id):
    distribution = service.get_task_length_distribution(engine_id)
    return jsonify({
        "values": distribution[0],
        "weights": distribution[1]
    })


@api.route('/engines/<engine_id>/scaling', methods=['POST'])
def get_scaling_decision(engine_id):
    content = request.get_json(silent=True)
    try:
        return jsonify(service.get_scaling_decision(
            engine_id,
            content["active_threads"],
            content["tasks_history"]
        ))

    except KeyError:
        abort(404)


if __name__ == '__main__':
    api.run()
