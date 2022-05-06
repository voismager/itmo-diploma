from flask import Flask, request, jsonify, abort

from scaling_service import ScalingService

api = Flask(__name__)
service = ScalingService()


@api.route('/engines', methods=['POST'])
def create_scaling_engine():
    content = request.get_json()

    engine = service.create_engine(
        content["worker_setup_delay_ms"],
        content["delta_ms"],
        content["measurement_frequency_ms"],
        content["task_length_order"],
        content["params"]
    )

    return jsonify({
        "id": engine.id,
        "measurement_frequency_ms": engine.measurement_frequency_ms,
        "task_length_distribution_interval_ms": engine.history.task_lengths_histogram.interval_length,
        "params": engine.params
    })


@api.route('/engines/<engine_id>/stats', methods=['GET'])
def get_stats(engine_id):
    stats = service.get_stats(engine_id)
    return jsonify(stats)


@api.route('/engines/<engine_id>/scaling', methods=['POST'])
def get_scaling_decision(engine_id):
    content = request.get_json(silent=True)
    try:
        return jsonify(service.get_scaling_decision(
            engine_id,
            content["active_threads"],
            content["tasks_history"],
            content["last_timestamp"]
        ))

    except KeyError:
        abort(404)


if __name__ == '__main__':
    api.run()
