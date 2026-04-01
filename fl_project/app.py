
"""
Flask Backend API for Energy-Aware Federated Learning Dashboard.

Endpoints:
  GET  /              - Serve the web UI
  POST /api/train     - Start FL training (async, returns immediately)
  POST /api/nsga2     - Run NSGA-II optimization (async, returns immediately)
  GET  /api/status    - Get training/optimization status
  GET  /api/results   - Get training results
  GET  /api/clients   - Get client device info
  GET  /api/pareto    - Get Pareto front data
"""

import os
import sys
import json
import threading
import traceback

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.federated_engine import FederatedEngine
from data.generate_har_data import generate_all_data

app = Flask(__name__, static_folder='static')

# Global state
engine = None
training_lock = threading.Lock()
training_status = {'state': 'idle', 'progress': 0, 'message': ''}


def ensure_data_exists():
    """Generate sensor data if not already present."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    check_file = os.path.join(data_dir, 'client_0.csv')
    if not os.path.exists(check_file):
        print("Generating HAR sensor data...")
        generate_all_data(data_dir)
        print("Data generation complete.")


# ---------- STATIC ROUTES ----------

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)


# ---------- API ROUTES ----------

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify(training_status)


@app.route('/api/train', methods=['POST'])
def start_training():
    global engine, training_status

    if not training_lock.acquire(blocking=False):
        return jsonify({'error': 'Training already in progress'}), 409

    try:
        config = request.json or {}
        num_rounds = config.get('num_rounds', 15)
        clients_per_round = config.get('clients_per_round', 4)
        local_epochs = config.get('local_epochs', 3)
        learning_rate = config.get('learning_rate', 0.01)
        noise_scale = config.get('noise_scale', 0.5)

        training_status = {'state': 'training', 'progress': 0,
                           'message': 'Initializing...'}

        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        engine = FederatedEngine(
            data_dir=data_dir,
            num_rounds=num_rounds,
            clients_per_round=clients_per_round,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            noise_scale=noise_scale,
        )
        engine.initialize()
        training_status['message'] = 'Training started'

        def train_loop():
            global training_status
            try:
                for r in range(num_rounds):
                    result = engine.train_single_round(r)
                    progress = int((r + 1) / num_rounds * 100)
                    training_status = {
                        'state': 'training',
                        'progress': progress,
                        'message': f'Round {r + 1}/{num_rounds} — Acc: {result["global_accuracy"]:.4f}',
                        'current_round': result,
                    }

                engine.training_complete = True
                training_status = {
                    'state': 'complete',
                    'progress': 100,
                    'message': 'Training complete!',
                    'training_complete': True,
                }
            except Exception as e:
                traceback.print_exc()
                training_status = {'state': 'error', 'progress': 0,
                                   'message': str(e)}
            finally:
                training_lock.release()

        thread = threading.Thread(target=train_loop, daemon=True)
        thread.start()

        return jsonify({'status': 'started', 'message': 'Training started in background'}), 202

    except Exception as e:
        traceback.print_exc()
        training_status = {'state': 'error', 'progress': 0, 'message': str(e)}
        training_lock.release()
        return jsonify({'error': str(e)}), 500


@app.route('/api/nsga2', methods=['POST'])
def run_nsga2():
    global engine, training_status

    if engine is None or not getattr(engine, 'training_complete', False):
        return jsonify({'error': 'Run training first'}), 400

    if not training_lock.acquire(blocking=False):
        return jsonify({'error': 'Optimization already in progress'}), 409

    try:
        config = request.json or {}
        pop_size = config.get('pop_size', 12)
        generations = config.get('generations', 8)

        training_status = {'state': 'optimizing', 'progress': 0,
                           'message': 'Running NSGA-II...'}

        def nsga2_loop():
            global training_status
            try:
                pareto_results = engine.run_nsga2_optimization(pop_size, generations)
                training_status = {
                    'state': 'complete',
                    'progress': 100,
                    'message': 'Optimization complete!',
                    'nsga2_complete': True,
                }
            except Exception as e:
                traceback.print_exc()
                training_status = {
                    'state': 'error',
                    'progress': 0,
                    'message': f'NSGA-II failed: {e}',
                }
            finally:
                training_lock.release()

        thread = threading.Thread(target=nsga2_loop, daemon=True)
        thread.start()

        return jsonify({'status': 'started', 'message': 'NSGA-II started in background'}), 202

    except Exception as e:
        traceback.print_exc()
        training_status = {'state': 'error', 'progress': 0, 'message': str(e)}
        training_lock.release()
        return jsonify({'error': str(e)}), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    if engine is None:
        return jsonify({'error': 'No training results yet'}), 404
    try:
        return jsonify(engine.get_all_results())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clients', methods=['GET'])
def get_clients():
    if engine is None or engine.clients is None:
        return jsonify({'error': 'No clients initialized'}), 404
    try:
        return jsonify([c.to_dict() for c in engine.clients])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pareto', methods=['GET'])
def get_pareto():
    if engine is None or engine.pareto_results is None:
        return jsonify({'error': 'No NSGA-II results yet'}), 404
    try:
        return jsonify(engine.pareto_results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    ensure_data_exists()
    print("\n" + "=" * 50)
    print("  FL Dashboard: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
