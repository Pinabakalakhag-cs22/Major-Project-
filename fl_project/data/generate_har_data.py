"""
Generate synthetic HAR (Human Activity Recognition) sensor data
for 6 federated learning clients.
"""

import os
import csv
import random
import math


ACTIVITIES = ['walking', 'running', 'sitting', 'standing', 'cycling', 'lying']
NUM_CLIENTS = 6
SAMPLES_PER_CLIENT = 500
FEATURES = [
    'acc_x', 'acc_y', 'acc_z',
    'gyro_x', 'gyro_y', 'gyro_z',
    'mag_x', 'mag_y', 'mag_z',
    'heart_rate', 'temperature',
]

# Activity-specific mean signatures for realism
ACTIVITY_SIGNATURES = {
    'walking':  {'acc_x': 0.3, 'acc_y': 9.5, 'acc_z': 0.2, 'gyro_x': 0.1, 'gyro_y': 0.05, 'gyro_z': 0.1,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 90, 'temperature': 36.8},
    'running':  {'acc_x': 1.2, 'acc_y': 9.0, 'acc_z': 0.8, 'gyro_x': 0.5, 'gyro_y': 0.3, 'gyro_z': 0.4,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 150, 'temperature': 37.5},
    'sitting':  {'acc_x': 0.0, 'acc_y': 9.8, 'acc_z': 0.0, 'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 70, 'temperature': 36.5},
    'standing': {'acc_x': 0.0, 'acc_y': 9.8, 'acc_z': 0.0, 'gyro_x': 0.02, 'gyro_y': 0.01, 'gyro_z': 0.01,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 75, 'temperature': 36.6},
    'cycling':  {'acc_x': 0.5, 'acc_y': 9.2, 'acc_z': 0.3, 'gyro_x': 0.3, 'gyro_y': 0.2, 'gyro_z': 0.15,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 120, 'temperature': 37.2},
    'lying':    {'acc_x': 0.0, 'acc_y': 0.0, 'acc_z': 9.8, 'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
                 'mag_x': 20.0, 'mag_y': 5.0, 'mag_z': -40.0, 'heart_rate': 60, 'temperature': 36.4},
}

NOISE_SCALE = {
    'acc_x': 0.3, 'acc_y': 0.3, 'acc_z': 0.3,
    'gyro_x': 0.05, 'gyro_y': 0.05, 'gyro_z': 0.05,
    'mag_x': 2.0, 'mag_y': 2.0, 'mag_z': 2.0,
    'heart_rate': 5.0, 'temperature': 0.2,
}


def _gauss(mean, std):
    # Box-Muller transform (no numpy dependency)
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2.0 * math.log(max(u1, 1e-10))) * math.cos(2 * math.pi * u2)
    return mean + std * z


def generate_client_data(client_id, num_samples=SAMPLES_PER_CLIENT):
    """Generate one client's HAR sensor data rows."""
    random.seed(client_id * 42 + 7)
    rows = []
    for i in range(num_samples):
        activity = random.choice(ACTIVITIES)
        sig = ACTIVITY_SIGNATURES[activity]
        row = {'sample_id': i, 'client_id': client_id, 'activity': activity}
        for feat in FEATURES:
            row[feat] = round(_gauss(sig[feat], NOISE_SCALE[feat]), 4)
        rows.append(row)
    return rows


def generate_all_data(data_dir, num_clients=NUM_CLIENTS):
    """Write client_0.csv … client_N.csv into data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    fieldnames = ['sample_id', 'client_id', 'activity'] + FEATURES

    for cid in range(num_clients):
        rows = generate_client_data(cid)
        path = os.path.join(data_dir, f'client_{cid}.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated {num_clients} client CSV files in '{data_dir}'")


if __name__ == '__main__':
    generate_all_data('data')
