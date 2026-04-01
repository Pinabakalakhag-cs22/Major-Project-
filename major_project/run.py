"""
Entry point for the Energy-Aware Federated Learning Dashboard.
Generates data if needed and starts the Flask server.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.generate_har_data import generate_all_data
from app import app


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    check_file = os.path.join(data_dir, 'client_0.csv')

    if not os.path.exists(check_file):
        print("=" * 50)
        print("  Generating HAR Sensor Data...")
        print("=" * 50)
        generate_all_data(data_dir)

    print("\n" + "=" * 50)
    print("  Energy-Aware Federated Learning Dashboard")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
