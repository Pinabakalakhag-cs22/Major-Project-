"""Verify NSGA-II runs end-to-end without errors."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.federated_engine import FederatedEngine

print("=== NSGA-II End-to-End Test ===")

try:
    e = FederatedEngine(
        data_dir='data',
        num_rounds=3,
        clients_per_round=3,
        local_epochs=1,
        learning_rate=0.01,
        noise_scale=0.5
    )
    e.initialize()

    for r in range(3):
        result = e.train_single_round(r)
        print(f"  Round {r}: acc={result['global_accuracy']:.4f}")

    e.training_complete = True
    print("Training OK")

    # Run NSGA-II with small params
    pareto_results = e.run_nsga2_optimization(pop_size=6, generations=3)
    print(f"NSGA-II OK: {pareto_results['summary']['num_solutions']} Pareto solutions")

    # Verify get_all_results
    all_results = e.get_all_results()
    assert all_results['training_complete'] is True
    assert len(all_results['round_results']) == 3
    assert all_results['pareto_results'] is not None
    assert len(all_results['clients_final']) == 6
    print(f"get_all_results OK: keys={list(all_results.keys())}")

    print("\n=== ALL TESTS PASSED ===")
except Exception as ex:
    print(f"\n=== TEST FAILED ===")
    print(f"Error: {ex}")
    traceback.print_exc()
    sys.exit(1)
