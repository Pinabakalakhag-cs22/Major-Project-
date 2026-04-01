"""Debug test to verify engine works end-to-end."""
import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.federated_engine import FederatedEngine

e = FederatedEngine(
    data_dir='data',
    num_rounds=5,
    clients_per_round=4,
    local_epochs=2,
    learning_rate=0.01,
    noise_scale=0.5
)
e.initialize()

for r in range(5):
    try:
        result = e.train_single_round(r)
        acc = result["global_accuracy"]
        loss = result["global_loss"]
        energy = result["total_energy_mj"]
        print(f"Round {r}: acc={acc}, loss={loss}, energy={energy}")
    except Exception as ex:
        print(f"ERROR at round {r}: {ex}")
        traceback.print_exc()
        break

e.training_complete = True
results = e.get_all_results()
print(f"\ntraining_complete: {results['training_complete']}")
print(f"round_results count: {len(results['round_results'])}")
print(f"final accuracy: {results['final_metrics']['accuracy']}")
print(f"total energy: {results['final_metrics']['total_energy_mj']}")
