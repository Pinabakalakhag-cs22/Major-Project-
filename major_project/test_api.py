"""Verify training API returns valid results."""
import json
import urllib.request

data = json.dumps({'num_rounds': 5, 'clients_per_round': 4, 'local_epochs': 2, 'learning_rate': 0.01, 'noise_scale': 0.5}).encode()
req = urllib.request.Request('http://localhost:5000/api/train', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
print("OK" if result['training_complete'] else "FAIL")
print(f"Accuracy: {result['final_metrics']['accuracy']}")
print(f"Energy: {result['final_metrics']['total_energy_mj']} mJ")
print(f"Rounds: {len(result['round_results'])}")
print(f"Clients: {len(result['clients_final'])}")
for r in result['round_results']:
    print(f"  R{r['round']}: acc={r['global_accuracy']:.4f} loss={r['global_loss']:.4f} energy={r['total_energy_mj']} latency={r['total_latency_s']}")
