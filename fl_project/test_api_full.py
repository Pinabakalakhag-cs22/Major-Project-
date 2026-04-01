"""Quick API integration test."""
import urllib.request
import json
import time
import sys

BASE = 'http://localhost:5000'

def api_get(path):
    resp = urllib.request.urlopen(BASE + path, timeout=10)
    return resp.status, json.loads(resp.read())

def api_post(path, data=None):
    body = json.dumps(data or {}).encode()
    req = urllib.request.Request(BASE + path, data=body,
                                headers={'Content-Type': 'application/json'},
                                method='POST')
    resp = urllib.request.urlopen(req, timeout=10)
    return resp.status, json.loads(resp.read())

# 1. Start training
print("1. POST /api/train ...")
code, body = api_post('/api/train', {'num_rounds': 5})
print(f"   Status: {code}, Body: {body}")
assert code == 202, f"Expected 202, got {code}"

# 2. Poll until complete
print("2. Polling /api/status ...")
for i in range(60):
    time.sleep(1)
    code, status = api_get('/api/status')
    state = status.get('state', '')
    progress = status.get('progress', 0)
    msg = status.get('message', '')
    print(f"   [{i+1}s] state={state} progress={progress} msg={msg}")
    if state == 'complete':
        print("   >>> Training complete!")
        break
    if state == 'error':
        print(f"   >>> ERROR: {msg}")
        sys.exit(1)
else:
    print("   >>> TIMEOUT after 60s")
    sys.exit(1)

# 3. Get results
print("3. GET /api/results ...")
code, results = api_get('/api/results')
print(f"   Status: {code}")
print(f"   training_complete: {results.get('training_complete')}")
print(f"   round_results: {len(results.get('round_results', []))} rounds")
print(f"   final_metrics: {results.get('final_metrics')}")
print(f"   clients_final: {len(results.get('clients_final', []))} clients")
assert results['training_complete'] is True

# 4. Start NSGA-II
print("4. POST /api/nsga2 ...")
code, body = api_post('/api/nsga2', {'pop_size': 6, 'generations': 3})
print(f"   Status: {code}, Body: {body}")
assert code == 202, f"Expected 202, got {code}"

# 5. Poll until complete
print("5. Polling /api/status for NSGA-II ...")
for i in range(120):
    time.sleep(1)
    code, status = api_get('/api/status')
    state = status.get('state', '')
    msg = status.get('message', '')
    print(f"   [{i+1}s] state={state} msg={msg}")
    if state == 'complete':
        print("   >>> NSGA-II complete!")
        break
    if state == 'error':
        print(f"   >>> ERROR: {msg}")
        sys.exit(1)
else:
    print("   >>> TIMEOUT after 120s")
    sys.exit(1)

# 6. Get Pareto
print("6. GET /api/pareto ...")
code, pareto = api_get('/api/pareto')
print(f"   Status: {code}")
print(f"   Pareto solutions: {pareto.get('summary', {}).get('num_solutions', 0)}")
print(f"   Pareto front length: {len(pareto.get('pareto_front', []))}")

print("\n=== ALL API TESTS PASSED ===")
