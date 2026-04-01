"""
Enhanced FL Client with:
  - Real-time energy monitoring (battery %, CPU %, RAM %, temperature)
  - Adaptive client selection scoring
  - Model compression (8-bit quantization + top-k sparsification)
  - Dynamic Differential Privacy (noise decreases as rounds progress)
"""

import os, csv, math, random

ACTIVITIES = ['walking', 'running', 'sitting', 'standing', 'cycling', 'lying']
LABEL_MAP  = {a: i for i, a in enumerate(ACTIVITIES)}
FEATURES   = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z',
               'mag_x','mag_y','mag_z','heart_rate','temperature']
NUM_FEATURES = len(FEATURES)
NUM_CLASSES  = len(ACTIVITIES)

DEVICE_PROFILES = [
    {'type':'smartphone',  'cpu_freq_ghz':2.4,'battery_mah':4000,'tx_power_mw':100,'ram_mb':4096},
    {'type':'smartwatch',  'cpu_freq_ghz':1.0,'battery_mah':300, 'tx_power_mw':20, 'ram_mb':512},
    {'type':'tablet',      'cpu_freq_ghz':2.0,'battery_mah':8000,'tx_power_mw':150,'ram_mb':6144},
    {'type':'iot_sensor',  'cpu_freq_ghz':0.5,'battery_mah':500, 'tx_power_mw':10, 'ram_mb':256},
    {'type':'laptop',      'cpu_freq_ghz':3.2,'battery_mah':5000,'tx_power_mw':200,'ram_mb':8192},
    {'type':'edge_device', 'cpu_freq_ghz':1.8,'battery_mah':2000,'tx_power_mw':80, 'ram_mb':2048},
]

def _softmax(logits):
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e/s for e in exps]

def _cross_entropy(probs, label):
    return -math.log(max(probs[label], 1e-12))

# ── Model Compression ──────────────────────────────────────────────────────
def quantize_weights(weights, bits=8):
    flat = [w for row in weights for w in row]
    if not flat: return weights, 0
    mn, mx = min(flat), max(flat)
    scale = (mx - mn) / (2**bits - 1) if mx != mn else 1.0
    dequant = [[(round((w-mn)/scale))*scale+mn for w in row] for row in weights]
    saved = len(flat) * 4 - len(flat) * (bits//8)
    return dequant, saved

def sparsify_weights(weights, top_k_ratio=0.3):
    flat = sorted([(abs(weights[c][j]),c,j)
                   for c in range(len(weights)) for j in range(len(weights[c]))], reverse=True)
    k = max(1, int(len(flat)*top_k_ratio))
    keep = set((c,j) for _,c,j in flat[:k])
    sparse = [[weights[c][j] if (c,j) in keep else 0.0
               for j in range(len(weights[c]))] for c in range(len(weights))]
    saved = (len(flat)-k)*4
    return sparse, saved

# ── Dynamic DP ────────────────────────────────────────────────────────────
def compute_dp_sigma(round_idx, total_rounds, base=1.0, min_sigma=0.05):
    progress = round_idx / max(total_rounds-1, 1)
    return round(base*(1.0-progress)+min_sigma, 4)

# ── Client ────────────────────────────────────────────────────────────────
class FLClient:
    def __init__(self, client_id, data_dir, noise_scale=1.0, total_rounds=15):
        self.client_id   = client_id
        self.noise_scale = noise_scale
        self.total_rounds= total_rounds

        p = DEVICE_PROFILES[client_id % len(DEVICE_PROFILES)]
        self.device_type   = p['type']
        self.cpu_freq_ghz  = p['cpu_freq_ghz']
        self.battery_mah   = p['battery_mah']
        self.tx_power_mw   = p['tx_power_mw']
        self.ram_mb        = p['ram_mb']

        random.seed(client_id*17+5)
        self.battery_pct   = round(random.uniform(20,100),1)
        self.cpu_load_pct  = round(random.uniform(5,60),1)
        self.ram_used_pct  = round(random.uniform(20,75),1)
        self.temperature_c = round(random.uniform(30,55),1)
        self.network_mbps  = round(random.uniform(0.5,10.0),2)

        random.seed(client_id*13+3)
        self.weights = [[random.gauss(0,0.01) for _ in range(NUM_FEATURES)] for _ in range(NUM_CLASSES)]
        self.bias    = [0.0]*NUM_CLASSES
        self.X, self.y = self._load_data(data_dir)

        self.local_accuracy = 0.0
        self.local_loss     = 0.0
        self.energy_consumed_mj   = 0.0
        self.total_energy_mj      = 0.0
        self.latency_s            = 0.0
        self.rounds_participated  = 0
        self.bytes_sent           = 0
        self.bytes_saved          = 0
        self.dp_sigma             = noise_scale
        self.selection_score      = 0.0
        self.participation_history= []

    def _load_data(self, data_dir):
        path = os.path.join(data_dir, f'client_{self.client_id}.csv')
        X, y = [], []
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                X.append([float(row[ft]) for ft in FEATURES])
                y.append(LABEL_MAP[row['activity']])
        return X, y

    def compute_selection_score(self):
        bat  = self.battery_pct/100.0
        cpu  = (100-self.cpu_load_pct)/100.0
        ram  = (100-self.ram_used_pct)/100.0
        net  = min(self.network_mbps/10.0, 1.0)
        temp = max(0,(self.temperature_c-45)/30.0)
        s = 0.40*bat + 0.25*cpu + 0.20*ram + 0.15*net - 0.10*temp
        self.selection_score = round(max(0.0, min(1.0, s)), 4)
        return self.selection_score

    def _forward(self, x):
        logits = [self.bias[c]+sum(self.weights[c][j]*x[j] for j in range(NUM_FEATURES))
                  for c in range(NUM_CLASSES)]
        return _softmax(logits)

    def _predict(self, x):
        p = self._forward(x); return p.index(max(p))

    def train_local(self, global_weights, global_bias, lr, local_epochs, round_idx=0):
        self.weights = [row[:] for row in global_weights]
        self.bias    = global_bias[:]
        n = len(self.X)
        idx_list = list(range(n))

        for _ in range(local_epochs):
            random.shuffle(idx_list)
            total_loss = 0.0
            for idx in idx_list:
                x, lbl = self.X[idx], self.y[idx]
                probs = self._forward(x)
                total_loss += _cross_entropy(probs, lbl)
                for c in range(NUM_CLASSES):
                    d = probs[c]-(1.0 if c==lbl else 0.0)
                    self.bias[c] -= lr*d
                    for j in range(NUM_FEATURES):
                        self.weights[c][j] -= lr*d*x[j]

        # Dynamic DP
        sigma = compute_dp_sigma(round_idx, self.total_rounds, self.noise_scale)
        self.dp_sigma = sigma
        for c in range(NUM_CLASSES):
            self.bias[c] += random.gauss(0, sigma*0.01)
            for j in range(NUM_FEATURES):
                self.weights[c][j] += random.gauss(0, sigma*0.001)

        # Compression
        qw, saved_q = quantize_weights(self.weights)
        sw, saved_s = sparsify_weights(qw)
        self.bytes_saved = saved_q + saved_s
        model_bytes = (NUM_CLASSES*NUM_FEATURES+NUM_CLASSES)*4
        self.bytes_sent = max(16, model_bytes - self.bytes_saved)

        # Eval
        correct = sum(1 for i in range(n) if self._predict(self.X[i])==self.y[i])
        self.local_accuracy = correct/n
        self.local_loss     = total_loss/n

        # Energy & latency
        ops = n*local_epochs*NUM_FEATURES*NUM_CLASSES*2
        compute_t = ops/(self.cpu_freq_ghz*1e9)
        tx_t = self.bytes_sent/(max(0.1,self.network_mbps)*1e6/8)
        self.latency_s = round(compute_t+tx_t+random.uniform(0.01,0.05),4)
        self.energy_consumed_mj = round(
            compute_t*self.cpu_freq_ghz*0.5*1000 +
            tx_t*self.tx_power_mw + random.uniform(0.3,1.5), 4)
        self.total_energy_mj += self.energy_consumed_mj
        self.rounds_participated += 1
        self.participation_history.append(round_idx)

        # Drain device
        self.battery_pct   = max(1.0,  round(self.battery_pct  -random.uniform(0.5,2.0),1))
        self.cpu_load_pct  = min(99.0, round(self.cpu_load_pct +random.uniform(-5,10),1))
        self.ram_used_pct  = min(95.0, round(self.ram_used_pct +random.uniform(-2,5),1))
        self.temperature_c = round(self.temperature_c+random.uniform(-0.5,1.5),1)

        return {
            'weights': sw, 'bias': self.bias[:],
            'num_samples': n,
            'local_accuracy': round(self.local_accuracy,4),
            'local_loss':     round(self.local_loss,4),
            'energy_mj':  self.energy_consumed_mj,
            'latency_s':  self.latency_s,
            'bytes_sent': self.bytes_sent,
            'bytes_saved':self.bytes_saved,
            'dp_sigma':   sigma,
        }

    def evaluate(self, weights, bias):
        n = len(self.X)
        correct, loss = 0, 0.0
        for i in range(n):
            logits = [bias[c]+sum(weights[c][j]*self.X[i][j] for j in range(NUM_FEATURES))
                      for c in range(NUM_CLASSES)]
            probs = _softmax(logits)
            loss += _cross_entropy(probs, self.y[i])
            if probs.index(max(probs))==self.y[i]: correct+=1
        return correct/n, loss/n

    def to_dict(self):
        return {
            'client_id':    self.client_id,
            'device_type':  self.device_type,
            'cpu_freq_ghz': self.cpu_freq_ghz,
            'battery_mah':  self.battery_mah,
            'ram_mb':       self.ram_mb,
            'tx_power_mw':  self.tx_power_mw,
            'num_samples':  len(self.X),
            'battery_pct':  self.battery_pct,
            'cpu_load_pct': self.cpu_load_pct,
            'ram_used_pct': self.ram_used_pct,
            'temperature_c':self.temperature_c,
            'network_mbps': self.network_mbps,
            'selection_score': self.selection_score,
            'local_accuracy':  round(self.local_accuracy,4),
            'local_loss':      round(self.local_loss,4),
            'energy_consumed_mj': round(self.energy_consumed_mj,4),
            'total_energy_mj':    round(self.total_energy_mj,4),
            'latency_s':    round(self.latency_s,4),
            'bytes_sent':   self.bytes_sent,
            'bytes_saved':  self.bytes_saved,
            'rounds_participated': self.rounds_participated,
            'dp_sigma':     round(self.dp_sigma,4),
        }
