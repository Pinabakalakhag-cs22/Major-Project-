"""
FederatedEngine — full EA-FL pipeline:
  - Adaptive client selection (score-based)
  - FedAvg aggregation
  - NSGA-II multi-objective optimisation (energy, accuracy, communication)
  - Per-round communication & privacy metrics
"""

import os, math, random
from core.client import FLClient, NUM_FEATURES, NUM_CLASSES

NUM_CLIENTS = 6

# ── FedAvg ────────────────────────────────────────────────────────────────
def _weighted_avg(updates):
    total = sum(u['num_samples'] for u in updates)
    w = [[0.0]*NUM_FEATURES for _ in range(NUM_CLASSES)]
    b = [0.0]*NUM_CLASSES
    for u in updates:
        f = u['num_samples']/total
        for c in range(NUM_CLASSES):
            b[c] += f*u['bias'][c]
            for j in range(NUM_FEATURES):
                w[c][j] += f*u['weights'][c][j]
    return w, b

# ── NSGA-II helpers ───────────────────────────────────────────────────────
def _dominates(a, b):
    better = False
    for va,vb in zip(a,b):
        if va>vb: return False
        if va<vb: better=True
    return better

def _nds(objs):
    n = len(objs)
    dom_by=[[] for _ in range(n)]; cnt=[0]*n; fronts=[[]]
    for i in range(n):
        for j in range(n):
            if i==j: continue
            if _dominates(objs[i],objs[j]): dom_by[i].append(j)
            elif _dominates(objs[j],objs[i]): cnt[i]+=1
        if cnt[i]==0: fronts[0].append(i)
    k=0
    while fronts[k]:
        nxt=[]
        for i in fronts[k]:
            for j in dom_by[i]:
                cnt[j]-=1
                if cnt[j]==0: nxt.append(j)
        k+=1; fronts.append(nxt)
    return [f for f in fronts if f]

def _crowd(front, objs, m):
    dist={i:0.0 for i in front}
    for o in range(m):
        srt=sorted(front,key=lambda i:objs[i][o])
        dist[srt[0]]=dist[srt[-1]]=float('inf')
        rng=objs[srt[-1]][o]-objs[srt[0]][o]
        if rng==0: continue
        for k in range(1,len(srt)-1):
            dist[srt[k]]+=(objs[srt[k+1]][o]-objs[srt[k-1]][o])/rng
    return dist

def _nsga2_sel(pop,objs,sz):
    fronts=_nds(objs); np2,no2=[],[]
    for front in fronts:
        if len(np2)+len(front)<=sz:
            for i in front: np2.append(pop[i]); no2.append(objs[i])
        else:
            cd=_crowd(front,objs,len(objs[0]))
            for i in sorted(front,key=lambda i:-cd[i])[:sz-len(np2)]:
                np2.append(pop[i]); no2.append(objs[i])
            break
    return np2,no2

# ── Engine ────────────────────────────────────────────────────────────────
class FederatedEngine:
    def __init__(self, data_dir, num_rounds=15, clients_per_round=4,
                 local_epochs=3, learning_rate=0.01, noise_scale=1.0):
        self.data_dir          = data_dir
        self.num_rounds        = num_rounds
        self.clients_per_round = clients_per_round
        self.local_epochs      = local_epochs
        self.learning_rate     = learning_rate
        self.noise_scale       = noise_scale
        self.clients           = None
        self.global_weights    = None
        self.global_bias       = None
        self.round_results     = []
        self.training_complete = False
        self.pareto_results    = None

    def initialize(self):
        self.clients = [FLClient(i, self.data_dir, self.noise_scale, self.num_rounds)
                        for i in range(NUM_CLIENTS)]
        random.seed(0)
        self.global_weights = [[random.gauss(0,0.01) for _ in range(NUM_FEATURES)]
                               for _ in range(NUM_CLASSES)]
        self.global_bias = [0.0]*NUM_CLASSES
        self.round_results=[]; self.training_complete=False; self.pareto_results=None

    def train_single_round(self, round_idx):
        # Adaptive selection: score all clients, pick top clients_per_round
        for c in self.clients:
            c.compute_selection_score()
        ranked = sorted(self.clients, key=lambda c: c.selection_score, reverse=True)
        k = min(self.clients_per_round, len(self.clients))
        # Weighted random from top candidates to add diversity
        top = ranked[:max(k, k+2)]
        weights_sel = [c.selection_score+0.01 for c in top]
        selected = []
        pool = list(range(len(top)))
        while len(selected)<k and pool:
            total_w = sum(weights_sel[i] for i in pool)
            r = random.uniform(0, total_w)
            cum = 0
            for i in pool:
                cum += weights_sel[i]
                if cum >= r:
                    selected.append(top[i])
                    pool.remove(i)
                    break

        updates = [c.train_local(self.global_weights, self.global_bias,
                                 self.learning_rate, self.local_epochs, round_idx)
                   for c in selected]

        self.global_weights, self.global_bias = _weighted_avg(updates)

        accs, losses = [], []
        for c in self.clients:
            a,l = c.evaluate(self.global_weights, self.global_bias)
            accs.append(a); losses.append(l)

        g_acc  = sum(accs)/len(accs)
        g_loss = sum(losses)/len(losses)
        t_energy  = sum(u['energy_mj'] for u in updates)
        t_latency = max(u['latency_s'] for u in updates)
        t_bytes   = sum(u['bytes_sent'] for u in updates)
        t_saved   = sum(u['bytes_saved'] for u in updates)
        avg_sigma = sum(u['dp_sigma'] for u in updates)/len(updates)
        part_rate = round(len(selected)/NUM_CLIENTS*100, 1)

        result = {
            'round': round_idx,
            'global_accuracy':   round(g_acc,4),
            'global_loss':       round(g_loss,4),
            'total_energy_mj':   round(t_energy,4),
            'total_latency_s':   round(t_latency,4),
            'bytes_transmitted': t_bytes,
            'bytes_saved':       t_saved,
            'compression_ratio': round(t_saved/(t_bytes+t_saved)*100,1) if (t_bytes+t_saved)>0 else 0,
            'avg_dp_sigma':      round(avg_sigma,4),
            'participation_rate':part_rate,
            'num_selected':      len(selected),
            'client_updates': [{
                'client_id':      selected[i].client_id,
                'device_type':    selected[i].device_type,
                'local_accuracy': round(updates[i]['local_accuracy'],4),
                'energy_mj':      round(updates[i]['energy_mj'],4),
                'latency_s':      round(updates[i]['latency_s'],4),
                'bytes_sent':     updates[i]['bytes_sent'],
                'dp_sigma':       updates[i]['dp_sigma'],
                'selection_score':round(selected[i].selection_score,4),
                'battery_pct':    selected[i].battery_pct,
            } for i in range(len(selected))],
        }
        self.round_results.append(result)
        return result

    def run_nsga2_optimization(self, pop_size=12, generations=8):
        if not self.clients: raise RuntimeError("Not initialized")
        random.seed(42)
        k = min(self.clients_per_round, len(self.clients))

        def rand_ind():
            return sorted(random.sample(range(NUM_CLIENTS), k))

        def eval_ind(ind):
            sel = [self.clients[i] for i in ind]
            ups = [c.train_local(self.global_weights, self.global_bias,
                                 self.learning_rate, 1, round_idx=self.num_rounds-1)
                   for c in sel]
            w,b = _weighted_avg(ups)
            accs = [c.evaluate(w,b)[0] for c in self.clients]
            acc  = sum(accs)/len(accs)
            energy = sum(u['energy_mj'] for u in ups)
            comm   = sum(u['bytes_sent'] for u in ups)/1000.0
            return [round(energy,4), round(1.0-acc,4), round(comm,4)]

        def crossover(a,b):
            pool=list(set(a+b)); random.shuffle(pool)
            return sorted(pool[:k])

        def mutate(ind):
            new=ind[:]
            non_sel=[i for i in range(NUM_CLIENTS) if i not in new]
            if non_sel and random.random()<0.3:
                new[random.randrange(len(new))]=random.choice(non_sel)
                new=sorted(set(new))
                while len(new)<k:
                    rem=[i for i in range(NUM_CLIENTS) if i not in new]
                    if rem: new.append(random.choice(rem))
                new=sorted(new[:k])
            return new

        pop  = [rand_ind() for _ in range(pop_size)]
        objs = [eval_ind(ind) for ind in pop]

        for _ in range(generations):
            off  = [mutate(crossover(pop[random.randrange(len(pop))],
                                     pop[random.randrange(len(pop))])) for _ in range(pop_size)]
            oobj = [eval_ind(o) for o in off]
            pop,objs = _nsga2_sel(pop+off, objs+oobj, pop_size)

        fronts = _nds(objs)
        pf_idx = fronts[0] if fronts else []
        solutions = [{
            'client_ids': pop[i],
            'energy_mj':  objs[i][0],
            'accuracy':   round(1.0-objs[i][1],4),
            'comm_kb':    objs[i][2],
        } for i in pf_idx]
        solutions.sort(key=lambda x:-x['accuracy'])

        energies   = [s['energy_mj'] for s in solutions]
        accuracies = [s['accuracy']  for s in solutions]
        comms      = [s['comm_kb']   for s in solutions]

        self.pareto_results = {
            'pareto_front': solutions,
            'summary': {
                'num_solutions':  len(solutions),
                'min_energy_mj':  round(min(energies),4) if energies else 0,
                'max_energy_mj':  round(max(energies),4) if energies else 0,
                'min_accuracy':   round(min(accuracies),4) if accuracies else 0,
                'max_accuracy':   round(max(accuracies),4) if accuracies else 0,
                'avg_comm_kb':    round(sum(comms)/len(comms),2) if comms else 0,
                'generations':    generations,
                'pop_size':       pop_size,
            }
        }
        return self.pareto_results

    def get_all_results(self):
        if not self.round_results:
            return {'training_complete':False,'round_results':[],
                    'final_metrics':{},'clients_final':[],'pareto_results':None}
        last = self.round_results[-1]
        total_energy = sum(r['total_energy_mj'] for r in self.round_results)
        total_bytes  = sum(r['bytes_transmitted'] for r in self.round_results)
        total_saved  = sum(r['bytes_saved'] for r in self.round_results)
        return {
            'training_complete': self.training_complete,
            'round_results':     self.round_results,
            'final_metrics': {
                'accuracy':           last['global_accuracy'],
                'loss':               last['global_loss'],
                'total_energy_mj':    round(total_energy,4),
                'total_bytes_sent':   total_bytes,
                'total_bytes_saved':  total_saved,
                'overall_compression':round(total_saved/(total_bytes+total_saved)*100,1)
                                      if (total_bytes+total_saved)>0 else 0,
                'num_rounds':         len(self.round_results),
            },
            'clients_final':  [c.to_dict() for c in (self.clients or [])],
            'pareto_results': self.pareto_results,
        }
