"""Frozen DynaMix embedding of all patient recordings (resumable). Per (recording, model): windows (<=4) x K replicates -> per window: mean over replicates (and the two replicate halves A/B for diagnostics).
Seed for (model m, recording index r, window w): 7_000_000 + 1000*r + 10*w + m  (m: pretrained 0, random_a 1, random_b 2, random_c 3). K from outputs/dynamix/seed_budget.json."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
OUT = 'outputs'; MODS = ['pretrained', 'random_a', 'random_b', 'random_c']
def task(mi, r, key, K):
    import dm; fn = f'{OUT}/dynamix/emb/{MODS[mi]}/{key}.npz'
    if os.path.exists(fn): return fn, 0.0
    t = time.time(); m = dm.build(MODS[mi]); W = np.load(f'{OUT}/dynamix/windows/{key}.npz')['W']; res = []
    for w in range(len(W)): res.append(dm.embed_window(m, W[w], K, seed=7_000_000 + 1000 * r + 10 * w + mi))              # (K,26,12)
    R = np.stack(res) if res else np.zeros((0, K, 26, 12), np.float32); h = K // 2
    np.savez_compressed(fn + '.tmp.npz', mean=R.mean(1), halfA=R[:, :h].mean(1) if K > 1 else R.mean(1), halfB=R[:, h:].mean(1) if K > 1 else R.mean(1)); os.replace(fn + '.tmp.npz', fn); return fn, time.time() - t
if __name__ == '__main__':
    K = json.load(open(f'{OUT}/dynamix/seed_budget.json'))['chosen_K']; M = pd.read_csv(f'{OUT}/dynamix/window_manifest_patients_PRIVATE.csv').reset_index(drop=True); E = M[(M.status == 'ok') & (M.n_windows >= 1)]
    for mo in MODS: os.makedirs(f'{OUT}/dynamix/emb/{mo}', exist_ok=True)
    jobs = [(mi, int(i), r.key, K) for i, r in E.iterrows() for mi in range(4)]; t0 = time.time(); print('K', K, 'tasks', len(jobs), flush=True)
    res = Parallel(n_jobs=int(os.environ.get('NJ', 12)), verbose=5)(delayed(task)(*j) for j in jobs); tt = [x[1] for x in res if x[1] > 0]
    json.dump(dict(K=K, tasks=len(jobs), wall_s=round(time.time() - t0), summed_task_s=round(sum(tt)), mean_task_s=round(float(np.mean(tt)), 1) if tt else None), open(f'{OUT}/dynamix/embed_runtime.json', 'w'), indent=1); print('done', round(time.time() - t0), 's')
