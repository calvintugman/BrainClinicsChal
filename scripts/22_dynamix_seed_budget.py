"""Choose the gating-noise replicate budget K from an OUTCOME-BLIND sample of non-rTMS pool recordings (no labels exist for them).
Sample: first 20 eligible recordings (>=1 eligible 30-s window) of the fixed-seed (7) draw in window_manifest_seedsample; first window of each. 4 models x 16 replicates (seed 900000+recording index).
Stability criterion (fixed before any outcome scoring): for K in {1,2,4,8}, mean of replicates 0..K-1 (A) vs replicates K..2K-1 (B); per feature type (12), Pearson r across the 20 windows x 26 channels = 520 units;
K = smallest value for which, in ALL four models, median over the 12 feature types of r >= 0.95 AND min over feature types >= 0.90; if none, K = 8 (maximum tested)."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
OUT = 'outputs'
def one(model, key, idx):
    import dm; m = dm.build(model); W = np.load(f'{OUT}/dynamix/windows/{key}.npz')['W'][0]; t = time.time(); e = dm.embed_window(m, W, 16, seed=900000 + idx); return model, idx, e, time.time() - t
if __name__ == '__main__':
    M = pd.read_csv(f'{OUT}/dynamix/window_manifest_seedsample_PRIVATE.csv'); E = M[(M.status == 'ok') & (M.n_windows >= 1)].head(20); print('eligible sample recordings', len(E), 'of', len(M))
    jobs = [(mo, r.key, i) for mo in ('pretrained', 'random_a', 'random_b', 'random_c') for i, r in enumerate(E.itertuples())]
    res = Parallel(n_jobs=8)(delayed(one)(*j) for j in jobs); out = {}; stab = []; tms = []
    for mo in ('pretrained', 'random_a', 'random_b', 'random_c'):
        arr = np.stack([r[2] for r in sorted([x for x in res if x[0] == mo], key=lambda x: x[1])])        # (20,16,26,12)
        tms += [r[3] for r in res if r[0] == mo]
        for K in (1, 2, 4, 8):
            A = arr[:, :K].mean(1).reshape(-1, 12); B = arr[:, K:2*K].mean(1).reshape(-1, 12); r = [np.corrcoef(A[:, f], B[:, f])[0, 1] for f in range(12)]
            stab.append(dict(model=mo, K=K, median_r=float(np.nanmedian(r)), min_r=float(np.nanmin(r)), r_by_feature=[round(float(x), 4) for x in r]))
    S = pd.DataFrame(stab); S['ok'] = (S.median_r >= .95) & (S.min_r >= .90); good = [K for K in (1, 2, 4, 8) if S[S.K == K].ok.all()]; K = good[0] if good else 8
    out = dict(n_sample_recordings=len(E), n_units=int(len(E) * 26), chosen_K=int(K), criterion_met=bool(good), per_model_per_K=S.drop(columns='r_by_feature').to_dict('records'), mean_seconds_per_window_K16_contended=float(np.mean(tms)), note='K = number of independent gating-noise replicates averaged per window')
    json.dump(out, open(f'{OUT}/dynamix/seed_budget.json', 'w'), indent=1); S.to_csv(f'{OUT}/dynamix/seed_budget_stability.csv', index=False); print(S.drop(columns='r_by_feature').round(4).to_string(index=False)); print('chosen K', K, 'criterion met', bool(good))
