"""Chance-association permutation check for model D only (DEVIATION_PROTOCOL): 999 permutations, rng seed 21, folds regenerated from permuted labels, full refit, plus-one p. NOT a test of incremental value or superiority."""
import os, sys, json, time, glob
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
d33 = importlib.import_module('33_dev_outcomes')
def one(T, cols, yp): return float(np.mean([roc_auc_score(yp, d33.run_lr2(T, cols, yp, s)[0]) for s in d33.SEEDS]))
if __name__ == '__main__':
    T, cols, y, both, _ = d33.build_table(); M = pd.read_csv(sorted(glob.glob('runs/DEV_Remitter_*'))[-1] + '/metrics_by_seed.csv'); obs = float(M[M.model == 'D'].AUC.mean())
    rng = np.random.default_rng(21); perms = [rng.permutation(y) for _ in range(999)]; t0 = time.time(); null = np.array(Parallel(n_jobs=12)(delayed(one)(T, cols['D'], yp) for yp in perms))
    out = dict(observed_mean_AUC_D=obs, n_perm=999, rng_seed=21, p_value_plus_one=float((1 + (null >= obs).sum()) / 1000), null_mean=float(null.mean()), null_sd=float(null.std()), null_q95=float(np.quantile(null, .95)), runtime_s=round(time.time() - t0),
               refit='impute/scale, inner-CV C selection, threshold per repeat; folds regenerated from permuted labels (seeds 42/43/44); statistic = mean of 3 pooled-OOF AUCs', interpretation='chance association of model D with remission only; NOT a test of incremental value or superiority over A/B/C/E01c')
    json.dump(out, open('outputs/dev/permutation_D_remitter.json', 'w'), indent=1); print(json.dumps(out, indent=1))
