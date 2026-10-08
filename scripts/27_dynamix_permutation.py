"""One prespecified inferential procedure: label-permutation test of the pretrained DynaMix pipeline (Remitter, mean of 3 repeats' pooled-OOF AUC), 999 permutations (rng seed 11).
Each permuted label vector re-runs the full frozen pipeline per repeat: folds regenerated from the permuted labels with the same random_state rule, inner C selection, thresholds. Cached label-independent embeddings are reused. Plus-one p-value. NOT a superiority test vs E01c."""
import os, sys, json, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib, glob
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
d24 = importlib.import_module('24_dynamix_eval'); ev = d24.ev; NPERM = int(os.environ.get('NPERM', 999))
def one(T, yp):
    return float(np.mean([roc_auc_score(yp, d24.run_lr(T, list(T.columns), yp, s)[0]) for s in d24.SEEDS]))
if __name__ == '__main__':
    T = pd.read_pickle('outputs/dynamix/patient_features_pretrained_PRIVATE.pkl'); D, y, *_ = ev.load_all('Remitter'); assert (D.id.values == T.attrs['ids']).all()
    M = pd.read_csv(sorted(glob.glob('runs/DM_Remitter_*'))[-1] + '/metrics_by_seed.csv'); obs = float(M[M.model == 'pretrained'].AUC.mean())
    rng = np.random.default_rng(11); perms = [rng.permutation(y) for _ in range(NPERM)]; t0 = time.time()
    null = Parallel(n_jobs=12, verbose=2)(delayed(one)(T, yp) for yp in perms); null = np.array(null)
    out = dict(observed_mean_AUC=obs, n_perm=NPERM, rng_seed=11, p_value_plus_one=float((1 + (null >= obs).sum()) / (NPERM + 1)), null_mean=float(null.mean()), null_sd=float(null.std()), null_q95=float(np.quantile(null, .95)), null_q99=float(np.quantile(null, .99)), runtime_s=round(time.time() - t0),
               refit='full pipeline (impute/scale, inner-CV C selection, threshold) per repeat; folds regenerated from permuted labels with seeds 42/43/44; statistic = mean of 3 pooled-OOF AUCs', interpretation='association of the pretrained-DynaMix pipeline with remission beyond label-shuffled chance; not superiority over E01c; not evidence that pretraining helps')
    json.dump(out, open('outputs/dynamix/permutation_pretrained_remitter.json', 'w'), indent=1); pd.DataFrame({'null_mean_auc': null}).to_csv('outputs/dynamix/permutation_null.csv', index=False); print(json.dumps(out, indent=1))
