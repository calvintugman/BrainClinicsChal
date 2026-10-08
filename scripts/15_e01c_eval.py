"""One corrected reference: E01 under the v2 cleaning pipeline (E01c), same model/splits/threshold rule as E01. Versioned separately."""
import os, sys, json, time, hashlib, platform
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
ev = importlib.import_module('10_batch2_eval')
def one(endpoint, seed):
    os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; return ev.run_task('E01_ref', seed, endpoint)
if __name__ == '__main__':
    for endpoint in ('Remitter', 'Responder'):
        t0 = time.time(); res = Parallel(n_jobs=3)(delayed(one)(endpoint, s) for s in ev.SEEDS)
        D, y, *_ = ev.load_all(endpoint); rid = f'PIV_E01c_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
        rows = [dict(model='E01c', seed=r['seed'], runtime_s=round(r['runtime'], 1), n_features=r['n_features'], **ev.metrics(y, r['p'], r['dec'])) for r in res]; M = pd.DataFrame(rows); M.to_csv(f'{rd}/metrics_by_seed.csv', index=False)
        pd.DataFrame({'id': D.id, 'y': y, **{f'p_E01c_s{r["seed"]}': r['p'] for r in res}, **{f'dec_E01c_s{r["seed"]}': r['dec'] for r in res}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
        old = pd.read_csv(sorted(__import__('glob').glob(f'runs/R_rest_{endpoint}_*'))[-1] + '/predictions_PRIVATE.csv'); assert (old.id == D.id).all()
        # AUC paired bootstrap vs original E01 (same draws as earlier batches), conditional on saved predictions
        rng = np.random.default_rng(1); draws = rng.integers(0, 164, size=(2000, 164)); from sklearn.metrics import roc_auc_score as auc
        A1 = np.array([[auc(y[i], r['p'][i]) for r in res] for i in draws if len(np.unique(y[i])) > 1]); A0 = np.array([[auc(y[i], old[f'p_E01_s{s}'].values[i]) for s in ev.SEEDS] for i in draws if len(np.unique(y[i])) > 1])
        d = (A1 - A0).mean(1); iv = dict(mean_auc_E01c=float(M.AUC.mean()), mean_auc_E01=float(np.mean([auc(y, old[f'p_E01_s{s}']) for s in ev.SEEDS])), diff_mean=float(M.AUC.mean() - np.mean([auc(y, old[f'p_E01_s{s}']) for s in ev.SEEDS])), diff_lo=float(np.quantile(d, .025)), diff_hi=float(np.quantile(d, .975)),
                                         E01c_mean_auc_lo=float(np.quantile(A1.mean(1), .025)), E01c_mean_auc_hi=float(np.quantile(A1.mean(1), .975)))
        json.dump(iv, open(f'{rd}/auc_vs_original_E01.json', 'w'), indent=1)
        cfg = dict(run_id=rid, endpoint=endpoint, n=len(y), n_pos=int(y.sum()), parent='R_rest_' + endpoint, hypothesis='reference only: E01 recomputed under corrected (v2) cleaning', features='outputs/v2/features_v2_PRIVATE.csv', feature_fp=hashlib.sha256(open('outputs/v2/features_v2_PRIVATE.csv', 'rb').read()).hexdigest()[:16],
                   code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], clean_code_fp=hashlib.sha256(open('scripts/clean_v2.py', 'rb').read()).hexdigest()[:16], seeds=ev.SEEDS, started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), runtime_s=round(time.time() - t0), python=platform.python_version())
        json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1); print(rid); print(M[['model', 'seed', 'AUC', 'nPPV', 'sensitivity', 'PPV', 'Brier', 'n_pos_calls']].round(3).to_string(index=False)); print(json.dumps(iv))
