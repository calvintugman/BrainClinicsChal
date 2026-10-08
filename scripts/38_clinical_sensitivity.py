"""POST-HOC sensitivity to CLINICAL_PROTOCOL.md results (NOT frozen in advance; labelled exploratory, does not alter any A4 decision). Which part of K1 carries the signal?
S1 = K1 without the 4 missing-protocol patients (refit on 160, new folds on the subset); S2 = BDI_pre only; S3 = age+sex+BDI_pre; S4 = protocol indicators only. Same pipeline/seeds/grid/threshold rule."""
import os, sys, json, time, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score as auc
warnings.filterwarnings('ignore')
c = importlib.import_module('37_clinical_eval'); ev = c.ev; SEEDS = c.SEEDS
VARS = {'S1_K1_no_missing_protocol': (c.CLIN_K1, True), 'S2_BDI_pre_only': (['BDI_pre'], False), 'S3_age_sex_BDI': (['age', 'sex', 'BDI_pre'], False), 'S4_protocol_only': (['prot1', 'prot2', 'prot3'], False)}
def task(name, s, endpoint):
    T, cols, y = c.build(endpoint); v, drop = VARS[name]
    if drop: keep = T.prot_raw.notna().values; T, y = T[keep].reset_index(drop=True), y[keep]
    p, d, meta = c.run_lr(T, v, y, s); return name, s, p, d, y
if __name__ == '__main__':
    out = {}
    for endpoint in ('Remitter', 'Responder'):
        res = Parallel(n_jobs=12)(delayed(task)(n, s, endpoint) for n in VARS for s in SEEDS); rows = [dict(endpoint=endpoint, model=n, seed=s, **ev.metrics(y, p, d)) for n, s, p, d, y in res]; M = pd.DataFrame(rows)
        pd.set_option('display.width', 200); print(endpoint); print(M.groupby('model')[['AUC', 'nPPV', 'sensitivity']].mean().round(3).to_string()); print(M[['model', 'seed', 'N', 'AUC', 'nPPV', 'sensitivity']].round(3).to_string(index=False), flush=True)
        rid = f'CLINX_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}'); M.to_csv(f'runs/{rid}/metrics_by_seed.csv', index=False)
        json.dump(dict(run_id=rid, status='POST-HOC exploratory sensitivity to A4', code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], parent='CLIN_' + endpoint), open(f'runs/{rid}/config.json', 'w'), indent=1)
