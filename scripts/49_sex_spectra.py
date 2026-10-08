"""SEX_SPECTRA_PROTOCOL.md (A11). Run from first_tests/."""
import os, sys, json, time, hashlib, warnings, importlib
os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score as auc, balanced_accuracy_score as bacc
warnings.filterwarnings('ignore'); sm = importlib.import_module('48_spectra_models'); SEEDS = [42, 43, 44]
A = sm.cohort(True); y = A.gender.astype(int).values; MODELS = {'X0': sm.E01, 'X1': sm.SPEC_COLS, 'X2': sm.SPEC_COLS + sm.E01}
def one(m, seed):
    X = A[MODELS[m]].values; p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        g = GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=3000)), {'logisticregression__C': [1e-4, 1e-3, 1e-2, 0.1, 1]}, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring='neg_log_loss').fit(X[tr], y[tr]); p[te] = g.predict_proba(X[te])[:, 1]
    return m, seed, p
if __name__ == '__main__':
    t0 = time.time(); res = Parallel(n_jobs=9)(delayed(one)(m, s) for m in MODELS for s in SEEDS); R = {}
    for m in MODELS:
        P = [p for mm, s, p in res if mm == m]; row = {}
        for g in ('ALL', 'MDD', 'HEALTHY', 'OTHER'):
            ix = np.arange(len(y)) if g == 'ALL' else np.where(A.grp.values == g)[0]; row[g] = dict(n=len(ix), AUC=float(np.mean([auc(y[ix], p[ix]) for p in P])), BA=float(np.mean([bacc(y[ix], (p[ix] >= .5).astype(int)) for p in P])))
        R[m] = row
    best = max(('X1', 'X2'), key=lambda m: R[m]['ALL']['BA']); R['rule'] = dict(best=best, usable=bool(all(R[best][g]['BA'] >= .70 for g in ('MDD', 'HEALTHY', 'OTHER'))), improved=bool(R[best]['ALL']['BA'] >= R['X0']['ALL']['BA'] + .05))
    rid = f'SEXSPEC_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}'); json.dump(R, open(f'runs/{rid}/analysis.json', 'w'), indent=1)
    json.dump(dict(protocol='SEX_SPECTRA_PROTOCOL.md', protocol_fp=hashlib.sha256(open('SEX_SPECTRA_PROTOCOL.md', 'rb').read()).hexdigest()[:16], code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], n=len(y), runtime_s=round(time.time() - t0)), open(f'runs/{rid}/config.json', 'w'), indent=1)
    for m in MODELS: print(m, {g: (round(R[m][g]['AUC'], 3), round(R[m][g]['BA'], 3)) for g in ('ALL', 'MDD', 'HEALTHY', 'OTHER')})
    print(rid, R['rule'])
