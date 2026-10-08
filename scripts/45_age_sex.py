"""AGE_SEX_PROTOCOL.md (A8). Run from first_tests/: .venv/bin/python scripts/45_age_sex.py"""
import os, sys, json, time, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score as auc, balanced_accuracy_score as bacc
warnings.filterwarnings('ignore')
d7 = importlib.import_module('44_confound_eval'); SEEDS = [42, 43, 44]; OUT = 'outputs'
A, E65 = d7.load_all(); F81 = [c for c in pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv', nrows=1).columns if c != 'id']; assert len(F81) == 81
grp = np.where(A.formal_status == 'MDD', 'MDD', np.where(A.formal_status == 'HEALTHY', 'HEALTHY', 'OTHER'))
def make(target, kind, fs, seed):
    cv = (KFold(3, shuffle=True, random_state=seed) if kind == 'hgb' else KFold(5, shuffle=True, random_state=seed)) if target == 'age' else (StratifiedKFold(3, shuffle=True, random_state=seed) if kind == 'hgb' else StratifiedKFold(5, shuffle=True, random_state=seed))
    if target == 'age' and kind == 'lin': return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge()), {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=cv, scoring='neg_mean_squared_error')
    if target == 'sex' and kind == 'lin': return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(max_iter=2000)), {'logisticregression__C': [0.01, 0.1, 1, 10]}, cv=cv, scoring='neg_log_loss')
    H = HistGradientBoostingRegressor if target == 'age' else HistGradientBoostingClassifier; name = H.__name__.lower(); grid = {f'{name}__max_leaf_nodes': [3, 7], f'{name}__l2_regularization': [1, 10]}
    return GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), H(learning_rate=0.05, max_iter=150, early_stopping=False, min_samples_leaf=20, random_state=seed)), grid, cv=cv, scoring='neg_mean_squared_error' if target == 'age' else 'neg_log_loss')
def one(target, kind, fsname, seed):
    cols = E65 if fsname == 'F65' else F81; X = A[cols].values; y = A.age.values if target == 'age' else A.sex.astype(int).values
    outer = KFold(5, shuffle=True, random_state=seed).split(X) if target == 'age' else StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y); p = np.zeros(len(y))
    for tr, te in outer:
        g = make(target, kind, fsname, seed).fit(X[tr], y[tr]); p[te] = g.predict(X[te]) if target == 'age' else g.predict_proba(X[te])[:, 1]
    return target, kind, fsname, seed, p
def stats(target, y, p, idx):
    return dict(R2=float(r2_score(y[idx], p[idx])), MAE=float(np.abs(y[idx] - p[idx]).mean()), r=float(np.corrcoef(y[idx], p[idx])[0, 1])) if target == 'age' else dict(AUC=float(auc(y[idx], p[idx])), BA=float(bacc(y[idx], (p[idx] >= .5).astype(int))))
if __name__ == '__main__':
    t0 = time.time(); print('N', len(A), pd.Series(grp).value_counts().to_dict(), flush=True)
    res = Parallel(n_jobs=12)(delayed(one)(t, k, f, s) for t in ('age', 'sex') for k in ('lin', 'hgb') for f in ('F65', 'F81') for s in SEEDS); R = {}
    for t in ('age', 'sex'):
        y = A.age.values if t == 'age' else A.sex.astype(int).values
        for k in ('lin', 'hgb'):
            for f in ('F65', 'F81'):
                Ps = [p for tt, kk, ff, s, p in res if (tt, kk, ff) == (t, k, f)]; row = {}
                for sub in ('ALL', 'MDD', 'HEALTHY', 'OTHER'):
                    idx = np.arange(len(y)) if sub == 'ALL' else np.where(grp == sub)[0]; st = [stats(t, y, p, idx) for p in Ps]; row[sub] = dict(n=len(idx), **{m: float(np.mean([s_[m] for s_ in st])) for m in st[0]})
                draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y))); key = 'R2' if t == 'age' else 'BA'
                bs = [np.mean([stats(t, y, p, i)[key] for p in Ps]) for i in draws]; row['ALL']['ci_' + key] = [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]; R[f'{t}_{k}_{f}'] = row
    key = lambda t: 'R2' if t == 'age' else 'BA'
    for t in ('age', 'sex'):
        thr = .30 if t == 'age' else .70
        for name, row in R.items():
            if name.startswith(t): row['usable_all_subgroups'] = bool(all(row[s][key(t)] >= thr for s in ('MDD', 'HEALTHY', 'OTHER')))
    rid = f'AGESEX_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}'); json.dump(R, open(f'runs/{rid}/analysis.json', 'w'), indent=1); json.dump(dict(protocol='AGE_SEX_PROTOCOL.md', n=len(A), code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], runtime_s=round(time.time() - t0), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'runs/{rid}/config.json', 'w'), indent=1)
    for name, row in R.items(): print(name, {s: {m: (round(v, 3) if isinstance(v, float) else v) for m, v in row[s].items() if m != 'ci_R2' and m != 'ci_BA'} for s in ('ALL', 'MDD', 'HEALTHY', 'OTHER')}, 'usable', row['usable_all_subgroups'])
    print(rid)
