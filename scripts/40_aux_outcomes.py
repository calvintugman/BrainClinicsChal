"""A5 (descriptive only, see UPDATE in protocol): outcome models with EEG-estimated age/sex. Run from first_tests/."""
import os, sys, json, glob, time, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score as auc
warnings.filterwarnings('ignore'); c = importlib.import_module('37_clinical_eval'); ev = c.ev; OUT = 'outputs'; SEEDS = ev.SEEDS
T0, cols0, y0 = c.build('Remitter'); E01 = cols0['K1E'][len(c.CLIN_K1):]
wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'}); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv'); P['sessID'] = P.ses.astype(str).str[-1].astype(int); F = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv')
Pm = P.merge(wb[['id', 'sessID', 'age', 'gender']], on=['id', 'sessID'], how='left').merge(F[['id'] + E01], on='id', how='left'); assert not set(Pm.id) & set(T0.id)
Pm = Pm[Pm.age.notna() & Pm.gender.notna() & (Pm.age >= 18)]; Pm = Pm[Pm[E01].notna().any(axis=1)]; Pm = Pm[~Pm.duplicated(subset=E01, keep='first')].reset_index(drop=True)
tr, te = train_test_split(np.arange(len(Pm)), test_size=0.2, random_state=2027); pipe = lambda m: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), m)       # identical split/estimators as scripts/39
ga = GridSearchCV(pipe(Ridge()), {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=KFold(5, shuffle=True, random_state=0), scoring='neg_mean_squared_error').fit(Pm.loc[tr, E01].values, Pm.age.values[tr])
gs = GridSearchCV(pipe(LogisticRegression(max_iter=2000)), {'logisticregression__C': [0.01, 0.1, 1, 10]}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='neg_log_loss').fit(Pm.loc[tr, E01].values, Pm.gender.values[tr].astype(int))
def est(X): return ga.predict(X), gs.predict_proba(X)[:, 1]
T = T0.copy(); T['age_hat'], T['sex_hat'] = est(T[E01].values)
# M2: residualize each E01c feature on [age_hat, sex_hat]; regression fit on pool-training people only (their own estimates), NaNs ignored per feature
Ptr = Pm.loc[tr].copy(); Ptr['age_hat'], Ptr['sex_hat'] = est(Ptr[E01].values); Zp = Ptr[['age_hat', 'sex_hat']].values; Zt = T[['age_hat', 'sex_hat']].values
for col in E01:
    ok = Ptr[col].notna().values; lr = LinearRegression().fit(Zp[ok], Ptr[col].values[ok]); T['R_' + col] = T[col].values - lr.predict(Zt)
MODELS = {'M1': E01 + ['age_hat', 'sex_hat'], 'M2': ['R_' + x for x in E01], 'P2': ['age_hat', 'sex_hat']}
def task(m, s, ep):
    _, _, y = c.build(ep); p, d, meta = c.run_lr(T, MODELS[m], y, s); return m, s, p, d
if __name__ == '__main__':
    out = {}; t0 = time.time()
    for ep in ('Remitter', 'Responder'):
        _, _, y = c.build(ep); res = Parallel(n_jobs=9)(delayed(task)(m, s, ep) for m in MODELS for s in SEEDS); rid = f'AUX_{ep}_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}')
        bm = pd.read_csv(sorted(glob.glob(f'runs/PIV_E01c_{ep}_*'))[-1] + '/predictions_PRIVATE.csv'); assert (bm.id.values == T.id.values).all(); Pd = {m: {s: p for mm, s, p, d in res if mm == m} for m in MODELS}; Pd['E01c'] = {s: bm[f'p_E01c_s{s}'].values for s in SEEDS}
        M = pd.DataFrame([dict(model=m, seed=s, **ev.metrics(y, p, d)) for m, s, p, d in res]); M.to_csv(f'runs/{rid}/metrics_by_seed.csv', index=False)
        draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y))); draws = np.array([i for i in draws if len(np.unique(y[i])) > 1]); A = {m: np.array([[auc(y[i], Pd[m][s][i]) for s in SEEDS] for i in draws]) for m in Pd}; mean = {m: float(np.mean([auc(y, Pd[m][s]) for s in SEEDS])) for m in Pd}
        out[ep] = dict(mean_AUC=mean, per_repeat={m: [float(auc(y, Pd[m][s])) for s in SEEDS] for m in Pd}, vs_E01c={m: dict(diff=mean[m] - mean['E01c'], lo=float(np.quantile((A[m] - A['E01c']).mean(1), .025)), hi=float(np.quantile((A[m] - A['E01c']).mean(1), .975)), per_repeat=[float(auc(y, Pd[m][s]) - auc(y, Pd['E01c'][s])) for s in SEEDS]) for m in MODELS}, run_id=rid)
        json.dump(out[ep], open(f'runs/{rid}/analysis.json', 'w'), indent=1); json.dump(dict(protocol='EEG_ONLY_CLINICAL_ASSIST_PROTOCOL_DRAFT.md (UPDATE)', code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'runs/{rid}/config.json', 'w'), indent=1)
        print(ep, rid); print(M.groupby('model')[['AUC', 'nPPV', 'sensitivity']].mean().round(3).to_string()); print(json.dumps(out[ep]['vs_E01c'], indent=0))
