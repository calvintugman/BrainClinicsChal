"""A5 sections 1-2: EEG->age, EEG->sex (pool-trained) and EEG->BDI_pre (cross-fitted in patients) gates G1-G3. Outcome-blind: no Remitter/Responder label is used except to build the saved outer splits. Run from first_tests/."""
import os, sys, json, time, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import r2_score, roc_auc_score
warnings.filterwarnings('ignore')
c = importlib.import_module('37_clinical_eval'); OUT = 'outputs'; os.makedirs(f'{OUT}/aux', exist_ok=True)
T, cols, y = c.build('Remitter'); E01 = cols['K1E'][len(c.CLIN_K1):]; assert len(E01) == 65
wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'})
P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv'); P['sessID'] = P.ses.astype(str).str[-1].astype(int); F = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv')
Pm = P.merge(wb[['id', 'sessID', 'age', 'gender']], on=['id', 'sessID'], how='left').merge(F[['id'] + E01], on='id', how='left')
assert not set(Pm.id) & set(T.id), 'pool overlaps patients'
Pm = Pm[Pm.age.notna() & Pm.gender.notna() & (Pm.age >= 18)]; Pm = Pm[Pm[E01].notna().any(axis=1)]; Pm = Pm[~Pm.duplicated(subset=E01, keep='first')].reset_index(drop=True)
tr, te = train_test_split(np.arange(len(Pm)), test_size=0.2, random_state=2027); print('pool adults', len(Pm), 'train', len(tr), 'eval', len(te), 'patients', len(T), flush=True)
cv = KFold(5, shuffle=True, random_state=0)
pipe = lambda m: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), m)
res = dict(n_pool_adults=len(Pm), n_pool_train=len(tr), n_pool_eval=len(te), missing_frac_pool=float(Pm[E01].isna().mean().mean()), missing_frac_patients=float(T[E01].isna().mean().mean()), pool_age_range=[float(Pm.age.min()), float(Pm.age.max())])
# E-age
ga = GridSearchCV(pipe(Ridge()), {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=cv, scoring='neg_mean_squared_error').fit(Pm.loc[tr, E01].values, Pm.age.values[tr])
pe = ga.predict(Pm.loc[te, E01].values); pp = ga.predict(T[E01].values)
res['G1'] = dict(alpha=ga.best_params_['ridge__alpha'], heldout_pool_R2=float(r2_score(Pm.age.values[te], pe)), heldout_pool_MAE=float(np.abs(pe - Pm.age.values[te]).mean()), patients_spearman=float(spearmanr(pp, T.age)[0]), patients_MAE=float(np.abs(pp - T.age).mean()), patients_mean_age_hat=float(pp.mean()), patients_mean_age=float(T.age.mean()))
res['G1']['pass'] = bool(res['G1']['heldout_pool_R2'] >= 0.25 and res['G1']['patients_spearman'] >= 0.50)
# E-sex
gs = GridSearchCV(pipe(LogisticRegression(max_iter=2000)), {'logisticregression__C': [0.01, 0.1, 1, 10]}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='neg_log_loss').fit(Pm.loc[tr, E01].values, Pm.gender.values[tr].astype(int))
res['G2'] = dict(C=gs.best_params_['logisticregression__C'], heldout_pool_AUC=float(roc_auc_score(Pm.gender.values[te].astype(int), gs.predict_proba(Pm.loc[te, E01].values)[:, 1])), patients_AUC=float(roc_auc_score(T.sex.astype(int), gs.predict_proba(T[E01].values)[:, 1])))
res['G2']['pass'] = bool(res['G2']['heldout_pool_AUC'] >= 0.70 and res['G2']['patients_AUC'] >= 0.65)
# E-BDI cross-fitted over saved outer splits (outer test patients predicted by a model fit on outer-train only)
bdi = {}; X = T[E01].values; b = T.BDI_pre.values
for seed in (42, 43, 44):
    oof = np.zeros(len(b))
    for a, t in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        g = GridSearchCV(pipe(Ridge()), {'ridge__alpha': [0.01, 0.1, 1, 10, 100, 1000]}, cv=KFold(5, shuffle=True, random_state=seed), scoring='neg_mean_squared_error').fit(X[a], b[a]); oof[t] = g.predict(X[t])
    r = spearmanr(oof, b)[0]; draws = np.random.default_rng(1).integers(0, len(b), size=(2000, len(b))); bs = [spearmanr(oof[i], b[i])[0] for i in draws]
    bdi[seed] = dict(spearman=float(r), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), R2=float(r2_score(b, oof)))
res['G3'] = dict(by_seed=bdi, seed42_spearman=bdi[42]['spearman'], seed42_lower=bdi[42]['lo']); res['G3']['pass'] = bool(bdi[42]['spearman'] >= 0.30 and bdi[42]['lo'] > 0.10)
res['code_fp'] = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]; res['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); json.dump(res, open(f'{OUT}/aux/gates.json', 'w'), indent=1); print(json.dumps(res, indent=1))
