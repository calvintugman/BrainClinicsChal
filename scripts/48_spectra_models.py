"""SPECTRA_PROTOCOL.md (A10): richer EEG-only age model + full-spectrum remission model + descriptive brain-age gap. Run from first_tests/."""
import os, sys, json, glob, time, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from common import SCALP
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score as auc, log_loss
warnings.filterwarnings('ignore')
c = importlib.import_module('37_clinical_eval'); ev = c.ev; OUT = 'outputs'; SEEDS = [42, 43, 44]; ALPHAS = [0.1, 1, 10, 100, 1e3, 1e4, 1e5]; GRID_K = [1e-4, 1e-3, 1e-2, 0.1, 1]
_, cols0, _ = c.build('Remitter'); E01 = cols0['K1E'][len(c.CLIN_K1):]
# ---- spectra -> SPEC / SPECAVG
z = np.load(f'{OUT}/spectra/spectra_PRIVATE.npz'); f = z['f']; SP = {}
for cond in ('EC', 'EO'):
    X = z[cond]; B = np.stack([np.log(np.maximum(X[:, :, (f >= k) & (f < k + 1)].mean(2), 1e-6)) for k in range(1, 30)], axis=2)   # (n, 26, 29); NaN rows stay NaN
    SP[cond] = B
SPEC_COLS = [f'S_{cd}_{ch}_{k}' for cd in ('EC', 'EO') for ch in SCALP for k in range(1, 30)]
S = pd.DataFrame(np.hstack([SP['EC'].reshape(len(SP['EC']), -1), SP['EO'].reshape(len(SP['EO']), -1)]), columns=SPEC_COLS); S['id'] = z['ids']
AVG_COLS = [f'A_{cd}_{k}' for cd in ('EC', 'EO') for k in range(1, 30)]; S[AVG_COLS] = np.hstack([np.nanmean(SP['EC'], 1), np.nanmean(SP['EO'], 1)])
def cohort(adults=True):
    wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'}); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses'); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv')
    ses = pd.concat([led[['id', 'selected_ses']].rename(columns={'selected_ses': 'ses'}), P[['id', 'ses']]]).assign(sessID=lambda d: d.ses.astype(str).str[-1].astype(int)).drop_duplicates('id')
    Fp = pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv'); Fo = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv'); A = pd.concat([Fp.assign(src='patient'), Fo.assign(src='pool')], ignore_index=True)
    A = A.merge(ses[['id', 'sessID']], on='id').merge(wb[['id', 'sessID', 'formal_status', 'age', 'gender']], on=['id', 'sessID'], how='left'); A = A[A.age.notna() & A.gender.notna()]
    if adults: A = A[A.age >= 18]
    A = A[~A.duplicated(subset=E01, keep='first')].merge(S, on='id', how='inner'); A = A[A[SPEC_COLS].notna().any(axis=1)].sort_values('id').reset_index(drop=True)
    A['grp'] = np.where(A.formal_status == 'MDD', 'MDD', np.where(A.formal_status == 'HEALTHY', 'HEALTHY', 'OTHER')); return A
MODELS = {'S0': E01, 'S1': SPEC_COLS, 'S2': SPEC_COLS + E01, 'S3': AVG_COLS}
def age_run(A, m, seed):
    X = A[MODELS[m]].values; y = A.age.values; p = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        g = GridSearchCV(make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge()), {'ridge__alpha': ALPHAS}, cv=KFold(5, shuffle=True, random_state=seed), scoring='neg_mean_squared_error').fit(X[tr], y[tr]); p[te] = g.predict(X[te])
    return m, seed, p
def st(y, p): return dict(R2=float(r2_score(y, p)), MAE=float(np.abs(y - p).mean()), r=float(np.corrcoef(y, p)[0, 1]))
def age_block(A, models, tag):
    res = Parallel(n_jobs=12)(delayed(age_run)(A, m, s) for m in models for s in SEEDS); P = {m: [p for mm, s, p in res if mm == m] for m in models}; y = A.age.values; out = {}
    for m in models:
        row = {'ALL': {k: float(np.mean([st(y, p)[k] for p in P[m]])) for k in ('R2', 'MAE', 'r')}}; row['ALL']['n'] = len(y)
        for gname in ('MDD', 'HEALTHY', 'OTHER'):
            ix = np.where(A.grp.values == gname)[0]
            if len(ix) > 10: row[gname] = dict(n=len(ix), **{k: float(np.mean([st(y[ix], p[ix])[k] for p in P[m]])) for k in ('R2', 'MAE', 'r')})
        out[m] = row
    draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y)))
    for a in [m for m in models if m != 'S0']:
        d = [np.mean([r2_score(y[i], pa[i]) - r2_score(y[i], p0[i]) for pa, p0 in zip(P[a], P['S0'])]) for i in draws]; out[f'{a}-S0'] = dict(diff=out[a]['ALL']['R2'] - out['S0']['ALL']['R2'], lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)))
    print(tag, json.dumps(out, indent=1), flush=True); return out, P
def run_k(T, cols, y, seed):
    n = len(y); p = np.zeros(n); dec = np.zeros(n, int); data = {'T': T}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
        oof = {cc: np.zeros(len(tr)) for cc in GRID_K}
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr]):
            pr = ev.Prep(cols, False, True, seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for cc in GRID_K: oof[cc][b] = LogisticRegression(C=cc, max_iter=3000).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {cc: log_loss(y[tr], oof[cc]) for cc in GRID_K}; cb = min(ll, key=ll.get); thr = ev.pick_threshold(oof[cb], y[tr])
        pr = ev.Prep(cols, False, True, seed).fit(data, tr); mdl = LogisticRegression(C=cb, max_iter=3000).fit(pr.transform(data, tr), y[tr]); p[te] = mdl.predict_proba(pr.transform(data, te))[:, 1]; dec[te] = (p[te] >= thr).astype(int)
    return p, dec
def perm_one(i, T, y):
    yp = np.random.default_rng(71000 + i).permutation(y); return float(np.mean([auc(yp, run_k(T, SPEC_COLS, yp, s)[0]) for s in SEEDS]))
if __name__ == '__main__':
    t0 = time.time(); R = {}; A = cohort(True); print('adults', len(A), A.grp.value_counts().to_dict(), 'missing SPEC frac', round(float(A[SPEC_COLS].isna().mean().mean()), 3), flush=True)
    R['age_adults'], Pad = age_block(A, ['S0', 'S1', 'S2', 'S3'], 'AGE adults')
    Aall = cohort(False); R['age_all_ages'], _ = age_block(Aall, ['S0', 'S1'], f'AGE all ages (n={len(Aall)}, min age {Aall.age.min():.1f})'); R['age_all_ages']['n'] = len(Aall)
    best = max(('S1', 'S2'), key=lambda m: R['age_adults'][m]['ALL']['R2']); b = R['age_adults']
    R['age_rule'] = dict(best=best, improved=bool(b[best]['ALL']['R2'] >= b['S0']['ALL']['R2'] + .05 and b[f'{best}-S0']['lo'] > 0 and all(b[best][g]['R2'] >= .30 for g in ('MDD', 'HEALTHY', 'OTHER'))))
    # secondary: full-spectrum remission model on the 164 patients
    K = {}
    for ep in ('Remitter', 'Responder'):
        T, _, y = c.build(ep); T = T.merge(S, on='id', how='left'); assert (T.id.values == c.build(ep)[0].id.values).all()
        res = Parallel(n_jobs=3)(delayed(run_k)(T, SPEC_COLS, y, s) for s in SEEDS); bm = pd.read_csv(sorted(glob.glob(f'runs/PIV_E01c_{ep}_*'))[-1] + '/predictions_PRIVATE.csv'); assert (bm.id.values == T.id.values).all()
        aucs = [float(auc(y, p)) for p, d in res]; e01c = [float(auc(y, bm[f'p_E01c_s{s}'].values)) for s in SEEDS]; M = [ev.metrics(y, p, d) for p, d in res]
        K[ep] = dict(per_repeat_AUC=aucs, mean_AUC=float(np.mean(aucs)), E01c_per_repeat=e01c, gain_vs_E01c=[a - e for a, e in zip(aucs, e01c)], mean_nPPV=float(np.nanmean([m['nPPV'] for m in M])), mean_sens=float(np.mean([m['sensitivity'] for m in M])))
        K[ep]['funding_rule_met'] = bool(K[ep]['mean_AUC'] >= .60 and np.mean(K[ep]['gain_vs_E01c']) >= .05 and all(g > 0 for g in K[ep]['gain_vs_E01c']))
        if ep == 'Remitter' and K[ep]['funding_rule_met']:
            null = np.array(Parallel(n_jobs=12)(delayed(perm_one)(i, T, y) for i in range(999))); K[ep]['permutation'] = dict(p_plus_one=float((1 + (null >= K[ep]['mean_AUC']).sum()) / 1000), null_mean=float(null.mean()))
        print('K-SPEC', ep, json.dumps(K[ep]), flush=True)
    R['K_SPEC'] = K
    # descriptive, NOT EEG-only: brain-age gap of patients from the S1 adult model (out-of-fold)
    pat = A.src.values == 'patient'; pred = np.mean(Pad['S1'], axis=0); gap = pred - A.age.values; G = pd.DataFrame({'id': A.id[pat], 'gap': gap[pat], 'age': A.age.values[pat]})
    led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv')[['id', 'Remitter', 'Responder']]; G = G.merge(led, on='id'); BG = {}
    for ep in ('Remitter', 'Responder'):
        yy = G[ep].astype(int).values; a = []
        for s in SEEDS:
            p = np.zeros(len(yy))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(yy)), yy): sc = StandardScaler().fit(G.gap.values[tr, None]); p[te] = LogisticRegression().fit(sc.transform(G.gap.values[tr, None]), yy[tr]).predict_proba(sc.transform(G.gap.values[te, None]))[:, 1]
            a.append(float(auc(yy, p)))
        BG[ep] = dict(n=len(yy), spearman_gap=float(spearmanr(G.gap, yy)[0]), cv_AUC_mean=float(np.mean(a)), mean_gap_years=float(G.gap.mean()))
    R['brain_age_gap_NOT_EEG_ONLY'] = BG; print('brain-age gap', json.dumps(BG), flush=True)
    rid = f'SPEC_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}'); json.dump(R, open(f'runs/{rid}/analysis.json', 'w'), indent=1)
    json.dump(dict(protocol='SPECTRA_PROTOCOL.md', protocol_fp=hashlib.sha256(open('SPECTRA_PROTOCOL.md', 'rb').read()).hexdigest()[:16], code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], runtime_s=round(time.time() - t0), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'runs/{rid}/config.json', 'w'), indent=1); print(rid, R['age_rule'])
