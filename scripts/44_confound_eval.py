"""CONFOUND_PROTOCOL.md (A7). Run from first_tests/: .venv/bin/python scripts/44_confound_eval.py"""
import os, sys, json, time, hashlib, platform, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score as auc, balanced_accuracy_score as bacc
warnings.filterwarnings('ignore')
d6 = importlib.import_module('43_diag_eval'); c = d6.c; ev = d6.ev; OUT = 'outputs'; SEEDS = ev.SEEDS; run = d6.run; met = d6.met
CLINIC = ['ADHD', 'BURNOUT', 'CHRONIC PAIN', 'INSOMNIA', 'PARKINSON', 'TINNITUS']
def load_all():
    _, cols, _ = c.build('Remitter'); E01 = cols['K1E'][len(c.CLIN_K1):]
    wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'}); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses'); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv')
    ses = pd.concat([led[['id', 'selected_ses']].rename(columns={'selected_ses': 'ses'}), P[['id', 'ses']]]).assign(sessID=lambda d: d.ses.astype(str).str[-1].astype(int)).drop_duplicates('id')
    Fp = pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv'); Fo = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv'); A = pd.concat([Fp.assign(src='patient'), Fo.assign(src='pool')], ignore_index=True)
    A = A.merge(ses[['id', 'sessID']], on='id').merge(wb[['id', 'sessID', 'formal_status', 'age', 'gender']], on=['id', 'sessID'], how='left')
    A['num'] = A.id.str.replace('sub-', '').astype(float); A['sex'] = A.gender.astype(float); A['age'] = A.age.astype(float); A = A[A.age.notna() & A.sex.notna() & (A.age >= 18)]
    A = A[~A.duplicated(subset=E01, keep='first')].sort_values('id').reset_index(drop=True); return A, E01
def task(name, s, T, cols, y): p, d, meta = run(T, cols, y, s); return name, s, p, d
def perm_one(i, T, cols, y):
    yp = np.random.default_rng(61000 + i).permutation(y); return float(np.mean([bacc(yp, run(T, cols, yp, s)[1]) for s in SEEDS]))
def evaluate(tag, T, y, E01, models, nperm=0, extra=None):
    T = T.reset_index(drop=True); cols = {'EEG': E01, 'IDONLY': ['num'], 'DEMO': ['age', 'sex'], 'EEG+ID': E01 + ['num']}; cols = {m: cols[m] for m in models}
    res = Parallel(n_jobs=12)(delayed(task)(m, s, T, cols[m], y) for m in models for s in SEEDS); P = {m: {s: (p, d) for mm, s, p, d in res if mm == m} for m in models}
    draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y))); draws = np.array([i for i in draws if len(np.unique(y[i])) > 1])
    out = dict(task=tag, n=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum()), models={})
    for m in models:
        ba = np.array([[bacc(y[i], P[m][s][1][i]) for s in SEEDS] for i in draws]).mean(1); au = np.array([[auc(y[i], P[m][s][0][i]) for s in SEEDS] for i in draws]).mean(1)
        out['models'][m] = dict(mean_BA=float(np.mean([bacc(y, P[m][s][1]) for s in SEEDS])), BA_ci=[float(np.quantile(ba, .025)), float(np.quantile(ba, .975))], mean_AUC=float(np.mean([auc(y, P[m][s][0]) for s in SEEDS])), AUC_ci=[float(np.quantile(au, .025)), float(np.quantile(au, .975))],
                                sens=float(np.mean([met(y, *P[m][s])['sensitivity'] for s in SEEDS])), spec=float(np.mean([met(y, *P[m][s])['specificity'] for s in SEEDS])))
    if nperm:
        obs = out['models']['EEG']['mean_BA']; null = np.array(Parallel(n_jobs=12)(delayed(perm_one)(i, T, E01, y) for i in range(nperm))); out['permutation_EEG'] = dict(observed=obs, n_perm=nperm, rng_seed=61, null_mean=float(null.mean()), null_q95=float(np.quantile(null, .95)), p_plus_one=float((1 + (null >= obs).sum()) / (nperm + 1)))
    if extra: out.update(extra(P, y))
    print(json.dumps(out, indent=1), flush=True); return out, P
if __name__ == '__main__':
    t0 = time.time(); A, E01 = load_all(); R = {}; print('adults with features', len(A), A.formal_status.value_counts().to_dict(), flush=True)
    # T-A
    W = A[A.formal_status.isin(['MDD', 'HEALTHY']) & A.num.between(87_990_000, 88_080_000)].reset_index(drop=True); yA = (W.formal_status == 'MDD').astype(int).values
    prev = pd.read_csv(sorted(__import__('glob').glob('runs/DIAG_*'))[-1] + '/predictions_PRIVATE.csv').set_index('id')
    def extra_A(P, y):
        ids = W.id.values; m = prev.index.isin(ids); yy = prev.loc[m, 'y'].values; return dict(conditional_A6_predictions_in_window=dict(n=int(m.sum()), n_healthy=int((1 - yy).sum()), mean_BA=float(np.mean([bacc(yy, prev.loc[m, f'dec_D1_s{s}'].values) for s in SEEDS])), mean_AUC=float(np.mean([auc(yy, prev.loc[m, f'p_D1_s{s}'].values) for s in SEEDS]))))
    R['T-A'], _ = evaluate('T-A ID-window MDD vs HEALTHY', W, yA, E01, ['EEG', 'IDONLY', 'DEMO', 'EEG+ID'], nperm=999, extra=extra_A)
    # T-B
    H = A[A.formal_status == 'HEALTHY'].reset_index(drop=True); yB = (H.num > 87_973_000).astype(int).values; R['T-B'], _ = evaluate('T-B healthy block A (ID<=87973000) vs block B', H, yB, E01, ['EEG', 'DEMO'])
    # T-C, T-D
    Cn = A[A.formal_status.isin(CLINIC)]; M = A[A.formal_status == 'MDD']; Hh = A[A.formal_status == 'HEALTHY']
    TC = pd.concat([M, Cn]).reset_index(drop=True); yC = (TC.formal_status == 'MDD').astype(int).values; R['T-C'], _ = evaluate('T-C MDD vs clinic non-MDD', TC, yC, E01, ['EEG', 'IDONLY', 'DEMO'], nperm=999)
    TD = pd.concat([Hh, Cn]).reset_index(drop=True); yD = (TD.formal_status == 'HEALTHY').astype(int).values; R['T-D'], _ = evaluate('T-D HEALTHY vs clinic non-MDD', TD, yD, E01, ['EEG', 'IDONLY', 'DEMO'])
    # T-E
    Pt = A[(A.src == 'patient')].reset_index(drop=True); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')[['id', 'Remitter', 'Responder']]; Pt = Pt.merge(led, on='id'); TE = {}
    for ep in ('Remitter', 'Responder'):
        y = Pt[ep].astype(int).values; fo = [auc(y, Pt.num.values)]; rho = spearmanr(Pt.num.values, y)[0]; TE[ep] = dict(n=len(y), AUC_raw_ID=float(auc(y, Pt.num.values)), spearman=float(rho))
    R['T-E'] = TE; print(json.dumps(TE, indent=1))
    rid = f'CONF_{time.strftime("%Y%m%dT%H%M%S")}'; os.makedirs(f'runs/{rid}'); json.dump(R, open(f'runs/{rid}/analysis.json', 'w'), indent=1)
    m = lambda t, k: R[t]['models'][k]['mean_BA']
    A_ = R['T-A']; surv = bool(m('T-A', 'EEG') >= .65 and A_['models']['EEG']['BA_ci'][0] > .55 and A_['permutation_EEG']['p_plus_one'] < .05 and m('T-A', 'EEG') - m('T-A', 'IDONLY') >= .05 and m('T-C', 'EEG') >= .62 and R['T-C']['permutation_EEG']['p_plus_one'] < .05)
    batch = bool(m('T-B', 'EEG') >= .70 and not (m('T-A', 'EEG') >= .65 and A_['models']['EEG']['BA_ci'][0] > .55 and A_['permutation_EEG']['p_plus_one'] < .05))
    rules = dict(diagnostic_survives_confound_control=surv, batch_driven=batch, inconclusive=bool(not surv and not batch)); json.dump(dict(rules=rules, code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], protocol_fp=hashlib.sha256(open('CONFOUND_PROTOCOL.md', 'rb').read()).hexdigest()[:16], runtime_s=round(time.time() - t0), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'runs/{rid}/config.json', 'w'), indent=1); print(rid, rules)
