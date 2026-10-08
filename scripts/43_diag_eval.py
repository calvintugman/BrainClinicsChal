"""DIAGNOSTIC_PROTOCOL.md (A6): MDD vs HEALTHY, EEG-only D1 vs age+sex D0. Run from first_tests/: .venv/bin/python scripts/43_diag_eval.py"""
import os, sys, json, time, hashlib, platform, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score as auc, balanced_accuracy_score as bacc
warnings.filterwarnings('ignore')
c = importlib.import_module('37_clinical_eval'); ev = c.ev; OUT = 'outputs'; SEEDS = ev.SEEDS
def load():
    _, cols, _ = c.build('Remitter'); E01 = cols['K1E'][len(c.CLIN_K1):]
    wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'}); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses'); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv')
    ses = pd.concat([led[['id', 'selected_ses']].rename(columns={'selected_ses': 'ses'}), P[['id', 'ses']]]).assign(sessID=lambda d: d.ses.astype(str).str[-1].astype(int)).drop_duplicates('id')
    Fp = pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv'); Fo = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv'); A = pd.concat([Fp.assign(src='patient'), Fo.assign(src='pool')], ignore_index=True)
    A = A.merge(ses[['id', 'sessID']], on='id').merge(wb[['id', 'sessID', 'formal_status', 'age', 'gender']], on=['id', 'sessID'], how='left')
    A = A[A.formal_status.isin(['MDD', 'HEALTHY'])].copy(); A['y'] = (A.formal_status == 'MDD').astype(int); A = A[~A.duplicated(subset=E01, keep='first')].sort_values('id').reset_index(drop=True)
    assert A.id.is_unique and A.age.notna().all() and A.gender.notna().all(); A['sex'] = A.gender.astype(float); A['age'] = A.age.astype(float); return A, E01
def pick_ba(p, yy):
    best = None
    for t in np.unique(p):
        d = p >= t; ba = bacc(yy, d.astype(int))
        if best is None or ba > best[1] + 1e-12 or (abs(ba - best[1]) <= 1e-12 and abs(t - .5) < abs(best[0] - .5)): best = (t, ba)
    return best[0]
def run(T, cols, y, seed, mask=None):
    n = len(y); mask = np.ones(n, bool) if mask is None else mask; p = np.full(n, np.nan); dec = np.full(n, -1); data = {'T': T}; meta = []
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)):
        tr, te = tr[mask[tr]], te[mask[te]]; oof = {cc: np.zeros(len(tr)) for cc in ev.GRID_C}
        for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr]):
            pr = ev.Prep(cols, False, True, seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for cc in ev.GRID_C: oof[cc][b] = ev.make_model('lr', cc, seed).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {cc: log_loss(y[tr], oof[cc]) for cc in ev.GRID_C}; cb = min(ll, key=ll.get); thr = pick_ba(oof[cb], y[tr])
        pr = ev.Prep(cols, False, True, seed).fit(data, tr); m = ev.make_model('lr', cb, seed).fit(pr.transform(data, tr), y[tr]); p[te] = m.predict_proba(pr.transform(data, te))[:, 1]; dec[te] = (p[te] >= thr).astype(int); meta.append(dict(fold=k, thr=float(thr), C=cb))
    return p, dec, meta
def task(name, s, T, cols, y, mask=None): p, d, meta = run(T, cols, y, s, mask); return name, s, p, d, meta
def perm_one(i, T, cols, y):
    yp = np.random.default_rng(51000 + i).permutation(y); return float(np.mean([bacc(yp, run(T, cols, yp, s)[1]) for s in SEEDS]))
def met(yy, p, d):
    m = ev.metrics(yy, p, d); return dict(BA=bacc(yy, d), AUC=m['AUC'], sensitivity=m['sensitivity'], specificity=(m['TN'] / (m['TN'] + m['FP'])), PPV=m['PPV'], Brier=m['Brier'], TP=m['TP'], FP=m['FP'], TN=m['TN'], FN=m['FN'], N=m['N'], prevalence=m['prevalence'])
def matched(T, y):
    order = sorted(range(len(T)), key=lambda i: hashlib.sha256(T.id[i].encode()).hexdigest()); used = set(); pairs = []
    for i in [i for i in order if y[i] == 1]:
        cand = [j for j in order if y[j] == 0 and j not in used and T.sex[j] == T.sex[i] and abs(T.age[j] - T.age[i]) <= 5]
        if cand: j = min(cand, key=lambda j: abs(T.age[j] - T.age[i])); used.add(j); pairs.append((i, j))
    return np.array([k for pr in pairs for k in pr]), len(pairs)
if __name__ == '__main__':
    t0 = time.time(); T, E01 = load(); y = T.y.values; cols = {'D0': ['age', 'sex'], 'D1': E01}
    miss = T.groupby('src')[E01].apply(lambda d: float(d.isna().mean().mean())).to_dict(); comp = T[E01].notna().all(axis=1).values
    print('N', len(y), 'MDD', int(y.sum()), 'HEALTHY', int((1 - y).sum()), 'src', T.groupby(['formal_status', 'src']).size().to_dict(), 'missing frac', miss, 'complete', int(comp.sum()), flush=True)
    res = Parallel(n_jobs=6)(delayed(task)(m, s, T, cols[m], y) for m in ('D0', 'D1') for s in SEEDS); cc_res = Parallel(n_jobs=3)(delayed(task)('D1cc', s, T, E01, y, comp) for s in SEEDS)
    rid = f'DIAG_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
    rows = [dict(model=m, seed=s, **met(y, p, d)) for m, s, p, d, meta in res] + [dict(model='D1cc', seed=s, **met(y[comp], p[comp], d[comp])) for m, s, p, d, meta in cc_res]; M = pd.DataFrame(rows); M.to_csv(f'{rd}/metrics_by_seed.csv', index=False)
    P = {m: {s: (p, d) for mm, s, p, d, meta in res if mm == m} for m in ('D0', 'D1')}
    pd.DataFrame({'id': T.id, 'y': y, 'src': T.src, **{f'p_{m}_s{s}': P[m][s][0] for m in P for s in SEEDS}, **{f'dec_{m}_s{s}': P[m][s][1] for m in P for s in SEEDS}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
    draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y))); draws = np.array([i for i in draws if len(np.unique(y[i])) > 1])
    BA = {m: np.array([[bacc(y[i], P[m][s][1][i]) for s in SEEDS] for i in draws]) for m in P}; AU = {m: np.array([[auc(y[i], P[m][s][0][i]) for s in SEEDS] for i in draws]) for m in P}
    mean_ba = {m: float(np.mean([bacc(y, P[m][s][1]) for s in SEEDS])) for m in P}; mean_auc = {m: float(np.mean([auc(y, P[m][s][0]) for s in SEEDS])) for m in P}
    ana = dict(n=len(y), n_mdd=int(y.sum()), n_complete=int(comp.sum()), missing_frac_by_source=miss, mean_BA=mean_ba, mean_AUC=mean_auc, per_repeat_BA={m: [float(bacc(y, P[m][s][1])) for s in SEEDS] for m in P},
               BA_interval={m: [float(np.quantile(BA[m].mean(1), .025)), float(np.quantile(BA[m].mean(1), .975))] for m in P}, AUC_interval={m: [float(np.quantile(AU[m].mean(1), .025)), float(np.quantile(AU[m].mean(1), .975))] for m in P},
               D1_minus_D0_BA=dict(mean=mean_ba['D1'] - mean_ba['D0'], per_repeat=[float(bacc(y, P['D1'][s][1]) - bacc(y, P['D0'][s][1])) for s in SEEDS], lo=float(np.quantile((BA['D1'] - BA['D0']).mean(1), .025)), hi=float(np.quantile((BA['D1'] - BA['D0']).mean(1), .975))),
               complete_case_D1_mean_BA=float(np.mean([bacc(y[comp], d[comp]) for m, s, p, d, meta in cc_res])))
    pe = np.mean([P['D1'][s][0] for s in SEEDS], axis=0); ana['D1_score_vs_age_within_class'] = {k: float(spearmanr(pe[y == v], T.age.values[y == v])[0]) for k, v in (('HEALTHY', 0), ('MDD', 1))}
    idx, npairs = matched(T, y); ana['matched_subset'] = dict(n_pairs=npairs, mean_BA_D1=float(np.mean([bacc(y[idx], P['D1'][s][1][idx]) for s in SEEDS])), mean_BA_D0=float(np.mean([bacc(y[idx], P['D0'][s][1][idx]) for s in SEEDS])), mean_AUC_D1=float(np.mean([auc(y[idx], P['D1'][s][0][idx]) for s in SEEDS])), age_mdd=float(T.age.values[idx][y[idx] == 1].mean()), age_healthy=float(T.age.values[idx][y[idx] == 0].mean()))
    nperm = 999; null = np.array(Parallel(n_jobs=12)(delayed(perm_one)(i, T, E01, y) for i in range(nperm))); obs = mean_ba['D1']
    ana['permutation_D1'] = dict(observed=obs, n_perm=nperm, rng_seed=51, null_mean=float(null.mean()), null_sd=float(null.std()), null_q95=float(np.quantile(null, .95)), p_plus_one=float((1 + (null >= obs).sum()) / (nperm + 1)))
    b = ana; b['rules'] = dict(signal_beyond_demographics=bool(mean_ba['D1'] >= .70 and b['permutation_D1']['p_plus_one'] < .05 and b['D1_minus_D0_BA']['mean'] >= .05 and b['D1_minus_D0_BA']['lo'] > 0), leaderboard_scale=bool(mean_ba['D1'] >= .62 and b['permutation_D1']['p_plus_one'] < .05), close_arm=bool(mean_ba['D1'] < .62 or abs(mean_ba['D1'] - mean_ba['D0']) <= .05))
    json.dump(ana, open(f'{rd}/analysis.json', 'w'), indent=1); json.dump(dict(run_id=rid, protocol='DIAGNOSTIC_PROTOCOL.md', protocol_fp=hashlib.sha256(open('DIAGNOSTIC_PROTOCOL.md', 'rb').read()).hexdigest()[:16], code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], seeds=SEEDS, runtime_s=round(time.time() - t0), python=platform.python_version(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), open(f'{rd}/config.json', 'w'), indent=1)
    pd.set_option('display.width', 200); print(rid); print(M[['model', 'seed', 'BA', 'AUC', 'sensitivity', 'specificity', 'N']].round(3).to_string(index=False)); print(json.dumps(ana, indent=1))
