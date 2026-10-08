"""CLINICAL_PROTOCOL.md (A4): clinical-variable ceiling, EEG increment and E01c tracking check. Run from first_tests/: .venv/bin/python scripts/37_clinical_eval.py"""
import os, sys, json, time, hashlib, platform, glob, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from scipy.stats import spearmanr, kruskal
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score as auc
warnings.filterwarnings('ignore')
ev = importlib.import_module('10_batch2_eval'); OUT = 'outputs'; SEEDS = ev.SEEDS; MODELS = ['K0', 'K1', 'K1E']
CLIN_K0 = ['age', 'sex']; CLIN_K1 = ['age', 'sex', 'BDI_pre', 'prot1', 'prot2', 'prot3']

def build(endpoint):
    D, y, e01, E11, cov = ev.load_all(endpoint); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')[['id', 'selected_ses']]
    wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx').rename(columns={'TDBRAIN_ID': 'id'}); wb = wb[wb.Dataset == 'MDD-rTMS']
    led['sessID'] = led.selected_ses.str[-1].astype(int); W = led.merge(wb, on=['id', 'sessID'], how='left'); assert len(W) == len(led) == len(D) and W.age.notna().all() and W.gender.notna().all() and W.BDI_pre.notna().all(), 'clinical variables incomplete'
    assert W.id.is_unique; prot = pd.to_numeric(W['rTMS PROTOCOL'], errors='coerce')
    C = pd.DataFrame({'id': W.id, 'age': W.age.astype(float), 'sex': W.gender.astype(float), 'BDI_pre': W.BDI_pre.astype(float), 'prot_raw': prot, 'prot1': (prot == 1).astype(float), 'prot2': (prot == 2).astype(float), 'prot3': (prot == 3).astype(float)})
    T = D.merge(C, on='id', how='left'); assert len(T) == len(D) and (T.id.values == D.id.values).all() and T.age.notna().all()
    cols = {'K0': CLIN_K0, 'K1': CLIN_K1, 'K1E': CLIN_K1 + e01}; return T, cols, y

def run_lr(T, cols, y, seed):
    n = len(y); folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)); p = np.zeros(n); dec = np.zeros(n, int); meta = []; data = {'T': T}
    for k, (tr, te) in enumerate(folds):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr])); oof = {c: np.zeros(len(tr)) for c in ev.GRID_C}
        for a, b in inner:
            pr = ev.Prep(cols, False, True, seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for c in ev.GRID_C: oof[c][b] = ev.make_model('lr', c, seed).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {c: log_loss(y[tr], oof[c]) for c in ev.GRID_C}; cb = min(ll, key=ll.get); thr = ev.pick_threshold(oof[cb], y[tr])
        pr = ev.Prep(cols, False, True, seed).fit(data, tr); mdl = ev.make_model('lr', cb, seed).fit(pr.transform(data, tr), y[tr]); p[te] = mdl.predict_proba(pr.transform(data, te))[:, 1]; dec[te] = (p[te] >= thr).astype(int); meta.append(dict(fold=k, thr=float(thr), C=cb))
    return p, dec, meta

def task(m, s, endpoint):
    T, cols, y = build(endpoint); t0 = time.time(); p, d, meta = run_lr(T, cols[m], y, s); return m, s, p, d, meta, time.time() - t0
def perm_one(i, y, seed_rng):
    T, cols, _ = build('Remitter'); yp = np.random.default_rng(seed_rng + i).permutation(y); return float(np.mean([auc(yp, run_lr(T, cols['K1'], yp, s)[0]) for s in SEEDS]))

def describe(T, y, name):
    rows = []
    def add(var, grp, mask): rows.append(dict(endpoint=name, variable=var, group=grp, n=int(mask.sum()), n_pos=int(y[mask].sum()), rate=round(float(y[mask].mean()), 3) if mask.sum() else np.nan))
    for v, lab in ((0, 'female'), (1, 'male')): add('sex', lab, (T.sex == v).values)
    for v in (1, 2, 3): add('protocol', str(v), (T.prot_raw == v).values)
    add('protocol', 'missing', T.prot_raw.isna().values)
    for v in ('age', 'BDI_pre'):
        q = pd.qcut(T[v], 3, labels=False, duplicates='drop')
        for t in sorted(q.unique()): add(v, f'tertile{int(t) + 1} [{T[v][q == t].min():.0f}-{T[v][q == t].max():.0f}]', (q == t).values)
    return pd.DataFrame(rows)

if __name__ == '__main__':
    for endpoint in ('Remitter', 'Responder'):
        t0 = time.time(); T, cols, y = build(endpoint); print(endpoint, 'N', len(y), 'pos', int(y.sum()), 'features', {m: len(c) for m, c in cols.items()}, flush=True)
        res = Parallel(n_jobs=9)(delayed(task)(m, s, endpoint) for m in MODELS for s in SEEDS); rid = f'CLIN_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
        rows = [dict(model=m, seed=s, runtime_s=round(rt, 1), n_features=len(cols[m]), **ev.metrics(y, p, d)) for m, s, p, d, meta, rt in res]; M = pd.DataFrame(rows); M.to_csv(f'{rd}/metrics_by_seed.csv', index=False)
        pd.DataFrame({'id': T.id, 'y': y, **{f'p_{m}_s{s}': p for m, s, p, d, meta, rt in res}, **{f'dec_{m}_s{s}': d for m, s, p, d, meta, rt in res}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
        pd.DataFrame([dict(model=m, seed=s, **x) for m, s, p, d, meta, rt in res for x in meta]).to_csv(f'{rd}/fold_thresholds_C.csv', index=False)
        describe(T, y, endpoint).to_csv(f'{rd}/descriptive_by_group.csv', index=False)
        # benchmark + paired bootstrap of mean-AUC differences (same draws rule as earlier stages; conditional on saved predictions)
        bm = pd.read_csv(sorted(glob.glob(f'runs/PIV_E01c_{endpoint}_*'))[-1] + '/predictions_PRIVATE.csv'); assert (bm.id.values == T.id.values).all() and (bm.y.values == y).all()
        P = {m: {s: p for mm, s, p, d, meta, rt in res if mm == m} for m in MODELS}; P['E01c'] = {s: bm[f'p_E01c_s{s}'].values for s in SEEDS}
        draws = rng = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y))); draws = np.array([i for i in draws if len(np.unique(y[i])) > 1])
        A = {m: np.array([[auc(y[i], P[m][s][i]) for s in SEEDS] for i in draws]) for m in P}; mean_auc = {m: float(np.mean([auc(y, P[m][s]) for s in SEEDS])) for m in P}
        per_rep = {m: [float(auc(y, P[m][s])) for s in SEEDS] for m in P}; comps = {}
        for a, b in (('K1', 'E01c'), ('K1E', 'K1'), ('K1E', 'E01c'), ('K0', 'K1')):
            d = (A[a] - A[b]).mean(1); comps[f'{a}-{b}'] = dict(mean_diff=mean_auc[a] - mean_auc[b], per_repeat=[x - z for x, z in zip(per_rep[a], per_rep[b])], lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)))
        ana = dict(endpoint=endpoint, mean_auc=mean_auc, per_repeat_auc=per_rep, paired_bootstrap=comps, n_draws=len(draws))
        if endpoint == 'Remitter':
            obs = mean_auc['K1']; nperm = 999; null = np.array(Parallel(n_jobs=12)(delayed(perm_one)(i, y, 31000) for i in range(nperm))); ana['permutation_K1'] = dict(observed=obs, n_perm=nperm, rng_seed=31, null_mean=float(null.mean()), null_sd=float(null.std()), null_q95=float(np.quantile(null, .95)), p_plus_one=float((1 + (null >= obs).sum()) / (nperm + 1)), interpretation='chance association of K1 only; not a test of incremental value')
            pe = np.mean([P['E01c'][s] for s in SEEDS], axis=0); trk = {}
            for v in ('age', 'sex', 'BDI_pre'):
                x = T[v].values; r = spearmanr(pe, x)[0]; bs = [spearmanr(pe[i], x[i])[0] for i in np.random.default_rng(1).integers(0, len(y), size=(2000, len(y)))]; trk[v] = dict(rho=float(r), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
            ok = T.prot_raw.notna().values; g = [pe[ok & (T.prot_raw.values == k)] for k in (1, 2, 3)]; kw = kruskal(*g); trk['protocol'] = dict(H=float(kw.statistic), p=float(kw.pvalue), n=int(ok.sum()), group_means=[float(x.mean()) for x in g]); ana['E01c_tracking'] = trk
        json.dump(ana, open(f'{rd}/analysis.json', 'w'), indent=1)
        cfg = dict(run_id=rid, endpoint=endpoint, n=len(y), n_pos=int(y.sum()), protocol='CLINICAL_PROTOCOL.md', variables=CLIN_K1, seeds=SEEDS, grid_C=ev.GRID_C, parent='PIV_E01c_' + endpoint, code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], feature_fp=hashlib.sha256(open('outputs/v2/features_v2_PRIVATE.csv', 'rb').read()).hexdigest()[:16],
                   runtime_s=round(time.time() - t0), python=platform.python_version(), started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())); json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1)
        pd.set_option('display.width', 220); print(rid, 'runtime', cfg['runtime_s']); print(M[['model', 'seed', 'AUC', 'nPPV', 'sensitivity', 'PPV', 'Brier', 'n_pos_calls']].round(3).to_string(index=False)); print(json.dumps(ana, indent=1), flush=True)
        print(pd.read_csv(f'{rd}/descriptive_by_group.csv').to_string(index=False), flush=True)
