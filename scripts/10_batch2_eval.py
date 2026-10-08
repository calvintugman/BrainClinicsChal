"""Batch-2 evaluation (BATCH2_PROTOCOL.md). Usage: python scripts/10_batch2_eval.py Remitter|Responder [--check]
All learned transforms are fit only on indices passed to Prep.fit (training indices of the current inner/outer split)."""
import sys, os, json, time, hashlib, platform, glob, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from pyriemann.tangentspace import TangentSpace
from b2feat import build, trace_norm, E13_NAMES
warnings.filterwarnings('ignore')
OUT = 'outputs'; SEEDS = [42, 43, 44]; GRID_C = [0.01, 0.1, 1, 10]; GRID_H = [(l, r) for l in (3, 7) for r in (1, 10)]
ROIS = ['frontal', 'central', 'posterior']; BANDS4 = ['delta', 'theta', 'alpha', 'beta']

def load_all(endpoint):
    F = pd.read_csv(os.environ.get('FEATURES_PATH', f'{OUT}/features_PRIVATE.csv')); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')
    D = F.merge(led[['id', 'Responder', 'Remitter']], on='id').sort_values('id').reset_index(drop=True); y = D[endpoint].astype(int).values
    psd_cols = [c for c in F.columns if (c.startswith(('EC_', 'EO_')) and c.rsplit('_', 1)[-1] in ('logabs', 'rel', 'IAF')) or c.startswith('ECEO_')]
    extra = [c for c in F.columns if c.endswith(('_FAA', '_entropy'))] + [c for c in F.columns if c.startswith('conn_') and not c.endswith(('_h1', '_h2'))]
    e01 = psd_cols + extra; assert len(e01) == 65
    z = np.load(f'{OUT}/batch2_cov_PRIVATE.npz'); order = pd.Series(range(len(z['ids'])), index=z['ids']).reindex(D.id).values; cov = z['cov_full'][order]
    sp = pd.read_csv(f'{OUT}/batch2_specparam_fits_PRIVATE.csv'); sp = sp[~sp.key.str.contains('_h')]; E11 = []
    for cond in ('EC', 'EO'):
        for r in ROIS:
            s = sp[sp.key == f'{cond}_{r}'].set_index('id').reindex(D.id)
            for nm, col in (('exp', 'exp'), ('off', 'off'), ('cf', 'cf'), ('pw', 'pw')):
                v = s[col].where(s.ok.fillna(False).astype(bool)).values; D[f'E11_{cond}_{r}_{nm}'] = v
                E11.append(f'E11_{cond}_{r}_{nm}')
    return D, y, e01, E11, cov

def make_data(D, cov, e01, E11, gain=1.0):
    """table for a given common gain; returns dict of arrays. Power-type features shift, others invariant (verified in checks)."""
    T = D.copy()
    if gain != 1.0:
        for c in T.columns:
            if c.endswith('_logabs'): T[c] = T[c] + 2 * np.log(gain)
        for c in E11:
            if c.endswith('_off'): T[c] = T[c] + 2 * np.log10(gain)
        for c in [c for c in T.columns if c.startswith('ECEO_')]: pass                       # EC - EO contrast: gain cancels
    pw = []
    for r in ROIS:
        for b in BANDS4: T[f'E13P_{r}_{b}'] = T[f'EO_{r}_{b}_logabs'] - T[f'EC_{r}_{b}_logabs']; pw.append(f'E13P_{r}_{b}')
    c_, E13 = build(cov, gain)
    for j, nm in enumerate(E13_NAMES): T[nm] = E13[:, j]
    COVN = trace_norm(c_) if True else None
    return dict(T=T, COVN=COVN)

SPEC = {  # name: (table cols builder, uses_cov_block, model)
    'E10': ('e01', False, 'hgb'), 'E11-only': ('e11', False, 'lr'), 'E11-add': ('e01+e11', False, 'lr'), 'E12-add': ('e01', True, 'lr'),
    'E13-power': ('e13p', False, 'lr'), 'E13-relative': ('e13', False, 'lr'), 'E13-add': ('e01+e13', False, 'lr'), 'E01_ref': ('e01', False, 'lr')}

class Prep:
    def __init__(self, cols, use_cov, linear, seed): self.cols, self.use_cov, self.linear, self.seed = cols, use_cov, linear, seed
    def fit(self, data, tr):
        X = data['T'][self.cols].values[tr]; self.imp = SimpleImputer(strategy='median').fit(X); Xi = self.imp.transform(X)
        if self.use_cov:
            self.ts = []; C = data['COVN'][tr]
            for c in range(2):
                for b in range(3):
                    ok = ~np.isnan(C[:, c, b]).any(axis=(1, 2)); t = TangentSpace(metric='riemann', tsupdate=False).fit(C[ok, c, b]); self.ts.append(t)
                    assert np.abs(t.transform(t.reference_[None])).max() < 1e-6     # reference maps to zero vector (used for missing)
            Z = self._tangent(data, tr); self.zs = StandardScaler().fit(Z); self.pca = PCA(n_components=20, svd_solver='full', random_state=0).fit(self.zs.transform(Z)); Xi = np.hstack([Xi, self.pca.transform(self.zs.transform(Z))])
        self.sc = StandardScaler().fit(Xi) if self.linear else None
        return self
    def _tangent(self, data, idx):
        C = data['COVN'][idx]; out = []
        for k, (c, b) in enumerate([(c, b) for c in range(2) for b in range(3)]):
            ok = ~np.isnan(C[:, c, b]).any(axis=(1, 2)); Zk = np.zeros((len(idx), self.ts[k].transform(self.ts[k].reference_[None]).shape[1]))
            if ok.any(): Zk[ok] = self.ts[k].transform(C[ok, c, b])
            out.append(Zk)
        return np.hstack(out)
    def transform(self, data, idx):
        X = self.imp.transform(data['T'][self.cols].values[idx])
        if self.use_cov: X = np.hstack([X, self.pca.transform(self.zs.transform(self._tangent(data, idx)))])
        return self.sc.transform(X) if self.linear else X

def make_model(kind, cand, seed):
    if kind == 'lr': return LogisticRegression(C=cand, max_iter=2000)
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, early_stopping=False, min_samples_leaf=20, max_leaf_nodes=cand[0], l2_regularization=cand[1], random_state=seed)
def pick_threshold(p, yy):
    best = None
    for t in np.unique(p):
        d = p >= t; tp = (d & (yy == 1)).sum(); sens = tp / (yy == 1).sum(); ppv = tp / d.sum() if d.sum() else np.nan
        if sens >= 0.5 and (best is None or (ppv, sens) > (best[1], best[2])): best = (t, ppv, sens)
    return best[0] if best else p.min()
def metrics(yy, p, dec):
    tp = int(((dec == 1) & (yy == 1)).sum()); fp = int(((dec == 1) & (yy == 0)).sum()); tn = int(((dec == 0) & (yy == 0)).sum()); fn = int(((dec == 0) & (yy == 1)).sum())
    prev = yy.mean(); sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan; ppv = tp / (tp + fp) if tp + fp else np.nan
    return dict(AUC=roc_auc_score(yy, p), PPV=ppv, sensitivity=sens, nPPV=100 * ppv / prev if ppv == ppv else np.nan, Brier=brier_score_loss(yy, p), balanced_accuracy=(sens + spec) / 2, n_pos_calls=tp + fp, TP=tp, FP=fp, TN=tn, FN=fn, N=len(yy), prevalence=prev)

def run_task(name, seed, endpoint, gains=()):
    t0 = time.time(); D, y, e01, E11, cov = load_all(endpoint); data = make_data(D, cov, e01, E11)
    colspec, use_cov, kind = SPEC[name]; e13 = E13_NAMES; e13p = [f'E13P_{r}_{b}' for r in ROIS for b in BANDS4]
    cols = {'e01': e01, 'e11': E11, 'e01+e11': e01 + E11, 'e13p': e13p, 'e13': e13, 'e01+e13': e01 + e13}[colspec]; cands = GRID_C if kind == 'lr' else GRID_H
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)); p = np.zeros(len(y)); dec = np.zeros(len(y), int); meta = []; gp = {g: np.zeros(len(y)) for g in gains}; gd = {g: np.zeros(len(y), int) for g in gains}
    for k, (tr, te) in enumerate(folds):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr])); oof = {c: np.zeros(len(tr)) for c in cands}
        for a, b in inner:
            pr = Prep(cols, use_cov, kind == 'lr', seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for c in cands: oof[c][b] = make_model(kind, c, seed).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {c: log_loss(y[tr], oof[c]) for c in cands}; cb = min(ll, key=ll.get); thr = pick_threshold(oof[cb], y[tr])
        pr = Prep(cols, use_cov, kind == 'lr', seed).fit(data, tr); mdl = make_model(kind, cb, seed).fit(pr.transform(data, tr), y[tr]); p[te] = mdl.predict_proba(pr.transform(data, te))[:, 1]; dec[te] = (p[te] >= thr).astype(int)
        for g in gains:
            dg = make_data(D, cov, e01, E11, gain=g); gp[g][te] = mdl.predict_proba(pr.transform(dg, te))[:, 1]; gd[g][te] = (gp[g][te] >= thr).astype(int)
        meta.append(dict(fold=k, thr=float(thr), cand=str(cb), n_test=len(te), n_feat_in=len(cols) + (20 if use_cov else 0)))
    return dict(name=name, seed=seed, p=p, dec=dec, meta=meta, gp=gp, gd=gd, runtime=time.time() - t0, n_features=len(cols) + (20 if use_cov else 0))

if __name__ == '__main__':
    endpoint = sys.argv[1]; names = ['E01_ref'] + [n for n in SPEC if n != 'E01_ref']; t0 = time.time()
    gains = (0.5, 2.0) if endpoint == 'Remitter' else ()
    jobs = [(n, s) for n in names for s in SEEDS]
    res = Parallel(n_jobs=10, verbose=3)(delayed(run_task)(n, s, endpoint, gains if s == 42 else ()) for n, s in jobs)
    D, y, *_ = load_all(endpoint); rid = f'B2_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
    rows = [dict(model=r['name'], seed=r['seed'], runtime_s=round(r['runtime'], 1), n_features=r['n_features'], **metrics(y, r['p'], r['dec'])) for r in res]
    pd.DataFrame(rows).to_csv(f'{rd}/metrics_by_seed.csv', index=False)
    pd.DataFrame({'id': D.id, 'y': y, **{f'p_{r["name"]}_s{r["seed"]}': r['p'] for r in res}, **{f'dec_{r["name"]}_s{r["seed"]}': r['dec'] for r in res}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
    pd.DataFrame([dict(model=r['name'], seed=r['seed'], **m) for r in res for m in r['meta']]).to_csv(f'{rd}/fold_thresholds_candidates.csv', index=False)
    g = [dict(model=r['name'], gain=gg, id=D.id.values[i], p_base=r['p'][i], p_gain=r['gp'][gg][i], dec_base=r['dec'][i], dec_gain=r['gd'][gg][i]) for r in res if r['seed'] == 42 for gg in gains for i in range(len(y))]
    if g: pd.DataFrame(g).to_csv(f'{rd}/gain_sensitivity_PRIVATE.csv', index=False)
    cfg = dict(run_id=rid, endpoint=endpoint, n=len(y), n_pos=int(y.sum()), seeds=SEEDS, grid_C=GRID_C, grid_hgb=GRID_H, protocol='BATCH2_PROTOCOL.md',
               code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], feature_fp={f: hashlib.sha256(open(f'{OUT}/{f}', 'rb').read()).hexdigest()[:16] for f in ('features_PRIVATE.csv', 'batch2_cov_PRIVATE.npz', 'batch2_specparam_fits_PRIVATE.csv')},
               command=' '.join(sys.argv), python=platform.python_version(), runtime_s=round(time.time() - t0), started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    import sklearn, pyriemann, specparam; cfg['versions'] = dict(sklearn=sklearn.__version__, pyriemann=pyriemann.__version__, specparam=specparam.__version__)
    json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1)
    pd.set_option('display.width', 220); print(rid, 'runtime', cfg['runtime_s']); print(pd.DataFrame(rows)[['model', 'seed', 'AUC', 'nPPV', 'sensitivity', 'Brier', 'n_features', 'runtime_s']].round(3).to_string(index=False))
