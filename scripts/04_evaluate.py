"""Locked evaluation E00 / E01-PSD / E01 (see PROTOCOL.md). Usage: python scripts/04_evaluate.py Remitter|Responder [--perm N]"""
import sys, os, json, hashlib, time, platform, subprocess, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
warnings.filterwarnings('ignore')
OUT = 'outputs'; GRID = [0.01, 0.1, 1, 10]; SEEDS = [42, 43, 44]; NBOOT = 2000
endpoint = sys.argv[1]; nperm = int(sys.argv[sys.argv.index('--perm') + 1]) if '--perm' in sys.argv else 0
F = pd.read_csv(f'{OUT}/features_PRIVATE.csv'); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')
D = F.merge(led[['id', 'Responder', 'Remitter']], on='id').sort_values('id').reset_index(drop=True)
y = D[endpoint].astype(int).values; ids = D.id.values
psd_cols = [c for c in F.columns if (c.startswith(('EC_', 'EO_')) and c.rsplit('_', 1)[-1] in ('logabs', 'rel', 'IAF')) or c.startswith('ECEO_')]
extra = [c for c in F.columns if c.endswith(('_FAA', '_entropy'))] + [c for c in F.columns if c.startswith('conn_') and not c.endswith(('_h1', '_h2'))]
SETS = {'E01-PSD': psd_cols, 'E01': psd_cols + extra}
assert len(psd_cols) == 53 and len(SETS['E01']) == 65, (len(psd_cols), len(SETS['E01']))

def pipe(C): return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=C, penalty='l2', max_iter=2000))
def pick_threshold(p, yy):
    best = None
    for t in np.unique(p):
        d = p >= t; tp = (d & (yy == 1)).sum(); sens = tp / (yy == 1).sum(); ppv = tp / d.sum() if d.sum() else np.nan
        if sens >= 0.5 and (best is None or (ppv, sens) > (best[1], best[2])): best = (t, ppv, sens)
    return best[0] if best else p.min()
def fit_predict(Xtr, ytr, Xte, seed):
    inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr)); ll = {}; oof_by = {}
    for C in GRID:
        oof = np.zeros(len(ytr))
        for a, b in inner: oof[b] = pipe(C).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        ll[C] = log_loss(ytr, oof); oof_by[C] = oof
    Cb = min(ll, key=ll.get); thr = pick_threshold(oof_by[Cb], ytr)
    return pipe(Cb).fit(Xtr, ytr).predict_proba(Xte)[:, 1], thr, Cb
def run_model(name, yy, seed):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(yy)), yy)); p = np.zeros(len(yy)); dec = np.zeros(len(yy), int); fo = np.zeros(len(yy), int); meta = []
    X = D[SETS[name]].values if name in SETS else None
    for k, (tr, te) in enumerate(folds):
        if name == 'E00': p[te] = yy[tr].mean(); dec[te] = 1; thr = np.nan; Cb = np.nan   # constant prevalence; sens=1>=0.5 so rule calls everyone positive
        else:
            p[te], thr, Cb = fit_predict(X[tr], yy[tr], X[te], seed); dec[te] = (p[te] >= thr).astype(int)
        fo[te] = k; meta.append(dict(fold=k, thr=float(thr), C=Cb, n_test=len(te), pos_test=int(yy[te].sum())))
    return p, dec, fo, meta
def metrics(yy, p, dec):
    tp = int(((dec == 1) & (yy == 1)).sum()); fp = int(((dec == 1) & (yy == 0)).sum()); tn = int(((dec == 0) & (yy == 0)).sum()); fn = int(((dec == 0) & (yy == 1)).sum())
    prev = yy.mean(); sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan; ppv = tp / (tp + fp) if tp + fp else np.nan
    return dict(AUC=roc_auc_score(yy, p) if len(np.unique(p)) > 1 else 0.5, PPV=ppv, sensitivity=sens, nPPV=100 * ppv / prev if ppv == ppv else np.nan,
                Brier=brier_score_loss(yy, p), balanced_accuracy=(sens + spec) / 2, n_pos_calls=tp + fp, TP=tp, FP=fp, TN=tn, FN=fn, N=len(yy), prevalence=prev)

t0 = time.time(); rid = f'R_rest_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
preds = {}; rows = []; splits = []; foldmeta = []
for s in SEEDS:
    for m in ['E00', 'E01-PSD', 'E01']:
        p, dec, fo, meta = run_model(m, y, s); preds[(m, s)] = (p, dec); rows.append(dict(model=m, seed=s, **metrics(y, p, dec)))
        foldmeta += [dict(model=m, seed=s, **d) for d in meta]
        if m == 'E00': splits.append(pd.DataFrame(dict(id=ids, seed=s, fold=fo)))
M = pd.DataFrame(rows); M.to_csv(f'{rd}/metrics_by_seed.csv', index=False); pd.DataFrame(foldmeta).to_csv(f'{rd}/fold_thresholds_C.csv', index=False)
pd.concat(splits).to_csv(f'{rd}/splits_PRIVATE.csv', index=False)
pd.DataFrame({'id': ids, 'y': y, **{f'p_{m}_s{s}': preds[(m, s)][0] for (m, s) in preds}, **{f'dec_{m}_s{s}': preds[(m, s)][1] for (m, s) in preds}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
# paired patient bootstrap, same draws for all models
rng = np.random.default_rng(1); B = []
for s in SEEDS:
    for b in range(NBOOT):
        pass
draws = rng.integers(0, len(y), size=(NBOOT, len(y))); boot = []
for s in SEEDS:
    for b, ix in enumerate(draws):
        if len(np.unique(y[ix])) < 2: continue
        rec = {'seed': s, 'b': b}
        for m in ['E00', 'E01-PSD', 'E01']:
            mm = metrics(y[ix], preds[(m, s)][0][ix], preds[(m, s)][1][ix]); rec.update({f'{m}|{k}': mm[k] for k in ('AUC', 'nPPV', 'sensitivity')})
        boot.append(rec)
boot = pd.DataFrame(boot); boot.to_csv(f'{rd}/bootstrap_draws.csv', index=False)
rows = []
for s in SEEDS:
    bs = boot[boot.seed == s]
    for k in ('AUC', 'nPPV', 'sensitivity'):
        for m in ['E00', 'E01-PSD', 'E01']:
            v = bs[f'{m}|{k}'].dropna(); rows.append(dict(seed=s, kind='model', name=m, metric=k, lo=v.quantile(.025), hi=v.quantile(.975), n_valid=len(v)))
        for a, b_ in [('E01-PSD', 'E00'), ('E01', 'E00'), ('E01', 'E01-PSD')]:
            d = (bs[f'{a}|{k}'] - bs[f'{b_}|{k}']).dropna(); rows.append(dict(seed=s, kind='diff', name=f'{a} - {b_}', metric=k, est=float(M[(M.model == a) & (M.seed == s)][k].iloc[0] - M[(M.model == b_) & (M.seed == s)][k].iloc[0]), lo=d.quantile(.025), hi=d.quantile(.975), n_valid=len(d)))
pd.DataFrame(rows).to_csv(f'{rd}/bootstrap_intervals.csv', index=False)
# permutation check (E01, seed 42 split design), full pipeline refit
if nperm:
    obs = float(M[(M.model == 'E01') & (M.seed == 42)].AUC.iloc[0]); rng2 = np.random.default_rng(7)
    perms = [rng2.permutation(y) for _ in range(nperm)]
    aucs = Parallel(n_jobs=12)(delayed(lambda yp: roc_auc_score(yp, run_model('E01', yp, 42)[0]))(yp) for yp in perms)
    json.dump(dict(observed_AUC=obs, n_perm=nperm, seed=7, perm_AUC_mean=float(np.mean(aucs)), perm_AUC_q95=float(np.quantile(aucs, .95)), p_value=float((1 + np.sum(np.array(aucs) >= obs)) / (nperm + 1)),
                   refit='full pipeline incl. inner C selection and threshold, patient labels permuted; split seed 42'), open(f'{rd}/permutation.json', 'w'), indent=1)
    pd.DataFrame({'perm_AUC': aucs}).to_csv(f'{rd}/permutation_aucs.csv', index=False)
code = hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]
cfg = dict(run_id=rid, endpoint=endpoint, n=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()), seeds=SEEDS, C_grid=GRID, n_boot=NBOOT, n_perm=nperm,
           feature_fp=hashlib.sha256(open(f'{OUT}/features_PRIVATE.csv', 'rb').read()).hexdigest()[:16], split_fp=hashlib.sha256(pd.concat(splits).to_csv(index=False).encode()).hexdigest()[:16],
           code_fp=code, schema={k: v for k, v in SETS.items()}, command=' '.join(sys.argv), python=platform.python_version(), host=platform.platform(), runtime_s=round(time.time() - t0))
import sklearn, mne, mne_connectivity; cfg['versions'] = dict(sklearn=sklearn.__version__, mne=mne.__version__, mne_connectivity=mne_connectivity.__version__, numpy=np.__version__, pandas=pd.__version__)
json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1)
pd.set_option('display.width', 220); print(rid, cfg['n'], cfg['n_pos'], 'runtime', cfg['runtime_s']); print(M.round(3).to_string(index=False))
