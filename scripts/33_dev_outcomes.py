"""DEVIATION_PROTOCOL section 5 (run only because F1 and F3 passed): exactly four outcome models A-D, Remitter primary / Responder secondary, + complete-case refit sensitivity."""
import os, sys, json, time, hashlib, platform, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
warnings.filterwarnings('ignore')
ev = importlib.import_module('10_batch2_eval'); dev = importlib.import_module('dev'); OUT = 'outputs'; SEEDS = [42, 43, 44]; MODELS = ['A', 'B', 'C', 'D']
dec = json.load(open(f'{OUT}/dev/feasibility_decision.json')); assert dec['proceed_to_remission_scoring'] is True, 'feasibility gate failed: no remission scoring'
def build_table():
    D, y, *_ = ev.load_all('Remitter'); ids = D.id.values; F = pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv').set_index('id').reindex(ids)
    X, tr, te = dev.load_reference(); Ec, Eo = X[dev.ECc].values, X[dev.EOc].values; lcR, scR = dev.level_shape(Ec[tr]); loR, soR = dev.level_shape(Eo[tr]); Ms = dev.Mapping(scR, soR); rsd = Ms.crossfit_resid_sd(scR, soR)     # reference-training only
    EcP, EoP = F[dev.ECc].values, F[dev.EOc].values; lc, sc = dev.level_shape(EcP); lo, so = dev.level_shape(EoP); both = np.isfinite(lc) & np.isfinite(lo)
    dev_ = np.full((len(ids), 12), np.nan); dev_[both] = (so[both] - Ms.predict(sc[both])) / rsd; contrast = np.full((len(ids), 12), np.nan); contrast[both] = so[both] - sc[both]
    T = pd.DataFrame({'EC_level': lc, 'EO_level': lo}); T.attrs['ids'] = ids
    for c in dev.ECc + dev.EOc: T['B_' + c] = F[c].values
    for j, n in enumerate(dev.NAMES): T[f'C_contrast_{n}'] = contrast[:, j]; T[f'D_deviation_{n}'] = dev_[:, j]
    cols = {'A': ['EC_level', 'EO_level'], 'B': ['B_' + c for c in dev.ECc + dev.EOc], 'C': ['EC_level', 'EO_level'] + [f'C_contrast_{n}' for n in dev.NAMES], 'D': ['EC_level', 'EO_level'] + [f'D_deviation_{n}' for n in dev.NAMES]}
    return T, cols, y, both, dict(mapping_alpha=Ms.alpha, resid_sd=rsd.tolist(), n_complete=int(both.sum()), n_EC_incomplete=int(np.isnan(lc).sum()), n_EO_incomplete=int(np.isnan(lo).sum()))
def run_lr2(T, cols, y, seed, mask=None):
    n = len(y); mask = np.ones(n, bool) if mask is None else mask; folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y)); p = np.full(n, np.nan); dec_ = np.full(n, -1); data = {'T': T}; meta = []
    for k, (tr, te) in enumerate(folds):
        tr, te = tr[mask[tr]], te[mask[te]]; inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr])); oof = {c: np.zeros(len(tr)) for c in ev.GRID_C}
        for a, b in inner:
            pr = ev.Prep(cols, False, True, seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for c in ev.GRID_C: oof[c][b] = ev.make_model('lr', c, seed).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {c: log_loss(y[tr], oof[c]) for c in ev.GRID_C}; cb = min(ll, key=ll.get); thr = ev.pick_threshold(oof[cb], y[tr])
        pr = ev.Prep(cols, False, True, seed).fit(data, tr); mdl = ev.make_model('lr', cb, seed).fit(pr.transform(data, tr), y[tr]); p[te] = mdl.predict_proba(pr.transform(data, te))[:, 1]; dec_[te] = (p[te] >= thr).astype(int); meta.append(dict(fold=k, thr=float(thr), C=cb))
    return p, dec_, meta
def task(m, s, endpoint, cc):
    T, cols, y, both, _ = build_table(); D, yy, *_ = ev.load_all(endpoint); t0 = time.time(); p, d, meta = run_lr2(T, cols[m], yy, s, both if cc else None); return m, s, p, d, meta, time.time() - t0
if __name__ == '__main__':
    T, cols, _, both, info = build_table(); print('features', {m: len(c) for m, c in cols.items()}, info, flush=True); T.to_pickle(f'{OUT}/dev/patient_features_PRIVATE.pkl'); json.dump(dict(cols=cols, **info), open(f'{OUT}/dev/feature_schema.json', 'w'), indent=1)
    for cc in (False, True):
        for endpoint in ('Remitter', 'Responder'):
            t0 = time.time(); res = Parallel(n_jobs=12)(delayed(task)(m, s, endpoint, cc) for m in MODELS for s in SEEDS); D, y, *_ = ev.load_all(endpoint); tag = 'DEVcc' if cc else 'DEV'; rid = f'{tag}_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
            keep = both if cc else np.ones(len(y), bool); rows = []
            for m, s, p, d, meta, rt in res:
                yy = y[keep]; rows.append(dict(model=m, seed=s, runtime_s=round(rt, 1), n_features=len(cols[m]), **ev.metrics(yy, p[keep], d[keep])))
            pd.DataFrame(rows).to_csv(f'{rd}/metrics_by_seed.csv', index=False)
            pd.DataFrame({'id': D.id, 'y': y, 'complete_case': both, **{f'p_{m}_s{s}': p for m, s, p, d, meta, rt in res}, **{f'dec_{m}_s{s}': d for m, s, p, d, meta, rt in res}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
            pd.DataFrame([dict(model=m, seed=s, **x) for m, s, p, d, meta, rt in res for x in meta]).to_csv(f'{rd}/fold_thresholds_C.csv', index=False)
            cfg = dict(run_id=rid, endpoint=endpoint, complete_case_refit=cc, n=int(keep.sum()), n_pos=int(y[keep].sum()), protocol='DEVIATION_PROTOCOL.md', feasibility=dec, schema=info, code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], feature_fp=hashlib.sha256(open(f'{OUT}/dev/patient_features_PRIVATE.pkl', 'rb').read()).hexdigest()[:16],
                       features_source_fp=hashlib.sha256(open(f'{OUT}/v2/features_v2_PRIVATE.csv', 'rb').read()).hexdigest()[:16], seeds=SEEDS, grid_C=ev.GRID_C, runtime_s=round(time.time() - t0), python=platform.python_version(), started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1); pd.set_option('display.width', 200); print(rid, 'N', cfg['n'], 'prev', round(y[keep].mean(), 3)); print(pd.DataFrame(rows)[['model', 'seed', 'AUC', 'nPPV', 'sensitivity', 'PPV', 'Brier', 'n_pos_calls']].round(3).to_string(index=False), flush=True)
