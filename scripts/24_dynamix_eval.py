"""Frozen DynaMix evaluation (DYNAMIX_PROTOCOL.md): 4 models x 3 repeats x 2 endpoints. median impute -> StandardScaler -> L2 LR, C {0.01,0.1,1,10} by inner log loss; threshold from inner OOF; same splits as all earlier runs."""
import os, sys, json, time, hashlib, platform, glob, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
import numpy as np, pandas as pd, importlib
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
warnings.filterwarnings('ignore')
ev = importlib.import_module('10_batch2_eval'); OUT = 'outputs'; MODS = ['pretrained', 'random_a', 'random_b', 'random_c']; SEEDS = [42, 43, 44]; CHN = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2']; FN = [f'z{i}' for i in range(10)] + ['gate_entropy', 'gate_max']
def patient_table(ids, model):
    cols = [f'dm_{c}_{ch}_{f}' for c in ('EC', 'EO') for ch in CHN for f in FN]; A = np.full((len(ids), 624), np.nan)
    for i, pid in enumerate(ids):
        for ci, c in enumerate(('EC', 'EO')):
            fn = f'{OUT}/dynamix/emb/{model}/pt_{pid}_{c}.npz'
            if os.path.exists(fn):
                m = np.load(fn)['mean']
                if len(m): A[i, ci*312:(ci+1)*312] = m.mean(0).reshape(-1)
    return pd.DataFrame(A, columns=cols)
def run_lr(T, cols, y, seed):
    cands = ev.GRID_C; folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)); p = np.zeros(len(y)); dec = np.zeros(len(y), int); meta = []; data = {'T': T}
    for k, (tr, te) in enumerate(folds):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y[tr])); oof = {c: np.zeros(len(tr)) for c in cands}
        for a, b in inner:
            pr = ev.Prep(cols, False, True, seed).fit(data, tr[a]); Xa, Xb = pr.transform(data, tr[a]), pr.transform(data, tr[b])
            for c in cands: oof[c][b] = ev.make_model('lr', c, seed).fit(Xa, y[tr][a]).predict_proba(Xb)[:, 1]
        ll = {c: log_loss(y[tr], oof[c]) for c in cands}; cb = min(ll, key=ll.get); thr = ev.pick_threshold(oof[cb], y[tr])
        pr = ev.Prep(cols, False, True, seed).fit(data, tr); mdl = ev.make_model('lr', cb, seed).fit(pr.transform(data, tr), y[tr]); p[te] = mdl.predict_proba(pr.transform(data, te))[:, 1]; dec[te] = (p[te] >= thr).astype(int); meta.append(dict(fold=k, thr=float(thr), C=cb))
    return p, dec, meta
def task(model, seed, endpoint):
    T = pd.read_pickle(f'{OUT}/dynamix/patient_features_{model}_PRIVATE.pkl'); D, y, *_ = ev.load_all(endpoint); assert (D.id.values == T.attrs['ids']).all()
    t0 = time.time(); p, dec, meta = run_lr(T, list(T.columns), y, seed); return model, seed, p, dec, meta, time.time() - t0
if __name__ == '__main__':
    D, y0, *_ = ev.load_all('Remitter'); ids = D.id.values; M = pd.read_csv(f'{OUT}/dynamix/window_manifest_patients_PRIVATE.csv')
    for mo in MODS: T = patient_table(ids, mo); T.attrs['ids'] = ids; T.to_pickle(f'{OUT}/dynamix/patient_features_{mo}_PRIVATE.pkl'); print(mo, 'features', T.shape, 'patients with any missing', int(T.isna().any(axis=1).sum()), 'both conditions available', int((~T.isna().any(axis=1)).sum()), flush=True)
    for endpoint in ('Remitter', 'Responder'):
        t0 = time.time(); res = Parallel(n_jobs=12)(delayed(task)(m, s, endpoint) for m in MODS for s in SEEDS); D, y, *_ = ev.load_all(endpoint)
        rid = f'DM_{endpoint}_{time.strftime("%Y%m%dT%H%M%S")}'; rd = f'runs/{rid}'; os.makedirs(rd)
        rows = [dict(model=m, seed=s, runtime_s=round(rt, 1), n_features=624, **ev.metrics(y, p, dec)) for m, s, p, dec, meta, rt in res]; pd.DataFrame(rows).to_csv(f'{rd}/metrics_by_seed.csv', index=False)
        pd.DataFrame({'id': D.id, 'y': y, **{f'p_{m}_s{s}': p for m, s, p, dec, meta, rt in res}, **{f'dec_{m}_s{s}': dec for m, s, p, dec, meta, rt in res}}).to_csv(f'{rd}/predictions_PRIVATE.csv', index=False)
        pd.DataFrame([dict(model=m, seed=s, **x) for m, s, p, dec, meta, rt in res for x in meta]).to_csv(f'{rd}/fold_thresholds_C.csv', index=False)
        em = json.load(open(f'{OUT}/dynamix/embed_runtime.json')) if os.path.exists(f'{OUT}/dynamix/embed_runtime.json') else {}
        cfg = dict(run_id=rid, endpoint=endpoint, n=len(y), n_pos=int(y.sum()), protocol='DYNAMIX_PROTOCOL.md', K=json.load(open(f'{OUT}/dynamix/seed_budget.json'))['chosen_K'], weights_sha256=hashlib.sha256(open('external_review/dynamix_reviewed/hf_6d.safetensors', 'rb').read()).hexdigest(), window_manifest_sha256=hashlib.sha256(open(f'{OUT}/dynamix/window_manifest_patients_PRIVATE.csv', 'rb').read()).hexdigest()[:16],
                   feature_fps={m: hashlib.sha256(open(f'{OUT}/dynamix/patient_features_{m}_PRIVATE.pkl', 'rb').read()).hexdigest()[:16] for m in MODS}, code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16], seeds=SEEDS, grid_C=ev.GRID_C, embed_runtime=em, runtime_s=round(time.time() - t0), python=platform.python_version(),
                   started_local=time.strftime('%Y-%m-%dT%H:%M:%S%z'), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), command=' '.join(sys.argv))
        import sklearn, torch; cfg['versions'] = dict(sklearn=sklearn.__version__, torch=torch.__version__); json.dump(cfg, open(f'{rd}/config.json', 'w'), indent=1)
        pd.set_option('display.width', 200); print(rid, 'runtime', cfg['runtime_s']); print(pd.DataFrame(rows)[['model', 'seed', 'AUC', 'nPPV', 'sensitivity', 'PPV', 'Brier', 'n_pos_calls']].round(3).to_string(index=False), flush=True)
