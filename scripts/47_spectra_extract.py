"""Label-blind per-channel mean PSD (v2 cleaning, same clean 4-s windows as 03_features) for every patient (session 1) and pool person. Resumable chunks.
Stores mean one-sided PSD (uV^2/Hz) per channel at 0.25 Hz resolution, 0-45 Hz, per condition. -> outputs/spectra/chunk_*.pkl, outputs/spectra/spectra_PRIVATE.npz"""
import os, sys, pickle, time, glob
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
OD = f'{OUT}/spectra'; FMAX = 45.0
def one(pid, ses):
    import importlib, warnings; warnings.filterwarnings('ignore'); f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2')
    out = dict(id=pid, ses=ses)
    for cond in ('EC', 'EO'):
        try:
            raw, log = cv2.clean(pid, ses, cond)
            if raw is None: out[cond + '_status'] = log.get('status', 'none'); continue
            bad = log['bad_ch'].split(';') if log.get('bad_ch') else []; x = raw.get_data(picks='eeg') * 1e6; st = f3.windows(x, bad)
            if len(st) < f3.MIN_WIN: out[cond + '_status'] = 'too_few_windows'; continue
            P, f, df = f3.band_feats(x, st, bad); m = f <= FMAX; out[cond] = P[:, :, m].mean(0).astype(np.float32); out['f'] = f[m]; out[cond + '_status'] = 'ok'; out[cond + '_nwin'] = len(st); out[cond + '_nbad'] = len(bad)
        except Exception as e: out[cond + '_status'] = 'error: ' + str(e)[:60]
    return out
if __name__ == '__main__':
    os.makedirs(OD, exist_ok=True); led = load_ledger()[['id', 'selected_ses']].rename(columns={'selected_ses': 'ses'}); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv')[['id', 'ses']]
    L = pd.concat([led.assign(src='patient'), P.assign(src='pool')], ignore_index=True); L.to_csv(f'{OD}/list_PRIVATE.csv', index=False); print('people', len(L), flush=True); t0 = time.time(); CH = 60
    for ci in range(0, len(L), CH):
        fn = f'{OD}/chunk_{ci:05d}.pkl'
        if os.path.exists(fn): continue
        res = Parallel(n_jobs=12)(delayed(one)(r.id, r.ses) for r in L.iloc[ci:ci + CH].itertuples()); pickle.dump(res, open(fn + '.tmp', 'wb')); os.replace(fn + '.tmp', fn); print(f'chunk {ci} done, {round(time.time() - t0)} s', flush=True)
    res = [r for fn in sorted(glob.glob(f'{OD}/chunk_*.pkl')) for r in pickle.load(open(fn, 'rb'))]; f = next(r['f'] for r in res if 'f' in r); n = len(res)
    EC = np.full((n, 26, len(f)), np.nan, np.float32); EO = EC.copy()
    for i, r in enumerate(res):
        if 'EC' in r: EC[i] = r['EC']
        if 'EO' in r: EO[i] = r['EO']
    np.savez_compressed(f'{OD}/spectra_PRIVATE.npz', ids=np.array([r['id'] for r in res]), f=f, EC=EC, EO=EO)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('EC', 'EO', 'f')} for r in res]).to_csv(f'{OD}/status_PRIVATE.csv', index=False); print('DONE', round(time.time() - t0), 's', flush=True)
