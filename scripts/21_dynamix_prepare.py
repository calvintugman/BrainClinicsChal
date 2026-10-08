"""DynaMix window preparation (outcome-blind). v2-cleaned (clean_v2) 500 Hz scalp EEG (26 ch, uV) -> bad 1-s blocks (any-channel peak-to-peak > 150 uV in the 1-s block), dilated by 1 s each side
-> maximal contiguous clean runs -> resample whole record to 100 Hz (mne FIR resample; 0.5-45 Hz band already applied) -> non-overlapping 30-s windows tiled from the start of each run (leftover <30 s of a run is discarded,
fragments from different runs are NEVER joined) -> first up to 4 windows chronologically per recording. Units uV; channel order = common.SCALP.
Usage: python scripts/21_dynamix_prepare.py patients | seedsample"""
import os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, mne
from joblib import Parallel, delayed
from common import *
import importlib; cv2 = importlib.import_module('clean_v2')
WIN_S = 30; MAXW = 4; PTP = 150.; MARGIN = 1; DIR = f'{OUT}/dynamix/windows'
def prep_one(tag, pid, ses, cond):
    key = f'{tag}_{pid}_{cond}'; rec = dict(tag=tag, id=pid, cond=cond, key=key)
    try: raw, log = cv2.clean(pid, ses, cond)
    except Exception as e: rec.update(status='error: ' + str(e)[:60], n_windows=0); return rec
    if raw is None: rec.update(status=log.get('status', 'failed'), n_windows=0); return rec
    raw.pick('eeg'); x = raw.get_data() * 1e6; nb = x.shape[1] // 500; bad = np.array([np.ptp(x[:, i*500:(i+1)*500], axis=1).max() > PTP for i in range(nb)])
    badd = bad.copy()
    for s in range(1, MARGIN + 1): badd[:-s] |= bad[s:]; badd[s:] |= bad[:-s]
    raw.resample(100., verbose=False); y = raw.get_data().T.astype(np.float32); runs = []; i = 0
    while i < nb:
        if badd[i]: i += 1; continue
        j = i
        while j < nb and not badd[j]: j += 1
        runs.append((i, j)); i = j
    starts = []
    for a, b in runs:
        s = a
        while s + WIN_S <= b: starts.append(s); s += WIN_S
    starts = sorted(starts)[:MAXW]; W = np.stack([y[s*100:(s + WIN_S)*100] for s in starts]) if starts else np.zeros((0, WIN_S*100, 26), np.float32)
    assert W.shape[1:] == (3000, 26) and np.isfinite(W).all()
    np.savez_compressed(f'{DIR}/{key}.npz', W=W, starts_s=np.array(starts)); rec.update(status='ok', n_windows=len(starts), starts_s=';'.join(map(str, starts)), n_clean_runs=len(runs), longest_run_s=max([b - a for a, b in runs], default=0), n_bad_blocks=int(bad.sum()), n_interp=log['n_bad'], n_ica_excl=log.get('n_ica_excl'))
    return rec
if __name__ == '__main__':
    mode = sys.argv[1]; t0 = time.time()
    if mode == 'patients':
        led = load_ledger(); jobs = [('pt', r['id'], r.selected_ses, c) for _, r in led.iterrows() for c in ('EC', 'EO')]
    else:
        P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv'); rng = np.random.default_rng(7); s = P.iloc[rng.choice(len(P), 24, replace=False)]; conds = rng.choice(['EC', 'EO'], 24)
        jobs = [('seed', r.id, r.ses, c) for r, c in zip(s.itertuples(), conds)]
    res = Parallel(n_jobs=int(os.environ.get('NJ', 8)))(delayed(prep_one)(*j) for j in jobs); R = pd.DataFrame(res); R.to_csv(f'{OUT}/dynamix/window_manifest_{mode}_PRIVATE.csv', index=False)
    print(mode, 'recordings', len(R), R.status.value_counts().to_dict(), 'windows dist', R.n_windows.value_counts().sort_index().to_dict(), 'elapsed', round(time.time() - t0)); 
