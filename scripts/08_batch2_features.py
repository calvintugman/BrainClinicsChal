"""Batch-2 recording-local, label-blind extraction: specparam fits (E11) and Ledoit-Wolf covariances (E12/E13). Frozen in BATCH2_PROTOCOL.md.
Usage: python scripts/08_batch2_features.py [N_DEBUG]"""
import os, sys, time, pickle, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd, mne
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from common import *
import importlib; f3 = importlib.import_module('03_features')
warnings.filterwarnings('ignore')
FS = 500; WIN = 2000; BANDS = {'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}; NEP = 20; PTP = 150.; MARGIN_S = 1
ROIS = f3.ROI

def specfit(f, p, lo=2, hi=40):
    from specparam import SpectralModel
    out = dict(ok=False, exp=np.nan, off=np.nan, cf=np.nan, pw=np.nan, r2=np.nan, n_peaks=np.nan)
    try:
        m = SpectralModel(aperiodic_mode='fixed', peak_width_limits=(1, 8), max_n_peaks=4, peak_threshold=2.0, verbose=False)
        m.fit(f, p, (lo, hi)); r2 = float(m.results.metrics.results['gof_rsquared']); ap = np.asarray(m.results.get_params('aperiodic')); pk = np.atleast_2d(m.results.get_params('peak'))
        out.update(r2=r2, n_peaks=0 if pk.size == 0 or np.isnan(pk).all() else len(pk))
        if not (np.isfinite(ap).all() and np.isfinite(r2) and r2 >= 0.90): return out
        out.update(ok=True, off=float(ap[0]), exp=float(ap[1]), pw=0.0)
        if pk.size and not np.isnan(pk).all():
            a = pk[(pk[:, 0] >= 8) & (pk[:, 0] <= 12)]
            if len(a): j = int(np.argmax(a[:, 1])); out.update(cf=float(a[j, 0]), pw=float(a[j, 1]))
    except Exception as e:
        out['err'] = str(e)[:60]
    return out

def valid_epochs(x):
    """non-overlapping 4 s epochs, valid iff ptp<=150 uV and no 1-s block with ptp>150 within +-1 s margin."""
    nb = x.shape[1] // FS; blk_bad = np.array([np.ptp(x[:, i*FS:(i+1)*FS], axis=1).max() > PTP for i in range(nb)])
    eps = []
    for s in range(0, x.shape[1] - WIN + 1, WIN):
        s0 = s // FS; lo = max(0, s0 - MARGIN_S); hi = min(nb, s0 + WIN // FS + MARGIN_S)
        if np.ptp(x[:, s:s+WIN], axis=1).max() <= PTP and not blk_bad[lo:hi].any(): eps.append(s)
    return eps

def lw_cov(seg_list):
    return np.mean([LedoitWolf().fit(s.T).covariance_ for s in seg_list], axis=0)

def process(pid, ses):
    res = dict(id=pid); cov = np.full((2, 2, 3, 26, 26), np.nan); logs = []; psd = {}; sp_rows = {}
    for ci, cond in enumerate(('EC', 'EO')):
        try: raw, log = f3.clean(pid, ses, cond)
        except Exception as e: logs.append(dict(id=pid, cond=cond, status='error')); continue
        if raw is None: logs.append(log); continue
        bad = log['bad_ch'].split(';') if log['bad_ch'] else []; x = raw.get_data(picks='eeg') * 1e6
        # ---- specparam on mean clean-window PSD per ROI (+ split halves with one-window gap)
        st = f3.windows(x, bad); log['n_clean_win'] = len(st)
        if len(st) >= f3.MIN_WIN:
            P, f, df = f3.band_feats(x, st, bad); ri = f3.roi_idx(bad); m = (f >= 2) & (f <= 40); n = len(st) // 2
            for r, ix in ri.items():
                full = P[:, ix].mean((0, 1)); h1 = P[:n-1][:, ix].mean((0, 1)); h2 = P[n:][:, ix].mean((0, 1))
                for tag, pp in (('', full), ('_h1', h1), ('_h2', h2)):
                    o = specfit(f[m], pp[m]); sp_rows[f'{cond}_{r}{tag}'] = o
                if cond == 'EO' or True: psd[(cond, r)] = (f[m], full[m])
        # ---- covariance epochs
        eps = valid_epochs(x); log['n_cov_epochs_valid'] = len(eps); log['n_bad'] = len(bad)
        if len(eps) >= NEP:
            eps = eps[:NEP]
            for bi, (b, (lo, hi)) in enumerate(BANDS.items()):
                xb = raw.copy().pick('eeg').filter(lo, hi, verbose=False).get_data() * 1e6
                segs = [xb[:, s:s+WIN] for s in eps]; cov[ci, 0, bi] = lw_cov(segs); cov[ci, 1, bi] = lw_cov(segs[0::2]); h2 = lw_cov(segs[1::2])
                cov[ci, 1, bi] = np.nan  # placeholder replaced below
                cov[ci, 1, bi] = lw_cov(segs[0::2]); res[f'_h2_{cond}_{b}'] = h2
        logs.append(log)
    return res, cov, logs, sp_rows, psd

if __name__ == '__main__':
    led = load_ledger(); n = int(sys.argv[1]) if len(sys.argv) > 1 else len(led); led = led.iloc[:n]; t0 = time.time()
    out = Parallel(n_jobs=10, verbose=2)(delayed(process)(r['id'], r.selected_ses) for _, r in led.iterrows())
    tag = '' if n >= 164 else '_debug'
    ids = [o[0]['id'] for o in out]; COV = np.stack([o[1] for o in out])        # (N, cond, [full, half1], band, 26, 26)
    H2 = np.full((len(out), 2, 3, 26, 26), np.nan)
    for i, o in enumerate(out):
        for ci, cond in enumerate(('EC', 'EO')):
            for bi, b in enumerate(BANDS):
                if f'_h2_{cond}_{b}' in o[0]: H2[i, ci, bi] = o[0][f'_h2_{cond}_{b}']
    np.savez(f'{OUT}/batch2_cov_PRIVATE{tag}.npz', ids=np.array(ids), cov_full=COV[:, :, 0], cov_h1=COV[:, :, 1], cov_h2=H2)
    pd.DataFrame([l for o in out for l in o[2]]).to_csv(f'{OUT}/batch2_extract_log_PRIVATE{tag}.csv', index=False)
    rows = []
    for o in out:
        for k, v in o[3].items(): rows.append(dict(id=o[0]['id'], key=k, **v))
    pd.DataFrame(rows).to_csv(f'{OUT}/batch2_specparam_fits_PRIVATE{tag}.csv', index=False)
    pickle.dump({o[0]['id']: o[4] for o in out}, open(f'{OUT}/batch2_roi_psd_PRIVATE{tag}.pkl', 'wb'))
    print('elapsed s', round(time.time() - t0))
