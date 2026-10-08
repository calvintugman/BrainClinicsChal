"""Per-recording cleaning + conventional resting features (label-blind, recording-local). Frozen settings in PROTOCOL.md.
Usage: python scripts/03_features.py [N_DEBUG]  -> outputs/features_PRIVATE.csv, outputs/windows_PRIVATE.pkl, outputs/cleaning_log_PRIVATE.csv"""
import os, sys, time, pickle, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd, mne
from joblib import Parallel, delayed
from common import *
warnings.filterwarnings('ignore')
FS = 500; WIN = 4 * FS; STEP = 2 * FS                       # 4 s Hann windows, 50% overlap
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12), 'beta': (12, 30)}   # half-open [lo, hi)
ROI = {'frontal': ['F7','F3','Fz','F4','F8'], 'central': ['FC3','FCz','FC4','C3','Cz','C4'], 'posterior': ['P3','Pz','P4','O1','Oz','O2']}
PTP_UV = 150.; MIN_WIN = 20; MIN_CONN_EP = 20; N_CONN_EP = 20; FLOOR_UV2 = 1e-6
CONN_BANDS = {'theta': (4, 8), 'alpha': (8, 12)}

def clean(pid, ses, cond):
    raw, pad = load_raw(pid, ses, cond)
    raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
    raw.notch_filter(50., notch_widths=2., picks=['eeg', 'eog', 'ecg'], verbose=False)     # line frequency 50 Hz (metadata + spectra)
    raw.filter(0.5, 45., l_trans_bandwidth=0.5, h_trans_bandwidth=4., picks=['eeg', 'eog', 'ecg'], verbose=False)
    x = raw.get_data(picks='eeg') * 1e6
    nb = x.shape[1] // FS; bs = x[:, :nb*FS].reshape(26, nb, FS).std(2); med = np.median(bs, 1); g = np.median(med)
    bad = [SCALP[i] for i in range(26) if med[i] > 3*g or med[i] < g/4]
    log = dict(id=pid, cond=cond, pad_s=pad, dur_s=raw.n_times/FS, bad_ch=';'.join(bad), n_bad=len(bad))
    if len(bad) > 3: log['status'] = 'too_many_bad_channels'; return None, log
    raw.info['bads'] = bad
    if bad: raw.interpolate_bads(reset_bads=True, verbose=False)
    # ICA on 1 Hz high-passed copy; EOG/ECG components by correlation (z>3), max 3 excluded
    rica = raw.copy().filter(1., None, picks=['eeg', 'eog', 'ecg'], verbose=False)
    x1 = rica.get_data(picks='eeg') * 1e6; ptp = np.ptp(x1[:, :nb*FS].reshape(26, nb, FS), axis=2).max(0)
    ann = mne.Annotations([i for i in range(nb) if ptp[i] > 200], 1., 'BAD_amp', orig_time=None)
    rica.set_annotations(ann)
    ncomp = min(20, 26 - len(bad) - 1)
    ica = mne.preprocessing.ICA(n_components=ncomp, method='fastica', random_state=42, max_iter=1000)
    try:
        ica.fit(rica, picks='eeg', reject_by_annotation=True); conv = int(ica.n_iter_)
        eogn = np.zeros((2, 0))
        rica_e = rica.copy()
        v = rica.get_data(picks=['VPVA']) - rica.get_data(picks=['VNVB']); h = rica.get_data(picks=['HPHL']) - rica.get_data(picks=['HNHR'])
        info = mne.create_info(['VEOG', 'HEOG'], FS, 'eog'); rica_e.add_channels([mne.io.RawArray(np.vstack([v, h]), info, verbose=False)], force_update_info=True)
        ex = set()
        for ch in ('VEOG', 'HEOG'):
            idx, sc = ica.find_bads_eog(rica_e, ch_name=ch, threshold=3.0, verbose=False); ex |= set(idx)
        idx, sc = ica.find_bads_ecg(rica_e, ch_name='Erbs', method='correlation', threshold=3.0, verbose=False); ex_ecg = set(idx)
        ex |= ex_ecg
        ica.exclude = sorted(ex)[:3]; log.update(ica_ok=True, ica_iter=conv, ica_excl=';'.join(map(str, ica.exclude)), n_ica_excl=len(ica.exclude), ica_ncomp=ncomp)
        ica.apply(raw, verbose=False)
    except Exception as e:
        log.update(ica_ok=False, ica_err=str(e)[:60], n_ica_excl=0)
    log['status'] = 'ok'
    return raw, log

def windows(x, interp_idx):
    """clean 4-s windows (50% overlap): returns indices of retained starts; rejection by scalp peak-to-peak > PTP_UV."""
    starts = np.arange(0, x.shape[1] - WIN + 1, STEP)
    keep = [s for s in starts if np.ptp(x[:, s:s+WIN], axis=1).max() <= PTP_UV]
    return np.array(keep, int)

def band_feats(x, starts, bad_names):
    win = np.hanning(WIN); scale = 1 / (FS * (win**2).sum())
    P = []   # (n_win, 26, n_freq) PSD in uV^2/Hz (one-sided)
    for s in starts:
        seg = x[:, s:s+WIN]; seg = seg - seg.mean(1, keepdims=True)           # constant detrend
        F = np.fft.rfft(seg * win, axis=1); p = (np.abs(F)**2) * scale; p[:, 1:-1] *= 2; P.append(p)
    P = np.array(P); f = np.fft.rfftfreq(WIN, 1/FS); df = f[1] - f[0]
    return P, f, df

def roi_idx(bad):
    return {k: [SCALP.index(c) for c in v if c not in bad] for k, v in ROI.items()}

def band_power(P, f, df, lo, hi):   # (n_win, 26) integrated power in uV^2, bins in [lo, hi)
    m = (f >= lo) & (f < hi); return P[:, :, m].sum(2) * df

def spectral_entropy(P, f):
    m = (f >= 1) & (f < 30); p = P[:, :, m].mean(0); p = p / p.sum(1, keepdims=True); return float((-(p*np.log(p)).sum(1) / np.log(m.sum())).mean())

def conn_feats(x, bad, seed_epochs=None):
    from mne_connectivity import spectral_connectivity_epochs
    use = [i for i, c in enumerate(SCALP) if c not in bad]
    starts = np.arange(0, x.shape[1] - WIN + 1, WIN)                       # non-overlapping 4 s epochs
    ep = [s for s in starts if np.ptp(x[:, s:s+WIN], axis=1).max() <= PTP_UV]
    if len(ep) < MIN_CONN_EP: return None, len(ep)
    ep = ep[:N_CONN_EP]
    def run(sel):
        data = np.array([x[use][:, s:s+WIN] for s in sel]) * 1e-6
        out = {}
        for b, (lo, hi) in CONN_BANDS.items():
            c = spectral_connectivity_epochs(data, method='wpli2_debiased', sfreq=FS, fmin=lo, fmax=hi, faverage=True, mode='multitaper', verbose=False)
            M = np.zeros((len(use), len(use))); a = c.get_data(output='dense')[:, :, 0]; M = a + a.T
            names = [SCALP[i] for i in use]; fi = [names.index(c) for c in ROI['frontal'] if c in names]; pi = [names.index(c) for c in ROI['posterior'] if c in names]
            def within(ix): return float(np.mean([M[i, j] for ii, i in enumerate(ix) for j in ix[ii+1:]]))
            out[f'conn_{b}_all'] = float(M[np.triu_indices(len(use), 1)].mean()); out[f'conn_{b}_frontal'] = within(fi); out[f'conn_{b}_posterior'] = within(pi)
            out[f'conn_{b}_between'] = float(M[np.ix_(fi, pi)].mean())
        return out
    full = run(ep); h1 = run(ep[0::2]); h2 = run(ep[1::2])
    return (full, h1, h2), len(ep)

def process(pid, ses):
    t0 = time.time(); feats = {'id': pid}; logs = []; win = {}
    for cond in ('EC', 'EO'):
        try:
            raw, log = clean(pid, ses, cond)
        except Exception as e:
            logs.append(dict(id=pid, cond=cond, status='error: ' + str(e)[:80])); continue
        if raw is None: logs.append(log); continue
        bad = log['bad_ch'].split(';') if log['bad_ch'] else []
        x = raw.get_data(picks='eeg') * 1e6
        st = windows(x, bad); log['n_clean_win'] = len(st); log['clean_s'] = len(st) * 2 + 2 if len(st) else 0
        if len(st) < MIN_WIN: log['status'] = 'too_few_windows'; logs.append(log); continue
        P, f, df = band_feats(x, st, bad); ri = roi_idx(bad)
        bp = {b: band_power(P, f, df, *lim) for b, lim in BANDS.items()}          # uV^2 per window x channel
        tot = sum(bp.values())
        W = {}
        for r, ix in ri.items():
            for b in BANDS:
                ab = np.maximum(bp[b][:, ix].mean(1), FLOOR_UV2); rl = ab / np.maximum(tot[:, ix].mean(1), FLOOR_UV2)
                W[(r, b)] = (np.log(ab), rl)                                        # per-window values for split-half
                feats[f'{cond}_{r}_{b}_logabs'] = float(np.log(np.maximum(bp[b][:, ix].mean(1).mean(), FLOOR_UV2)))
                feats[f'{cond}_{r}_{b}_rel'] = float(np.mean(rl))
        # FAA (needs both F3 and F4 non-interpolated)
        if 'F3' not in bad and 'F4' not in bad:
            a = np.maximum(bp['alpha'][:, [SCALP.index('F3'), SCALP.index('F4')]].mean(0), FLOOR_UV2); feats[f'{cond}_FAA'] = float(np.log(a[1]) - np.log(a[0]))
            wf = np.log(np.maximum(bp['alpha'][:, SCALP.index('F4')], FLOOR_UV2)) - np.log(np.maximum(bp['alpha'][:, SCALP.index('F3')], FLOOR_UV2)); W['FAA'] = wf
        feats[f'{cond}_entropy'] = spectral_entropy(P, f)
        # IAF: peak of posterior-ROI mean PSD (EC primary) in 7-13 Hz, must be interior local max with >=1.2x mean of 6-7 & 13-14 Hz flanks
        pm = P[:, ri['posterior']].mean((0, 1)); m = (f >= 7) & (f <= 13); ff = f[m]; pp = pm[m]; k = int(np.argmax(pp)); fl = pm[((f >= 6) & (f < 7)) | ((f > 13) & (f <= 14))].mean()
        feats[f'{cond}_IAF'] = float(ff[k]) if 0 < k < len(pp) - 1 and pp[k] >= 1.2 * fl else np.nan
        log.update({k_: float(v) for k_, v in dict(n_roi_frontal=len(ri['frontal']), n_roi_posterior=len(ri['posterior'])).items()})
        win[cond] = W
        if cond == 'EO':
            c, ne = conn_feats(x, bad); log['n_conn_ep'] = ne
            if c is not None:
                feats.update(c[0]); feats.update({k + '_h1': v for k, v in c[1].items()}); feats.update({k + '_h2': v for k, v in c[2].items()})
        logs.append(log)
    if 'EC_posterior_alpha_logabs' in feats and 'EO_posterior_alpha_logabs' in feats:
        for r in ROI: feats[f'ECEO_{r}_alpha_contrast'] = feats[f'EC_{r}_alpha_logabs'] - feats[f'EO_{r}_alpha_logabs']
    return feats, logs, win, time.time() - t0

if __name__ == '__main__':
    led = load_ledger(); n = int(sys.argv[1]) if len(sys.argv) > 1 else len(led); led = led.iloc[:n]
    t0 = time.time()
    res = Parallel(n_jobs=12, verbose=5)(delayed(process)(r['id'], r.selected_ses) for _, r in led.iterrows())
    pd.DataFrame([r[0] for r in res]).to_csv(f'{OUT}/features_PRIVATE.csv' if n == len(pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')) else f'{OUT}/features_debug_PRIVATE.csv', index=False)
    pd.DataFrame([l for r in res for l in r[1]]).to_csv(f'{OUT}/cleaning_log_PRIVATE.csv' if n >= 164 else f'{OUT}/cleaning_log_debug_PRIVATE.csv', index=False)
    pickle.dump({r[0]['id']: r[2] for r in res}, open(f'{OUT}/windows_PRIVATE.pkl' if n >= 164 else f'{OUT}/windows_debug_PRIVATE.pkl', 'wb'))
    print('elapsed s', round(time.time() - t0))
