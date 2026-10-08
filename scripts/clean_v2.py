"""Corrected cleaning (v2) - outcome-blind, frozen before any new outcome scoring. Differences from 03_features.clean (v1):
 (1) ICA convergence: n_iter_ == max_iter counts as NON-convergence; retry once (random_state 43, max_iter 3000); still non-converged -> recording marked ica_failed (features missing, no silent fallback).
 (2) Component selection by EVIDENCE, not index: candidate = EOG-correlation z>3 (MNE find_bads_eog on bipolar VEOG/HEOG, as v1) AND frontal-dominant topography
     (mean |weight| at Fp1,Fp2,F7,F8 >= 1.5 x mean |weight| over all channels). Accepted candidates are ranked by max |r|, at most 3 kept (rank order, not index); candidates beyond the cap are logged.
 (3) ECG/'Erbs' criterion dropped (Erbs is an unvalidated proxy channel; heart-artifact removal not claimed). Padding trimming, notch, band-pass, bad-channel interpolation unchanged.
 (4) Rank diagnostics logged: effective rank of the post-ICA scalp data (eigenvalues > 1e-6 of max)."""
import numpy as np, mne
from common import *
import importlib; f3 = importlib.import_module('03_features')
FS = 500; FRONT = ['Fp1', 'Fp2', 'F7', 'F8']; Z_THR = 3.0; FRONT_RATIO = 1.5; CAP = 3

def _fit_ica(rica, ncomp, seed, max_iter):
    ica = mne.preprocessing.ICA(n_components=ncomp, method='fastica', random_state=seed, max_iter=max_iter)
    ica.fit(rica, picks='eeg', reject_by_annotation=True); return ica, int(ica.n_iter_) < max_iter, int(ica.n_iter_)

def clean(pid, ses, cond, return_ica=False):
    raw, pad = load_raw(pid, ses, cond); raw.set_montage('standard_1020', match_case=False, on_missing='ignore')
    raw.notch_filter(50., notch_widths=2., picks=['eeg', 'eog', 'ecg'], verbose=False)
    raw.filter(0.5, 45., l_trans_bandwidth=0.5, h_trans_bandwidth=4., picks=['eeg', 'eog', 'ecg'], verbose=False)
    x = raw.get_data(picks='eeg') * 1e6; nb = x.shape[1] // FS; bs = x[:, :nb*FS].reshape(26, nb, FS).std(2); med = np.median(bs, 1); g = np.median(med)
    bad = [SCALP[i] for i in range(26) if med[i] > 3*g or med[i] < g/4]
    log = dict(id=pid, cond=cond, pad_s=pad, dur_s=raw.n_times/FS, bad_ch=';'.join(bad), n_bad=len(bad), version='v2')
    if len(bad) > 3: log['status'] = 'too_many_bad_channels'; return (None, log, None) if return_ica else (None, log)
    raw.info['bads'] = bad
    if bad: raw.interpolate_bads(reset_bads=True, verbose=False)
    rica = raw.copy().filter(1., None, picks=['eeg', 'eog', 'ecg'], verbose=False)
    x1 = rica.get_data(picks='eeg') * 1e6; ptp = np.ptp(x1[:, :nb*FS].reshape(26, nb, FS), axis=2).max(0)
    rica.set_annotations(mne.Annotations([i for i in range(nb) if ptp[i] > 200], 1., 'BAD_amp', orig_time=None))
    ncomp = min(20, 26 - len(bad) - 1); ica = None; conv = False
    for seed, mi in ((42, 1000), (43, 3000)):
        ica, conv, nit = _fit_ica(rica, ncomp, seed, mi); log.update(ica_seed=seed, ica_iter=nit, ica_converged=conv, ica_retried=(seed == 43))
        if conv: break
    if not conv:
        log.update(ica_ok=False, status='ica_failed', n_ica_excl=0); return (None, log, None) if return_ica else (None, log)
    v = rica.get_data(picks=['VPVA']) - rica.get_data(picks=['VNVB']); h = rica.get_data(picks=['HPHL']) - rica.get_data(picks=['HNHR'])
    r2 = rica.copy(); r2.add_channels([mne.io.RawArray(np.vstack([v, h]), mne.create_info(['VEOG', 'HEOG'], FS, 'eog'), verbose=False)], force_update_info=True)
    W = np.abs(ica.get_components()); allm = W.mean(0); fi = [SCALP.index(c) for c in FRONT]; ratio = W[fi].mean(0) / allm
    cand = {}
    for ch in ('VEOG', 'HEOG'):
        idx, sc = ica.find_bads_eog(r2, ch_name=ch, threshold=Z_THR, verbose=False); sc = np.asarray(sc)
        for k in idx: cand[k] = max(cand.get(k, 0), float(np.abs(sc[k] if sc.ndim == 1 else sc.max(0)[k])))
    accepted = {k: s for k, s in cand.items() if ratio[k] >= FRONT_RATIO}; order = sorted(accepted, key=lambda k: -accepted[k]); excl = order[:CAP]
    ica.exclude = sorted(excl); ica.apply(raw, verbose=False)
    Xc = raw.get_data(picks='eeg'); ev = np.linalg.eigvalsh(np.cov(Xc)); eff_rank = int((ev > 1e-6 * ev.max()).sum())
    log.update(ica_ok=True, ica_ncomp=ncomp, n_candidates_z=len(cand), n_candidates_accepted=len(accepted), n_ica_excl=len(excl), ica_excl=';'.join(map(str, excl)), n_cand_beyond_cap=max(0, len(accepted) - CAP),
               cand_rejected_not_frontal=len(cand) - len(accepted), eff_rank=eff_rank, status='ok')
    return (raw, log, (ica, cand, ratio)) if return_ica else (raw, log)
