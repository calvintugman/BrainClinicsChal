"""Step 1: (a) which patients/conditions became missing under v2, (b) conditional sensitivity of the E01->E01c AUC change to availability change (descriptive, on saved predictions),
(c) expanded component evidence (time course, spectrum, eye alignment) for the QC sample. clean_v2 is NOT modified; nothing here changes cleaning."""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, mne
from scipy.signal import welch, find_peaks
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'scripts')
from common import *
import importlib; f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2')
FIG = 'outputs/figures_dynamix'; os.makedirs(FIG + '/plotted_data', exist_ok=True); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}); cap = {}
def save(fig, n, c): fig.savefig(f'{FIG}/{n}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{n}.pdf', bbox_inches='tight'); plt.close(fig); cap[n] = c
# ---------- (a) availability changes
v1 = pd.read_csv('outputs/cleaning_log_PRIVATE.csv'); v2 = pd.read_csv('outputs/v2/cleaning_log_v2_PRIVATE.csv'); F1 = pd.read_csv('outputs/features_PRIVATE.csv'); F2 = pd.read_csv('outputs/v2/features_v2_PRIVATE.csv')
m = v1[['id', 'cond', 'status']].merge(v2[['id', 'cond', 'status']], on=['id', 'cond'], suffixes=('_v1', '_v2')); m['transition'] = m.status_v1 + ' -> ' + m.status_v2
T = m.transition.value_counts().rename_axis('transition').reset_index(name='recordings'); T.to_csv(f'{FIG}/plotted_data/status_transitions.csv', index=False)
# E01 feature columns
sys.path.insert(0, 'scripts'); ev = importlib.import_module('10_batch2_eval'); os.environ.pop('FEATURES_PATH', None); D1, y, e01, E11, cov = ev.load_all('Remitter'); os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; D2, y2, *_ = ev.load_all('Remitter'); os.environ.pop('FEATURES_PATH', None)
assert (D1.id == D2.id).all() and (y == y2).all()
na1 = D1[e01].isna().values; na2 = D2[e01].isna().values; changed = (na1 != na2).any(1); newly_missing = (~na1 & na2).any(1); newly_avail = (na1 & ~na2).any(1)
# patient-level private table
cond_chg = {}
for c in ('EC', 'EO'):
    s = m[m.cond == c].set_index('id').reindex(D1.id); cond_chg[c] = (s.status_v1 != s.status_v2).values
AV = pd.DataFrame({'id': D1.id, 'availability_changed': changed, 'newly_missing_any': newly_missing, 'newly_available_any': newly_avail, 'EC_status_changed': cond_chg['EC'], 'EO_status_changed': cond_chg['EO'], 'n_e01_na_v1': na1.sum(1), 'n_e01_na_v2': na2.sum(1)})
AV.to_csv('outputs/v2/availability_changes_PRIVATE.csv', index=False)
summ = dict(recordings_status_transitions=T.set_index('transition').recordings.to_dict(), patients_total=len(AV), patients_availability_changed=int(changed.sum()), patients_newly_missing_any_feature=int(newly_missing.sum()), patients_newly_available_any_feature=int(newly_avail.sum()),
            patients_EC_status_changed=int(cond_chg['EC'].sum()), patients_EO_status_changed=int(cond_chg['EO'].sum()), changed_patients_remitters=int(y[changed].sum()))
# ---------- (b) conditional sensitivity analysis on saved predictions
Z = (D2[e01] - D1[e01]).abs() / D1[e01].std(); mag = Z.mean(1).values; nz = mag[mag > 0]; thr = np.median(nz); tert = np.where(mag == 0, 'low', np.where(mag <= thr, 'mid', 'high'))       # label-free magnitude of feature change (mean |dz|)
rng = np.random.default_rng(1); NB = 2000
def sens(endpoint):
    P0 = pd.read_csv(sorted(glob.glob(f'runs/R_rest_{endpoint}_*'))[-1] + '/predictions_PRIVATE.csv'); P1 = pd.read_csv(sorted(glob.glob(f'runs/PIV_E01c_{endpoint}_*'))[-1] + '/predictions_PRIVATE.csv'); assert (P0.id == D1.id).all() and (P1.id == D1.id).all(); yy = P0.y.values
    groups = {'all patients': np.ones(len(yy), bool), 'availability unchanged': ~changed, 'availability changed': changed, 'features identical (zero change)': tert == 'low', 'change below median of changed': tert == 'mid', 'change above median of changed': tert == 'high'}
    rows = []
    for g, msk in groups.items():
        idx = np.flatnonzero(msk); a0 = [roc_auc_score(yy[idx], P0[f'p_E01_s{s}'].values[idx]) if len(set(yy[idx])) > 1 else np.nan for s in (42, 43, 44)]; a1 = [roc_auc_score(yy[idx], P1[f'p_E01c_s{s}'].values[idx]) if len(set(yy[idx])) > 1 else np.nan for s in (42, 43, 44)]
        d = []
        for b in range(NB):
            ix = rng.choice(idx, len(idx)); 
            if len(set(yy[ix])) < 2: continue
            d.append(np.mean([roc_auc_score(yy[ix], P1[f'p_E01c_s{s}'].values[ix]) - roc_auc_score(yy[ix], P0[f'p_E01_s{s}'].values[ix]) for s in (42, 43, 44)]))
        rows.append(dict(endpoint=endpoint, group=g, n=len(idx), n_pos=int(yy[idx].sum()), mean_auc_E01=np.mean(a0), mean_auc_E01c=np.mean(a1), mean_diff=np.mean(a1) - np.mean(a0), diff_lo=np.quantile(d, .025), diff_hi=np.quantile(d, .975)))
    return pd.DataFrame(rows)
SENS = pd.concat([sens(e) for e in ('Remitter', 'Responder')]); SENS.to_csv(f'{FIG}/plotted_data/conditional_sensitivity_E01_vs_E01c.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, e in zip(axs, ('Remitter', 'Responder')):
    s = SENS[SENS.endpoint == e].reset_index(drop=True)
    for i, r in s.iterrows(): ax.plot([r.diff_lo, r.diff_hi], [i, i], color='#0072B2', lw=2, alpha=.5); ax.plot(r.mean_diff, i, 'D', color='#0072B2', mec='k')
    ax.axvline(0, color='k'); ax.set_yticks(range(len(s))); ax.set_yticklabels([f'{g} (n={n})' for g, n in zip(s.group, s.n)], fontsize=7); ax.invert_yaxis(); ax.set_xlabel(f'{e}: mean AUC change E01c - E01'); ax.set_title(e, loc='left', fontsize=9)
fig.tight_layout(); save(fig, 'figD1_conditional_sensitivity', 'CONDITIONAL SENSITIVITY ANALYSIS on saved out-of-fold predictions (not causal attribution, not a retrained model): mean-over-repeats AUC change from E01 (original cleaning) to E01c (corrected) within patient subsets defined label-free: whether any E01 feature changed between available/missing, and three groups by the mean absolute standardized feature change (zero / below / above the median of the non-zero changes). Diamond = difference; line = 95% bootstrap interval resampling patients within the subset (conditional on saved predictions). Small subsets give wide intervals; subgroup AUCs pool predictions from models trained on all patients, so differences cannot be attributed to the availability change itself.')
json.dump(summ, open('outputs/v2/availability_summary.json', 'w'), indent=1); print(json.dumps(summ, indent=1)); print(SENS.round(3).to_string(index=False))
# ---------- (c) component evidence
def pre_ica(pid, ses, cond):
    raw, pad = load_raw(pid, ses, cond); raw.set_montage('standard_1020', match_case=False, on_missing='ignore'); raw.notch_filter(50., notch_widths=2., picks=['eeg', 'eog', 'ecg'], verbose=False)
    raw.filter(0.5, 45., l_trans_bandwidth=0.5, h_trans_bandwidth=4., picks=['eeg', 'eog', 'ecg'], verbose=False); x = raw.get_data(picks='eeg') * 1e6; nb = x.shape[1] // 500; bs = x[:, :nb*500].reshape(26, nb, 500).std(2); med = np.median(bs, 1); g = np.median(med)
    bad = [SCALP[i] for i in range(26) if med[i] > 3*g or med[i] < g/4]; raw.info['bads'] = bad
    if bad: raw.interpolate_bads(reset_bads=True, verbose=False)
    rica = raw.copy().filter(1., None, picks=['eeg', 'eog', 'ecg'], verbose=False); return raw, rica
def evidence(pid, ses, cond):
    r2, l2, extra = cv2.clean(pid, ses, cond, return_ica=True)
    if r2 is None or extra is None: return None
    ica, cand, ratio = extra
    raw, rica = pre_ica(pid, ses, cond); chk = raw.copy(); ica.apply(chk, verbose=False); same = bool(np.allclose(chk.get_data(picks='eeg'), r2.get_data(picks='eeg'), atol=1e-12))
    S = ica.get_sources(rica).get_data(); v = (rica.get_data(picks=['VPVA']) - rica.get_data(picks=['VNVB']))[0]; h = (rica.get_data(picks=['HPHL']) - rica.get_data(picks=['HNHR']))[0]
    zv = (v - np.median(v)) / (1.4826 * np.median(np.abs(v - np.median(v)))); pk, _ = find_peaks(zv, height=5, distance=500)                     # my blink detector (documented, not MNE's)
    out = []
    for k in [int(s) for s in str(l2['ica_excl']).split(';') if s != '']:
        s = S[k]; rv = np.corrcoef(s, v)[0, 1]; rh = np.corrcoef(s, h)[0, 1]; ep = [s[p - 500:p + 500] for p in pk if p >= 500 and p + 500 <= len(s)]; ev = [v[p - 500:p + 500] for p in pk if p >= 500 and p + 500 <= len(s)]
        avg = np.mean(ep, 0) if ep else np.full(1000, np.nan); base = np.std(avg[:250]) if ep else np.nan; peak = np.max(np.abs(avg[400:700])) if ep else np.nan; ratio_b = peak / base if ep and base > 0 else np.nan
        f, p_ = welch(s, 500, nperseg=2000); low = p_[(f >= .5) & (f < 4)].sum() / p_[(f >= .5) & (f < 45)].sum()
        out.append(dict(id=pid, cond=cond, ic=k, abs_r_VEOG=abs(rv), abs_r_HEOG=abs(rh), n_blinks=len(ep), blink_locked_peak_to_baseline_SD=ratio_b, low_freq_fraction=low, mne_corr_score=cand.get(k), frontal_ratio=ratio[k],
                        unresolved=bool(not (ratio_b >= 3) and abs(rh) < 0.3), pipeline_reproduction_exact=same, _s=s, _v=v, _h=h, _pk=pk, _avg=avg, _evavg=np.mean(ev, 0) if ep else None))
    return out
Q = pd.read_csv('outputs/pivot_qc_sample_PRIVATE.csv'); led = load_ledger().set_index('id'); allrows = []; ex = []
from joblib import Parallel, delayed
res = Parallel(n_jobs=3)(delayed(evidence)(r.id, led.loc[r.id, 'selected_ses'], r.cond) for r in Q.itertuples()); 
for r, o in zip(Q.itertuples(), res):
    if o: allrows += o
E = pd.DataFrame(allrows); pub = E.drop(columns=[c for c in E.columns if c.startswith('_') or c == 'id']); pub.to_csv(f'{FIG}/plotted_data/component_evidence_qc_sample.csv', index=False)
summ2 = dict(qc_recordings_with_v2_ok=int(E[['id', 'cond']].drop_duplicates().shape[0]), removed_components=len(E), pipeline_reproduction_exact_all=bool(E.pipeline_reproduction_exact.all()), unresolved_components=int(E.unresolved.sum()), blink_evidence_ratio_ge3=int((E.blink_locked_peak_to_baseline_SD >= 3).sum()), abs_r_HEOG_ge_0_3=int((E.abs_r_HEOG >= .3).sum()),
             rule='unresolved = blink-locked peak/baseline SD < 3 AND |r_HEOG| < 0.3 (documentation only; cleaning not altered)')
json.dump(summ2, open('outputs/v2/component_evidence_summary.json', 'w'), indent=1); print(json.dumps(summ2, indent=1))
# 3 fixed-rule examples (same as figP2: first id per stratum with v2 ok)
exq = []
for st in ('v1_ica_hit_max_iter', 'v1_cap3', 'interpolated'):
    q = Q[(Q.stratum == st) & (Q.v2_status == 'ok')].sort_values('id').iloc[0]; exq.append((q.id, q.cond))
for n, (pid, cond) in enumerate(exq, 1):
    sub = E[(E.id == pid) & (E.cond == cond)].reset_index(drop=True); k = len(sub)
    fig, axs = plt.subplots(max(k, 1), 3, figsize=(12, 2.6 * max(k, 1)), squeeze=False)
    for i, r in sub.iterrows():
        t = np.arange(5000) / 500; a = axs[i, 0]; a.plot(t, r._s[10000:15000] / np.std(r._s), color='#D55E00', lw=.6, label=f'IC{r.ic}'); a.plot(t, r._v[10000:15000] / np.std(r._v) - 6, color='#7f7f7f', lw=.6, label='VEOG'); a.plot(t, r._h[10000:15000] / np.std(r._h) - 12, color='#56B4E9', lw=.6, label='HEOG')
        for p in r._pk: 
            if 10000 <= p < 15000: a.axvline((p - 10000) / 500, color='k', lw=.4, alpha=.4)
        a.set_yticks([]); a.set_xlabel('Time (s), 10 s; lines = detected blinks'); a.legend(frameon=False, fontsize=6, ncol=3, loc='upper right'); a.set_title(f'IC{r.ic}: |r| VEOG {r.abs_r_VEOG:.2f}, HEOG {r.abs_r_HEOG:.2f}', fontsize=8, loc='left')
        f, p_ = welch(r._s, 500, nperseg=2000); axs[i, 1].semilogy(f, p_, color='#D55E00'); axs[i, 1].set_xlim(0, 45); axs[i, 1].set_xlabel('Frequency (Hz)'); axs[i, 1].set_ylabel('Source PSD (a.u.)'); axs[i, 1].set_title(f'low-freq (0.5-4 Hz) fraction {r.low_freq_fraction:.2f}', fontsize=8, loc='left')
        tt = np.arange(-500, 500) / 500; axs[i, 2].plot(tt, r._avg / np.std(r._s), color='#D55E00', label=f'IC{r.ic}'); 
        if r._evavg is not None: axs[i, 2].plot(tt, r._evavg / np.std(r._v), color='#7f7f7f', label='VEOG')
        axs[i, 2].axvline(0, color='k', lw=.5); axs[i, 2].set_xlabel(f'Time from blink peak (s); n={r.n_blinks}'); axs[i, 2].set_ylabel('Blink-locked average (SD units of whole record)'); axs[i, 2].legend(frameon=False, fontsize=6)
        axs[i, 2].set_title(f'peak/baseline SD {r.blink_locked_peak_to_baseline_SD:.1f}' + ('  UNRESOLVED' if r.unresolved else ''), fontsize=8, loc='left')
    fig.tight_layout(); save(fig, f'figD2_component_evidence_example{n}', f'Example {n} (same fixed-rule recording as figure P2; ID not shown): for each component removed by v2 - time course with VEOG/HEOG (bipolar) and detected blinks; source spectrum; blink-locked average of the component and VEOG. This is documentation only; cleaning was not changed. Components marked UNRESOLVED have no blink-locked response (peak/baseline SD < 3) and weak HEOG correlation (|r| < 0.3), so their removal rests on correlation outlier + frontal topography alone and remains unverified.')
json.dump(cap, open(f'{FIG}/captions_diag.json', 'w'), indent=1)
