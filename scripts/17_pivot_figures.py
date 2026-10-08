"""Figures for PIVOT_FEASIBILITY_REPORT.html (label-free except the E01c table). Fixed example rule: first (by id) recording of each QC stratum with v2 status ok."""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, mne
from scipy.signal import welch
sys.path.insert(0, 'scripts')
from common import *
import importlib; f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2')
FIG = 'outputs/figures_pivot'; os.makedirs(FIG + '/plotted_data', exist_ok=True); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False}); cap = {}
def save(fig, n, c): fig.savefig(f'{FIG}/{n}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{n}.pdf', bbox_inches='tight'); plt.close(fig); cap[n] = c
# P1 groups
S = json.load(open('outputs/pivot_manifest_summary.json')); G = pd.DataFrame(S['discovery_groups']); lab = {'healthy_verified_by_workbook_indication': 'Healthy (workbook indication)', 'other_discovery_unlabeled_for_remission': 'Other discovery (no remission label)', 'other_treatment_cohort(ADHD_NF)': 'ADHD neurofeedback cohort', 'other_treatment_cohort(ADHD_NF/OCD)': 'ADHD/OCD', 'other_treatment_cohort(OCD)': 'OCD treatment cohort', 'rTMS_excluded_timing': 'rTMS, excluded (timing)', 'rTMS_labeled_164': 'rTMS labeled (analysis cohort)'}
G['label'] = G.group.map(lab); G = G.sort_values('people'); fig, ax = plt.subplots(figsize=(8, 3.4)); ax.barh(G.label, G.people, color=['#0072B2' if 'analysis' in l else '#999999' for l in G.label])
for i, r in enumerate(G.itertuples()): ax.text(r.people + 8, i, f'{r.people} people / {r.recordings} recordings', va='center', fontsize=8)
ax.set_xlim(0, 1250); ax.set_xlabel('People in TDBRAIN_Dataset_V3_1'); fig.tight_layout()
save(fig, 'figP1_participant_groups', f'Participant groups in the discovery download ({S["n_discovery_people"]} people), kept separate. Only the blue group has remission labels. Other discovery people are NOT assumed healthy and have no remission label; {S["reference_pool_for_EC_EO_extension_candidates"]} non-rTMS people have both EC and EO recordings (candidate reference pool, not yet a validated one). Replication sets (104 + 60 recordings) have zero ID overlap with discovery or the workbook and are not shown or used. Fingerprint check: {S["hash_check"]["identical_content_across_sets"]} identical files across sets; 2 healthy people have identical ses-1/ses-2 EC files (not in the labeled cohort).')
# QC examples
Q = pd.read_csv('outputs/pivot_qc_sample_PRIVATE.csv'); led = load_ledger().set_index('id'); ex = []
for st in ('v1_ica_hit_max_iter', 'v1_cap3', 'interpolated'):
    q = Q[(Q.stratum == st) & (Q.v2_status == 'ok')].sort_values('id'); ex.append(q.iloc[0])
fig, axs = plt.subplots(3, 3, figsize=(12, 8)); fig2, axs2 = plt.subplots(3, 4, figsize=(11, 7)); rows = []
for k, q in enumerate(ex):
    ses = led.loc[q.id, 'selected_ses']; raw0, pad = load_raw(q.id, ses, q.cond); r1, l1 = f3.clean(q.id, ses, q.cond); r2, l2, (ica, cand, ratio) = cv2.clean(q.id, ses, q.cond, return_ica=True)
    x0 = raw0.get_data(picks='eeg') * 1e6; x1 = r1.get_data(picks='eeg') * 1e6; x2 = r2.get_data(picks='eeg') * 1e6; t = np.arange(5000) / 500
    for j, (ch, nm) in enumerate(((SCALP.index('Fp1'), 'Fp1'), (SCALP.index('Fz'), 'Fz'))):
        a = x0[ch, 10000:15000]; axs[k, 0].plot(t, a - a.mean() - j * 200, color='#999', lw=.5); axs[k, 0].plot(t, x1[ch, 10000:15000] - j * 200, color='#0072B2', lw=.6); axs[k, 0].plot(t, x2[ch, 10000:15000] - j * 200, color='#D55E00', lw=.6, ls='-'); axs[k, 0].text(-.1, -j * 200, nm, ha='right', fontsize=7)
    axs[k, 0].set_yticks([]); axs[k, 0].set_xlabel('Time (s), 10 s excerpt, 200 uV spacing'); axs[k, 0].set_title(f'Example {k+1}: {q.stratum.replace("v1_", "")} (grey raw, blue v1, orange v2)', fontsize=8, loc='left')
    for xx, c, l in ((x0 - x0.mean(1, keepdims=True), '#999', 'raw'), (x1, '#0072B2', 'v1'), (x2, '#D55E00', 'v2')):
        f, p = welch(xx, 500, nperseg=2000, axis=1); axs[k, 1].semilogy(f, p.mean(0), color=c, label=l)
    axs[k, 1].set_xlim(0, 60); axs[k, 1].set_xlabel('Frequency (Hz)'); axs[k, 1].set_ylabel('PSD (uV$^2$/Hz)'); axs[k, 1].legend(frameon=False, fontsize=7)
    ev = {nm: np.sort(np.linalg.eigvalsh(np.cov(xx)))[::-1] for nm, xx in (('v1', x1), ('v2', x2))}
    for nm, c in (('v1', '#0072B2'), ('v2', '#D55E00')): axs[k, 2].semilogy(ev[nm] / ev[nm][0], color=c, label=nm)
    axs[k, 2].set_xlabel('Eigenvalue rank (26-ch covariance)'); axs[k, 2].set_ylabel('Relative eigenvalue'); axs[k, 2].legend(frameon=False, fontsize=7)
    excl = [int(s) for s in str(l2['ica_excl']).split(';') if s != '']; comp = ica.get_components()
    for j in range(4):
        a = axs2[k, j]; a.set_xticks([]); a.set_yticks([])
        if j < len(excl):
            mne.viz.plot_topomap(comp[:, excl[j]], r2.info, axes=a, show=False, sphere=0.1); a.set_title(f'IC{excl[j]}  |r|={cand[excl[j]]:.2f}\nfrontal ratio {ratio[excl[j]]:.1f}', fontsize=7)
        else: a.axis('off')
    axs2[k, 0].set_ylabel(f'Example {k+1}', fontsize=8)
    rows.append(dict(example=k + 1, stratum=q.stratum, v1_excluded_by_index=q.v1_excl, v2_excluded_by_evidence=q.v2_excl, v1_iter=q.v1_ica_iter, v2_iter=q.v2_iter, v2_retried=q.v2_retried, n_interp=q.n_interp, v1_eff_rank=int((np.linalg.eigvalsh(np.cov(x1)) > 1e-6 * np.linalg.eigvalsh(np.cov(x1)).max()).sum()), v2_eff_rank=q.v2_eff_rank))
fig.tight_layout(); fig2.tight_layout(); pd.DataFrame(rows).to_csv(f'{FIG}/plotted_data/qc_examples_anonymous.csv', index=False)
save(fig, 'figP2_cleaning_examples', 'Cleaning comparison for three fixed-rule examples (first ID in each outcome-blind QC stratum with v2 status ok; IDs not shown). Left: raw (padding trimmed, demeaned), v1 and v2 cleaned traces (uV). Middle: mean scalp PSD. Right: relative eigenvalue spectrum of the cleaned 26-channel covariance (steep falloff = rank loss from ICA removal/interpolation, which affects covariance-ratio features). Neither pipeline is ground truth; agreement between them does not prove artifact removal.')
save(fig2, 'figP3_removed_components', 'Evidence for components removed by the corrected (v2) pipeline in the same three examples: scalp topography of each removed independent component (arbitrary sign), maximum absolute EOG correlation |r| (candidates are components whose correlation is an outlier, z>3, among all components of that recording) and frontal-dominance ratio (mean |weight| at Fp1/Fp2/F7/F8 over all channels; rule: outlier and ratio>=1.5, ranked by |r|, max 3). The frontal-ratio rule is crude: for example the third component of Example 2 has a diffuse, non-blink-like map yet passes, so these exclusions need visual review before any claim of artifact removal. v1 removed components by index order (see table in the report).')
# P4 v1 vs v2 summary
v1 = pd.read_csv('outputs/cleaning_log_PRIVATE.csv'); v2 = pd.read_csv('outputs/v2/cleaning_log_v2_PRIVATE.csv'); F1 = pd.read_csv('outputs/features_PRIVATE.csv'); F2 = pd.read_csv('outputs/v2/features_v2_PRIVATE.csv')
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.6))
b = np.arange(0, 5); axs[0].hist([v1.n_ica_excl.dropna(), v2.n_ica_excl.dropna()], bins=b - .5, label=['v1 (index order, cap 3, incl. ECG proxy)', 'v2 (evidence, cap 3)'], color=['#0072B2', '#D55E00']); axs[0].set_xlabel('ICA components removed per recording'); axs[0].set_ylabel('Recordings'); axs[0].set_ylim(0, 230); axs[0].legend(frameon=False, fontsize=7, loc='upper left')
st = pd.DataFrame({'v1': v1.status.value_counts(), 'v2': v2.status.value_counts()}).fillna(0).astype(int); st.T.plot.bar(stacked=True, ax=axs[1], color=['#D55E00', '#7f7f7f', '#009E73', '#E69F00']); axs[1].set_ylabel('Recordings'); axs[1].set_xticklabels(['v1', 'v2'], rotation=0); axs[1].set_ylim(0, 420); axs[1].legend(fontsize=7, frameon=False, loc='upper right', ncol=2)
cols = [c for c in F1.columns if c != 'id' and not c.endswith(('_h1', '_h2'))]; R = []
for c in cols:
    m = F1[c].notna() & F2[c].notna()
    if m.sum() > 20: R.append(dict(feature=c, r=np.corrcoef(F1.loc[m, c], F2.loc[m, c])[0, 1], n=int(m.sum()), group='FAA' if 'FAA' in c else 'connectivity' if c.startswith('conn_') else 'entropy/IAF' if c.endswith(('entropy', 'IAF')) else 'power/contrast'))
R = pd.DataFrame(R); R.to_csv(f'{FIG}/plotted_data/v1_v2_feature_agreement.csv', index=False); g = ['power/contrast', 'entropy/IAF', 'connectivity', 'FAA']; rg = np.random.default_rng(0)
for i, gg in enumerate(g): v = R[R.group == gg]; axs[2].scatter(v.r, np.full(len(v), i) + rg.uniform(-.15, .15, len(v)), s=12, color='#0072B2', alpha=.7); axs[2].text(1.005, i, f'median {v.r.median():.2f}\nmin {v.r.min():.2f}', va='center', fontsize=7)
axs[2].set_yticks(range(4)); axs[2].set_yticklabels([f'{x} ({(R.group==x).sum()})' for x in g]); axs[2].invert_yaxis(); axs[2].set_xlim(.6, 1.08); axs[2].set_xlabel('Across-patient correlation, v1 vs v2 cleaning')
fig.tight_layout(); save(fig, 'figP4_v1_v2_summary', f'Outcome-blind comparison of original (v1) and corrected (v2) cleaning over all 328 included recordings. Left: components removed. Middle: recording status (v2: 17 ICA fits that hit the iteration limit were retried once; 8 still failed and are missing, not silently kept). Right: per-feature agreement across 164 patients between the two cleaning views of the same recordings (dot = one feature). Power features agree almost perfectly; FAA and connectivity depend more on cleaning (lowest r {R.r.min():.2f}). No outcome labels were used here.')
json.dump(cap, open(f'{FIG}/captions.json', 'w'), indent=1); print('ok', list(cap))
