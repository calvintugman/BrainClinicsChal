"""Figures + REPORT.html from saved runs. Reproduce: .venv/bin/python scripts/05_report.py"""
import glob, json, os, base64, pickle, io, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, 'scripts')
OUT = 'outputs'; FIG = f'{OUT}/figures'; DATA = f'{FIG}/plotted_data'; os.makedirs(DATA, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 100})
COL = {'E00': '#7f7f7f', 'E01-PSD': '#0072B2', 'E01': '#D55E00'}; SEEDMK = {42: 'o', 43: 's', 44: '^'}
run = {e: sorted(glob.glob(f'runs/R_rest_{e}_*'))[-1] for e in ('Remitter', 'Responder')}
cfg = {e: json.load(open(f'{r}/config.json')) for e, r in run.items()}
M = {e: pd.read_csv(f'{r}/metrics_by_seed.csv') for e, r in run.items()}; BI = {e: pd.read_csv(f'{r}/bootstrap_intervals.csv') for e, r in run.items()}
P = {e: pd.read_csv(f'{r}/predictions_PRIVATE.csv') for e, r in run.items()}
cap = {}; saved = []
def interval_text():
    t = []
    for e in ('Remitter', 'Responder'):
        B = BI[e]; d = B[(B.kind == 'diff') & (B.name == 'E01 - E01-PSD')]; x = d[(d.lo > 0) | (d.hi < 0)]
        t.append(f"{e}: E01 minus E01-PSD interval excludes zero in {len(x)} of {len(d)} seed/metric cells" + (' (' + '; '.join(f"seed {int(r.seed)} {r.metric} {r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in x.itertuples()) + ')' if len(x) else ''))
        a = B[(B.kind == 'model') & (B.metric == 'AUC')]; y = a[(a.lo > .5) | (a.hi < .5)]
        t.append(f"{e}: absolute-AUC interval excludes 0.5 in {len(y)} of {len(a)} model/seed cells" + (' (' + '; '.join(f"{r.name} seed {int(r.seed)} [{r.lo:.3f}, {r.hi:.3f}] {'above' if r.lo > .5 else 'below'}" for r in y.itertuples()) + ')' if len(y) else ''))
    return ' '.join(t)
INTERVALS = interval_text()
def save(fig, name, caption):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); plt.close(fig); cap[name] = caption; saved.append(name)

# ---- Fig 1 cohort / availability
inv = json.load(open(f'{OUT}/inventory_summary.json')); cl = pd.read_csv(f'{OUT}/cleaning_log_PRIVATE.csv'); F = pd.read_csv(f'{OUT}/features_PRIVATE.csv')
rows = [('Workbook rows (MDD-rTMS)', inv['rows']), ('Independent people', inv['people']), ('Included: session 1, EC + EO files', inv['included']),
        ('Usable EC features', int((cl[cl.cond == 'EC'].status == 'ok').sum())), ('Usable EO features', int((cl[cl.cond == 'EO'].status == 'ok').sum())),
        ('Usable EO connectivity (>=20 clean epochs)', int(F.conn_alpha_all.notna().sum())), ('rTMS patients with oddball EEG', 0)]
fig, ax = plt.subplots(figsize=(7.5, 3.4)); ax.barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color='#0072B2')
for i, r in enumerate(rows[::-1]): ax.text(r[1] + 2, i, str(r[1]), va='center')
ax.set_xlabel('Number of people (rows for first bar)'); ax.set_xlim(0, 200)
save(fig, 'fig1_cohort', f'Cohort and recording availability (rTMS treatment patients). One bar = a count of people (first bar: workbook rows, which include 8 repeat sessions). Excluded: 3 people with only a session-2 workbook row (pretreatment timing not established). Oddball EEG exists locally only for 129 healthy participants, so zero rTMS patients have task EEG. Overlapping, not mutually exclusive, subsets.')
pd.DataFrame(rows, columns=['item', 'count']).to_csv(f'{DATA}/fig1.csv', index=False)

# ---- Fig 2 model comparison (primary figure)
fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2), sharey='row')
metr = [('AUC', 'AUC', .5), ('nPPV', 'nPPV (% of base rate)', 100), ('sensitivity', 'Sensitivity', None)]; exp = []
for r, e in enumerate(('Remitter', 'Responder')):
    for c, (k, lab, ref) in enumerate(metr):
        ax = axs[r, c]
        for mi, m in enumerate(['E00', 'E01-PSD', 'E01']):
            for s in (42, 43, 44):
                v = M[e][(M[e].model == m) & (M[e].seed == s)][k].iloc[0]; b = BI[e][(BI[e].kind == 'model') & (BI[e].name == m) & (BI[e].metric == k) & (BI[e].seed == s)].iloc[0]
                y = mi + (s - 43) * 0.18; ax.plot([b.lo, b.hi], [y, y], color=COL[m], lw=1.5, alpha=.7); ax.plot(v, y, SEEDMK[s], color=COL[m], ms=5)
                exp.append(dict(endpoint=e, model=m, seed=s, metric=k, value=v, lo=b.lo, hi=b.hi))
        if ref is not None: ax.axvline(ref, color='k', ls=':', lw=1)
        ax.set_yticks(range(3)); ax.set_yticklabels(['E00 no-info', 'E01-PSD', 'E01']); ax.invert_yaxis(); ax.set_xlabel(lab)
        if c == 0: ax.set_title(f'{e} (N={cfg[e]["n"]}, prevalence {cfg[e]["prevalence"]:.2f})', loc='left', fontsize=9)
axs[0, 2].plot([], [], 'o', color='k', label='seed 42 (primary)'); axs[0, 2].plot([], [], 's', color='k', label='seed 43'); axs[0, 2].plot([], [], '^', color='k', label='seed 44'); axs[0, 2].legend(fontsize=7, loc='lower left', frameon=False)
fig.tight_layout(); pd.DataFrame(exp).to_csv(f'{DATA}/fig2.csv', index=False)
save(fig, 'fig2_model_comparison', 'Model comparison, same 164 patients and identical splits. Dot = pooled out-of-fold metric for one CV repeat (markers: seed 42/43/44, all shown, none selected); line = 95% paired patient-bootstrap interval, conditional on the saved predictions (does not include training/selection uncertainty; repeats share the same patients, so they are not independent). Dotted lines: AUC 0.5 and nPPV 100 (= base rate). nPPV = 100 x PPV / prevalence, a percent of the base rate, not accuracy. E00 calls everyone positive at its threshold (sensitivity 1, nPPV 100); its AUC below 0.5 reflects fold-to-fold prevalence variation, not information. Numeric interval check (generated from the interval file): ' + INTERVALS + ' A below-0.5 interval means worse than chance, not a useful predictor.')

# ---- Fig 3 paired differences E01 - E01-PSD
fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True); exp = []
for c, (k, lab, _) in enumerate(metr):
    ax = axs[c]
    for ei, e in enumerate(('Remitter', 'Responder')):
        for s in (42, 43, 44):
            d = BI[e][(BI[e].kind == 'diff') & (BI[e].name == 'E01 - E01-PSD') & (BI[e].metric == k) & (BI[e].seed == s)].iloc[0]; y = ei + (s - 43) * 0.18
            ax.plot([d.lo, d.hi], [y, y], color='#D55E00' if ei == 0 else '#009E73', lw=1.5); ax.plot(d.est, y, SEEDMK[s], color='#D55E00' if ei == 0 else '#009E73', ms=5)
            exp.append(dict(endpoint=e, seed=s, metric=k, diff=d.est, lo=d.lo, hi=d.hi))
    ax.axvline(0, color='k', lw=1); ax.set_xlabel(f'{lab}: E01 minus E01-PSD\n(<0 worse, >0 better)'); ax.set_yticks([0, 1]); ax.set_yticklabels(['Remitter', 'Responder']); ax.invert_yaxis()
fig.tight_layout(); pd.DataFrame(exp).to_csv(f'{DATA}/fig3.csv', index=False)
save(fig, 'fig3_added_value', 'Paired difference in out-of-fold performance when FAA, spectral entropy and the 8 connectivity summaries are added to the spectral panel (E01 vs E01-PSD), same patients and splits. Dot = difference for one CV repeat (marker = seed); line = 95% paired patient-bootstrap interval, conditional on saved predictions. Vertical line = zero improvement. Numeric interval check (generated): ' + INTERVALS + ' Improvement over the spectral-only comparator is not the same as prediction above chance: E01 absolute AUC is 0.49-0.54. This comparison cannot say which added feature contributed.')

# ---- Fig 4 prediction behaviour (Remitter, seed 42)
p = P['Remitter']; fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={'width_ratios': [1.3, 1, 1]})
rng = np.random.default_rng(0)
for yv, col in ((0, '#0072B2'), (1, '#D55E00')):
    v = p[p.y == yv]['p_E01_s42']; axs[0].scatter(yv + rng.uniform(-.12, .12, len(v)), v, s=12, color=col, alpha=.7); axs[0].hlines(v.median(), yv - .25, yv + .25, color='k')
axs[0].set_xticks([0, 1]); axs[0].set_xticklabels([f'Not remitter\n(n={int((p.y==0).sum())})', f'Remitter\n(n={int((p.y==1).sum())})']); axs[0].set_ylabel('Out-of-fold P(remission), E01'); axs[0].set_title('E01, seed 42', loc='left', fontsize=9)
for ax, m in zip(axs[1:], ['E01-PSD', 'E01']):
    d = p[f'dec_{m}_s42']; cm = np.array([[((d == 0) & (p.y == 0)).sum(), ((d == 1) & (p.y == 0)).sum()], [((d == 0) & (p.y == 1)).sum(), ((d == 1) & (p.y == 1)).sum()]])
    ax.imshow(cm, cmap='Blues'); [ax.text(j, i, cm[i, j], ha='center', va='center', color='k') for i in range(2) for j in range(2)]
    ax.set_xticks([0, 1]); ax.set_xticklabels(['called no', 'called yes']); ax.set_yticks([0, 1]); ax.set_yticklabels(['not remitter', 'remitter']); ax.set_title(f'{m}, seed 42', loc='left', fontsize=9)
fig.tight_layout(); p[['y', 'p_E01_s42', 'dec_E01_s42', 'dec_E01-PSD_s42']].to_csv(f'{DATA}/fig4_anonymous.csv', index=False)
save(fig, 'fig4_predictions', 'Remitter endpoint, seed 42 only (the prespecified first repeat). Left: each dot is one patient (anonymous), out-of-fold probability from E01 by true outcome; bar = median. The two distributions overlap almost completely. Middle/right: confusion matrices (rows = truth, columns = decision) at thresholds chosen inside training folds only (max PPV subject to sensitivity >= 50%). False positives = patients predicted to remit who did not; false negatives = remitters missed. Both models call most patients positive, so PPV stays near the 0.48 base rate.')

# ---- Fig 5 reliability
W = pickle.load(open(f'{OUT}/windows_PRIVATE.pkl', 'rb')); rel = []
def sb(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); r = spearmanr(a[m], b[m])[0]; return 2 * r / (1 + r), int(m.sum())
for cond in ('EC', 'EO'):
    for key in [(r, b) for r in ('frontal', 'central', 'posterior') for b in ('delta', 'theta', 'alpha', 'beta')]:
        for typ, ix in (('log abs power', 0), ('relative power', 1)):
            h1, h2 = [], []
            for pid, w in W.items():
                if cond in w: v = w[cond][key][ix]; n = len(v) // 2; h1.append(v[:n-1].mean()); h2.append(v[n:].mean())
                else: h1.append(np.nan); h2.append(np.nan)
            r_, n_ = sb(np.array(h1), np.array(h2)); rel.append(dict(group=typ, feature=f'{cond}_{key[0]}_{key[1]}', r_sb=r_, n=n_))
    h1, h2 = [], []
    for pid, w in W.items():
        if cond in w and 'FAA' in w[cond]: v = w[cond]['FAA']; n = len(v) // 2; h1.append(v[:n-1].mean()); h2.append(v[n:].mean())
        else: h1.append(np.nan); h2.append(np.nan)
    r_, n_ = sb(np.array(h1), np.array(h2)); rel.append(dict(group='FAA', feature=f'{cond}_FAA', r_sb=r_, n=n_))
for b in ('theta', 'alpha'):
    for s in ('all', 'frontal', 'posterior', 'between'):
        r_, n_ = sb(F[f'conn_{b}_{s}_h1'].values, F[f'conn_{b}_{s}_h2'].values); rel.append(dict(group='EO connectivity (dwPLI)', feature=f'conn_{b}_{s}', r_sb=r_, n=n_))
R = pd.DataFrame(rel); R.to_csv(f'{DATA}/fig5_reliability.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={'width_ratios': [1.7, 1]}); grp = ['log abs power', 'relative power', 'FAA', 'EO connectivity (dwPLI)']
for i, g in enumerate(grp):
    v = R[R.group == g]; axs[0].scatter(v.r_sb, np.full(len(v), i) + rng.uniform(-.15, .15, len(v)), s=18, color='#0072B2', alpha=.8); axs[0].text(1.02, i, f'median {v.r_sb.median():.2f}', va='center', fontsize=8)
axs[0].set_yticks(range(4)); axs[0].set_yticklabels(['log abs. power\n(n=24)', 'relative power\n(n=24)', 'FAA (n=2)', 'EO connectivity\n(n=8)']); axs[0].invert_yaxis(); axs[0].set_xlim(-0.3, 1.25); axs[0].axvline(0, color='k', lw=.8); axs[0].set_xlabel('Split-half reliability (Spearman-Brown corrected r)')
axs[1].hist(cl[cl.status == 'ok'].clean_s, bins=20, color='#009E73'); axs[1].set_xlabel('Retained clean EEG per recording (s)'); axs[1].set_ylabel('Recordings')
fig.tight_layout(); cl[['cond', 'clean_s', 'n_clean_win', 'n_conn_ep', 'n_bad', 'n_ica_excl']].to_csv(f'{DATA}/fig5_qc_anonymous.csv', index=False)
save(fig, 'fig5_reliability', 'Measurement reliability of the conventional features (no reconfiguration-cost measure exists in this pilot). Left: one dot = one feature; reliability = correlation across 164 patients between the first and second half of each patient\'s clean windows (connectivity: interleaved halves of 20 epochs), Spearman-Brown corrected. Halves are separated by one window to avoid overlap; patients lacking a condition or feature are excluded per feature (valid N 162-163 for spectral/FAA, 155 for connectivity; see plotted-data CSV), never imputed. Spectral features are highly reliable; FAA and especially connectivity, which uses only 10 epochs per half, are lower. IAF and entropy are not included here. Right: retained clean EEG per recording (padding, rejected windows). Split-half reliability measures stability within a session, not clinical value.')

# ---- Fig 6 cleaning example (frozen rule: lowest participant ID among included, EO)
from common import *
import importlib; f3 = importlib.import_module('03_features')
led = load_ledger().sort_values('id'); pid, ses = led.iloc[0]['id'], led.iloc[0].selected_ses
raw0, _ = load_raw(pid, ses, 'EO'); raw1, lg = f3.clean(pid, ses, 'EO')
x0 = raw0.get_data(picks='eeg') * 1e6; x1 = raw1.get_data(picks='eeg') * 1e6; t = np.arange(10 * 500) / 500
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={'width_ratios': [1.3, 1]})
for ch, off, lab in ((SCALP.index('Fp1'), 0, 'Fp1'), (SCALP.index('Fz'), 1, 'Fz'), (SCALP.index('Pz'), 2, 'Pz')):
    a = x0[ch, 20*500:30*500]; axs[0].plot(t, a - a.mean() - off * 160, color='#7f7f7f', lw=.6); axs[0].plot(t, x1[ch, 20*500:30*500] - off * 160 + 0, color='#D55E00', lw=.7); axs[0].text(-0.2, -off * 160, lab, ha='right', va='center', fontsize=8)
axs[0].set_yticks([]); axs[0].set_xlabel('Time (s), 10 s excerpt'); axs[0].set_title('Raw (demeaned, grey) vs cleaned 0.5-45 Hz + ICA (orange); scale: 160 uV spacing', fontsize=8, loc='left')
from scipy.signal import welch
for xx, c, l in ((x0 - x0.mean(1, keepdims=True), '#7f7f7f', 'raw (padding trimmed)'), (x1, '#D55E00', 'cleaned')):
    fq, pw = welch(xx, 500, nperseg=2000, axis=1); axs[1].semilogy(fq, pw.mean(0), color=c, label=l)
for (lo, hi), nm in zip(f3.BANDS.values(), f3.BANDS): axs[1].axvspan(lo, hi, color='k', alpha=.04 if nm in ('delta', 'alpha') else .0)
axs[1].set_xlim(0, 60); axs[1].set_xlabel('Frequency (Hz)'); axs[1].set_ylabel('PSD (uV$^2$/Hz), mean of 26 scalp ch.'); axs[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
save(fig, 'fig6_cleaning_example', f'Illustrative cleaning example (not evaluation evidence): the lowest-ID included patient (representative-selection rule fixed in advance, EO recording; ID not shown). Left: three channels, raw vs cleaned. Right: Welch PSD averaged over 26 scalp channels before/after (50 Hz notch, 0.5-45 Hz band-pass, interpolation of {lg["n_bad"]} channel(s), ICA excluding {lg.get("n_ica_excl", 0)} component(s)). Absolute amplitudes in microvolts; raw data carry large DC offsets (hundreds to thousands of uV). Only the analyses that ran are shown; no connectivity heatmaps, ERPs or microstate maps exist in this pilot.')

# ---- tables + index + HTML
def tab(e):
    d = M[e].copy(); d['seed'] = d.seed.astype(int); return d[['model', 'seed', 'AUC', 'PPV', 'sensitivity', 'nPPV', 'Brier', 'balanced_accuracy', 'n_pos_calls', 'TP', 'FP', 'TN', 'FN']].round(3).to_html(index=False, na_rep='undefined')
perm = json.load(open(f'{run["Remitter"]}/permutation.json'))
idx = [dict(figure=n, png=f'{FIG}/{n}.png', pdf=f'{FIG}/{n}.pdf', runs=[cfg[e]['run_id'] for e in cfg], split_fp={e: cfg[e]['split_fp'] for e in cfg}, feature_fp=cfg['Remitter']['feature_fp']) for n in saved]
json.dump(idx, open(f'{FIG}/figure_index.json', 'w'), indent=1)
img = lambda n: f'<figure><img src="data:image/png;base64,{base64.b64encode(open(f"{FIG}/{n}.png","rb").read()).decode()}" style="max-width:100%"><figcaption>{cap[n]}</figcaption></figure>'
html = f'''<!doctype html><meta charset="utf-8"><title>TDBRAIN resting-only pilot</title><style>body{{font:15px/1.5 system-ui;max-width:1000px;margin:2em auto;padding:0 1em}}figcaption{{font-size:13px;color:#333}}table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:2px 6px}}.box{{background:#fff4e5;border-left:4px solid #D55E00;padding:.5em 1em}}</style>
<h1>Resting-EEG baseline for rTMS remission (TDBRAIN) &mdash; pilot report</h1>
<div class="box"><b>Bottom line:</b> in this pilot, conventional resting EEG features (spectral power, frontal alpha asymmetry, entropy, wPLI connectivity) did <b>not</b> predict remission or response better than chance (cross-validated AUC about 0.4&ndash;0.54; label-permutation p = {perm["p_value"]:.2f} for E01 seed 42). Adding FAA/entropy/connectivity to the spectral panel did not make E01 better than chance (AUC 0.49&ndash;0.54); against the spectral-only panel it was not consistently better (see numeric check in the figure captions; one repeat, seed 44, shows a gain because E01-PSD itself scored below chance there). This is a negative baseline result in 164 patients, not evidence for or against the proposed oddball idea.</div>
<h2>Question and status</h2><p>The original question (does task-vs-rest EEG reconfiguration add predictive information?) <b>cannot be tested</b>: the downloaded rTMS patients have eyes-open and eyes-closed resting EEG only; no oddball EEG exists for them (all 162 local oddball files belong to 129 healthy participants). Experiments E02&ndash;E05 are <i>blocked_missing_rTMS_oddball</i> and untested. This report covers only the conventional resting baseline: E00 (no information: training prevalence), E01-PSD (53 spectral features: band power in 3 regions for EC/EO, individual alpha frequency, EC/EO alpha contrast) and E01 (65 features: adds FAA, spectral entropy, 8 EO theta/alpha connectivity summaries). Prediction target: Remitter (primary), Responder (secondary). These are established controls, not a novel contribution.</p>
<h2>Cohort</h2><p>164 independent patients (session 1, assuming it precedes treatment per dataset convention &mdash; <b>not individually verified</b>; no dates available), {cfg["Remitter"]["n_pos"]} remitters ({cfg["Remitter"]["prevalence"]*100:.0f}%), {cfg["Responder"]["n_pos"]} responders ({cfg["Responder"]["prevalence"]*100:.0f}%). 3 people with only a session-2 workbook row were excluded. All recordings: ~120 s at 500 Hz, 26 scalp EEG channels. Features were computed once, outcome-blind; models used 5-fold patient-level cross-validation (3 repeats), one regularized logistic regression, thresholds from training data only.</p>
{img("fig2_model_comparison")}{img("fig3_added_value")}{img("fig5_reliability")}
<h2>Results table (Remitter; pooled out-of-fold, each seed shown)</h2>{tab("Remitter")}<p>Permutation check (E01, seed-42 splits, 200 label permutations, full pipeline refit): observed AUC {perm["observed_AUC"]:.3f}, permutation mean {perm["perm_AUC_mean"]:.3f}, 95th percentile {perm["perm_AUC_q95"]:.3f}, p = {perm["p_value"]:.2f}. Even a small p would remain preliminary in this cohort size.</p>
<h2>Results table (Responder, secondary)</h2>{tab("Responder")}
{img("fig4_predictions")}{img("fig1_cohort")}{img("fig6_cleaning_example")}
<h2>Limitations</h2><ul><li>Small cohort (164): AUC intervals are wide (about &plusmn;0.1); a modest true effect could be missed.</li><li>Session-1-as-baseline is a convention, not verified timing.</li><li>26 scalp channels, linked-mastoid reference, ~2 min per condition; connectivity uses only 20 four-second epochs (9 patients lacked enough clean epochs and were imputed).</li><li>Bipolar EOG pairing (VPVA&minus;VNVB, HPHL&minus;HNHR) is an assumption from channel names. Only EC/EO and no gamma.</li><li>Bootstrap intervals are conditional on saved out-of-fold predictions; CV repeats share the same patients.</li><li>E00 pooled AUC (0.49) is below 0.5 purely from fold-to-fold prevalence variation.</li><li>Connectivity matrices were not saved and no heatmaps were produced; IAF/entropy reliability not shown.</li><li>The 104 + 60 replication recordings were not used (blinded, headers only).</li></ul>
<h2>Next decision</h2><p>Stop the conventional-baseline direction here. The oddball hypothesis remains untested; the next step is to ask whether matched pretreatment rTMS oddball data exist (the dataset authors historically described oddball data as available on request; not confirmed for these patients, and no one has been contacted). A new EC/EO feature would need its own hypothesis and a protocol amendment.</p>'''
open('REPORT.html', 'w').write(html); print('ok', saved)
