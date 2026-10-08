"""Batch-2 analysis: bootstrap, screening rule, reliability, gain sensitivity, figures, BATCH2_REPORT.html. Run from first_tests/."""
import sys, os, glob, json, base64
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from importlib import import_module
OUT = 'outputs'; FIG = f'{OUT}/figures_batch2'; DATA = f'{FIG}/plotted_data'; os.makedirs(DATA, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
B2 = {e: sorted(glob.glob(f'runs/B2_{e}_*'))[-1] for e in ('Remitter', 'Responder')}; B1 = {e: sorted(glob.glob(f'runs/R_rest_{e}_*'))[-1] for e in ('Remitter', 'Responder')}
CH = ['E10', 'E11-only', 'E11-add', 'E12-add', 'E13-power', 'E13-relative', 'E13-add']; REF = ['E00', 'E01-PSD', 'E01']; SEEDS = [42, 43, 44]
COL = {'E00': '#7f7f7f', 'E01-PSD': '#999999', 'E01': '#000000', 'E10': '#D55E00', 'E11-only': '#0072B2', 'E11-add': '#56B4E9', 'E12-add': '#009E73', 'E13-power': '#CC79A7', 'E13-relative': '#E69F00', 'E13-add': '#8c564b'}
MK = {42: 'o', 43: 's', 44: '^'}
cfgs = {e: json.load(open(f'{r}/config.json')) for e, r in B2.items()}
cap = {}; saved = []
def save(fig, name, caption):
    fig.savefig(f'{FIG}/{name}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight'); plt.close(fig); cap[name] = caption; saved.append(name)

# ---- predictions: all models in one dict
PRED = {}; Y = {}
for e in B2:
    p1 = pd.read_csv(f'{B1[e]}/predictions_PRIVATE.csv'); p2 = pd.read_csv(f'{B2[e]}/predictions_PRIVATE.csv'); assert (p1.id == p2.id).all() and (p1.y == p2.y).all(); Y[e] = p1.y.values; d = {}
    for s in SEEDS:
        for m in REF: d[(m, s)] = (p1[f'p_{m}_s{s}'].values, p1[f'dec_{m}_s{s}'].values)
        for m in CH: d[(m, s)] = (p2[f'p_{m}_s{s}'].values, p2[f'dec_{m}_s{s}'].values)
    PRED[e] = d
# ---- point metrics for all
def pm(y, p, d):
    tp = ((d == 1) & (y == 1)).sum(); fp = ((d == 1) & (y == 0)).sum(); tn = ((d == 0) & (y == 0)).sum(); fn = ((d == 0) & (y == 1)).sum()
    ppv = tp / (tp + fp) if tp + fp else np.nan; sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(AUC=roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5, PPV=ppv, sensitivity=sens, nPPV=100 * ppv / y.mean() if ppv == ppv else np.nan, Brier=float(np.mean((p - y) ** 2)), balanced_accuracy=(sens + spec) / 2, n_pos_calls=int(tp + fp), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
rows = [dict(endpoint=e, model=m, seed=s, **pm(Y[e], *PRED[e][(m, s)])) for e in PRED for m in REF + CH for s in SEEDS]; ALL = pd.DataFrame(rows); ALL.to_csv(f'{DATA}/all_metrics.csv', index=False)
# ---- bootstrap: same draws as baseline (rng seed 1), joint across repeats
rng = np.random.default_rng(1); NB = 2000; draws = rng.integers(0, 164, size=(NB, 164)); BOOT = {}
for e in PRED:
    y = Y[e]; A = {m: np.full((NB, 3), np.nan) for m in REF + CH}; NP = {m: np.full((NB, 3), np.nan) for m in REF + CH}
    for b, ix in enumerate(draws):
        if len(np.unique(y[ix])) < 2: continue
        for m in REF + CH:
            for si, s in enumerate(SEEDS):
                p, d = PRED[e][(m, s)]; A[m][b, si] = roc_auc_score(y[ix], p[ix]) if len(np.unique(p[ix])) > 1 else 0.5
                tp = ((d[ix] == 1) & (y[ix] == 1)).sum(); npos = (d[ix] == 1).sum(); NP[m][b, si] = 100 * (tp / npos) / y[ix].mean() if npos else np.nan
    BOOT[e] = (A, NP)
q = lambda v: (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975)))
S = []   # screening + intervals
for e in PRED:
    A, NP = BOOT[e]; sub = ALL[ALL.endpoint == e]
    for m in REF + CH:
        aucs = sub[sub.model == m].set_index('seed').AUC; gains = [aucs[s] - sub[(sub.model == 'E01') & (sub.seed == s)].AUC.iloc[0] for s in SEEDS]
        mean_auc = A[m].mean(1); mean_gain = (A[m] - A['E01']).mean(1)
        S.append(dict(endpoint=e, model=m, auc_s42=aucs[42], auc_s43=aucs[43], auc_s44=aucs[44], mean_auc=aucs.mean(), mean_auc_lo=q(mean_auc)[0], mean_auc_hi=q(mean_auc)[1], mean_gain_vs_E01=float(np.mean(gains)), gain_lo=q(mean_gain)[0], gain_hi=q(mean_gain)[1],
                      gain_s42=gains[0], gain_s43=gains[1], gain_s44=gains[2], all_gains_positive=bool(all(g > 0 for g in gains)),
                      qualifies=bool(e == 'Remitter' and m in CH and aucs.mean() >= 0.60 and np.mean(gains) >= 0.05 and all(g > 0 for g in gains))))
SCR = pd.DataFrame(S); SCR.to_csv(f'{DATA}/screening_and_intervals.csv', index=False)
extra = [dict(endpoint=e, contrast='E13-relative minus E13-power', mean_diff=float((ALL[(ALL.endpoint == e) & (ALL.model == 'E13-relative')].AUC.mean() - ALL[(ALL.endpoint == e) & (ALL.model == 'E13-power')].AUC.mean())),
              lo=q((BOOT[e][0]['E13-relative'] - BOOT[e][0]['E13-power']).mean(1))[0], hi=q((BOOT[e][0]['E13-relative'] - BOOT[e][0]['E13-power']).mean(1))[1]) for e in PRED]
pd.DataFrame(extra).to_csv(f'{DATA}/e13_relative_vs_power.csv', index=False)
# per-seed bootstrap intervals for nPPV and AUC (model level)
BI = []
for e in PRED:
    A, NP = BOOT[e]
    for m in REF + CH:
        for si, s in enumerate(SEEDS):
            BI.append(dict(endpoint=e, model=m, seed=s, auc_lo=q(A[m][:, si])[0], auc_hi=q(A[m][:, si])[1], nppv_lo=q(NP[m][:, si])[0], nppv_hi=q(NP[m][:, si])[1], auc_gain_lo=q(A[m][:, si] - A['E01'][:, si])[0], auc_gain_hi=q(A[m][:, si] - A['E01'][:, si])[1]))
BI = pd.DataFrame(BI); BI.to_csv(f'{DATA}/bootstrap_intervals_by_seed.csv', index=False)

def stat_text(e):
    s = SCR[(SCR.endpoint == e) & SCR.model.isin(CH)]; ex = s[(s.gain_lo > 0) | (s.gain_hi < 0)]; ab = s[(s.mean_auc_lo > .5) | (s.mean_auc_hi < .5)]
    t = f"{e}, mean-over-repeats AUC gain over E01: interval excludes zero for {len(ex)} of {len(s)} challengers" + (' (' + '; '.join(f"{r.model} {r.mean_gain_vs_E01:+.3f} [{r.gain_lo:+.3f}, {r.gain_hi:+.3f}]" for r in ex.itertuples()) + ')' if len(ex) else '') + '. '
    t += f"Mean-over-repeats absolute AUC interval excludes 0.5 for {len(ab)} of {len(s)} challengers" + (' (' + '; '.join(f"{r.model} {r.mean_auc:.3f} [{r.mean_auc_lo:.3f}, {r.mean_auc_hi:.3f}] {'above' if r.mean_auc_lo > .5 else 'below'} 0.5" for r in ab.itertuples()) + ')' if len(ab) else '') + '.'
    return t
TXT = {e: stat_text(e) for e in PRED}

# ---- Fig B1 overall comparison
order = REF + CH; fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, e in zip(axs, ('Remitter', 'Responder')):
    for i, m in enumerate(order):
        for s in SEEDS:
            v = ALL[(ALL.endpoint == e) & (ALL.model == m) & (ALL.seed == s)].AUC.iloc[0]; ax.plot(v, i + (s - 43) * .15, MK[s], color=COL[m], ms=4, alpha=.8)
        r = SCR[(SCR.endpoint == e) & (SCR.model == m)].iloc[0]; ax.plot([r.mean_auc_lo, r.mean_auc_hi], [i, i], color=COL[m], lw=2, alpha=.5); ax.plot(r.mean_auc, i, 'D', color=COL[m], ms=7, mec='k')
    ax.axvline(.5, color='k', ls=':'); ax.axvline(.6, color='#999', ls='--', lw=.8); ax.set_xlabel('Outer-fold AUC'); ax.set_title(f'{e} (N=164, prevalence {Y[e].mean():.2f})', loc='left', fontsize=9)
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
save(fig, 'figB1_all_vs_E01', 'All seven challengers with the three baseline models, same 164 patients and identical splits. Small markers = pooled out-of-fold AUC of each CV repeat (circle 42, square 43, triangle 44; all shown); diamond + line = mean over the three repeats with a 95% paired patient-bootstrap interval (patients resampled jointly across repeats; conditional on saved predictions, not corrected for choosing among 7 methods). Dotted line AUC 0.5; dashed line 0.60 (triage threshold, not a clinical standard). ' + TXT['Remitter'] + ' ' + TXT['Responder'])
# ---- Fig B2 paired gains vs E01 + E13 control
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1, 1, .9]})
for ax, e in zip(axs[:2], ('Remitter', 'Responder')):
    for i, m in enumerate(CH):
        r = SCR[(SCR.endpoint == e) & (SCR.model == m)].iloc[0]
        for s, g in zip(SEEDS, (r.gain_s42, r.gain_s43, r.gain_s44)): ax.plot(g, i + (s - 43) * .15, MK[s], color=COL[m], ms=4, alpha=.8)
        ax.plot([r.gain_lo, r.gain_hi], [i, i], color=COL[m], lw=2, alpha=.5); ax.plot(r.mean_gain_vs_E01, i, 'D', color=COL[m], ms=7, mec='k')
    ax.axvline(0, color='k'); ax.axvline(.05, color='#999', ls='--', lw=.8); ax.set_yticks(range(len(CH))); ax.set_yticklabels(CH); ax.invert_yaxis(); ax.set_xlabel('AUC change vs E01\n(<0 worse, >0 better)'); ax.set_title(e, loc='left', fontsize=9)
ax = axs[2]
for j, e in enumerate(('Remitter', 'Responder')):
    r = [x for x in extra if x['endpoint'] == e][0]; ax.plot([r['lo'], r['hi']], [j, j], color='#E69F00', lw=2, alpha=.6); ax.plot(r['mean_diff'], j, 'D', color='#E69F00', mec='k', ms=7)
ax.axvline(0, color='k'); ax.set_yticks([0, 1]); ax.set_yticklabels(['Remitter', 'Responder']); ax.invert_yaxis(); ax.set_xlabel('E13-relative minus E13-power\n(mean AUC; <0 = covariance summaries worse)'); ax.set_title('Specific control', loc='left', fontsize=9)
fig.tight_layout()
save(fig, 'figB2_paired_changes', 'Paired AUC change against E01 (left, middle; same patients/splits; markers per repeat, diamond + line = mean over repeats with joint patient-bootstrap interval, dashed line = +0.05 triage gain) and E13-relative minus its compact control E13-power (right). A change near zero or negative means no added value. ' + TXT['Remitter'] + ' ' + TXT['Responder'] + ' Improvement over a comparator is a different claim from prediction above chance.')
# ---- Fig B3 operating behavior
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
for ax, (k, lab) in zip(axs, (('nPPV', 'nPPV (% of base rate)'), ('sensitivity', 'Sensitivity'), ('Brier', 'Brier score (lower is better)'))):
    for i, m in enumerate(order):
        for s in SEEDS: ax.plot(ALL[(ALL.endpoint == 'Remitter') & (ALL.model == m) & (ALL.seed == s)][k].iloc[0], i + (s - 43) * .15, MK[s], color=COL[m], ms=4.5)
    ax.set_xlabel(lab)
    if k == 'nPPV': ax.axvline(100, color='k', ls=':')
    if k == 'Brier': ax.axvline(Y['Remitter'].mean() * (1 - Y['Remitter'].mean()), color='k', ls=':')
axs[0].set_yticks(range(len(order))); axs[0].set_yticklabels(order); axs[0].invert_yaxis()
save(fig, 'figB3_operating_behavior', 'Remitter endpoint: operating behavior at training-selected thresholds (max PPV subject to sensitivity >= 50%) and Brier score, every method and repeat shown, no method omitted. Marker = one CV repeat. Dotted: nPPV 100 (= base rate) and the Brier score of a constant prevalence predictor. nPPV = 100 x PPV / prevalence (percent of base rate, not accuracy). Values scatter around the no-information reference; no method has a clearly better nPPV or Brier score.')
ALL[ALL.endpoint == 'Remitter'][['model', 'seed', 'AUC', 'PPV', 'sensitivity', 'nPPV', 'Brier', 'balanced_accuracy', 'n_pos_calls', 'TP', 'FP', 'TN', 'FN']].to_csv(f'{DATA}/figB3_table.csv', index=False)

# ---- reliability: E11 parameters, E13 relative features, E12 covariance distance ratio
b2f = import_module('b2feat'); from pyriemann.utils.distance import distance_riemann
sp = pd.read_csv(f'{OUT}/batch2_specparam_fits_PRIVATE.csv'); REL = []
def sbr(a, b):
    m = ~(np.isnan(a) | np.isnan(b)); r = spearmanr(a[m], b[m])[0]; return 2 * r / (1 + r), int(m.sum())
ids = sorted(sp.id.unique())
for nm in ('exp', 'off', 'cf', 'pw'):
    for cond in ('EC', 'EO'):
        for r in ('frontal', 'central', 'posterior'):
            g = {}
            for tag in ('_h1', '_h2'):
                s = sp[sp.key == f'{cond}_{r}{tag}'].set_index('id').reindex(ids); g[tag] = s[nm].where(s.ok.fillna(False).astype(bool)).values
            rr, n = sbr(g['_h1'], g['_h2']); REL.append(dict(group={'exp': 'aperiodic exponent', 'off': 'aperiodic offset', 'cf': 'alpha peak CF', 'pw': 'alpha peak height'}[nm], feature=f'{cond}_{r}_{nm}', r_sb=rr, n=n))
z = np.load(f'{OUT}/batch2_cov_PRIVATE.npz'); h1, h2, full = z['cov_h1'], z['cov_h2'], z['cov_full']
E1 = np.array([b2f.eigen_feats(h1[i, 1], h1[i, 0]) for i in range(len(h1))]); E2 = np.array([b2f.eigen_feats(h2[i, 1], h2[i, 0]) for i in range(len(h2))])
for j, nm in enumerate(b2f.E13_NAMES):
    rr, n = sbr(E1[:, j], E2[:, j]); REL.append(dict(group='E13 eigen summaries', feature=nm, r_sb=rr, n=n))
REL = pd.DataFrame(REL); REL.to_csv(f'{DATA}/figB6_reliability.csv', index=False)
# E12: within-person split-half Riemannian distance vs between-person distance (trace-normalized, EO alpha etc.)
dist = []
rng2 = np.random.default_rng(0)
for c in range(2):
    for b in range(3):
        okp = [i for i in range(len(h1)) if not (np.isnan(h1[i, c, b]).any() or np.isnan(h2[i, c, b]).any())]; Nn = b2f.trace_norm
        w = [distance_riemann(Nn(h1[i, c, b]), Nn(h2[i, c, b])) for i in okp]; bt = [distance_riemann(Nn(h1[i, c, b]), Nn(h2[j, c, b])) for i, j in zip(okp, rng2.permutation(okp))]
        dist.append(dict(cond=['EC', 'EO'][c], band=['theta', 'alpha', 'beta'][b], n=len(okp), within_median=float(np.median(w)), between_median=float(np.median(bt)), ratio=float(np.median(w) / np.median(bt))))
DIST = pd.DataFrame(dist); DIST.to_csv(f'{DATA}/figB6_cov_distance.csv', index=False)
# ---- gain sensitivity
G = pd.read_csv(f'{B2["Remitter"]}/gain_sensitivity_PRIVATE.csv'); G['abs_dp'] = (G.p_gain - G.p_base).abs(); G['flip'] = (G.dec_gain != G.dec_base)
GS = G.groupby(['model', 'gain']).agg(mean_abs_dp=('abs_dp', 'mean'), max_abs_dp=('abs_dp', 'max'), pct_decision_flips=('flip', lambda x: 100 * x.mean())).reset_index(); GS.to_csv(f'{DATA}/figB6_gain_sensitivity.csv', index=False)
fig, axs = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={'width_ratios': [1.3, .8, 1]})
grps = ['aperiodic exponent', 'aperiodic offset', 'alpha peak CF', 'alpha peak height', 'E13 eigen summaries']; rg = np.random.default_rng(0)
for i, g in enumerate(grps):
    v = REL[REL.group == g]; axs[0].scatter(v.r_sb, np.full(len(v), i) + rg.uniform(-.15, .15, len(v)), s=16, color='#0072B2', alpha=.8); axs[0].text(1.04, i, f'median {v.r_sb.median():.2f}\nN {v.n.min()}-{v.n.max()}', va='center', fontsize=7)
axs[0].set_yticks(range(len(grps))); axs[0].set_yticklabels(['Exponent (6)', 'Offset (6)', 'Alpha CF (6)', 'Alpha height (6)', 'E13 eigen (12)']); axs[0].invert_yaxis(); axs[0].set_xlim(-.4, 1.35); axs[0].axvline(0, color='k', lw=.8); axs[0].set_xlabel('Split-half reliability\n(Spearman-Brown r)')
lab = [f'{r.cond} {r.band}' for r in DIST.itertuples()]; axs[1].barh(range(6), DIST.ratio, color='#009E73'); axs[1].set_yticks(range(6)); axs[1].set_yticklabels(lab); axs[1].invert_yaxis(); axs[1].axvline(1, color='k', ls=':'); axs[1].set_xlabel('Within-person split-half /\nbetween-person Riemannian distance\n(<1 = person-specific)')
for m, off in zip(['E01_ref', 'E10', 'E11-add', 'E12-add', 'E13-power', 'E13-relative', 'E13-add'], range(7)):
    for gi, g in enumerate((0.5, 2.0)):
        r = GS[(GS.model == m) & (GS.gain == g)]; 
        if len(r): axs[2].plot(r.mean_abs_dp.iloc[0], off + (gi - .5) * .3, 'o' if g == .5 else 's', color=COL.get(m.replace('E01_ref', 'E01'), 'k'), ms=5)
axs[2].set_yticks(range(7)); axs[2].set_yticklabels(['E01', 'E10', 'E11-add', 'E12-add', 'E13-power', 'E13-relative', 'E13-add']); axs[2].invert_yaxis(); axs[2].set_xlabel('Mean |change in P(remission)|\ncommon gain 0.5 (o) / 2 (square)'); axs[2].set_xlim(-0.005, GS.mean_abs_dp.max() * 1.25)
fig.tight_layout()
save(fig, 'figB6_reliability_robustness', 'Left: split-half reliability of the batch-2 features (one dot = one feature; halves = first/second clean-window halves with a one-window gap for spectral fits, interleaved 10-epoch halves for covariances; patients missing either half excluded, never imputed; N shown). Middle: E12 covariance shape stability (median within-person split-half Riemannian distance divided by distance to a random other person; lower = more person-specific). Right: LABEL-FREE gain sensitivity, not clinical evidence: frozen seed-42 fold models applied to test patients after a common gain of 0.5 or 2 on both conditions; only E13-power and E13-relative are fully invariant (change exactly 0: contrasts and eigenvalue ratios cancel the gain); E12-add and E13-add still include the E01 absolute-power features, which shift, as do E01, E10 and E11-add (the E12 own tangent features are trace-normalized, hence invariant). Middle-panel values reflect two recordings from the same session and so overstate between-session stability; high reliability of E13 features does NOT mean they are physiological: the SD of ln eigenvalues correlates with the number of ICA components removed (Spearman 0.38-0.44) and with interpolated channels (0.40-0.45; outputs/e13_processing_confound_check.json), so part of the signal is cleaning-induced rank structure. This does not simulate real acquisition shifts.')
# ---- Fig B4 specparam example + counts; Fig B5 covariance illustration (fixed rule: lowest ID with accepted EO posterior fit / complete covariances)
import pickle
PSD = pickle.load(open(f'{OUT}/batch2_roi_psd_PRIVATE.pkl', 'rb')); full_fits = sp[~sp.key.str.contains('_h')]
ex_id = sorted(full_fits[(full_fits.key == 'EO_posterior') & full_fits.ok & full_fits.cf.notna()].id)[0]; f, pw = PSD[ex_id][('EO', 'posterior')]
from specparam import SpectralModel
m_ = SpectralModel(aperiodic_mode='fixed', peak_width_limits=(1, 8), max_n_peaks=4, peak_threshold=2.0, verbose=False); m_.fit(f, pw, (2, 40))
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), gridspec_kw={'width_ratios': [1.2, .9, .9]})
ap = m_.results.get_params('aperiodic'); axs[0].semilogy(f, pw, color='k', label='measured PSD'); axs[0].semilogy(f, 10 ** (ap[0] - ap[1] * np.log10(f)), '--', color='#D55E00', label='aperiodic fit'); axs[0].semilogy(f, 10 ** m_.results.model.modeled_spectrum, color='#0072B2', label='full model') if hasattr(m_.results.model, 'modeled_spectrum') else None
axs[0].set_xlabel('Frequency (Hz)'); axs[0].set_ylabel('PSD (uV$^2$/Hz)'); axs[0].legend(frameon=False, fontsize=8); axs[0].set_title('Example EO posterior-ROI fit (lowest accepted ID)', loc='left', fontsize=8)
cnt = full_fits.assign(res=np.where(full_fits.ok, 'accepted', 'rejected (R$^2$<0.90 or non-finite)')).groupby(['key', 'res']).size().unstack(fill_value=0); cnt.plot.barh(stacked=True, ax=axs[1], color=['#009E73', '#D55E00'], legend=True); axs[1].legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(.5, -.18), ncol=1); axs[1].set_xlabel('Fits (recording x ROI)'); axs[1].set_ylabel('')
axs[2].hist(full_fits.r2.dropna(), bins=30, color='#0072B2'); axs[2].axvline(.9, color='k', ls='--'); axs[2].set_xlabel('Fit R$^2$ (rejection threshold 0.90)'); axs[2].set_ylabel('Fits')
fig.tight_layout()
nacc = int(full_fits.ok.sum()); ntot = len(full_fits); noalpha = int((full_fits.ok & full_fits.cf.isna()).sum())
save(fig, 'figB4_specparam_fits', f'Spectral parameterization (specparam 2.0.0rc7; fixed aperiodic, 2-40 Hz, peaks <= 4, width 1-8 Hz, threshold 2). Left: example fit for a patient chosen by a fixed rule (lowest ID with an accepted EO posterior fit and an alpha peak; ID not shown). Middle: accepted vs rejected fits per condition/ROI; overall {nacc} of {ntot} accepted ({100*nacc/ntot:.0f}%), {noalpha} accepted fits had no alpha peak (height 0, CF missing). Right: R$^2$ distribution with the pre-set 0.90 cut. Exponent is a spectral slope, not a direct measure of excitation/inhibition.')
ci = sorted(i for i in range(len(full)) if not np.isnan(full[i]).any())[0]; Nn = b2f.trace_norm
fig, axs = plt.subplots(1, 4, figsize=(14, 3.6))
for k, (c, nm) in enumerate(((0, 'EC'), (1, 'EO'))):
    M = Nn(full[ci, c, 1]); im = axs[k].imshow(M / np.abs(M).max(), cmap='RdBu_r', vmin=-1, vmax=1); axs[k].set_title(f'{nm} alpha covariance\n(trace-normalized, scaled to max)', fontsize=8); axs[k].set_xticks([]); axs[k].set_yticks([])
fig.colorbar(im, ax=axs[1], fraction=.046)
from scipy.linalg import eigh
for b, (nm, col) in enumerate((('theta', '#0072B2'), ('alpha', '#D55E00'), ('beta', '#009E73'))): axs[2].plot(np.sort(np.log(eigh(full[ci, 1, b], full[ci, 0, b], eigvals_only=True))), color=col, label=nm)
axs[2].axhline(0, color='k', lw=.8); axs[2].set_xlabel('Eigenvalue rank'); axs[2].set_ylabel('ln generalized eigenvalue\n(EO relative to EC)'); axs[2].legend(frameon=False, fontsize=8); axs[2].set_title('Example patient', fontsize=8)
allE = np.array([b2f.eigen_feats(full[i, 1], full[i, 0]) for i in range(len(full))]); axs[3].hist(allE[:, 5][~np.isnan(allE[:, 5])], bins=25, color='#E69F00'); axs[3].set_xlabel('SD of ln eigenvalues, alpha (all patients)'); axs[3].set_ylabel('Patients')
fig.tight_layout()
save(fig, 'figB5_covariance_illustration', 'Illustration only (not evaluation): lowest-ID patient with complete covariances (fixed rule; ID not shown). Left two: alpha-band 26-channel covariance (Ledoit-Wolf shrinkage, trace-normalized) for eyes-closed and eyes-open; entries scaled to the maximum magnitude, dimensionless. Third: ln generalized eigenvalues of C_EO relative to C_EC for three bands (0 = no relative change). Right: distribution of one E13 summary (SD of ln eigenvalues, alpha) across patients with complete data. These describe relative covariance differences between two separate resting recordings, not a time-locked transition and not biological energy.')
# ---- Fig B7 runtime / features
rt = pd.concat([pd.read_csv(f'{B2[e]}/metrics_by_seed.csv').assign(endpoint=e) for e in B2]); R = rt[rt.endpoint == 'Remitter'].groupby('model').agg(n_features=('n_features', 'first'), runtime_s=('runtime_s', 'mean')).reindex(['E01_ref'] + CH)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for m, r in R.iterrows(): ax.scatter(r.n_features, r.runtime_s, s=60, color=COL.get(m.replace('E01_ref', 'E01'), 'k')); ax.annotate(m.replace('_ref', ''), (r.n_features, r.runtime_s), textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xlabel('Input features to classifier'); ax.set_ylabel('Mean runtime per repeat (s, one core)'); ax.set_yscale('log'); fig.tight_layout(); R.to_csv(f'{DATA}/figB7_complexity.csv')
save(fig, 'figB7_complexity', 'Complexity: classifier input features (E12-add includes 20 PCA scores) vs mean wall time per CV repeat (five outer folds with nested tuning, one core). The covariance method is ~10x costlier than linear models; extraction cost (~4.5 min for all patients) is excluded.')
# ---- tables + HTML
fmt = lambda df: df.round(3).to_html(index=False, na_rep='undefined')
tabs = {e: SCR[(SCR.endpoint == e)][['model', 'auc_s42', 'auc_s43', 'auc_s44', 'mean_auc', 'mean_auc_lo', 'mean_auc_hi', 'mean_gain_vs_E01', 'gain_lo', 'gain_hi', 'gain_s42', 'gain_s43', 'gain_s44', 'all_gains_positive', 'qualifies']] for e in PRED}
nq = int(SCR[(SCR.endpoint == 'Remitter')].qualifies.sum()); best = SCR[(SCR.endpoint == 'Remitter') & SCR.model.isin(CH)].sort_values('mean_auc').iloc[-1]
idx = [dict(figure=n, png=f'{FIG}/{n}.png', pdf=f'{FIG}/{n}.pdf', runs=[cfgs[e]['run_id'] for e in cfgs] + [os.path.basename(B1[e]) for e in B1]) for n in saved]; json.dump(idx, open(f'{FIG}/figure_index.json', 'w'), indent=1)
img = lambda n: f'<figure><img src="data:image/png;base64,{base64.b64encode(open(f"{FIG}/{n}.png","rb").read()).decode()}" style="max-width:100%"><figcaption>{cap[n]}</figcaption></figure>'
html = f'''<!doctype html><meta charset="utf-8"><title>Batch 2 exploratory EEG experiments</title><style>body{{font:15px/1.5 system-ui;max-width:1050px;margin:2em auto;padding:0 1em}}figcaption{{font-size:13px;color:#333}}table{{border-collapse:collapse;font-size:11px}}td,th{{border:1px solid #ccc;padding:2px 5px}}.box{{background:#fff4e5;border-left:4px solid #D55E00;padding:.5em 1em}}</style>
<h1>Batch 2 (exploratory): four experiment families on resting TDBRAIN EEG</h1>
<div class="box"><b>Bottom line:</b> {nq} of 7 challengers met the pre-set triage rule (mean remission AUC &ge; 0.60, gain over E01 &ge; 0.05, positive gain in every repeat). The highest mean remission AUC was {best.model} at {best.mean_auc:.3f}. No method clearly beats chance or E01; none needed (or earned) the batch-aware permutation. This batch is <b>exploratory</b> (chosen after seeing the baseline), uses the same 164 patients, and is conditional on session 1 being pretreatment. The oddball experiments E02&ndash;E05 remain blocked and untested.</div>
<p>Audit corrections to the baseline report are in AUDIT.md and the updated REPORT.html (earlier caption claimed all E01-minus-E01-PSD intervals crossed zero; remission seed-44 did not, because E01-PSD itself scored below chance there).</p>
{img("figB1_all_vs_E01")}{img("figB2_paired_changes")}{img("figB6_reliability_robustness")}
<h2>Exact numbers: Remitter (primary)</h2>{fmt(tabs["Remitter"])}<h2>Responder (secondary)</h2>{fmt(tabs["Responder"])}
<p>Challenger metrics for every repeat (PPV, sensitivity, nPPV, Brier, balanced accuracy, TP/FP/TN/FN): plotted_data/all_metrics.csv. E01_ref is E01 re-implemented in the batch-2 framework; predictions are identical to the baseline (max probability difference 0).</p>
{img("figB3_operating_behavior")}{img("figB4_specparam_fits")}{img("figB5_covariance_illustration")}{img("figB7_complexity")}
<h2>What each test taught us</h2><ul><li><b>E10 (nonlinear):</b> boosting on the same 65 features did not help (AUC below E01 in most repeats): no usable interaction signal at this sample size.</li><li><b>E11 (periodic/aperiodic):</b> parameter fits are good (see figure) but neither alone nor added to E01 does it predict remission.</li><li><b>E12 (covariance tangent space):</b> adds spatial detail and ~10x runtime without any gain.</li><li><b>E13 (EC/EO reactivity):</b> relative covariance summaries are not better than the compact power-contrast control (see B2 right), and are invariant to gain by construction. Reliability is very high (median r 0.99) but confounded with cleaning rank loss (B6), so it is not evidence of a physiological signal.</li></ul>
<h2>Limitations</h2><ul><li>Exploratory, post-baseline; the same 164 people were reused (new splits are not untouched validation); bootstrap intervals are conditional on saved predictions and uncorrected for 7 methods.</li><li>Session-1 timing assumed. ICA used up to 3 components (cap rule arbitrary), 17 non-converged fits (AUDIT.md). Interpolated channels retained in covariance space; covariances are rank-reduced by ICA/interpolation, so the extreme eigenvalue summaries may reflect shrinkage floors.</li><li>E13 spread features track ICA/interpolation burden (see B6 caption), so the covariance-reactivity measure is partly a processing artifact.</li><li>Covariance features exist for 141 of 164 patients (others imputed/mapped to the reference); connectivity for 155; E11 fits rejected in 9% of cases.</li><li>Prior-art comparisons used abstract-level reading only; two spec links did not resolve to the described studies (BATCH2_PROTOCOL.md).</li></ul>
<h2>Decision</h2><p>Stop this batch: no challenger qualified. Do not add variants or ensembles. Resting-only conventional and covariance features appear exhausted for this cohort; the open lever remains obtaining pretreatment oddball (or larger labeled resting) data.</p>'''
open('BATCH2_REPORT.html', 'w').write(html)
json.dump(dict(qualifying=nq, best=best.model, best_mean_auc=float(best.mean_auc)), open(f'{FIG}/screening_result.json', 'w')); print(SCR[SCR.endpoint == 'Remitter'][['model', 'mean_auc', 'mean_gain_vs_E01', 'all_gains_positive', 'qualifies']].round(3).to_string(index=False)); print(TXT)
