"""DynaMix analysis + figures (figures_dynamix/). Conditional on saved predictions; one inferential procedure = permutation test (27)."""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
OUT = 'outputs'; FIG = f'{OUT}/figures_dynamix'; DATA = f'{FIG}/plotted_data'; os.makedirs(DATA, exist_ok=True); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
SEEDS = [42, 43, 44]; MODS = ['E01c', 'pretrained', 'random_a', 'random_b', 'random_c']; COL = {'E01c': '#000000', 'pretrained': '#D55E00', 'random_a': '#56B4E9', 'random_b': '#0072B2', 'random_c': '#009E73'}; MK = {42: 'o', 43: 's', 44: '^'}
cap = {}
def save(fig, n, c): fig.savefig(f'{FIG}/{n}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{n}.pdf', bbox_inches='tight'); plt.close(fig); cap[n] = c
RUN = {e: sorted(glob.glob(f'runs/DM_{e}_*'))[-1] for e in ('Remitter', 'Responder')}; E01C = {e: sorted(glob.glob(f'runs/PIV_E01c_{e}_*'))[-1] for e in RUN}
PRED, Y = {}, {}
for e in RUN:
    a = pd.read_csv(f'{RUN[e]}/predictions_PRIVATE.csv'); b = pd.read_csv(f'{E01C[e]}/predictions_PRIVATE.csv'); assert (a.id == b.id).all() and (a.y == b.y).all(); Y[e] = a.y.values; d = {}
    for s in SEEDS:
        d[('E01c', s)] = (b[f'p_E01c_s{s}'].values, b[f'dec_E01c_s{s}'].values)
        for m in MODS[1:]: d[(m, s)] = (a[f'p_{m}_s{s}'].values, a[f'dec_{m}_s{s}'].values)
    PRED[e] = d
def pm(y, p, d):
    tp = ((d == 1) & (y == 1)).sum(); fp = ((d == 1) & (y == 0)).sum(); tn = ((d == 0) & (y == 0)).sum(); fn = ((d == 0) & (y == 1)).sum(); ppv = tp / (tp + fp) if tp + fp else np.nan; sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p), PPV=ppv, sensitivity=sens, nPPV=100 * ppv / y.mean() if ppv == ppv else np.nan, Brier=float(np.mean((p - y) ** 2)), balanced_accuracy=(sens + spec) / 2, n_pos_calls=int(tp + fp), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
ALL = pd.DataFrame([dict(endpoint=e, model=m, seed=s, N=len(Y[e]), prevalence=Y[e].mean(), **pm(Y[e], *PRED[e][(m, s)])) for e in PRED for m in MODS for s in SEEDS]); ALL.to_csv(f'{DATA}/all_metrics.csv', index=False)
# bootstrap: identical draws, joint across repeats
NB = 2000; draws = np.random.default_rng(1).integers(0, 164, size=(NB, 164)); q = lambda v: (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))); BOOT = {}
for e in PRED:
    y = Y[e]; A = {m: np.full((NB, 3), np.nan) for m in MODS}; LL = {m: np.zeros(164) for m in MODS}
    for m in MODS:
        for si, s in enumerate(SEEDS): p = np.clip(PRED[e][(m, s)][0], 1e-6, 1 - 1e-6); LL[m] += -(y * np.log(p) + (1 - y) * np.log(1 - p)) / 3
    for b, ix in enumerate(draws):
        if len(set(y[ix])) < 2: continue
        for m in MODS:
            for si, s in enumerate(SEEDS): A[m][b, si] = roc_auc_score(y[ix], PRED[e][(m, s)][0][ix])
    BOOT[e] = (A, LL)
rows = []
for e in PRED:
    A, LL = BOOT[e]; sub = ALL[ALL.endpoint == e]
    for m in MODS:
        a = sub[sub.model == m].AUC.values; g = a - sub[sub.model == 'E01c'].AUC.values; mean_boot = A[m].mean(1)
        rows.append(dict(endpoint=e, model=m, auc_s42=a[0], auc_s43=a[1], auc_s44=a[2], mean_auc=a.mean(), mean_auc_lo=q(mean_boot)[0], mean_auc_hi=q(mean_boot)[1], gain_vs_E01c=g.mean(), gain_lo=q((A[m] - A['E01c']).mean(1))[0], gain_hi=q((A[m] - A['E01c']).mean(1))[1], all_gains_positive=bool((g > 0).all()), mean_logloss=float(LL[m].mean())))
SCR = pd.DataFrame(rows); SCR['triage_pass'] = (SCR.mean_auc >= .60) & (SCR.gain_vs_E01c >= .05) & SCR.all_gains_positive & (SCR.model != 'E01c'); SCR.to_csv(f'{DATA}/summary_and_intervals.csv', index=False)
PAIR = []
for e in PRED:
    A, LL = BOOT[e]
    for m in MODS[2:] + ['E01c']:
        d_auc = (A['pretrained'] - A[m]).mean(1); ll = LL['pretrained'] - LL[m]; llb = [ll[ix].mean() for ix in draws]; sub = ALL[ALL.endpoint == e]
        PAIR.append(dict(endpoint=e, comparison=f'pretrained - {m}', mean_auc_diff=float(sub[sub.model == 'pretrained'].AUC.mean() - sub[sub.model == m].AUC.mean()), auc_lo=q(d_auc)[0], auc_hi=q(d_auc)[1], mean_patient_logloss_diff=float(ll.mean()), ll_lo=q(llb)[0], ll_hi=q(llb)[1], n_patients_pretrained_lower_loss=int((ll < 0).sum())))
PAIR = pd.DataFrame(PAIR); PAIR.to_csv(f'{DATA}/paired_differences.csv', index=False)
perm = json.load(open(f'{OUT}/dynamix/permutation_pretrained_remitter.json')) if os.path.exists(f'{OUT}/dynamix/permutation_pretrained_remitter.json') else None
def ivtxt(e):
    s = SCR[(SCR.endpoint == e)]; t = []
    for m in MODS[1:]: r = s[s.model == m].iloc[0]; t.append(f'{m} mean AUC {r.mean_auc:.3f} [{r.mean_auc_lo:.3f}, {r.mean_auc_hi:.3f}] ' + ('(interval excludes 0.5)' if r.mean_auc_lo > .5 or r.mean_auc_hi < .5 else '(includes 0.5)') + f'; gain vs E01c {r.gain_vs_E01c:+.3f} [{r.gain_lo:+.3f}, {r.gain_hi:+.3f}]' + (' (excludes 0)' if r.gain_lo > 0 or r.gain_hi < 0 else ' (includes 0)'))
    r0 = s[s.model == 'E01c'].iloc[0]; return f'{e}: E01c mean AUC {r0.mean_auc:.3f} [{r0.mean_auc_lo:.3f}, {r0.mean_auc_hi:.3f}]; ' + '; '.join(t) + '.'
TXT = {e: ivtxt(e) for e in PRED}
# ---- figD3 comparison
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, e in zip(axs, ('Remitter', 'Responder')):
    for i, m in enumerate(MODS):
        r = SCR[(SCR.endpoint == e) & (SCR.model == m)].iloc[0]
        for s, v in zip(SEEDS, (r.auc_s42, r.auc_s43, r.auc_s44)): ax.plot(v, i + (s - 43) * .15, MK[s], color=COL[m], ms=4.5, alpha=.85)
        ax.plot([r.mean_auc_lo, r.mean_auc_hi], [i, i], color=COL[m], lw=2, alpha=.5); ax.plot(r.mean_auc, i, 'D', color=COL[m], ms=7, mec='k')
    ax.axvline(.5, color='k', ls=':'); ax.axvline(.6, color='#999', ls='--', lw=.8); ax.set_xlabel('Outer-fold AUC'); ax.set_title(f'{e} (N=164, prevalence {Y[e].mean():.2f})', loc='left', fontsize=9)
axs[0].set_yticks(range(5)); axs[0].set_yticklabels(['E01c (saved)', 'DynaMix pretrained', 'DynaMix random A', 'DynaMix random B', 'DynaMix random C']); axs[0].invert_yaxis()
save(fig, 'figD3_model_comparison', 'Same 164 patients, identical splits. Small markers = pooled out-of-fold AUC per CV repeat (circle 42, square 43, triangle 44; all shown); diamond + line = mean over repeats with a 95% paired patient-bootstrap interval (patients resampled jointly across repeats; conditional on saved predictions; excludes training/tuning uncertainty). Dotted: AUC 0.5; dashed: 0.60 triage line. Numeric interval check (generated): ' + TXT['Remitter'] + ' ' + TXT['Responder'] + ' The three random models are separate fits of the same architecture; none is averaged with another or selected.')
# ---- figD4 paired
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6)); labs = [f'pretrained - {m}' for m in ['random_a', 'random_b', 'random_c', 'E01c']]
for k, e in enumerate(('Remitter', 'Responder')):
    p = PAIR[PAIR.endpoint == e].set_index('comparison').loc[labs].reset_index()
    for i, r in p.iterrows(): axs[k].plot([r.auc_lo, r.auc_hi], [i, i], color='#D55E00', lw=2, alpha=.5); axs[k].plot(r.mean_auc_diff, i, 'D', color='#D55E00', mec='k')
    axs[k].axvline(0, color='k'); axs[k].set_yticks(range(4)); axs[k].set_yticklabels([l.replace('pretrained - ', 'pretrained minus ') for l in labs]); axs[k].invert_yaxis(); axs[k].set_xlabel(f'{e}: mean AUC difference\n(>0 = pretrained higher)'); axs[k].set_title(e, loc='left', fontsize=9)
p = PAIR[PAIR.endpoint == 'Remitter'].set_index('comparison').loc[labs].reset_index()
for i, r in p.iterrows(): axs[2].plot([r.ll_lo, r.ll_hi], [i, i], color='#7f7f7f', lw=2, alpha=.6); axs[2].plot(r.mean_patient_logloss_diff, i, 'D', color='#7f7f7f', mec='k')
axs[2].axvline(0, color='k'); axs[2].set_yticks(range(4)); axs[2].set_yticklabels(['' for _ in labs]); axs[2].invert_yaxis(); axs[2].set_xlabel('Remitter: mean per-patient log-loss difference\n(<0 = pretrained lower loss)'); fig.tight_layout()
save(fig, 'figD4_paired_differences', 'Paired differences on the same patients and splits, pretrained DynaMix minus each comparator (three separate random initializations, and saved E01c). Left/middle: mean-over-repeats AUC difference with joint patient-bootstrap interval. Right: patient-level paired difference in log-loss averaged over the three repeats (each patient contributes one number), Remitter. Intervals are conditional on saved predictions. A difference to a random model addresses whether pretraining changes these features, not whether the features predict; a difference to E01c is descriptive, not a superiority test.')
# ---- figD5 ROC/PR
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4)); y = Y['Remitter']
for m in MODS:
    for s in SEEDS:
        p = PRED['Remitter'][(m, s)][0]; f, t, _ = roc_curve(y, p); axs[0].plot(f, t, color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5, label=m if s == 42 else None); pr, rc, _ = precision_recall_curve(y, p); axs[1].plot(rc, pr, color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5)
axs[0].plot([0, 1], [0, 1], 'k:', lw=1); axs[0].set_xlabel('False-positive rate'); axs[0].set_ylabel('True-positive rate'); axs[0].legend(frameon=False, fontsize=7); axs[1].axhline(y.mean(), color='k', ls=':'); axs[1].set_xlabel('Recall (sensitivity)'); axs[1].set_ylabel('Precision (PPV)'); axs[1].set_ylim(0, 1); fig.tight_layout()
save(fig, 'figD5_roc_pr', 'Remitter, pooled out-of-fold predictions. Thick = repeat 42 (prespecified primary), thin = repeats 43/44 (not selected). Left: ROC. Right: precision-recall; dotted = prevalence (0.48). Descriptive curves only; thresholds were fixed inside training folds and are not chosen from these curves. Curves near the diagonal/prevalence line indicate no useful ranking.')
# ---- figD6 calibration
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, m in zip(axs, ['pretrained', 'E01c']):
    for s in SEEDS:
        p = PRED['Remitter'][(m, s)][0]; q5 = pd.qcut(p, 5, labels=False, duplicates='drop'); g = pd.DataFrame({'p': p, 'y': y, 'b': q5}).groupby('b').agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size')); ax.plot(g.p, g.y, 'o-', color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5, ms=4)
        if s == 42: [ax.text(a, b + .02, f'n={n}', fontsize=6, ha='center') for a, b, n in zip(g.p, g.y, g.n)]
    ax.plot([0, 1], [0, 1], 'k:'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('Mean predicted P(remission), 5 equal-count bins'); ax.set_ylabel('Observed remission fraction'); ax.set_title(m, loc='left', fontsize=9)
fig.tight_layout(); save(fig, 'figD6_calibration', 'Remitter calibration (reliability) of pooled out-of-fold probabilities in five equal-count bins (thick = repeat 42, thin = 43/44; n per bin shown for 42). Points near the dotted diagonal would indicate calibrated probabilities; with ~33 patients per bin the observed fractions carry roughly +/-0.09 binomial noise.')
# ---- representation diagnostics (label-free)
CHN = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2']; FN = [f'z{i}' for i in range(10)] + ['gate_entropy', 'gate_max']
M = pd.read_csv(f'{OUT}/dynamix/window_manifest_patients_PRIVATE.csv'); E = M[(M.status == 'ok') & (M.n_windows >= 2)]; REP = []
def sb(a, b):
    ok = ~(np.isnan(a) | np.isnan(b)); r = spearmanr(a[ok], b[ok])[0]; return 2 * r / (1 + r) if r == r else np.nan
for mo in ['pretrained', 'random_a', 'random_b', 'random_c']:
    H1, H2, WA, WB, ICCs = [], [], [], [], []
    for r in E.itertuples():
        z = np.load(f'{OUT}/dynamix/emb/{mo}/{r.key}.npz'); w = z['mean']; n = len(w); k = (n + 1) // 2; H1.append(w[:k].mean(0)); H2.append(w[k:].mean(0)); WA.append(z['halfA'].reshape(-1, 26, 12)); WB.append(z['halfB'].reshape(-1, 26, 12)); ICCs.append(w)
    H1, H2 = np.array(H1), np.array(H2); WA, WB = np.concatenate(WA), np.concatenate(WB)
    # ICC(1) across windows within recording (one-way): per (channel, feature), recordings with >=2 windows
    for f in range(12):
        rs = [sb(H1[:, c, f], H2[:, c, f]) for c in range(26)]
        rep = [np.corrcoef(WA[:, c, f], WB[:, c, f])[0, 1] for c in range(26)]
        icc = []
        for c in range(26):
            groups = [w[:, c, f] for w in ICCs]; k_ = np.mean([len(g) for g in groups]); gm = np.mean(np.concatenate(groups)); msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (len(groups) - 1); msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (sum(len(g) for g in groups) - len(groups)); icc.append((msb - msw) / (msb + (k_ - 1) * msw) if msb + (k_ - 1) * msw > 0 else np.nan)
        REP.append(dict(model=mo, feature=FN[f], split_half_SB_median=float(np.nanmedian(rs)), replicate_halves_r_median=float(np.nanmedian(rep)), icc_windows_median=float(np.nanmedian(icc)), n_recordings=len(E)))
REP = pd.DataFrame(REP); REP.to_csv(f'{DATA}/representation_stability.csv', index=False)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (col, ttl) in zip(axs, (('split_half_SB_median', 'Patient-level split-half reliability\n(first vs second windows; SB-corrected Spearman; median over 26 channels)'), ('icc_windows_median', 'Between-recording share of variance\n(ICC(1) across windows; median over channels)'), ('replicate_halves_r_median', 'Gating-noise stability\n(replicates 1-4 vs 5-8; Pearson across windows)'))):
    for mo in ['pretrained', 'random_a', 'random_b', 'random_c']:
        d = REP[REP.model == mo]; ax.plot(d[col], range(12), 'o', color=COL[mo], label=mo, ms=5, alpha=.85)
    ax.set_yticks(range(12)); ax.set_yticklabels(FN); ax.invert_yaxis(); ax.set_xlabel(ttl, fontsize=8); ax.axvline(0, color='k', lw=.6)
axs[0].legend(frameon=False, fontsize=7, loc='lower left'); fig.tight_layout()
save(fig, 'figD7_representation_stability', 'Label-free diagnostics of the DynaMix summaries (one dot = one of the 12 feature types, median over 26 channels; recordings with >=2 eligible windows pooled over EC and EO per model; no outcome data). Left: whether a patient\'s first-window summary predicts the later windows. Middle: how much variance lies between recordings versus across windows of the same recording. Right: numerical stability to gating noise. These speak to representation stability only; stable features are not necessarily predictive, and amplitude information is discarded by the per-series standardization.')
json.dump(cap, open(f'{FIG}/captions_dm.json', 'w'), indent=1)
print(SCR[['endpoint', 'model', 'auc_s42', 'auc_s43', 'auc_s44', 'mean_auc', 'mean_auc_lo', 'mean_auc_hi', 'gain_vs_E01c', 'gain_lo', 'gain_hi', 'all_gains_positive', 'triage_pass']].round(3).to_string(index=False)); print(PAIR.round(3).to_string(index=False)); print(TXT); print(REP.round(3).to_string(index=False))
