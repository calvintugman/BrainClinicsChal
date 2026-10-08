"""Analysis + figures for the deviation experiment (conditional on saved predictions). Decision rules as frozen in DEVIATION_PROTOCOL.md."""
import sys, os, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
OUT = 'outputs'; FIG = f'{OUT}/figures_dev'; DATA = f'{FIG}/plotted_data'; plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
SEEDS = [42, 43, 44]; MODS = ['E01c', 'A', 'B', 'C', 'D']; LAB = {'E01c': 'E01c (saved)', 'A': 'A: EC+EO levels', 'B': 'B: 24 log powers', 'C': 'C: levels + contrast', 'D': 'D: levels + deviation (primary)'}
COL = {'E01c': '#000000', 'A': '#56B4E9', 'B': '#009E73', 'C': '#E69F00', 'D': '#D55E00'}; MK = {42: 'o', 43: 's', 44: '^'}; cap = {}
def save(fig, n, c): fig.savefig(f'{FIG}/{n}.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/{n}.pdf', bbox_inches='tight'); plt.close(fig); cap[n] = c
def load(prefix, ep):
    a = pd.read_csv(sorted(glob.glob(f'runs/{prefix}_{ep}_*'))[-1] + '/predictions_PRIVATE.csv'); return a
EP = ('Remitter', 'Responder'); FULL = {e: load('DEV', e) for e in EP}; CC = {e: load('DEVcc', e) for e in EP}; E01C = {e: pd.read_csv(sorted(glob.glob(f'runs/PIV_E01c_{e}_*'))[-1] + '/predictions_PRIVATE.csv') for e in EP}
for e in EP: assert (FULL[e].id == E01C[e].id).all() and (FULL[e].y == E01C[e].y).all()
Y = {e: FULL[e].y.values for e in EP}; COMPLETE = FULL['Remitter'].complete_case.values
PRED = {e: {} for e in EP}
for e in EP:
    for s in SEEDS:
        PRED[e][('E01c', s)] = (E01C[e][f'p_E01c_s{s}'].values, E01C[e][f'dec_E01c_s{s}'].values)
        for m in MODS[1:]: PRED[e][(m, s)] = (FULL[e][f'p_{m}_s{s}'].values, FULL[e][f'dec_{m}_s{s}'].values)
def pm(y, p, d):
    tp = ((d == 1) & (y == 1)).sum(); fp = ((d == 1) & (y == 0)).sum(); tn = ((d == 0) & (y == 0)).sum(); fn = ((d == 0) & (y == 1)).sum(); ppv = tp / (tp + fp) if tp + fp else np.nan; sens = tp / (tp + fn); spec = tn / (tn + fp)
    return dict(AUC=roc_auc_score(y, p), AP=average_precision_score(y, p), PPV=ppv, sensitivity=sens, nPPV=100 * ppv / y.mean() if ppv == ppv else np.nan, Brier=float(np.mean((p - y) ** 2)), balanced_accuracy=(sens + spec) / 2, n_pos_calls=int(tp + fp), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
ALL = pd.DataFrame([dict(endpoint=e, model=m, seed=s, N=len(Y[e]), prevalence=Y[e].mean(), **pm(Y[e], *PRED[e][(m, s)])) for e in EP for m in MODS for s in SEEDS]); ALL.to_csv(f'{DATA}/all_metrics.csv', index=False)
NB = 2000; draws = np.random.default_rng(1).integers(0, 164, size=(NB, 164)); q = lambda v: (float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))); BOOT = {}
for e in EP:
    y = Y[e]; A = {m: np.full((NB, 3), np.nan) for m in MODS}; LL = {m: np.zeros(164) for m in MODS}
    for m in MODS:
        for s in SEEDS: p = np.clip(PRED[e][(m, s)][0], 1e-6, 1 - 1e-6); LL[m] += -(y * np.log(p) + (1 - y) * np.log(1 - p)) / 3
    for b, ix in enumerate(draws):
        if len(set(y[ix])) < 2: continue
        for m in MODS:
            for si, s in enumerate(SEEDS): A[m][b, si] = roc_auc_score(y[ix], PRED[e][(m, s)][0][ix])
    BOOT[e] = (A, LL)
S = []
for e in EP:
    A, LL = BOOT[e]; sub = ALL[ALL.endpoint == e]
    for m in MODS:
        a = sub[sub.model == m].AUC.values; g = a - sub[sub.model == 'E01c'].AUC.values; mb = A[m].mean(1); gb = (A[m] - A['E01c']).mean(1)
        S.append(dict(endpoint=e, model=m, auc_s42=a[0], auc_s43=a[1], auc_s44=a[2], mean_auc=a.mean(), mean_auc_lo=q(mb)[0], mean_auc_hi=q(mb)[1], gain_vs_E01c=g.mean(), gain_lo=q(gb)[0], gain_hi=q(gb)[1], all_gains_vs_E01c_positive=bool((g > 0).all()), mean_logloss=float(LL[m].mean())))
SCR = pd.DataFrame(S); SCR.to_csv(f'{DATA}/summary_and_intervals.csv', index=False)
PAIR = []
for e in EP:
    A, LL = BOOT[e]; sub = ALL[ALL.endpoint == e]
    for m in ['A', 'B', 'C', 'E01c']:
        da = sub[sub.model == 'D'].AUC.values - sub[sub.model == m].AUC.values; db = (A['D'] - A[m]).mean(1); ll = LL['D'] - LL[m]; llb = [ll[ix].mean() for ix in draws]
        PAIR.append(dict(endpoint=e, comparison=f'D - {m}', auc_diff_s42=da[0], auc_diff_s43=da[1], auc_diff_s44=da[2], mean_auc_diff=da.mean(), auc_lo=q(db)[0], auc_hi=q(db)[1], all_repeats_positive=bool((da > 0).all()), mean_patient_logloss_diff=float(ll.mean()), ll_lo=q(llb)[0], ll_hi=q(llb)[1], n_patients_D_lower_loss=int((ll < 0).sum())))
PAIR = pd.DataFrame(PAIR); PAIR.to_csv(f'{DATA}/paired_differences.csv', index=False)
# ---- frozen decision rules (Remitter)
r = SCR[(SCR.endpoint == 'Remitter') & (SCR.model == 'D')].iloc[0]; pr = PAIR[PAIR.endpoint == 'Remitter'].set_index('comparison')
funding = bool(r.mean_auc >= .60 and r.gain_vs_E01c >= .05 and r.all_gains_vs_E01c_positive)
def beats(c): x = pr.loc[f'D - {c}']; return bool(x.mean_auc_diff >= .05 and x.all_repeats_positive and x.auc_lo > 0)
helps = bool(funding and beats('A') and beats('C'))
DEC = dict(funding_D_vs_E01c=dict(mean_AUC_D=float(r.mean_auc), gain_vs_E01c=float(r.gain_vs_E01c), all_repeats_positive=bool(r.all_gains_vs_E01c_positive), threshold_mean_AUC=0.60, threshold_gain=0.05, met=funding),
           learned_reference_helps=dict(beats_A=beats('A'), beats_C=beats('C'), D_minus_A=float(pr.loc['D - A'].mean_auc_diff), D_minus_C=float(pr.loc['D - C'].mean_auc_diff), rule='funding AND (vs A and vs C separately: mean gain>=0.05, positive every repeat, interval excludes 0)', claim_supported=helps),
           extension_status='CLOSED (candidate D failed the funding rule)' if not funding else 'candidate passed funding rule; see helps flag', permutation=json.load(open(f'{OUT}/dev/permutation_D_remitter.json')))
json.dump(DEC, open(f'{OUT}/dev/decision_rules_result.json', 'w'), indent=1); print(json.dumps(DEC, indent=1))
# ---- complete-case sensitivity
rows = []
for e in EP:
    y = Y[e]; idx = np.flatnonzero(COMPLETE)
    for m in MODS:
        for s in SEEDS:
            pf = PRED[e][(m, s)][0]; row = dict(endpoint=e, model=m, seed=s, n=len(idx), auc_restricted_saved=roc_auc_score(y[idx], pf[idx]))
            if m != 'E01c': row['auc_refit_complete_only'] = roc_auc_score(y[idx], CC[e][f'p_{m}_s{s}'].values[idx])
            rows.append(row)
CCT = pd.DataFrame(rows); CCT.to_csv(f'{DATA}/complete_case_sensitivity.csv', index=False)
CCS = CCT.groupby(['endpoint', 'model'])[['auc_restricted_saved', 'auc_refit_complete_only']].mean().reset_index(); CCS.to_csv(f'{DATA}/complete_case_summary.csv', index=False)
# ---- figures
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, e in zip(axs, EP):
    for i, m in enumerate(MODS):
        rr = SCR[(SCR.endpoint == e) & (SCR.model == m)].iloc[0]
        for s, v in zip(SEEDS, (rr.auc_s42, rr.auc_s43, rr.auc_s44)): ax.plot(v, i + (s - 43) * .15, MK[s], color=COL[m], ms=4.5, alpha=.85)
        ax.plot([rr.mean_auc_lo, rr.mean_auc_hi], [i, i], color=COL[m], lw=2, alpha=.5); ax.plot(rr.mean_auc, i, 'D', color=COL[m], ms=7, mec='k')
    ax.axvline(.5, color='k', ls=':'); ax.axvline(.6, color='#999', ls='--', lw=.8); ax.set_xlabel('Outer-fold AUC'); ax.set_title(f'{e} (N=164, prevalence {Y[e].mean():.2f})', loc='left', fontsize=9)
axs[0].set_yticks(range(5)); axs[0].set_yticklabels([LAB[m] for m in MODS], fontsize=8); axs[0].invert_yaxis()
def ivtxt(e):
    s = SCR[SCR.endpoint == e]; t = []
    for m in MODS:
        x = s[s.model == m].iloc[0]; t.append(f'{m} {x.mean_auc:.3f} [{x.mean_auc_lo:.3f}, {x.mean_auc_hi:.3f}] ' + ('(excludes 0.5)' if x.mean_auc_lo > .5 or x.mean_auc_hi < .5 else '(includes 0.5)'))
    return f'{e}: ' + '; '.join(t) + '.'
TXT = {e: ivtxt(e) for e in EP}
save(fig, 'figV1_model_comparison', 'Same 164 patients and saved splits. Small markers = pooled out-of-fold AUC per CV repeat (circle 42, square 43, triangle 44; all shown); diamond + line = mean over repeats with joint paired patient-bootstrap 95% interval (conditional on saved predictions; excludes training/tuning uncertainty). Dotted 0.5; dashed 0.60 funding line. Generated interval check: ' + TXT['Remitter'] + ' ' + TXT['Responder'])
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6)); comps = ['A', 'B', 'C', 'E01c']
for k, e in enumerate(EP):
    p = PAIR[PAIR.endpoint == e].set_index('comparison').loc[[f'D - {c}' for c in comps]].reset_index()
    for i, rr in p.iterrows():
        axs[k].plot([rr.auc_lo, rr.auc_hi], [i, i], color='#D55E00', lw=2, alpha=.5); axs[k].plot(rr.mean_auc_diff, i, 'D', color='#D55E00', mec='k')
        for s, v in zip(SEEDS, (rr.auc_diff_s42, rr.auc_diff_s43, rr.auc_diff_s44)): axs[k].plot(v, i + (s - 43) * .13, MK[s], color='#D55E00', ms=3.5, alpha=.7)
    axs[k].axvline(0, color='k'); axs[k].axvline(.05, color='#999', ls='--', lw=.8); axs[k].set_yticks(range(4)); axs[k].set_yticklabels([f'D minus {c}' for c in comps]); axs[k].invert_yaxis(); axs[k].set_xlabel(f'{e}: AUC difference\n(>0 = D higher; dashed = +0.05 rule)'); axs[k].set_title(e, loc='left', fontsize=9)
p = PAIR[PAIR.endpoint == 'Remitter'].set_index('comparison').loc[[f'D - {c}' for c in comps]].reset_index()
for i, rr in p.iterrows(): axs[2].plot([rr.ll_lo, rr.ll_hi], [i, i], color='#7f7f7f', lw=2, alpha=.6); axs[2].plot(rr.mean_patient_logloss_diff, i, 'D', color='#7f7f7f', mec='k')
axs[2].axvline(0, color='k'); axs[2].set_yticks(range(4)); axs[2].set_yticklabels([]); axs[2].invert_yaxis(); axs[2].set_xlabel('Remitter: mean per-patient log-loss difference\n(<0 = D lower loss)'); fig.tight_layout()
save(fig, 'figV2_paired_differences', 'Candidate D (levels + EO-minus-predicted-EO shape) minus each comparator, all shown, none selected. Left/middle: mean-over-repeats AUC difference with joint patient-bootstrap interval (small markers = individual repeats). Right: patient-level paired log-loss difference averaged over repeats (Remitter). Conditional on saved predictions. A positive AUC difference against A/C addresses incremental information; a chance-association permutation test cannot.')
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4)); y = Y['Remitter']
for m in MODS:
    for s in SEEDS:
        p = PRED['Remitter'][(m, s)][0]; f, t, _ = roc_curve(y, p); axs[0].plot(f, t, color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5, label=LAB[m] if s == 42 else None); pp, rc, _ = precision_recall_curve(y, p); axs[1].plot(rc, pp, color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5)
axs[0].plot([0, 1], [0, 1], 'k:', lw=1); axs[0].set_xlabel('False-positive rate'); axs[0].set_ylabel('True-positive rate'); axs[0].legend(frameon=False, fontsize=6); axs[1].axhline(y.mean(), color='k', ls=':'); axs[1].set_xlabel('Recall (sensitivity)'); axs[1].set_ylabel('Precision (PPV)'); axs[1].set_ylim(0, 1); fig.tight_layout()
save(fig, 'figV3_roc_pr', 'Remitter pooled out-of-fold predictions; thick = repeat 42 (prespecified primary), thin = 43/44. Dotted: chance / prevalence 0.48. Descriptive only; thresholds were fixed inside training folds.')
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, m in zip(axs, ['D', 'E01c']):
    for s in SEEDS:
        p = PRED['Remitter'][(m, s)][0]; b = pd.qcut(p, 5, labels=False, duplicates='drop'); g = pd.DataFrame({'p': p, 'y': y, 'b': b}).groupby('b').agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size')); ax.plot(g.p, g.y, 'o-', color=COL[m], lw=1.6 if s == 42 else .6, alpha=1 if s == 42 else .5, ms=4)
        if s == 42: [ax.text(a, b_ + .02, f'n={n}', fontsize=6, ha='center') for a, b_, n in zip(g.p, g.y, g.n)]
    ax.plot([0, 1], [0, 1], 'k:'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('Mean predicted P(remission), 5 equal-count bins'); ax.set_ylabel('Observed remission fraction'); ax.set_title(LAB[m], loc='left', fontsize=9)
fig.tight_layout(); save(fig, 'figV4_calibration', 'Remitter calibration of pooled out-of-fold probabilities in five equal-count bins (thick = repeat 42; n per bin for 42). About +/-0.09 binomial noise per bin.')
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
for ax, e in zip(axs, EP):
    c = CCS[CCS.endpoint == e].set_index('model').reindex(MODS)
    ax.plot(c.auc_restricted_saved, range(5), 'o', color='#0072B2', label='saved predictions, restricted to complete patients'); ax.plot(c.auc_refit_complete_only, range(5), 's', color='#D55E00', label='refit on complete patients only')
    ax.axvline(.5, color='k', ls=':'); ax.set_xlabel(f'{e}: mean AUC over repeats (N={int(COMPLETE.sum())} complete)'); ax.set_title(e, loc='left', fontsize=9)
axs[0].set_yticks(range(5)); axs[0].set_yticklabels([LAB[m] for m in MODS], fontsize=8); axs[0].invert_yaxis(); axs[0].legend(frameon=False, fontsize=7, loc='lower left'); fig.tight_layout()
save(fig, 'figV5_complete_case', f'Complete-case sensitivity (patients with complete EC and EO vectors, N={int(COMPLETE.sum())} of 164). Blue: saved full-cohort predictions evaluated on complete patients only (descriptive; E01c shown for the same patients). Orange: models A-D refit with the saved outer folds restricted to complete patients (E01c not refit). Mean AUC over three repeats; no intervals shown (small N).')
json.dump(cap, open(f'{FIG}/captions_dev.json', 'w'), indent=1)
pd.set_option('display.width', 220); print(SCR[['endpoint', 'model', 'auc_s42', 'auc_s43', 'auc_s44', 'mean_auc', 'mean_auc_lo', 'mean_auc_hi', 'gain_vs_E01c', 'gain_lo', 'gain_hi']].round(3).to_string(index=False)); print(PAIR.round(3).to_string(index=False)); print(CCS.round(3).to_string(index=False))
