"""DEVIATION_PROTOCOL sections 2a/2b + criteria F1/F2. Outcome-blind; follow-up analysis on the already-examined 152-person reference evaluation set."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import dev
FIG = 'outputs/figures_dev'; plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
X, tr, te = dev.load_reference(); Ec, Eo = X[dev.ECc].values, X[dev.EOc].values; lc, sc = dev.level_shape(Ec); lo, so = dev.level_shape(Eo); assert np.isfinite(sc).all() and np.isfinite(so).all()
out = dict(n_reference_total=len(X), n_train=len(tr), n_eval=len(te))
# identity check of the uncentered mapping from the previous stage
Mu = dev.Mapping(Ec[tr], Eo[tr]); pu = Mu.predict(Ec[te]); zO = (Eo[te] - Mu.mo) / Mu.so; r2z = 1 - ((zO - (pu - Mu.mo) / Mu.so) ** 2).mean(0) / (zO ** 2).mean(0)
prev = json.load(open('outputs/pool/reference_mapping_summary.json')); out['uncentered_identity_check'] = dict(alpha=Mu.alpha, alpha_previous=prev['alpha_selected'], mean_R2=float(r2z.mean()), mean_R2_previous=prev['mean_R2']); assert Mu.alpha == prev['alpha_selected'] and abs(r2z.mean() - prev['mean_R2']) < 1e-9
rng = np.random.default_rng(1); BS = rng.integers(0, len(te), size=(2000, len(te))); ci = lambda v: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
# ---- 2a shape mapping vs (a) mean shape, (b) persistence + mean shift
Ms = dev.Mapping(sc[tr], so[tr]); pred = Ms.predict(sc[te]); pa = np.tile(so[tr].mean(0), (len(te), 1)); pb = sc[te] + (so[tr] - sc[tr]).mean(0)
per = {k: ((so[te] - p) ** 2) for k, p in (('ridge_shape', pred), ('a_mean_EO_shape', pa), ('b_persistence_plus_mean_shift', pb))}; pm = {k: v.mean(1) for k, v in per.items()}
def paired(base, model):
    d = pm[base] - pm[model]; db = d[BS].mean(1); return dict(mean_MSE_base=float(pm[base].mean()), mean_MSE_model=float(pm[model].mean()), mean_reduction=float(d.mean()), reduction_ci=ci(db), relative_reduction=float(d.mean() / pm[base].mean()), relative_ci=ci(db / pm[base][BS].mean(1)))
pa_, pb_ = paired('a_mean_EO_shape', 'ridge_shape'), paired('b_persistence_plus_mean_shift', 'ridge_shape'); stronger = 'a_mean_EO_shape' if pm['a_mean_EO_shape'].mean() < pm['b_persistence_plus_mean_shift'].mean() else 'b_persistence_plus_mean_shift'
rel_vs_stronger = float(1 - pm['ridge_shape'].mean() / pm[stronger].mean())
out['shape_mapping'] = dict(alpha=Ms.alpha, vs_a_mean_shape=pa_, vs_b_persistence=pb_, stronger_baseline=stronger, relative_reduction_vs_stronger=rel_vs_stronger, aggregate_MSE={k: float(v.mean()) for k, v in pm.items()}, per_element_MSE={n: {k: float(per[k][:, j].mean()) for k in per} for j, n in enumerate(dev.NAMES)},
                          note='MSE in squared natural-log-power units; per-person mean over 12 elements; bootstrap over the 152 evaluation people, conditional on the fixed fit')
F1 = bool(pm['ridge_shape'].mean() < min(pm['a_mean_EO_shape'].mean(), pm['b_persistence_plus_mean_shift'].mean()) and pa_['reduction_ci'][0] > 0 and pb_['reduction_ci'][0] > 0 and rel_vs_stronger >= 0.05)
# ---- 2b uncentered mapping vs amplitude-only (OLS of each EO element on mean EC log power)
A = np.column_stack([np.ones(len(tr)), lc[tr]]); coef = np.linalg.lstsq(A, Eo[tr], rcond=None)[0]; pamp = np.column_stack([np.ones(len(te)), lc[te]]) @ coef
pm2 = {'ridge_uncentered': ((Eo[te] - pu) ** 2).mean(1), 'amplitude_only': ((Eo[te] - pamp) ** 2).mean(1)}; d2 = pm2['amplitude_only'] - pm2['ridge_uncentered']; d2b = d2[BS].mean(1)
out['uncentered_vs_amplitude_only'] = dict(MSE_ridge=float(pm2['ridge_uncentered'].mean()), MSE_amplitude_only=float(pm2['amplitude_only'].mean()), mean_reduction=float(d2.mean()), reduction_ci=ci(d2b), relative_reduction=float(d2.mean() / pm2['amplitude_only'].mean()),
                                         per_element_MSE={n: dict(ridge=float(((Eo[te][:, j] - pu[:, j]) ** 2).mean()), amplitude_only=float(((Eo[te][:, j] - pamp[:, j]) ** 2).mean())) for j, n in enumerate(dev.NAMES)})
F2 = bool(out['uncentered_vs_amplitude_only']['reduction_ci'][0] > 0 and out['uncentered_vs_amplitude_only']['relative_reduction'] >= 0.05)
out['criteria'] = dict(F1_shape_mapping_improvement_GATING=F1, F2_uncentered_beats_amplitude_only_INFORMATIONAL=F2, F1_detail='ridge MSE < both baselines, both paired intervals exclude 0, relative reduction vs stronger baseline >= 5%', variance_of_level_share=float(np.var(lc) / np.var(Ec, axis=0).mean()) if False else None)
# context (not gating): how much of EO log-power variance is carried by level vs shape
out['context'] = dict(var_share_level_in_EO=float(12 * np.var(lo[te]) / np.var(Eo[te], axis=0).sum()), corr_EC_EO_level=float(np.corrcoef(lc, lo)[0, 1]))
json.dump(out, open('outputs/dev/reference_diagnostics.json', 'w'), indent=1); print(json.dumps({k: out[k] for k in ('identity' if False else 'uncentered_identity_check', 'criteria', 'context')}, indent=1)); print(json.dumps({k: v for k, v in out['shape_mapping'].items() if k != 'per_element_MSE'}, indent=1)); print(json.dumps({k: v for k, v in out['uncentered_vs_amplitude_only'].items() if k != 'per_element_MSE'}, indent=1))
# ---- figure
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.8), gridspec_kw={'width_ratios': [1, 1.4, 1.4]})
items = [('shape ridge vs (a) mean EO shape', pa_['reduction_ci'], pa_['mean_reduction']), ('shape ridge vs (b) persistence + mean shift', pb_['reduction_ci'], pb_['mean_reduction']), ('uncentered ridge vs amplitude-only (informational)', out['uncentered_vs_amplitude_only']['reduction_ci'], out['uncentered_vs_amplitude_only']['mean_reduction'])]
for i, (l, c, m) in enumerate(items): axs[0].plot(c, [i, i], color='#0072B2', lw=2, alpha=.5); axs[0].plot(m, i, 'D', color='#0072B2', mec='k')
axs[0].axvline(0, color='k'); axs[0].set_yticks(range(3)); axs[0].set_yticklabels([x[0] for x in items], fontsize=7); axs[0].invert_yaxis(); axs[0].set_xlabel('Paired held-out MSE reduction\n(>0 = ridge better; ln-power$^2$)')
y = np.arange(12); S = out['shape_mapping']['per_element_MSE']
axs[1].barh(y - .27, [S[n]['ridge_shape'] for n in dev.NAMES], .27, color='#0072B2', label='shape ridge'); axs[1].barh(y, [S[n]['a_mean_EO_shape'] for n in dev.NAMES], .27, color='#999999', label='(a) mean EO shape'); axs[1].barh(y + .27, [S[n]['b_persistence_plus_mean_shift'] for n in dev.NAMES], .27, color='#E69F00', label='(b) persistence + mean shift')
axs[1].set_yticks(y); axs[1].set_yticklabels(dev.NAMES, fontsize=7); axs[1].invert_yaxis(); axs[1].set_xlabel('Held-out MSE per element (shape; ln-power$^2$)'); axs[1].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=3)
U = out['uncentered_vs_amplitude_only']['per_element_MSE']; axs[2].barh(y - .18, [U[n]['ridge'] for n in dev.NAMES], .36, color='#0072B2', label='uncentered ridge'); axs[2].barh(y + .18, [U[n]['amplitude_only'] for n in dev.NAMES], .36, color='#D55E00', label='amplitude-only (mean EC log power)')
axs[2].set_yticks(y); axs[2].set_yticklabels(dev.NAMES, fontsize=7); axs[2].invert_yaxis(); axs[2].set_xlabel('Held-out MSE per element (uncentered; ln-power$^2$)'); axs[2].legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2); fig.tight_layout()
fig.savefig(f'{FIG}/figF1_mapping_vs_baselines.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/figF1_mapping_vs_baselines.pdf', bbox_inches='tight')
pd.DataFrame([dict(element=n, **{k: v for k, v in S[n].items()}) for n in dev.NAMES]).to_csv(f'{FIG}/plotted_data/shape_mapping_per_element_mse.csv', index=False)
