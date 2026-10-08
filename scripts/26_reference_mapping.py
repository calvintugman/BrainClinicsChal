"""Outcome-blind reference-pool EC->EO ridge mapping (PIVOT_PROTOCOL_DRAFT section 4 as amended). NO remission labels are read. No alternative mappings are tried.
Pool: non-treatment, non-rTMS, non-replication discovery people (see 19_pool_features.py), one session per person (lowest session with EC and EO), complete cases for the 12 EC and 12 EO log-abs power features (3 ROIs x 4 bands, v2 pipeline).
Fixed split: np.random.default_rng(2026) permutation, first 20% of people = held-out test. Ridge alpha grid {0.01,0.1,1,10,100,1000}, tuned ONLY by 5-fold CV inside the training people (mean MSE over the 12 standardized outputs).
Baseline: training-mean EO predictor. R2 = 1 - MSE_model / MSE_training_mean on held-out people. Shuffled-pair control: EO rows permuted among training people (20 shuffles, seeds 0..19), alpha tuned within training, evaluated on true held-out pairs."""
import os, sys, json, glob, numpy as np, pandas as pd, hashlib, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
OUT = 'outputs'; FIG = f'{OUT}/figures_dynamix'; os.makedirs(FIG + '/plotted_data', exist_ok=True); plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
ROIS = ['frontal', 'central', 'posterior']; BANDS = ['delta', 'theta', 'alpha', 'beta']; ECc = [f'EC_{r}_{b}_logabs' for r in ROIS for b in BANDS]; EOc = [f'EO_{r}_{b}_logabs' for r in ROIS for b in BANDS]; GRID = [0.01, 0.1, 1, 10, 100, 1000]
F = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv'); L = pd.read_csv(f'{OUT}/pool/cleaning_log_pool_v2_PRIVATE.csv'); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv')
led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv'); wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx', dtype={'TDBRAIN_ID': str}); treat = set(wb[wb.Dataset.notna()].TDBRAIN_ID)
assert not (set(P.id) & treat) and not (set(P.id) & set(led.id)), 'pool overlaps treatment cohorts'
exc = {'pool_people_listed': len(P), 'recording_statuses': L.status.value_counts().to_dict()}; F = F.set_index('id').reindex(P.id)
cc = F[ECc + EOc].notna().all(1); exc['excluded_missing_EC_or_EO_features'] = int((~cc).sum()); X = F[cc]
# duplicates: identical feature vectors across different people (would overweight) -> keep first
dup = X.duplicated(subset=ECc + EOc, keep='first'); exc['dropped_exact_duplicate_feature_vectors'] = int(dup.sum()); X = X[~dup]; exc['people_used'] = len(X)
rng = np.random.default_rng(2026); perm = rng.permutation(len(X)); nte = int(round(.2 * len(X))); te, tr = perm[:nte], perm[nte:]; exc.update(train_people=len(tr), heldout_people=len(te))
Xe, Xo = X[ECc].values, X[EOc].values; mu_e, sd_e = Xe[tr].mean(0), Xe[tr].std(0); mu_o, sd_o = Xo[tr].mean(0), Xo[tr].std(0)
zE = (Xe - mu_e) / sd_e; zO = (Xo - mu_o) / sd_o
def fit_tune(Xin, Yin):
    cv = KFold(5, shuffle=True, random_state=0); sc = {}
    for a in GRID: sc[a] = np.mean([np.mean((Ridge(alpha=a).fit(Xin[i], Yin[i]).predict(Xin[j]) - Yin[j]) ** 2) for i, j in cv.split(Xin)])
    a = min(sc, key=sc.get); return Ridge(alpha=a).fit(Xin, Yin), a, sc
mdl, alpha, cvs = fit_tune(zE[tr], zO[tr]); pred = mdl.predict(zE[te]); mse_mean = ((zO[te] - 0.0) ** 2).mean(0)             # training mean = 0 in z units
mse_mod = ((zO[te] - pred) ** 2).mean(0); r2 = 1 - mse_mod / mse_mean
# shuffled-pair control
sh = []
for s in range(20):
    pr = np.random.default_rng(s).permutation(len(tr)); m2, a2, _ = fit_tune(zE[tr], zO[tr][pr]); p2 = m2.predict(zE[te]); sh.append(dict(seed=s, alpha=a2, mean_r2=float((1 - ((zO[te] - p2) ** 2).mean(0) / mse_mean).mean()), pooled_mse_ratio=float(((zO[te] - p2) ** 2).mean() / mse_mean.mean()), r2_by_feature=(1 - ((zO[te] - p2) ** 2).mean(0) / mse_mean).tolist()))
SH = pd.DataFrame(sh)
# bootstrap over held-out people (conditional on the fixed fit)
bs = np.random.default_rng(1).integers(0, nte, size=(2000, nte)); r2b = np.array([1 - ((zO[te][i] - pred[i]) ** 2).mean(0) / (zO[te][i] ** 2).mean(0) for i in bs])
rows = [dict(feature=c.replace('EO_', ''), mse_model_z=mse_mod[k], mse_trainmean_z=mse_mean[k], R2=r2[k], R2_lo=np.quantile(r2b[:, k], .025), R2_hi=np.quantile(r2b[:, k], .975), mse_model_logpower=float(((Xo[te][:, k] - (pred[:, k] * sd_o[k] + mu_o[k])) ** 2).mean()), mse_trainmean_logpower=float(((Xo[te][:, k] - mu_o[k]) ** 2).mean()), shuffled_R2_mean=float(np.mean([x[k] for x in SH.r2_by_feature])), shuffled_R2_min=float(np.min([x[k] for x in SH.r2_by_feature])), shuffled_R2_max=float(np.max([x[k] for x in SH.r2_by_feature]))) for k, c in enumerate(EOc)]
R = pd.DataFrame(rows); R.to_csv(f'{FIG}/plotted_data/reference_mapping_per_feature.csv', index=False)
agg = dict(**exc, alpha_selected=alpha, cv_mse_by_alpha={str(k): float(v) for k, v in cvs.items()}, mean_R2=float(r2.mean()), mean_R2_ci=[float(np.quantile(r2b.mean(1), .025)), float(np.quantile(r2b.mean(1), .975))], pooled_mse_ratio_vs_trainmean=float(mse_mod.mean() / mse_mean.mean()), features_with_R2_gt0=int((r2 > 0).sum()),
           shuffled_mean_R2_mean=float(SH.mean_r2.mean()), shuffled_mean_R2_range=[float(SH.mean_r2.min()), float(SH.mean_r2.max())], shuffled_pooled_mse_ratio_mean=float(SH.pooled_mse_ratio.mean()), heldout_ids_sha256=hashlib.sha256(','.join(sorted(X.index[te])).encode()).hexdigest()[:16], shuffles=20,
           pool_group_counts=P.set_index('id').loc[X.index].group.value_counts().to_dict(), no_outcome_labels_read=True)
json.dump(agg, open(f'{OUT}/pool/reference_mapping_summary.json', 'w'), indent=1); pd.DataFrame({'id': X.index[te]}).to_csv(f'{OUT}/pool/heldout_reference_ids_PRIVATE.csv', index=False); print(json.dumps(agg, indent=1))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={'width_ratios': [1.7, 1]}); y = np.arange(12)
axs[0].barh(y - .18, R.R2, .36, xerr=[R.R2 - R.R2_lo, R.R2_hi - R.R2], color='#0072B2', label='ridge EC->EO (held-out)', error_kw=dict(lw=1)); axs[0].barh(y + .18, R.shuffled_R2_mean, .36, xerr=[R.shuffled_R2_mean - R.shuffled_R2_min, R.shuffled_R2_max - R.shuffled_R2_mean], color='#999999', label='shuffled-pair control (mean, min-max of 20)', error_kw=dict(lw=1))
axs[0].set_yticks(y); axs[0].set_yticklabels(R.feature.str.replace('_logabs', ''), fontsize=7); axs[0].invert_yaxis(); axs[0].axvline(0, color='k'); axs[0].set_xlabel('Held-out R$^2$ vs training-mean predictor (>0 = better than mean)'); axs[0].legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2)
axs[1].hist(SH.mean_r2, bins=10, color='#999999'); axs[1].axvline(r2.mean(), color='#0072B2', lw=2, label=f'real pairs: {r2.mean():.3f}'); axs[1].set_xlabel('Mean held-out R$^2$ over 12 outputs'); axs[1].set_ylabel('Shuffles'); axs[1].legend(frameon=False, fontsize=8); fig.tight_layout()
fig.savefig(f'{FIG}/figR1_reference_mapping.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/figR1_reference_mapping.pdf', bbox_inches='tight')
