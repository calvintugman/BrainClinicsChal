"""DEVIATION_PROTOCOL section 4: reference applicability (descriptive; aggregate outputs only). Diagnoses/demographics are never predictors."""
import sys, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, 'scripts'); import dev
OUT = 'outputs'; FIG = 'outputs/figures_dev'; plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
X, tr, te = dev.load_reference(); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv'); L = pd.read_csv(f'{OUT}/pool/cleaning_log_pool_v2_PRIVATE.csv')
wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx', dtype={'TDBRAIN_ID': str}).drop_duplicates('TDBRAIN_ID').set_index('TDBRAIN_ID')
P['included'] = P.id.isin(X.index); P['partition'] = np.where(~P.included, 'excluded (missing EC/EO features)', np.where(P.id.isin(set(X.index[tr])), 'reference-training', 'reference-evaluation'))
for c in ('age', 'gender', 'indication', 'sessSeason', 'sessTime'): P[c] = P.id.map(wb[c])
L1 = L.pivot(index='id', columns='cond', values='n_clean_win'); P['clean_win_EC'] = P.id.map(L1['EC']); P['clean_win_EO'] = P.id.map(L1['EO'])
for c in ('n_bad', 'n_ica_excl'): lp = L.pivot(index='id', columns='cond', values=c); P[f'{c}_EC'] = P.id.map(lp['EC']); P[f'{c}_EO'] = P.id.map(lp['EO'])
P['healthy'] = P.group.eq('healthy_verified_by_workbook_indication'); P['female'] = P.gender.eq(0)
def smd(a, b): a, b = pd.Series(a).dropna(), pd.Series(b).dropna(); s = np.sqrt((a.var() + b.var()) / 2); return float((a.mean() - b.mean()) / s) if s > 0 else np.nan
rows = []
for lab, g in P.groupby('partition'):
    rows.append(dict(group=lab, n=len(g), age_median=g.age.median(), age_iqr=f'{g.age.quantile(.25):.0f}-{g.age.quantile(.75):.0f}', female_pct=100 * g.female.mean(), workbook_healthy_pct=100 * g.healthy.mean(), clean_windows_EC_median=g.clean_win_EC.median(), clean_windows_EO_median=g.clean_win_EO.median(), interp_ch_EC_mean=g.n_bad_EC.mean(), interp_ch_EO_mean=g.n_bad_EO.mean(), ica_removed_EC_mean=g.n_ica_excl_EC.mean(), ica_removed_EO_mean=g.n_ica_excl_EO.mean()))
INC = pd.DataFrame(rows); INC.to_csv(f'{FIG}/plotted_data/reference_included_vs_excluded.csv', index=False)
inc, exc = P[P.included], P[~P.included]; cmp = {k: smd(inc[k], exc[k]) for k in ['age', 'female', 'clean_win_EC', 'clean_win_EO', 'n_bad_EC', 'n_bad_EO', 'n_ica_excl_EC', 'n_ica_excl_EO']}; cmp['healthy_pct_included_minus_excluded'] = float(100 * (inc.healthy.mean() - exc.healthy.mean()))
topind = P.assign(ind=P.indication.fillna('NA')).groupby(['partition', 'ind']).size().rename('n').reset_index(); topind = topind.sort_values('n', ascending=False).groupby('partition').head(6); topind.to_csv(f'{FIG}/plotted_data/reference_indication_counts_top6.csv', index=False)
# patients vs reference-training: levels, shapes, coverage, missingness
F = pd.read_csv(f'{OUT}/v2/features_v2_PRIVATE.csv').set_index('id'); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses'); F = F.reindex(sorted(led.id)); EcP, EoP = F[dev.ECc].values, F[dev.EOc].values
lcP, scP = dev.level_shape(EcP); loP, soP = dev.level_shape(EoP); Ec, Eo = X[dev.ECc].values, X[dev.EOc].values; lcR, scR = dev.level_shape(Ec[tr]); loR, soR = dev.level_shape(Eo[tr])
cov = []
def row(name, p, r): 
    lo_, hi_ = np.percentile(r, [1, 99]); pv = p[np.isfinite(p)]; cov.append(dict(variable=name, n_patients=len(pv), reference_train_mean=float(r.mean()), reference_train_sd=float(r.std()), patient_mean=float(pv.mean()), SMD=float((pv.mean() - r.mean()) / np.sqrt((pv.var() + r.var()) / 2)), pct_inside_ref_1_99=float(100 * ((pv >= lo_) & (pv <= hi_)).mean())))
row('EC level', lcP, lcR); row('EO level', loP, loR)
for j, n in enumerate(dev.NAMES): row(f'EC shape {n}', scP[:, j], scR[:, j])
for j, n in enumerate(dev.NAMES): row(f'EO shape {n}', soP[:, j], soR[:, j])
COV = pd.DataFrame(cov); COV.to_csv(f'{FIG}/plotted_data/patient_coverage_vs_reference_train.csv', index=False)
miss = dict(patients_EC_incomplete=int(np.isnan(lcP).sum()), patients_EO_incomplete=int(np.isnan(loP).sum()), patients_either_incomplete=int((np.isnan(lcP) | np.isnan(loP)).sum()), patients_total=len(F), reference_pool_incomplete_pct=float(100 * (~P.included).mean()), reference_people_listed=len(P), reference_complete=int(P.included.sum()),
            patient_recording_status_pct={k: float(v) for k, v in (pd.read_csv(f'{OUT}/v2/cleaning_log_v2_PRIVATE.csv').status.value_counts(normalize=True) * 100).items()}, pool_recording_status_pct={k: float(v) for k, v in (L.status.value_counts(normalize=True) * 100).items()},
            patient_age_median=float(wb.loc[F.index, 'age'].median()), patient_female_pct=float(100 * (wb.loc[F.index, 'gender'] == 0).mean()), reference_train_age_median=float(P[P.partition == 'reference-training'].age.median()), reference_train_female_pct=float(100 * P[P.partition == 'reference-training'].female.mean()))
out = dict(included_vs_excluded_SMD=cmp, missingness=miss, coverage_min_pct=float(COV.pct_inside_ref_1_99.min()), coverage_median_pct=float(COV.pct_inside_ref_1_99.median()), n_variables_below_80pct=int((COV.pct_inside_ref_1_99 < 80).sum()), max_abs_SMD=float(COV.SMD.abs().max()), variables_below_80=COV[COV.pct_inside_ref_1_99 < 80].variable.tolist())
json.dump(out, open('outputs/dev/applicability.json', 'w'), indent=1, default=float); print(json.dumps(out, indent=1, default=float)); print(INC.round(2).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 5.2)); C = COV.iloc[::-1]
axs[0].barh(C.variable, C.pct_inside_ref_1_99, color=['#D55E00' if v < 80 else '#0072B2' for v in C.pct_inside_ref_1_99]); axs[0].axvline(98, color='k', ls=':', lw=.8); axs[0].axvline(80, color='#D55E00', ls='--', lw=.8); axs[0].set_xlabel('% of patients inside reference-training 1st-99th percentile range\n(98% expected if identical; dashed = 80% flag)'); axs[0].tick_params(axis='y', labelsize=6)
axs[1].barh(C.variable, C.SMD, color='#7f7f7f'); axs[1].axvline(0, color='k'); axs[1].set_xlabel('Standardized mean difference, patients minus reference-training'); axs[1].set_yticklabels([]); fig.tight_layout()
fig.savefig(f'{FIG}/figA1_coverage.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/figA1_coverage.pdf', bbox_inches='tight')
