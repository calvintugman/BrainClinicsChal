"""Programmatic audit checks (AUDIT.md). Prints results; writes outputs/audit_checks.json"""
import glob, json, numpy as np, pandas as pd, pickle
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy.stats import spearmanr
OUT = 'outputs'; res = {}
F = pd.read_csv(f'{OUT}/features_PRIVATE.csv'); led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv'); inc = led[led.selected_ses.notna()]
wb = pd.read_excel('../TDBRAIN_participants_V3.1.xlsx', dtype={'TDBRAIN_ID': str}); wb = wb[wb.Dataset == 'MDD-rTMS']
# 1 join / alignment / class mapping
res['1_unique_feature_ids'] = bool(F.id.is_unique); res['1_ledger_ids_unique'] = bool(inc.id.is_unique); res['1_feature_ids_eq_included'] = set(F.id) == set(inc.id)
for e in ('Remitter', 'Responder'):
    r = sorted(glob.glob(f'runs/R_rest_{e}_*'))[-1]; P = pd.read_csv(f'{r}/predictions_PRIVATE.csv')
    D = F.merge(inc[['id', e]], on='id').sort_values('id').reset_index(drop=True)
    res[f'1_{e}_pred_ids_match_sorted_features'] = bool((P.id.values == D.id.values).all()); res[f'1_{e}_y_matches_ledger'] = bool((P.y.values == D[e].astype(int).values).all())
    # independent label from workbook (raw rows, max over person)
    wl = wb.groupby('TDBRAIN_ID')[e].max(); res[f'1_{e}_y_matches_workbook'] = bool((P.set_index('id').y == wl.reindex(P.id).values).all())
    res[f'1_{e}_class_values'] = sorted(map(int, P.y.unique()))
    # direction check using a label-free logic: probability higher for fold prevalence? instead check predict_proba column mapping by AUC symmetry
    M = pd.read_csv(f'{r}/metrics_by_seed.csv'); rows = []
    for _, m in M.iterrows():
        p = P[f'p_{m.model}_s{int(m.seed)}'].values; d = P[f'dec_{m.model}_s{int(m.seed)}'].values; y = P.y.values
        tp = ((d == 1) & (y == 1)).sum(); fp = ((d == 1) & (y == 0)).sum(); tn = ((d == 0) & (y == 0)).sum(); fn = ((d == 0) & (y == 1)).sum()
        auc = roc_auc_score(y, p) if len(np.unique(p)) > 1 else 0.5
        rows.append(dict(model=m.model, seed=int(m.seed), auc_diff=abs(auc - m.AUC), brier_diff=abs(brier_score_loss(y, p) - m.Brier), counts_ok=bool((tp, fp, tn, fn) == (m.TP, m.FP, m.TN, m.FN)), ppv_diff=abs((tp / (tp + fp) if tp + fp else np.nan) - m.PPV) if tp + fp else 0))
    R = pd.DataFrame(rows); res[f'3_{e}_recompute_max_abs_diff'] = dict(auc=float(R.auc_diff.max()), brier=float(R.brier_diff.max()), ppv=float(R.ppv_diff.max()), counts_all_ok=bool(R.counts_ok.all()), n_rows=len(R))
    # fold ledger check: splits reproduced from seeds + patients disjoint
    S = pd.read_csv(f'{r}/splits_PRIVATE.csv'); res[f'2_{e}_each_patient_one_fold_per_seed'] = bool((S.groupby(['seed', 'id']).fold.nunique() == 1).all()); res[f'2_{e}_fold_class_counts'] = S.merge(P[['id', 'y']]).groupby(['seed', 'fold']).y.agg(['size', 'sum']).reset_index().values.tolist()[:5]
    # threshold-rule sanity: sensitivity achieved on test vs 0.5 target
    res[f'2_{e}_E01_min_test_sens'] = float(M[M.model == 'E01'].sensitivity.min())
# 7 caption reconciliation: recompute from numeric endpoints
stmts = []
for e in ('Remitter', 'Responder'):
    r = sorted(glob.glob(f'runs/R_rest_{e}_*'))[-1]; B = pd.read_csv(f'{r}/bootstrap_intervals.csv')
    for _, b in B[(B.kind == 'diff') & (B.name == 'E01 - E01-PSD')].iterrows():
        stmts.append(dict(endpoint=e, seed=int(b.seed), metric=b.metric, est=round(b.est, 3), lo=round(b.lo, 3), hi=round(b.hi, 3), excludes_zero=bool(b.lo > 0 or b.hi < 0), side='above' if b.lo > 0 else ('below' if b.hi < 0 else 'spans')))
    for _, b in B[(B.kind == 'model') & (B.metric == 'AUC')].iterrows():
        stmts.append(dict(endpoint=e, seed=int(b.seed), metric='AUC abs ' + b['name'], lo=round(b.lo, 3), hi=round(b.hi, 3), excludes_0_5=bool(b.lo > .5 or b.hi < .5), side='above .5' if b.lo > .5 else ('below .5' if b.hi < .5 else 'spans .5')))
S7 = pd.DataFrame(stmts); S7.to_csv(f'{OUT}/audit_interval_statements.csv', index=False); pd.set_option('display.width', 200); print(S7[(S7.get('excludes_zero') == True) | (S7.get('excludes_0_5') == True)].to_string(index=False))
# 6 reliability: valid N, half-boundary overlap (halves split by retained index, adjacent windows overlap 2 s)
W = pickle.load(open(f'{OUT}/windows_PRIVATE.pkl', 'rb')); n_valid = {}
for c in ('EC', 'EO'): n_valid[c] = sum(1 for w in W.values() if c in w)
res['6_patients_with_windows'] = n_valid; res['6_note'] = 'half split by retained-window index; boundary windows can overlap by up to 2 s (50% overlap) -> small optimistic bias; reliability fig recomputed with a 1-window gap in 07_audit_fixes'
cl = pd.read_csv(f'{OUT}/cleaning_log_PRIVATE.csv'); res['6_status_counts'] = cl.status.value_counts().to_dict(); res['6_ica_ok'] = cl.ica_ok.value_counts().to_dict(); res['6_ica_iter_max'] = float(cl.ica_iter.max()); res['6_n_ica_excl_dist'] = cl.n_ica_excl.value_counts().to_dict(); res['6_recordings_with_bad_ch'] = int((cl.n_bad > 0).sum()); res['6_ica_exclusion_at_cap3'] = int((cl.n_ica_excl == 3).sum())
res['6_nan_counts'] = {k: int(v) for k, v in F.isna().sum().items() if v}
json.dump(res, open(f'{OUT}/audit_checks.json', 'w'), indent=1, default=str); print(json.dumps({k: v for k, v in res.items() if not k.startswith('6_nan')}, indent=1, default=str))
