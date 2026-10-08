"""Outcome-blind QC sample (fixed seed) comparing v1 vs v2 cleaning. Output: outputs/pivot_qc_sample_PRIVATE.csv + figures."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from common import *
import importlib; f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2')
cl = pd.read_csv(f'{OUT}/cleaning_log_PRIVATE.csv'); rng = np.random.default_rng(0); pick = []
def take(df, n, tag): 
    d = df.sample(min(n, len(df)), random_state=0); pick.extend([(r.id, r.cond, tag) for r in d.itertuples()])
take(cl[cl.ica_iter >= 1000], 6, 'v1_ica_hit_max_iter'); take(cl[cl.n_ica_excl == 3], 6, 'v1_cap3'); take(cl[cl.n_bad > 0], 6, 'interpolated')
take(cl[cl.status == 'too_few_windows'], 3, 'v1_too_few_windows'); take(cl[(cl.n_bad == 0) & (cl.n_ica_excl <= 2) & (cl.ica_iter < 500) & (cl.status == 'ok')], 6, 'unflagged_control')
seen = set(); sample = [p for p in pick if not (p[:2] in seen or seen.add(p[:2]))]; led = load_ledger().set_index('id')
def run(pid, cond, tag):
    ses = led.loc[pid, 'selected_ses']; o = {}
    r1, l1 = f3.clean(pid, ses, cond); r2, l2 = cv2.clean(pid, ses, cond)
    rec = dict(id=pid, cond=cond, stratum=tag, v1_ica_iter=l1.get('ica_iter'), v1_excl=l1.get('ica_excl'), v1_n_excl=l1.get('n_ica_excl'), v2_status=l2.get('status'), v2_iter=l2.get('ica_iter'), v2_converged=l2.get('ica_converged'), v2_retried=l2.get('ica_retried'),
               v2_candidates_z=l2.get('n_candidates_z'), v2_accepted=l2.get('n_candidates_accepted'), v2_excl=l2.get('ica_excl'), v2_n_excl=l2.get('n_ica_excl'), v2_beyond_cap=l2.get('n_cand_beyond_cap'), v2_rejected_not_frontal=l2.get('cand_rejected_not_frontal'), v2_eff_rank=l2.get('eff_rank'), n_interp=l1.get('n_bad'))
    for nm, r in (('v1', r1), ('v2', r2)):
        if r is not None: x = r.get_data(picks='eeg') * 1e6; st = f3.windows(x, []); rec[f'{nm}_clean_win'] = len(st)
    return rec
rows = Parallel(n_jobs=10)(delayed(run)(*s) for s in sample); Q = pd.DataFrame(rows); Q.to_csv(f'{OUT}/pivot_qc_sample_PRIVATE.csv', index=False)
pd.set_option('display.width', 250, 'display.max_columns', 30); print(Q.drop(columns='id').to_string())
