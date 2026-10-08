"""Reference pool (non-treatment, non-rTMS, non-replication) v2 features. One session per person (lowest session with both EC and EO). Resumable chunks.
Excludes: all MDD-rTMS (incl. 3 timing-excluded), ADHD_NF, OCD, ADHD_NF/OCD treatment-cohort people; replication folders are never read."""
import os, sys, pickle, time, glob
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, 'scripts')
from common import *
def process_v2(pid, ses):
    import importlib; f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2'); f3.clean = cv2.clean
    feats, logs, win, rt = f3.process(pid, ses); return feats, logs
def pool_list():
    M = pd.read_csv(f'{OUT}/pivot_manifest_PRIVATE.csv'); M = M[M.group.isin(['healthy_verified_by_workbook_indication', 'other_discovery_unlabeled_for_remission'])]
    rows = []
    for pid, g in M.groupby('id'):
        for ses in sorted(g.ses.unique()):
            t = set(g[g.ses == ses].task)
            if {'restEC', 'restEO'} <= t: rows.append(dict(id=pid, ses=ses, group=g.group.iloc[0], n_sessions=g.ses.nunique())); break
    return pd.DataFrame(rows)
if __name__ == '__main__':
    P = pool_list(); P.to_csv(f'{OUT}/pool/pool_list_PRIVATE.csv', index=False); print('pool people', len(P), P.group.value_counts().to_dict(), flush=True)
    t0 = time.time(); CH = 100
    for ci in range(0, len(P), CH):
        fn = f'{OUT}/pool/chunk_{ci:05d}.pkl'
        if os.path.exists(fn): continue
        sub = P.iloc[ci:ci + CH]; res = Parallel(n_jobs=8)(delayed(process_v2)(r.id, r.ses) for r in sub.itertuples())
        pickle.dump(res, open(fn + '.tmp', 'wb')); os.replace(fn + '.tmp', fn); print(f'chunk {ci} done, elapsed {round(time.time()-t0)} s', flush=True)
    res = [r for fn in sorted(glob.glob(f'{OUT}/pool/chunk_*.pkl')) for r in pickle.load(open(fn, 'rb'))]
    pd.DataFrame([r[0] for r in res]).to_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv', index=False); pd.DataFrame([l for r in res for l in r[1]]).to_csv(f'{OUT}/pool/cleaning_log_pool_v2_PRIVATE.csv', index=False); print('DONE', round(time.time() - t0), 's')
