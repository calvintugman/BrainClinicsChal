"""Per-window ROI band POWERS (uV^2, before logs) for the 152 reference evaluation people (v2 cleaning, same windows as the features). For split-half deviation reliability."""
import os, sys, pickle, time
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
import dev, importlib
def one(pid, ses, cond):
    f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2')
    raw, log = cv2.clean(pid, ses, cond)
    if raw is None: return (pid, cond), None
    bad = log['bad_ch'].split(';') if log['bad_ch'] else []; x = raw.get_data(picks='eeg') * 1e6; st = f3.windows(x, bad)
    if len(st) == 0: return (pid, cond), None
    Pw, f, df = f3.band_feats(x, st, bad); ri = f3.roi_idx(bad); bp = {b: f3.band_power(Pw, f, df, *lim) for b, lim in f3.BANDS.items()}
    pw = np.stack([bp[b][:, ri[r]].mean(1) for r in dev.ROIS for b in dev.BANDS], 1)            # (n_win, 12) ROI-mean band power, uV^2, ROI-major
    return (pid, cond), dict(starts_s=st / 500., power=pw, n_bad=len(bad))
if __name__ == '__main__':
    X, tr, te = dev.load_reference(); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv').set_index('id'); ids = list(X.index[te]); t0 = time.time()
    res = Parallel(n_jobs=12, verbose=2)(delayed(one)(i, P.loc[i, 'ses'], c) for i in ids for c in ('EC', 'EO'))
    pickle.dump(dict(res), open(f'{OUT}/dev/eval_window_powers_PRIVATE.pkl', 'wb')); print('done', round(time.time() - t0), 's', sum(v is None for _, v in res), 'recordings without windows')
