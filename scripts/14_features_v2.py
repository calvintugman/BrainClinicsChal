"""Corrected-pipeline (v2) recording-local features: identical feature definitions to 03_features.py, cleaning from clean_v2. -> outputs/v2/"""
import os, sys, time, pickle
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from common import *
def process_v2(pid, ses):
    import importlib; f3 = importlib.import_module('03_features'); cv2 = importlib.import_module('clean_v2'); f3.clean = cv2.clean   # same feature code, corrected cleaning
    feats, logs, win, rt = f3.process(pid, ses); return feats, logs, win, rt
if __name__ == '__main__':
    led = load_ledger(); t0 = time.time()
    res = Parallel(n_jobs=10, verbose=2)(delayed(process_v2)(r['id'], r.selected_ses) for _, r in led.iterrows())
    pd.DataFrame([r[0] for r in res]).to_csv(f'{OUT}/v2/features_v2_PRIVATE.csv', index=False)
    pd.DataFrame([l for r in res for l in r[1]]).to_csv(f'{OUT}/v2/cleaning_log_v2_PRIVATE.csv', index=False)
    pickle.dump({r[0]['id']: r[2] for r in res}, open(f'{OUT}/v2/windows_v2_PRIVATE.pkl', 'wb')); print('elapsed s', round(time.time() - t0))
