"""Label-blind recording QC for included people (no outcome data read). Output: outputs/qc_recordings_PRIVATE.csv"""
import sys, time, numpy as np, pandas as pd
from scipy.signal import welch
from common import *
led = load_ledger(); rows = []; t0 = time.time()
for i, r in led.iterrows():
    for cond in ('EC', 'EO'):
        try:
            raw, pad = load_raw(r['id'], r.selected_ses, cond)
            x = raw.get_data(picks='eeg') * 1e6
            n = x.shape[1]
            xc = x - x.mean(1, keepdims=True)
            flat = (np.abs(np.diff(x, axis=1)) == 0).mean(1)                  # per-channel zero-diff fraction (raw)
            # 1-s block std of 1-45 Hz filtered, to judge amplitude/bad channels (padding already trimmed)
            f = raw.copy().pick('eeg').filter(1, 45, verbose=False).get_data() * 1e6
            nb = n // 500; bs = f[:, :nb*500].reshape(26, nb, 500).std(2)
            fq, P = welch(x, 500, nperseg=2000, axis=1)
            line = P[:, np.argmin(abs(fq - 50))] / P[:, (np.abs(fq - 50) > 3) & (np.abs(fq - 50) < 8)].mean(1)
            hf = P[:, (fq > 30) & (fq < 45)].sum(1) / P[:, (fq > 1) & (fq < 45)].sum(1)
            rows.append(dict(id=r['id'], cond=cond, ok=True, dur_s=n/500, pad_s=pad, dc_abs_median_uV=float(np.median(np.abs(x.mean(1)))),
                             n_flat_ch=int((flat > .05).sum()), med_std_uV=float(np.median(bs)), n_hi_std_ch=int((np.median(bs, 1) > 3*np.median(bs)).sum()),
                             n_lo_std_ch=int((np.median(bs, 1) < np.median(bs)/4).sum()), frac_blocks_gt100uV=float((bs.max(0) > 100).mean()),
                             max_line_ratio=float(line.max()), med_hf_frac=float(np.median(hf))))
        except Exception as e:
            rows.append(dict(id=r['id'], cond=cond, ok=False, err=str(e)[:80]))
    if i % 20 == 0: print(i, round(time.time()-t0), flush=True)
pd.DataFrame(rows).to_csv(f'{OUT}/qc_recordings_PRIVATE.csv', index=False)
