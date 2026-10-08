"""Shared loaders. Label-blind recording handling: 26 scalp EEG channels, trailing zero-padding trimmed before filtering."""
import numpy as np, mne, pandas as pd, os
mne.set_log_level('ERROR')
SCALP = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC3','FCz','FC4','T7','C3','Cz','C4','T8','CP3','CPz','CP4','P7','P3','Pz','P4','P8','O1','Oz','O2']
AUX = ['VPVA','VNVB','HPHL','HNHR','Erbs','Mass']  # EOG x4, ECG, EMG (audit/ICA only)
OUT = 'outputs'
def load_ledger():
    led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv'); return led[led.selected_ses.notna()].reset_index(drop=True)
def rec_path(pid, ses, cond):
    return f'../TDBRAIN_Dataset_V3_1/{pid}/{ses}/eeg/{pid}_{ses}_task-rest{cond}_eeg.bdf'
def load_raw(pid, ses, cond):
    """Returns (raw with 26 scalp + 6 aux channels typed correctly, V units, trimmed of trailing constant-valued padding (all scalp channels flat)), pad_s."""
    raw = mne.io.read_raw_bdf(rec_path(pid, ses, cond), preload=True)
    raw.pick(SCALP + AUX)
    d = raw.get_data()
    ch = np.flatnonzero((np.diff(d[:26], axis=1) != 0).any(0))   # any scalp channel changing sample-to-sample
    end = ch[-1] + 2 if len(ch) else 0                              # last changing sample, inclusive; constant tail = padding
    pad = (raw.n_times - end) / raw.info['sfreq']
    raw.crop(tmax=(end - 1) / raw.info['sfreq'])
    raw.set_channel_types({**{c: 'eog' for c in AUX[:4]}, 'Erbs': 'ecg', 'Mass': 'emg'})
    return raw, pad
