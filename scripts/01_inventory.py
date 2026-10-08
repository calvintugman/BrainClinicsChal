"""Inventory rTMS cohort: workbook parse, file manifest (headers only), inclusion ledger. No outcome-feature association is computed."""
import glob, re, json, hashlib, os, sys
import pandas as pd, mne
mne.set_log_level('ERROR')
DATA = os.environ.get('DATA_ROOT', os.path.abspath('..'))
XLSX = os.environ.get('PARTICIPANTS_XLSX', os.path.join(DATA, 'TDBRAIN_participants_V3.1.xlsx'))
OUT = os.environ.get('OUTPUT_ROOT', 'outputs')
os.makedirs(OUT, exist_ok=True)
sha = hashlib.sha256(open(XLSX, 'rb').read()).hexdigest()
df = pd.read_excel(XLSX, dtype={'TDBRAIN_ID': str})
t = df[df.Dataset == 'MDD-rTMS'].copy()
roots = ['TDBRAIN_Dataset_V3_1', 'TDBRAIN_MDD_Dx_ReplicationV3_1', 'TDBRAIN_Tx_Response_ReplicationV3_1']
rows = []
for r in roots:
    for p in glob.glob(f'{DATA}/{r}/sub-*/ses-*/eeg/*_eeg.bdf'):
        m = re.search(r'(sub-[^/]+)/(ses-\d+)/eeg/.*task-(\w+)_eeg', p)
        rows.append(dict(root=r, sub=m[1], ses=m[2], task=m[3], path=p, size=os.path.getsize(p), mtime=os.path.getmtime(p)))
inv = pd.DataFrame(rows)
ids = set(t.TDBRAIN_ID)
man = inv[inv['sub'].isin(ids)].copy()
# lightweight headers (no data load)
def hdr(p):
    raw = mne.io.read_raw_bdf(p, preload=False, verbose='ERROR')
    return dict(sfreq=raw.info['sfreq'], n_ch=len(raw.ch_names), dur_s=raw.n_times / raw.info['sfreq'],
                ch_types=json.dumps({k: v for k, v in zip(*__import__('numpy').unique(raw.get_channel_types(), return_counts=True))}, default=int),
                has_F3_F4=all(c in raw.ch_names for c in ('F3', 'F4')))
h = pd.DataFrame([hdr(p) for p in man.path], index=man.index)
man = pd.concat([man, h], axis=1)
man.to_csv(f'{OUT}/manifest_rtms.csv', index=False)
# ledger: one session per person
sess_rows = t[['TDBRAIN_ID', 'sessID', 'nrSessions', 'Responder', 'Remitter']]
led = []
for pid, g in sess_rows.groupby('TDBRAIN_ID'):
    f = man[man['sub'] == pid]
    has1 = (g.sessID == 1).any()
    tasks1 = set(f[f.ses == 'ses-1'].task)
    if not has1: reason, sel = 'only sessID=2 row; pretreatment timing unverified', None
    elif {'restEC', 'restEO'} <= tasks1: reason, sel = 'ok', 'ses-1'
    else: reason, sel = 'ses-1 missing EC/EO', None
    led.append(dict(id=pid, selected_ses=sel, reason=reason, Responder=g.Responder.max(), Remitter=g.Remitter.max(),
                    label_conflict=bool(g.Responder.nunique() > 1 or g.Remitter.nunique() > 1),
                    has_oddball=bool((f.task == 'oddball').any()), has_EO=bool((f.task == 'restEO').any()), has_EC=bool((f.task == 'restEC').any())))
led = pd.DataFrame(led); led.to_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv', index=False)
inc = led[led.selected_ses.notna()]
summ = dict(workbook_sha256=sha, rows=len(t), people=int(t.TDBRAIN_ID.nunique()), responders=int(led.Responder.sum()), remitters=int(led.Remitter.sum()),
            label_conflicts=int(led.label_conflict.sum()), included=len(inc), excluded=len(led) - len(inc),
            exclusion_reasons=led[led.selected_ses.isna()].reason.value_counts().to_dict(),
            included_remitters=int(inc.Remitter.sum()), people_with_oddball=int(led.has_oddball.sum()),
            oddball_files_total=int((inv.task == 'oddball').sum()),
            oddball_subjects_total=int(inv[inv.task == 'oddball']['sub'].nunique()),
            replication_ids_in_workbook=int(inv[inv.root != roots[0]]['sub'].isin(df.TDBRAIN_ID).sum()),
            manifest_sfreq=man.sfreq.value_counts().to_dict(), manifest_nch=man.n_ch.value_counts().to_dict(),
            dur_s_min_med_max=[float(man.dur_s.min()), float(man.dur_s.median()), float(man.dur_s.max())],
            F3F4_present_frac=float(man.has_F3_F4.mean()), raw_checksums_computed=False)
json.dump(summ, open(f'{OUT}/inventory_summary.json', 'w'), indent=1, default=str)
print(json.dumps(summ, indent=1, default=str))
