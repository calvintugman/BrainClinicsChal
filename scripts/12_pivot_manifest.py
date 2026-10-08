"""Participant-level manifest + overlap checks across discovery/replication folders (PIVOT step 2). Private outputs; summary json is shareable."""
import glob, re, os, json, hashlib, collections
import pandas as pd, numpy as np
DATA = '..'; OUT = 'outputs'
wb = pd.read_excel(f'{DATA}/TDBRAIN_participants_V3.1.xlsx', dtype={'TDBRAIN_ID': str})
roots = {'discovery': 'TDBRAIN_Dataset_V3_1', 'rep_Dx': 'TDBRAIN_MDD_Dx_ReplicationV3_1', 'rep_Tx': 'TDBRAIN_Tx_Response_ReplicationV3_1'}
rows = []
for k, r in roots.items():
    for p in glob.glob(f'{DATA}/{r}/sub-*/ses-*/eeg/*_eeg.bdf'):
        m = re.search(r'(sub-[^/]+)/(ses-\d+)/eeg/.*task-(\w+)_eeg', p); rows.append(dict(set=k, id=m[1], ses=m[2], task=m[3], path=p, size=os.path.getsize(p)))
F = pd.DataFrame(rows)
led = pd.read_csv(f'{OUT}/inclusion_ledger_PRIVATE.csv'); lab164 = set(led[led.selected_ses.notna()].id)
inc = wb.groupby('TDBRAIN_ID')[['indication', 'Dataset']].nunique(dropna=True); res_incons = int((inc > 1).any(axis=1).sum())
d = wb.groupby('TDBRAIN_ID')[['indication', 'Dataset']].first()
def grp(i):
    if i in lab164: return 'rTMS_labeled_164'
    if i in set(led.id): return 'rTMS_excluded_timing'
    ds = d.Dataset.get(i); ind = d.indication.get(i)
    if isinstance(ds, str): return 'other_treatment_cohort(' + ds + ')'
    if ind == 'HEALTHY': return 'healthy_verified_by_workbook_indication'
    return 'other_discovery_unlabeled_for_remission'
disc = F[F.set == 'discovery'].copy(); disc['group'] = disc.id.map(grp)
P = disc.groupby('group').agg(people=('id', 'nunique'), recordings=('id', 'size')).reset_index()
res = dict(workbook_people_with_inconsistent_indication_or_dataset_across_rows=res_incons, discovery_groups=P.to_dict('records'), n_discovery_people=int(disc.id.nunique()))
# ID overlaps
ids = {k: set(F[F.set == k].id) for k in roots}
res['id_overlap'] = {f'{a}&{b}': len(ids[a] & ids[b]) for a in roots for b in roots if a < b}
res['replication_ids_in_workbook'] = {k: len(ids[k] & set(wb.TDBRAIN_ID)) for k in ('rep_Dx', 'rep_Tx')}
# file overlap: cheap fingerprint (size + sha256 of first 1 MiB + last 64 KiB) for ALL files, full sha256 only for fingerprint matches
def fp(p):
    sz = os.path.getsize(p)
    with open(p, 'rb') as f:
        h = hashlib.sha256(f.read(1 << 20)); f.seek(max(0, sz - 65536)); h.update(f.read())
    return (sz, h.hexdigest())
F['fp'] = [fp(p) for p in F.path]; sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
g = F.groupby('fp').path.apply(list); dupg = g[g.map(len) > 1]; full_hashed = 0; cross = 0; within_set_dups = 0
setof = dict(zip(F.path, F.set))
for lst in dupg:
    hs = collections.defaultdict(list)
    for p in lst: hs[sha(p)].append(p); full_hashed += 1
    for h, ps in hs.items():
        if len(ps) > 1:
            if len({setof[p] for p in ps}) > 1: cross += 1
            else: within_set_dups += 1
res['hash_check'] = dict(files_fingerprinted=len(F), fingerprint_collision_groups=len(dupg), files_fully_hashed=full_hashed, identical_content_across_sets=cross, identical_content_within_a_set=within_set_dups,
                         method='size + sha256(first 1 MiB + last 64 KiB) for every file; full sha256 only for fingerprint-colliding files. Not a full-content checksum of all files.')
res['labeled_164'] = dict(n=len(lab164), timing_basis='session-1 by dataset convention; unverified', sessions_available=int(disc[disc.group == 'rTMS_labeled_164'].groupby('id').ses.nunique().gt(1).sum()), with_ses2_too=True)
res['oddball'] = {g: int(v) for g, v in disc[disc.task == 'oddball'].groupby('group').id.nunique().items()}
res['healthy_note'] = 'healthy = workbook indication==HEALTHY only; other discovery people are NOT assumed healthy or remission-labeled'
res['reference_pool_for_EC_EO_extension_candidates'] = int(disc[(~disc.group.isin(['rTMS_labeled_164', 'rTMS_excluded_timing'])) & disc.id.isin(set(disc[disc.task == 'restEC'].id)) & disc.id.isin(set(disc[disc.task == 'restEO'].id))].id.nunique())
disc.drop(columns='path').to_csv(f'{OUT}/pivot_manifest_PRIVATE.csv', index=False)
res['manifest_sha256'] = hashlib.sha256(open(f'{OUT}/pivot_manifest_PRIVATE.csv', 'rb').read()).hexdigest()
json.dump(res, open(f'{OUT}/pivot_manifest_summary.json', 'w'), indent=1); print(json.dumps(res, indent=1))
