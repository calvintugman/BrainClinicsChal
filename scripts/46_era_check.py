"""ERA_PROTOCOL.md (A9). Run from first_tests/."""
import os, sys, json, glob, hashlib, warnings
os.environ['FEATURES_PATH'] = 'outputs/v2/features_v2_PRIVATE.csv'; os.environ.setdefault('OMP_NUM_THREADS', '1'); sys.path.insert(0, 'scripts')
import numpy as np, pandas as pd, importlib
from scipy.stats import spearmanr, kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as auc
warnings.filterwarnings('ignore'); c = importlib.import_module('37_clinical_eval'); SEEDS = [42, 43, 44]
T, cols, y = c.build('Remitter'); num = T.id.str.replace('sub-', '').astype(float).values
bm = pd.read_csv(sorted(glob.glob('runs/PIV_E01c_Remitter_*'))[-1] + '/predictions_PRIVATE.csv'); assert (bm.id.values == T.id.values).all()
pe = np.mean([bm[f'p_E01c_s{s}'].values for s in SEEDS], axis=0); R = {}; draws = np.random.default_rng(1).integers(0, len(y), size=(2000, len(y)))
def sp(a, b): r = spearmanr(a, b)[0]; bs = [spearmanr(a[i], b[i])[0] for i in draws]; return dict(rho=float(r), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)))
R['a'] = {'E01c_score': sp(num, pe), 'age': sp(num, T.age.values), 'sex': sp(num, T.sex.values), 'BDI_pre': sp(num, T.BDI_pre.values)}
ok = T.prot_raw.notna().values; g = [num[ok & (T.prot_raw.values == k)] for k in (1, 2, 3)]; kw = kruskal(*g); R['a']['protocol'] = dict(H=float(kw.statistic), p=float(kw.pvalue), median_ID_by_protocol=[float(np.median(x)) for x in g], n=[len(x) for x in g])
led = pd.read_csv('outputs/inclusion_ledger_PRIVATE.csv').query('selected_ses==selected_ses')[['id', 'Remitter', 'Responder']].merge(T[['id']], on='id'); assert (led.id.values == T.id.values).all(); R['b'] = {}
for ep in ('Remitter', 'Responder'):
    yy = led[ep].astype(int).values; a = []
    for s in SEEDS:
        p = np.zeros(len(yy))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.zeros(len(yy)), yy): sc = StandardScaler().fit(num[tr][:, None]); p[te] = LogisticRegression(C=1).fit(sc.transform(num[tr][:, None]), yy[tr]).predict_proba(sc.transform(num[te][:, None]))[:, 1]
        a.append(float(auc(yy, p)))
    R['b'][ep] = dict(per_seed_AUC=a, mean_AUC=float(np.mean(a)))
q = pd.qcut(num, 3, labels=False); R['c'] = dict(pooled_E01c_AUC=float(np.mean([auc(y, bm[f'p_E01c_s{s}'].values) for s in SEEDS])), tertiles=[dict(tertile=int(t) + 1, n=int((q == t).sum()), n_pos=int(y[q == t].sum()), mean_AUC=float(np.mean([auc(y[q == t], bm[f'p_E01c_s{s}'].values[q == t]) for s in SEEDS]))) for t in range(3)])
R['rule'] = dict(tracks_era=bool(abs(R['a']['E01c_score']['rho']) >= .30 or (all(t['mean_AUC'] <= .55 for t in R['c']['tertiles']) and R['c']['pooled_E01c_AUC'] >= .57)))
rid = 'ERA_' + __import__('time').strftime('%Y%m%dT%H%M%S'); os.makedirs(f'runs/{rid}'); json.dump(dict(**R, code_fp=hashlib.sha256(open(__file__, 'rb').read()).hexdigest()[:16]), open(f'runs/{rid}/analysis.json', 'w'), indent=1); print(rid); print(json.dumps(R, indent=1))
