"""DEVIATION_PROTOCOL 2c + criterion F3 + final feasibility decision (F1 and F3 gating). Outcome-blind."""
import sys, json, pickle, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
sys.path.insert(0, 'scripts'); import dev
FIG = 'outputs/figures_dev'; plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
X, tr, te = dev.load_reference(); Ec, Eo = X[dev.ECc].values, X[dev.EOc].values; lc, sc = dev.level_shape(Ec); lo, so = dev.level_shape(Eo)
Ms = dev.Mapping(sc[tr], so[tr]); rsd = Ms.crossfit_resid_sd(sc[tr], so[tr])                      # fixed mapping + cross-fitted reference-training residual SD
W = pickle.load(open('outputs/dev/eval_window_powers_PRIVATE.pkl', 'rb')); ids = list(X.index[te]); MINW = 5
def halves(rec):
    if rec is None: return None
    st = rec['starts_s']; pw = rec['power']; t0, t1 = st.min(), st.max() + 4.0; mid = (t0 + t1) / 2; h1 = np.flatnonzero(st + 4.0 <= mid); h2 = np.flatnonzero(st >= mid)
    if len(h1) < MINW or len(h2) < MINW: return None
    def cover(ix): iv = sorted((st[i], st[i] + 4.0) for i in ix); tot = 0; cur = None; 
    def union(ix):
        tot = 0; end = -1
        for s, e in sorted((st[i], st[i] + 4.0) for i in ix): tot += max(0, e - max(s, end)); end = max(end, e)
        return tot
    return dict(p1=pw[h1].mean(0), p2=pw[h2].mean(0), n1=len(h1), n2=len(h2), dur1=union(h1), dur2=union(h2))     # linear power averaged over windows, THEN log below
rows = []; D1 = []; D2 = []; C1 = []; C2 = []; S1 = []; S2 = []; dur = []; elig = []
for pid in ids:
    a, b = halves(W.get((pid, 'EC'))), halves(W.get((pid, 'EO')))
    if a is None or b is None: elig.append(False); continue
    elig.append(True); res = {}
    for h, k in ((1, 'p1'), (2, 'p2')):
        l_c, s_c = dev.level_shape(np.log(a[k])[None]); l_o, s_o = dev.level_shape(np.log(b[k])[None]); pred = Ms.predict(s_c); res[h] = dict(dev=((s_o - pred) / rsd)[0], contrast=(s_o - s_c)[0], shape_EO=s_o[0], shape_EC=s_c[0])
    D1.append(res[1]['dev']); D2.append(res[2]['dev']); C1.append(res[1]['contrast']); C2.append(res[2]['contrast']); S1.append(res[1]['shape_EO']); S2.append(res[2]['shape_EO']); dur.append([a['dur1'], a['dur2'], b['dur1'], b['dur2'], a['n1'], a['n2'], b['n1'], b['n2']])
D1, D2, C1, C2, S1, S2, dur = map(np.array, (D1, D2, C1, C2, S1, S2, dur)); n = len(D1); rng = np.random.default_rng(1); BS = rng.integers(0, n, size=(2000, n))
def rel(A, B):
    r = np.array([spearmanr(A[:, j], B[:, j])[0] for j in range(12)]); sb = 2 * r / (1 + r); lo = []; hi = []
    boots = np.array([[ (lambda x: 2 * x / (1 + x))(spearmanr(A[i, j], B[i, j])[0]) for j in range(12)] for i in BS[:1000]])
    return sb, np.nanquantile(boots, .025, 0), np.nanquantile(boots, .975, 0)
sbD, loD, hiD = rel(D1, D2); sbC, loC, hiC = rel(C1, C2); sbS, loS, hiS = rel(S1, S2)
R = pd.DataFrame(dict(element=dev.NAMES, deviation_SB=sbD, deviation_lo=loD, deviation_hi=hiD, contrast_SB=sbC, contrast_lo=loC, contrast_hi=hiC, EOshape_SB=sbS, EOshape_lo=loS, EOshape_hi=hiS)); R.to_csv(f'{FIG}/plotted_data/deviation_reliability.csv', index=False)
medD = float(np.nanmedian(sbD)); n_ok = int((loD >= 0.30).sum()); F3 = bool(n >= 100 and medD >= 0.50 and n_ok >= 8)
ref = json.load(open('outputs/dev/reference_diagnostics.json')); F1 = ref['criteria']['F1_shape_mapping_improvement_GATING']
dec = dict(F1_shape_mapping_improvement=F1, F3_deviation_reliability=F3, proceed_to_remission_scoring=bool(F1 and F3), n_evaluation_people=len(ids), n_eligible_both_conditions_ge5_windows_per_half=int(n), median_SB_deviation=medD, n_elements_lower_bound_ge_0_30=n_ok,
           median_SB_contrast=float(np.nanmedian(sbC)), median_SB_EO_shape=float(np.nanmedian(sbS)), retained_duration_s=dict(EC_half1_median=float(np.median(dur[:, 0])), EC_half2_median=float(np.median(dur[:, 1])), EO_half1_median=float(np.median(dur[:, 2])), EO_half2_median=float(np.median(dur[:, 3])), min_any_half=float(dur[:, :4].min()), IQR_EC_half1=[float(np.quantile(dur[:, 0], .25)), float(np.quantile(dur[:, 0], .75))]),
           windows_per_half_median=float(np.median(dur[:, 4:])), recordings_without_windows=int(sum(v is None for v in W.values())), thresholds=dict(N=100, median_SB=0.50, n_elements=8, lower_bound=0.30))
json.dump(dec, open('outputs/dev/feasibility_decision.json', 'w'), indent=1); print(json.dumps(dec, indent=1)); print(R.round(3).to_string(index=False))
fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1.5, 1]}); y = np.arange(12)
for off, (col, lab, c) in zip((-.22, 0, .22), (('deviation', 'deviation (EO shape - predicted)', '#D55E00'), ('contrast', 'simple contrast (EO shape - EC shape)', '#0072B2'), ('EOshape', 'EO shape alone', '#999999'))):
    v = R[f'{col}_SB']; axs[0].errorbar(v, y + off, xerr=[v - R[f'{col}_lo'], R[f'{col}_hi'] - v], fmt='o', color=c, ms=4, lw=1, label=lab)
axs[0].axvline(.5, color='k', ls='--', lw=.8); axs[0].axvline(.3, color='#999', ls=':', lw=.8); axs[0].set_yticks(y); axs[0].set_yticklabels(dev.NAMES, fontsize=7); axs[0].invert_yaxis(); axs[0].set_xlabel('Split-half reliability (Spearman-Brown corrected; 95% bootstrap over people)'); axs[0].legend(frameon=False, fontsize=7, loc='lower left')
axs[1].hist(dur[:, 0], bins=15, alpha=.7, color='#0072B2', label='EC half 1'); axs[1].hist(dur[:, 2], bins=15, alpha=.7, color='#D55E00', label='EO half 1'); axs[1].set_xlabel('Retained clean seconds in a half (union of windows)'); axs[1].set_ylabel('People'); axs[1].legend(frameon=False, fontsize=7); fig.tight_layout()
fig.savefig(f'{FIG}/figF2_deviation_reliability.png', dpi=300, bbox_inches='tight'); fig.savefig(f'{FIG}/figF2_deviation_reliability.pdf', bbox_inches='tight')
