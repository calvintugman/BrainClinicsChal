"""Shared helpers for the EC->EO deviation experiment (DEVIATION_PROTOCOL.md). Reference partition identical to 26_reference_mapping.py."""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
OUT = 'outputs'; ROIS = ['frontal', 'central', 'posterior']; BANDS = ['delta', 'theta', 'alpha', 'beta']; GRID = [0.01, 0.1, 1, 10, 100, 1000]
ECc = [f'EC_{r}_{b}_logabs' for r in ROIS for b in BANDS]; EOc = [f'EO_{r}_{b}_logabs' for r in ROIS for b in BANDS]; NAMES = [f'{r}_{b}' for r in ROIS for b in BANDS]
def load_reference():
    F = pd.read_csv(f'{OUT}/pool/features_pool_v2_PRIVATE.csv'); P = pd.read_csv(f'{OUT}/pool/pool_list_PRIVATE.csv'); F = F.set_index('id').reindex(P.id)
    cc = F[ECc + EOc].notna().all(1); X = F[cc]; X = X[~X.duplicated(subset=ECc + EOc, keep='first')]
    perm = np.random.default_rng(2026).permutation(len(X)); nte = int(round(.2 * len(X))); return X, perm[nte:], perm[:nte]            # X, train idx, eval idx
def level_shape(M):
    """M (n,12) log powers -> level (n,), shape (n,12). Incomplete vectors (any NaN) -> NaN level and NaN shape (no partial centering)."""
    M = np.asarray(M, float); ok = np.isfinite(M).all(1); lev = np.full(len(M), np.nan); sh = np.full(M.shape, np.nan)
    lev[ok] = M[ok].mean(1); sh[ok] = M[ok] - lev[ok, None]; return lev, sh
def fit_tune(Xin, Yin):
    cv = KFold(5, shuffle=True, random_state=0); sc = {a: np.mean([np.mean((Ridge(alpha=a).fit(Xin[i], Yin[i]).predict(Xin[j]) - Yin[j]) ** 2) for i, j in cv.split(Xin)]) for a in GRID}
    a = min(sc, key=sc.get); return Ridge(alpha=a).fit(Xin, Yin), a, sc
class Mapping:
    """ridge on z-scored inputs/outputs; fitted ONLY on reference-training. kind: 'shape' (EC shape->EO shape) or 'raw' (EC logs -> EO logs, uncentered as in the earlier stage)."""
    def __init__(self, Ein, Eout): self.mi, self.si = Ein.mean(0), Ein.std(0); self.mo, self.so = Eout.mean(0), Eout.std(0); self.model, self.alpha, self.cv = fit_tune((Ein - self.mi) / self.si, (Eout - self.mo) / self.so)
    def predict(self, Ein): return self.model.predict((np.asarray(Ein, float) - self.mi) / self.si) * self.so + self.mo
    def crossfit_resid_sd(self, Ein, Eout):
        cv = KFold(5, shuffle=True, random_state=0); R = np.zeros_like(Eout)
        for i, j in cv.split(Ein):
            m = Mapping.__new__(Mapping); m.mi, m.si, m.mo, m.so = Ein[i].mean(0), Ein[i].std(0), Eout[i].mean(0), Eout[i].std(0); mod = Ridge(alpha=self.alpha).fit((Ein[i] - m.mi) / m.si, (Eout[i] - m.mo) / m.so)
            R[j] = Eout[j] - (mod.predict((Ein[j] - m.mi) / m.si) * m.so + m.mo)
        return R.std(0)
