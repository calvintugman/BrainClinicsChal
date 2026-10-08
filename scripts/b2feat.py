"""Batch-2 feature derivation shared by checks/eval (label-free, recording-local)."""
import numpy as np, pandas as pd
from scipy.linalg import eigh
BANDS = ['theta', 'alpha', 'beta']
def eigen_feats(C_eo, C_ec):
    """12 -> per band: mean, pop-SD, min, max of ln generalized eigenvalues of C_EO v = lambda C_EC v. C_*: (3,26,26) or NaN."""
    out = []
    for b in range(3):
        a, c = C_eo[b], C_ec[b]
        if np.isnan(a).any() or np.isnan(c).any(): out += [np.nan] * 4; continue
        lam = eigh(a, c, eigvals_only=True); assert (lam > 0).all(); l = np.log(lam); out += [l.mean(), l.std(ddof=0), l.min(), l.max()]
    return np.array(out)
E13_NAMES = [f'E13_{b}_{s}' for b in BANDS for s in ('mean', 'sd', 'min', 'max')]
def trace_norm(C): return C / np.trace(C, axis1=-2, axis2=-1)[..., None, None]
def build(cov_full, gain=1.0):
    """cov_full (N,2,3,26,26) [cond EC,EO]; gain applies to both conditions (amplitude gain -> cov * gain^2)."""
    cov = cov_full * gain**2
    E13 = np.array([eigen_feats(cov[i, 1], cov[i, 0]) for i in range(len(cov))])
    return cov, E13
