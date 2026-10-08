"""Pre-scoring checks for E13 (synthetic) + real-data covariance diagnostics. Label-free."""
import numpy as np, json, sys
sys.path.insert(0, 'scripts')
from b2feat import *
from sklearn.covariance import LedoitWolf
rng = np.random.default_rng(0); res = {}
def rand_cov(n=26, T=2000, rank=None):
    A = rng.normal(size=(n, n)); x = A @ rng.normal(size=(n, T)); return LedoitWolf().fit(x.T).covariance_
C = np.stack([rand_cov() for _ in range(3)])
f_same = eigen_feats(C, C); res['identical_matrices_max_abs_log_lambda'] = float(np.abs(f_same).max())
C2 = np.stack([rand_cov() for _ in range(3)]); f0 = eigen_feats(C2, C); f_gain = eigen_feats(C2 * 4.0, C * 4.0)
res['gain_x2_max_abs_feature_change'] = float(np.abs(f0 - f_gain).max())
# invariance to a known-eigen design: C_EO = V diag(l) V' with C_EC = I-like -> recovers log l
n = 26; Q, _ = np.linalg.qr(rng.normal(size=(n, n))); l = np.exp(rng.normal(size=n)); A = Q @ np.diag(l) @ Q.T; B = np.eye(n)
f = eigen_feats(np.stack([A] * 3), np.stack([B] * 3)); res['known_eigen_recovery_err'] = float(abs(f[0] - np.log(l).mean()) + abs(f[1] - np.log(l).std()) + abs(f[2] - np.log(l).min()) + abs(f[3] - np.log(l).max()))
assert res['identical_matrices_max_abs_log_lambda'] < 1e-8 and res['gain_x2_max_abs_feature_change'] < 1e-8 and res['known_eigen_recovery_err'] < 1e-8
# real covariances: PD, symmetry, conditioning, effective rank
z = np.load('outputs/batch2_cov_PRIVATE.npz'); cov = z['cov_full']; ok = ~np.isnan(cov).any(axis=(3, 4)); eig = []
for i in range(cov.shape[0]):
    for c in range(2):
        for b in range(3):
            if ok[i, c, b]: M = cov[i, c, b]; assert np.allclose(M, M.T, rtol=1e-8, atol=1e-8); e = np.linalg.eigvalsh(M); assert e.min() > 0; eig.append((e.min() / e.max(), (e / e.max() > 1e-4).sum(), c, b))
eig = np.array(eig); res['n_real_matrices'] = len(eig); res['all_symmetric_pd'] = True; res['cond_number_median'] = float(np.median(1 / eig[:, 0])); res['cond_number_max'] = float((1 / eig[:, 0]).max())
res['eff_rank_1e-4_median_min_max'] = [float(np.median(eig[:, 1])), float(eig[:, 1].min()), float(eig[:, 1].max())]
_, E13 = build(cov); res['E13_nan_frac'] = float(np.isnan(E13).any(1).mean()); res['E13_gain_0.5_2_max_abs_change'] = float(max(np.nanmax(np.abs(build(cov, g)[1] - E13)) for g in (0.5, 2.0)))
d = np.nanmin(E13[:, [2, 6, 10]]), np.nanmax(E13[:, [3, 7, 11]]); res['E13_min_log_lambda_overall'] = float(d[0]); res['E13_max_log_lambda_overall'] = float(d[1])
json.dump(res, open('outputs/synthetic_checks.json', 'w'), indent=1); print(json.dumps(res, indent=1))
