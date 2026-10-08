"""Feasibility only (NO outcome scoring): one-recording DynaMix inference with reviewed code + safetensors weights (+ random-weight control).
Source: github DurstewitzLab/DynaMix-python (main as fetched 2026-10-05, files hashed in external_review/dynamix_reviewed/SHA256SUMS.txt; commit SHA not recorded), HF DurstewitzLab/dynamix v1.0."""
import sys, os, json, time, resource, tracemalloc
sys.path.insert(0, 'external_review/dynamix_reviewed'); sys.path.insert(0, 'scripts')
import numpy as np, torch
from safetensors.torch import load_file
from dynamix_rev.core import DynaMix
from dynamix_rev.preprocessing import DataPreprocessor
from common import *
import importlib; cv2 = importlib.import_module('clean_v2')
torch.set_num_threads(4); out = {}
cfg = json.load(open('external_review/dynamix_reviewed/hf_config_6d-alrnn-v1.0.json'))['architecture']
def build(pretrained):
    torch.manual_seed(0); np.random.seed(0)
    m = DynaMix(M=cfg['M'], N=cfg['N'], Experts=cfg['Experts'], P=cfg['P'], hidden_dim=cfg['hidden_dim'], expert_type=cfg['expert_type'], probabilistic_expert=cfg['probabilistic_expert'])
    if pretrained:
        sd = load_file('external_review/dynamix_reviewed/hf_6d.safetensors'); r = m.load_state_dict(sd, strict=True); out['load_state_dict'] = str(r)
    return m.eval()
M1, M0 = build(True), build(False); out['n_params'] = sum(p.numel() for p in M1.parameters()); out['arch'] = cfg
# one recording: lowest-ID labeled patient, EO, v2-cleaned, 26 scalp channels, 100 Hz, 30 s context
led = load_ledger().sort_values('id'); pid, ses = led.iloc[0]['id'], led.iloc[0].selected_ses
raw, lg = cv2.clean(pid, ses, 'EO'); raw.pick('eeg'); raw.resample(100., verbose=False); X = raw.get_data() * 1e6          # (26, T) uV
ctx = torch.tensor(X[:, 1000:4000].T, dtype=torch.float32)                          # (3000, 26): 30 s at 100 Hz
out['context'] = dict(shape=list(ctx.shape), fs=100, seconds=30, units='uV (per-series standardized inside preprocessor => amplitude discarded)')
def embed(model, ctx, horizon=200, seed=0):
    """My extraction (undocumented in the package): treat each channel as a univariate series (feature_dim 26 > N=6 would trigger this reshaping anyway);
    returns per-series [mean latent z over forecast horizon (M), mean expert-gate entropy, mean max gate weight]."""
    torch.manual_seed(seed)
    c = ctx.T.unsqueeze(-1).permute(1, 0, 2)                                         # (T, 26 series, 1)
    pre = DataPreprocessor(standardize=True, power_transform=False, detrending=False, preprocessing_method='pos_embedding')
    ce, ic = pre.preprocess(c, model.N, None)                                         # (T', 26, N)
    z = torch.matmul(ic, model.B).t(); z[:model.N, :] = ic.t()
    zs, ent, mx = [], [], []
    with torch.no_grad():
        cnn = model.precompute_cnn(ce)
        for t in range(horizon):
            w = model.gating_network(ce, z, precomputed_cnn=cnn); ent.append(-(w * torch.log(w + 1e-12)).sum(0)); mx.append(w.max(0).values)
            z = model(z, ce, precomputed_cnn=cnn); zs.append(z.clone())
    Z = torch.stack(zs)                                                              # (H, M, 26)
    return torch.cat([Z.mean(0).T, torch.stack(ent).mean(0)[:, None], torch.stack(mx).mean(0)[:, None]], 1).numpy(), Z
tracemalloc.start(); t0 = time.time(); r0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
E1, Z1 = embed(M1, ctx); t1 = time.time() - t0; E1b, _ = embed(M1, ctx, seed=0); E1c, _ = embed(M1, ctx, seed=1); E0, _ = embed(M0, ctx)
out['pretrained_embedding_shape'] = list(E1.shape); out['all_finite'] = bool(np.isfinite(E1).all() and np.isfinite(E0).all())
out['time_one_recording_30s_26series_s'] = round(t1, 2); out['peak_rss_MB'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024); out['python_alloc_peak_MB'] = round(tracemalloc.get_traced_memory()[1] / 1e6, 1)
out['same_seed_reproducible_max_abs_diff'] = float(np.abs(E1 - E1b).max()); out['different_seed_max_abs_diff'] = float(np.abs(E1 - E1c).max())          # gating adds noise: inference is stochastic unless seeded
out['pretrained_vs_random_embedding_mean_abs_diff'] = float(np.abs(E1 - E0).mean()); out['embedding_feature_std_across_series'] = np.std(E1, 0).round(3).tolist()
n_rec = 164 * 2; out['estimate'] = dict(one_30s_window_s=round(t1, 1), full_120s_as_4_nonoverlap_windows_per_recording_s=round(4 * t1, 1), cohort_164_pts_EC_EO_s=round(n_rec * 4 * t1), cohort_hours=round(n_rec * 4 * t1 / 3600, 2),
                                         discovery_reference_pool_1129_people_EC_EO_hours=round(1129 * 2 * 4 * t1 / 3600, 1), note='single process, 4 torch threads; this window/horizon is a placeholder choice, not a recommended protocol')
json.dump(out, open('outputs/dynamix_feasibility.json', 'w'), indent=1); print(json.dumps(out, indent=1))
# figure: context vs forecast for one channel (illustration of inference only)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); ch = SCALP.index('Pz'); seg = ctx[-300:, ch].numpy(); seg = (seg - ctx[:, ch].mean().item()) / ctx[:, ch].std().item()
ax[0].plot(np.arange(-300, 0) / 100, seg, color='k', lw=.8); ax[0].plot(np.arange(200) / 100, Z1[:, 0, ch].numpy(), color='#D55E00'); ax[0].set_xlabel('Time (s) relative to forecast start'); ax[0].set_ylabel('Standardized (z-score) Pz'); ax[0].set_title('Last 3 s context (black) and model forecast of 1st latent (orange)', fontsize=8)
ax[1].bar(range(E1.shape[1]), E1.std(0), color='#0072B2'); ax[1].set_xlabel('Feature index (10 latent means, gate entropy, max gate)'); ax[1].set_ylabel('SD across 26 channels'); fig.tight_layout(); os.makedirs('outputs/figures_pivot', exist_ok=True); fig.savefig('outputs/figures_pivot/fig_dynamix_inference.png', dpi=300, bbox_inches='tight'); fig.savefig('outputs/figures_pivot/fig_dynamix_inference.pdf', bbox_inches='tight')
