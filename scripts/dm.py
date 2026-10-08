"""DynaMix extraction (FROZEN spec in DYNAMIX_PROTOCOL.md). Uses reviewed code in external_review/dynamix_reviewed (hashes in SHA256SUMS.txt).
Exact extraction: window (T=3000 samples @100 Hz, 26 channels, uV) -> each channel an independent univariate series, per-series z-scoring + positional embedding (package default) -> N=6 dims.
Initial latent z_0: z[:6]=embedded last context sample, z[6:]=B@(that sample) as in DynaMixForecaster._init_latent_state. For t=0..H-1 (H=200): w_t = gating(context, z_t) with ONE noise draw, z_{t+1} = sum_e w_t[e]*expert_e(z_t).
Per series features (12): mean over t=1..H of z_t[0..9] (post-update states, 10 values); mean over t=0..H-1 of gate entropy -sum w log w; mean over t=0..H-1 of max_e w_t[e].
K independent noise replicates run as batch (B=26*K), driven by a single torch generator seeded per (model, recording, window). Embedding is deterministic (positional sinusoids at autocorrelation period)."""
import os, sys, json, hashlib
sys.path.insert(0, 'external_review/dynamix_reviewed')
import numpy as np, torch
from safetensors.torch import load_file
from dynamix_rev.core import DynaMix
from dynamix_rev.preprocessing import DataPreprocessor
H = 200; CFG = json.load(open('external_review/dynamix_reviewed/hf_config_6d-alrnn-v1.0.json'))['architecture']; WEIGHTS = 'external_review/dynamix_reviewed/hf_6d.safetensors'
MODELS = {'pretrained': None, 'random_a': 101, 'random_b': 102, 'random_c': 103}      # random-weight init seeds (torch+numpy) for the same architecture
def build(name):
    torch.set_num_threads(1)
    seed = MODELS[name]; torch.manual_seed(seed if seed is not None else 0); np.random.seed(seed if seed is not None else 0)
    m = DynaMix(M=CFG['M'], N=CFG['N'], Experts=CFG['Experts'], P=CFG['P'], hidden_dim=CFG['hidden_dim'], expert_type=CFG['expert_type'], probabilistic_expert=CFG['probabilistic_expert'])
    if seed is None: m.load_state_dict(load_file(WEIGHTS), strict=True)
    return m.eval()
def weights_sha256(): return hashlib.sha256(open(WEIGHTS, 'rb').read()).hexdigest()
@torch.no_grad()
def embed_window(model, win, K, seed):
    """win: (T,26) float array (uV). Returns (K, 26, 12) float32."""
    torch.manual_seed(int(seed)); x = torch.tensor(win, dtype=torch.float32).unsqueeze(-1).permute(0, 1, 2)          # (T, 26 series, 1)
    pre = DataPreprocessor(standardize=True, power_transform=False, detrending=False, preprocessing_method='pos_embedding'); ce, ic = pre.preprocess(x, model.N, None)   # ce (T',26,6); ic (26,6)
    cnn = model.precompute_cnn(ce); S = ce.shape[1]
    ce = ce.repeat(1, K, 1); cnn = cnn.repeat(1, K, 1); ic = ic.repeat(K, 1)                                                      # batch = replicate-major: index k*26 + series
    z = torch.matmul(ic, model.B).t(); z[:model.N, :] = ic.t()
    Zs = torch.zeros(model.M, z.shape[1]); ent = torch.zeros(z.shape[1]); mx = torch.zeros(z.shape[1])
    for t in range(H):
        w = model.gating_network(ce, z, precomputed_cnn=cnn)                                                                      # (Experts, B): one noise draw
        ent += -(w * torch.log(w + 1e-12)).sum(0); mx += w.max(0).values
        z = torch.sum(torch.stack([model.experts[i](z) * w[i].unsqueeze(0) for i in range(model.Experts)], 0), 0); Zs += z        # identical to DynaMix.step, reusing w
    out = torch.cat([Zs.t() / H, (ent / H)[:, None], (mx / H)[:, None]], 1)                                                       # (B,12)
    return out.reshape(K, S, 12).numpy().astype(np.float32)
FEATURE_NAMES = [f'z{i}' for i in range(10)] + ['gate_entropy', 'gate_max']
