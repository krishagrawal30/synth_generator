"""
synth_core.py
=============
Shared library behind the Streamlit app. It mirrors the notebook
(Synthetic_Clinical_Data_AE_VAE_GAN.ipynb) so models exported from the notebook load here unchanged.

Contents
  1. Cohort preparation  (UCI Diabetes 130-US Hospitals -> modelling table)
  2. TabularEncoder      (log + z-score for counts, one-hot for categories; JSON round-trip)
  3. Models              (Autoencoder, VAE, WGAN-GP generator / critic)
  4. Training            (with optional progress callbacks, used by the live-training page)
  5. Generation          (AE + latent GMM, VAE, WGAN-GP, plus rejection-sampled "what-if" cohorts)
  6. Reference statistics and evaluation (fidelity, relationships, utility, privacy audit)

The app never needs real patient rows: fidelity is measured against aggregate reference statistics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_sample_weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["AE+GMM", "VAE", "WGAN-GP"]

NUM_COLS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
TARGET = "early_readmit"
QI = ["age", "gender", "race", "payer_code", "medical_specialty", "diag_1"]
SUBGROUP_COLS = ["age", "race", "diag_1"]
EXPIRED_HOSPICE = {11, 13, 14, 19, 20, 21}


# ════════════════════════════════════════════════════════════════════
# 1 · Cohort preparation
# ════════════════════════════════════════════════════════════════════
def icd9_group(code) -> str:
    if pd.isna(code):
        return "Missing"
    c = str(code).strip()
    if c in ("", "?", "nan", "None"):
        return "Missing"
    if c[0] in "VvEe":
        return "Other"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    if 250 <= v < 251: return "Diabetes"
    if 390 <= v < 460 or int(v) == 785: return "Circulatory"
    if 460 <= v < 520 or int(v) == 786: return "Respiratory"
    if 520 <= v < 580 or int(v) == 787: return "Digestive"
    if 580 <= v < 630 or int(v) == 788: return "Genitourinary"
    if 800 <= v < 1000: return "Injury"
    if 710 <= v < 740: return "Musculoskeletal"
    if 140 <= v < 240: return "Neoplasms"
    return "Other"


REQUIRED_RAW = {"encounter_id", "patient_nbr", "readmitted", "age", "gender", "race", "time_in_hospital",
                "discharge_disposition_id", "diag_1", "diag_2", "diag_3"}


def prepare_cohort(raw: pd.DataFrame, max_rows: int | None = None, seed: int = 42):
    """Raw UCI table -> (modelling table, cohort-funnel steps, dropped near-constant columns)."""
    missing = REQUIRED_RAW - set(raw.columns)
    if missing:
        raise ValueError(f"This does not look like diabetic_data.csv - missing columns: {sorted(missing)}")
    steps = []
    df = raw.copy().replace("?", np.nan)
    steps.append(("raw encounters", len(df)))
    df = df[df["gender"].isin(["Male", "Female"])]
    steps.append(("valid gender", len(df)))
    disp = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    df = df[~disp.isin(EXPIRED_HOSPICE)]
    steps.append(("excluding expired / hospice discharges", len(df)))
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    steps.append(("first encounter per patient", len(df)))

    df[TARGET] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int).astype(str)
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].map(icd9_group)
    for c in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int).astype(str)
    for c in ["max_glu_serum", "A1Cresult"]:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    df = df.drop(columns=[c for c in ["encounter_id", "patient_nbr", "weight", "readmitted"] if c in df.columns])
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    cat_cols = [c for c in df.columns if c not in NUM_COLS]
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    dropped = [c for c in cat_cols if c != TARGET and df[c].value_counts(normalize=True).iloc[0] > 0.99]
    df = df.drop(columns=dropped)
    df = df.reset_index(drop=True)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
    return df, steps, dropped


# ════════════════════════════════════════════════════════════════════
# 2 · Tabular encoder
# ════════════════════════════════════════════════════════════════════
class TabularEncoder:
    OTHER = "__other__"

    def __init__(self, num_cols, cat_cols, min_freq=0.005):
        self.num_cols, self.cat_cols, self.min_freq = list(num_cols), list(cat_cols), min_freq

    # -- layout -----------------------------------------------------
    def _layout(self):
        self.n_num, self.groups, start = len(self.num_cols), [], len(self.num_cols)
        for c in self.cat_cols:
            k = len(self.cats[c])
            self.groups.append((c, start, start + k))
            start += k
        self.dim = start

    def fit(self, df):
        self.num_min = {c: int(df[c].min()) for c in self.num_cols}
        self.num_max = {c: int(df[c].max()) for c in self.num_cols}
        x = np.log1p(df[self.num_cols].values.astype(float))
        self.mean, sd = x.mean(0), x.std(0)
        sd[sd == 0] = 1.0
        self.std = sd
        self.cats, self.fallback = {}, {}
        for c in self.cat_cols:
            vc = df[c].astype(str).value_counts(normalize=True)
            keep = sorted(vc[vc >= self.min_freq].index.tolist())
            rare = bool((vc < self.min_freq).any())
            self.cats[c] = keep + ([self.OTHER] if rare else [])
            self.fallback[c] = self.OTHER if rare else vc.index[0]
        self._layout()
        return self

    # -- (de)serialisation -------------------------------------------
    def to_dict(self):
        return dict(num_cols=self.num_cols, cat_cols=self.cat_cols, min_freq=self.min_freq,
                    num_min=self.num_min, num_max=self.num_max,
                    scale_mean=np.asarray(self.mean).tolist(), scale_std=np.asarray(self.std).tolist(),
                    cats=self.cats, fallback=self.fallback)

    @classmethod
    def from_dict(cls, d):
        e = cls(d["num_cols"], d["cat_cols"], d.get("min_freq", 0.005))
        e.num_min = {k: int(v) for k, v in d["num_min"].items()}
        e.num_max = {k: int(v) for k, v in d["num_max"].items()}
        e.mean, e.std = np.asarray(d["scale_mean"], float), np.asarray(d["scale_std"], float)
        e.cats, e.fallback = d["cats"], d["fallback"]
        e._layout()
        return e

    # -- transforms --------------------------------------------------
    def prepare(self, df):
        out = df[self.num_cols + self.cat_cols].copy()
        for c in self.cat_cols:
            out[c] = out[c].astype(str)
            out[c] = out[c].where(out[c].isin(self.cats[c]), self.fallback[c])
        for c in self.num_cols:
            out[c] = out[c].clip(self.num_min[c], self.num_max[c]).astype(int)
        return out.reset_index(drop=True)

    def transform(self, df):
        d = self.prepare(df)
        num = (np.log1p(d[self.num_cols].values.astype(float)) - self.mean) / self.std
        blocks = [num.astype(np.float32)]
        for c in self.cat_cols:
            idx = pd.Categorical(d[c], categories=self.cats[c]).codes
            oh = np.zeros((len(d), len(self.cats[c])), dtype=np.float32)
            oh[np.arange(len(d)), idx] = 1.0
            blocks.append(oh)
        return np.hstack(blocks)

    def inverse_transform(self, mat):
        mat = np.asarray(mat)
        z = mat[:, :self.n_num] * self.std + self.mean
        lo = np.log1p([self.num_min[c] for c in self.num_cols])
        hi = np.log1p([self.num_max[c] for c in self.num_cols])
        num = np.rint(np.expm1(np.clip(z, lo, hi))).astype(int)
        out = pd.DataFrame(num, columns=self.num_cols)
        for c, s, e in self.groups:
            out[c] = np.array(self.cats[c])[mat[:, s:e].argmax(1)]
        return out[self.num_cols + self.cat_cols]

    def split_xy(self, X):
        s, e = [(s, e) for c, s, e in self.groups if c == TARGET][0]
        mask = np.ones(X.shape[1], bool)
        mask[s:e] = False
        return X[:, mask], X[:, s:e].argmax(1)


# ════════════════════════════════════════════════════════════════════
# 3 · Models (identical to the notebook so checkpoints load unchanged)
# ════════════════════════════════════════════════════════════════════
def recon_loss(out, x, enc):
    n = enc.n_num
    mse = F.mse_loss(out[:, :n], x[:, :n], reduction="none").sum(1)
    ce = 0.0
    for _, s, e in enc.groups:
        ce = ce - (x[:, s:e] * F.log_softmax(out[:, s:e], dim=1)).sum(1)
    return (mse + ce).mean()


class AutoEncoder(nn.Module):
    def __init__(self, d_in, d_lat, h=(256, 128)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]), nn.ReLU(), nn.Linear(h[1], d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, h[1]), nn.ReLU(), nn.Linear(h[1], h[0]), nn.ReLU(), nn.Linear(h[0], d_in))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class VAE(nn.Module):
    def __init__(self, d_in, d_lat, h=(256, 128)):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, h[0]), nn.ReLU(), nn.Linear(h[0], h[1]), nn.ReLU())
        self.mu, self.logvar = nn.Linear(h[1], d_lat), nn.Linear(h[1], d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, h[1]), nn.ReLU(), nn.Linear(h[1], h[0]), nn.ReLU(), nn.Linear(h[0], d_in))

    def encode(self, x):
        h = self.trunk(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.fc, self.bn = nn.Linear(h, h), nn.BatchNorm1d(h)

    def forward(self, x):
        return x + F.relu(self.bn(self.fc(x)))


class Generator(nn.Module):
    def __init__(self, z_dim, d_out, h=256, blocks=2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(z_dim, h), nn.BatchNorm1d(h), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(h) for _ in range(blocks)])
        self.out = nn.Linear(h, d_out)

    def forward(self, z):
        return self.out(self.blocks(self.inp(z)))


class Critic(nn.Module):
    def __init__(self, d_in, h=(256, 128)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h[0]), nn.LeakyReLU(0.2), nn.Linear(h[0], h[1]), nn.LeakyReLU(0.2), nn.Linear(h[1], 1))

    def forward(self, x):
        return self.net(x)


def gan_activate(raw, enc, tau=0.2, hard=False):
    parts = [raw[:, :enc.n_num]]
    for _, s, e in enc.groups:
        parts.append(F.gumbel_softmax(raw[:, s:e], tau=tau, hard=hard))
    return torch.cat(parts, 1)


# ════════════════════════════════════════════════════════════════════
# 4 · Training (progress callbacks keep the UI responsive)
# ════════════════════════════════════════════════════════════════════
def _loader(X, bs):
    return DataLoader(TensorDataset(X), batch_size=min(bs, len(X)), shuffle=True, drop_last=len(X) > bs)


def train_ae(model, X, enc, epochs, lr, bs, progress=None):
    opt, loader, hist = torch.optim.Adam(model.parameters(), lr=lr), _loader(X, bs), []
    model.train()
    for ep in range(epochs):
        tot = n = 0
        for (xb,) in loader:
            out, _ = model(xb)
            loss = recon_loss(out, xb, enc)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        hist.append(tot / n)
        if progress: progress((ep + 1) / epochs, f"Autoencoder · epoch {ep + 1}/{epochs} · loss {hist[-1]:.2f}")
    return hist


def train_vae(model, X, enc, epochs, lr, bs, beta, anneal_frac=0.3, progress=None):
    opt, loader, hist = torch.optim.Adam(model.parameters(), lr=lr), _loader(X, bs), []
    warm = max(1, int(anneal_frac * epochs))
    model.train()
    for ep in range(epochs):
        b = beta * min(1.0, (ep + 1) / warm)
        tr = tk = n = 0
        for (xb,) in loader:
            out, mu, lv = model(xb)
            rec = recon_loss(out, xb, enc)
            kl = (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1)).mean()
            loss = rec + b * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tr += rec.item() * len(xb); tk += kl.item() * len(xb); n += len(xb)
        hist.append((tr / n, tk / n))
        if progress: progress((ep + 1) / epochs, f"VAE · epoch {ep + 1}/{epochs} · recon {hist[-1][0]:.2f} · KL {hist[-1][1]:.2f}")
    return hist


def _gradient_penalty(D, real, fake):
    a = torch.rand(real.size(0), 1, device=real.device)
    inter = (a * real + (1 - a) * fake).requires_grad_(True)
    d = D(inter)
    g = torch.autograd.grad(d, inter, torch.ones_like(d), create_graph=True, retain_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()


def train_wgan_gp(G, D, X, enc, steps, bs, z_dim, n_critic=5, lr=2e-4, gp_w=10.0, progress=None):
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    hist, N = [], len(X)
    bs = min(bs, N)
    G.train(); D.train()
    for step in range(steps):
        for _ in range(n_critic):
            real = X[torch.randint(0, N, (bs,), device=X.device)]
            with torch.no_grad():
                fake = gan_activate(G(torch.randn(bs, z_dim, device=X.device)), enc)
            wd = D(real).mean() - D(fake).mean()
            loss_d = -wd + gp_w * _gradient_penalty(D, real, fake)
            oD.zero_grad(); loss_d.backward(); oD.step()
        fake = gan_activate(G(torch.randn(bs, z_dim, device=X.device)), enc)
        loss_g = -D(fake).mean()
        oG.zero_grad(); loss_g.backward(); oG.step()
        hist.append((loss_d.item(), loss_g.item(), wd.item()))
        if progress and (step % 5 == 0 or step == steps - 1):
            progress((step + 1) / steps, f"WGAN-GP · step {step + 1}/{steps} · Wasserstein est. {wd.item():.3f}")
    return hist


# ════════════════════════════════════════════════════════════════════
# 5 · Generation
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def logits_to_matrix(out, enc, res_std=None):
    out = out.detach().cpu().float()
    n = enc.n_num
    mat = torch.zeros_like(out)
    num = out[:, :n]
    if res_std is not None:
        num = num + torch.randn_like(num) * res_std.cpu()
    mat[:, :n] = num
    rows = torch.arange(len(out))
    for _, s, e in enc.groups:
        idx = torch.multinomial(F.softmax(out[:, s:e], dim=1), 1).squeeze(1)
        mat[rows, s + idx] = 1.0
    return mat.numpy()


def residual_std(model, X, enc, kind):
    """Per-numeric-column reconstruction residual (used to add Gaussian decoder noise when sampling)."""
    model.eval()
    with torch.no_grad():
        Xs = X[:20000]
        out = model(Xs)[0] if kind == "ae" else model.dec(model.encode(Xs)[0])
        return (out[:, :enc.n_num] - Xs[:, :enc.n_num]).std(0)


class Bundle:
    """Everything needed to generate: encoder + the three trained generators."""

    def __init__(self, enc, ae, ae_res, gmm, vae, vae_res, G, latent_dim, z_dim, meta=None):
        self.enc, self.ae, self.ae_res, self.gmm = enc, ae, ae_res, gmm
        self.vae, self.vae_res, self.G = vae, vae_res, G
        self.latent_dim, self.z_dim, self.meta = latent_dim, z_dim, meta or {}


def sample_gmm(gmm, n, rng):
    """Sample latent codes from the Gaussian mixture (jittered Cholesky: robust to tiny asymmetries / near-singular covariances)."""
    w, mu, cov = gmm["w"], gmm["mu"], gmm["cov"]
    d = mu.shape[1]
    comp = rng.choice(len(w), size=n, p=w / w.sum())
    z = np.zeros((n, d))
    for k in np.unique(comp):
        idx = comp == k
        c = (cov[k] + cov[k].T) / 2 + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(c)
        z[idx] = mu[k] + rng.standard_normal((int(idx.sum()), d)) @ L.T
    return z


@torch.no_grad()
def generate(bundle: Bundle, name: str, n: int, seed: int = 0, batch: int = 8192) -> pd.DataFrame:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    for m in (bundle.ae, bundle.vae, bundle.G):
        m.eval()
    mats = []
    for i in range(0, n, batch):
        m = min(batch, n - i)
        if name == "AE+GMM":
            z = torch.tensor(sample_gmm(bundle.gmm, m, rng), dtype=torch.float32, device=DEVICE)
            mats.append(logits_to_matrix(bundle.ae.dec(z), bundle.enc, bundle.ae_res))
        elif name == "VAE":
            z = torch.randn(m, bundle.latent_dim, device=DEVICE)
            mats.append(logits_to_matrix(bundle.vae.dec(z), bundle.enc, bundle.vae_res))
        else:
            z = torch.randn(m, bundle.z_dim, device=DEVICE)
            mats.append(logits_to_matrix(bundle.G(z), bundle.enc))
    return bundle.enc.inverse_transform(np.vstack(mats))


def generate_filtered(bundle, name, n, filters=None, seed=0, max_rounds=40, batch=5000):
    """Rejection-sampled 'what-if' cohorts. filters = {column: iterable of allowed categories}.
    Returns (table, rows_tried). The table may hold fewer than n rows if the cohort is very rare."""
    if not filters:
        return generate(bundle, name, n, seed), n
    kept, got, tried = [], 0, 0
    for rnd in range(max_rounds):
        d = generate(bundle, name, batch, seed + rnd)
        tried += len(d)
        mask = np.ones(len(d), bool)
        for c, vals in filters.items():
            mask &= d[c].isin(list(vals)).values
        kept.append(d[mask])
        got += int(mask.sum())
        if got >= n:
            break
    out = pd.concat(kept).head(n).reset_index(drop=True)
    return out, tried


def load_bundle(art_dir) -> Bundle:
    art = Path(art_dir)
    enc = TabularEncoder.from_dict(json.loads((art / "encoder.json").read_text()))
    meta = json.loads((art / "meta.json").read_text())
    ld, zd = int(meta["latent_dim"]), int(meta["gan_z_dim"])
    ck = torch.load(art / "ae.pt", map_location=DEVICE, weights_only=True)
    ae = AutoEncoder(enc.dim, ld).to(DEVICE); ae.load_state_dict(ck["state"]); ae.eval()
    gmm = dict(w=ck["gmm_w"].numpy().astype(float), mu=ck["gmm_mu"].numpy().astype(float), cov=ck["gmm_cov"].numpy().astype(float))
    ae_res = ck["res_std"].to(DEVICE)
    ck = torch.load(art / "vae.pt", map_location=DEVICE, weights_only=True)
    vae = VAE(enc.dim, ld).to(DEVICE); vae.load_state_dict(ck["state"]); vae.eval()
    vae_res = ck["res_std"].to(DEVICE)
    ck = torch.load(art / "gan.pt", map_location=DEVICE, weights_only=True)
    G = Generator(zd, enc.dim).to(DEVICE); G.load_state_dict(ck["state"]); G.eval()
    return Bundle(enc, ae, ae_res, gmm, vae, vae_res, G, ld, zd, meta)


def train_bundle(real_train: pd.DataFrame, enc: TabularEncoder, cfg: dict, seed: int = 42, progress=None):
    """Train all three generators on a prepared training table. Returns (Bundle, histories, seconds)."""
    torch.manual_seed(seed); np.random.seed(seed)
    X = torch.tensor(enc.transform(real_train), dtype=torch.float32, device=DEVICE)
    hist, secs = {}, {}

    def span(lo, hi):
        return (lambda f, t: progress(lo + f * (hi - lo), t)) if progress else None

    t0 = time.time()
    ae = AutoEncoder(enc.dim, cfg["latent_dim"]).to(DEVICE)
    hist["AE+GMM"] = train_ae(ae, X, enc, cfg["ae_epochs"], cfg["lr"], cfg["batch_size"], span(0.0, 0.25))
    ae.eval()
    with torch.no_grad():
        Z = ae.enc(X).cpu().numpy()
    k = max(1, min(cfg["gmm_components"], len(Z) // 50))
    g = GaussianMixture(k, covariance_type="full", random_state=seed, reg_covar=1e-4).fit(Z)
    gmm = dict(w=g.weights_, mu=g.means_, cov=g.covariances_)
    ae_res = residual_std(ae, X, enc, "ae")
    secs["AE+GMM"] = time.time() - t0

    t0 = time.time()
    vae = VAE(enc.dim, cfg["latent_dim"]).to(DEVICE)
    hist["VAE"] = train_vae(vae, X, enc, cfg["vae_epochs"], cfg["lr"], cfg["batch_size"], cfg["vae_beta"],
                            cfg.get("kl_anneal_frac", 0.3), span(0.25, 0.5))
    vae_res = residual_std(vae, X, enc, "vae")
    secs["VAE"] = time.time() - t0

    t0 = time.time()
    G, D = Generator(cfg["gan_z_dim"], enc.dim).to(DEVICE), Critic(enc.dim).to(DEVICE)
    hist["WGAN-GP"] = train_wgan_gp(G, D, X, enc, cfg["gan_steps"], cfg["batch_size"], cfg["gan_z_dim"],
                                    cfg.get("gan_n_critic", 5), cfg.get("gan_lr", 2e-4), cfg.get("gan_gp", 10.0), span(0.5, 1.0))
    G.eval()
    secs["WGAN-GP"] = time.time() - t0
    return Bundle(enc, ae, ae_res, gmm, vae, vae_res, G, cfg["latent_dim"], cfg["gan_z_dim"]), hist, secs


def marginal_baseline(real: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sanity floor: every column sampled independently (keeps histograms, destroys relationships)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({c: rng.choice(real[c].values, size=n, replace=True) for c in real.columns})


# ════════════════════════════════════════════════════════════════════
# 6 · Reference statistics + evaluation
# ════════════════════════════════════════════════════════════════════
def tvd_from_props(real_prop: dict, syn: pd.Series) -> float:
    sp = syn.value_counts(normalize=True)
    keys = set(real_prop) | set(sp.index.astype(str))
    return 0.5 * sum(abs(real_prop.get(k, 0.0) - float(sp.get(k, 0.0))) for k in keys)


def ks_from_pmf(pmf: dict, values) -> float:
    vals = pd.Series(values)
    sp = vals.value_counts(normalize=True)
    keys = sorted(set(int(k) for k in pmf) | set(int(k) for k in sp.index))
    cr = np.cumsum([pmf.get(str(k), 0.0) for k in keys])
    cs = np.cumsum([float(sp.get(k, 0.0)) for k in keys])
    return float(np.max(np.abs(cr - cs)))


def column_scores_ref(ref: dict, syn: pd.DataFrame, enc: TabularEncoder) -> pd.DataFrame:
    rows = [(c, "numeric", 1 - ks_from_pmf(ref["num_pmf"][c], syn[c])) for c in enc.num_cols]
    rows += [(c, "categorical", 1 - tvd_from_props(ref["cat_prop"][c], syn[c].astype(str))) for c in enc.cat_cols]
    return pd.DataFrame(rows, columns=["column", "type", "score"])


def cramers_v(x, y):
    kx, ky = x.max() + 1, y.max() + 1
    if kx < 2 or ky < 2:
        return 0.0
    ct = np.bincount(x * ky + y, minlength=kx * ky).reshape(kx, ky).astype(float)
    n = ct.sum()
    exp = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - exp) ** 2 / np.where(exp == 0, 1, exp)).sum()
    phi2 = max(0.0, chi2 / n - (kx - 1) * (ky - 1) / (n - 1))
    kxc, kyc = kx - (kx - 1) ** 2 / (n - 1), ky - (ky - 1) ** 2 / (n - 1)
    d = min(kxc - 1, kyc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def corr_ratio(codes, vals):
    vals = vals.astype(float)
    tot = ((vals - vals.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k)
    sm = np.bincount(codes, weights=vals, minlength=k)
    means = np.divide(sm, cnt, out=np.zeros(k), where=cnt > 0)
    return float(np.sqrt((cnt * (means - vals.mean()) ** 2).sum() / tot))


def association_matrix(d: pd.DataFrame, enc: TabularEncoder, n=None, seed=0) -> pd.DataFrame:
    if n and len(d) > n:
        d = d.sample(n, random_state=seed)
    cols = enc.num_cols + enc.cat_cols
    codes = {c: pd.factorize(d[c])[0] for c in enc.cat_cols}
    vals = {c: np.log1p(d[c].values.astype(float)) for c in enc.num_cols}
    M = np.eye(len(cols))
    with np.errstate(all="ignore"):
        for i, a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                b = cols[j]
                if a in vals and b in vals: v = abs(np.corrcoef(vals[a], vals[b])[0, 1])
                elif a in vals: v = corr_ratio(codes[b], vals[a])
                elif b in vals: v = corr_ratio(codes[a], vals[b])
                else: v = cramers_v(codes[a], codes[b])
                M[i, j] = M[j, i] = 0.0 if np.isnan(v) else v
    return pd.DataFrame(M, index=cols, columns=cols)


def relationship_score(Ar: pd.DataFrame, As: pd.DataFrame, thr=0.10):
    iu = np.triu_indices(len(Ar), 1)
    r, s = Ar.values[iu], As.values[iu]
    mask = r >= thr
    if mask.sum() < 5:
        mask = r >= np.sort(r)[-5]
    return float(max(0.0, 1 - np.abs(r[mask] - s[mask]).mean() / r[mask].mean())), int(mask.sum())


def group_rates(d, col):
    return d.groupby(col)[TARGET].apply(lambda s: (s == "1").mean())


def build_reference(real: pd.DataFrame, enc: TabularEncoder, assoc_n=15000) -> dict:
    """Aggregate statistics of the real training table - no patient-level rows."""
    A = association_matrix(real, enc, assoc_n)
    sub_cols = [c for c in SUBGROUP_COLS if c in enc.cat_cols]
    return dict(
        n_train=int(len(real)), target_rate=float((real[TARGET] == "1").mean()),
        num_pmf={c: {str(k): float(v) for k, v in real[c].value_counts(normalize=True).items()} for c in enc.num_cols},
        cat_prop={c: {str(k): float(v) for k, v in real[c].value_counts(normalize=True).items()} for c in enc.cat_cols},
        assoc_cols=list(A.columns), assoc=A.round(5).values.tolist(),
        subgroup_rates={c: {str(k): float(v) for k, v in group_rates(real, c).items()} for c in sub_cols},
        subgroup_share={c: {str(k): float(v) for k, v in real[c].value_counts(normalize=True).items()} for c in sub_cols},
    )


def subgroup_score_ref(ref: dict, syn: pd.DataFrame, col: str) -> float:
    r = pd.Series(ref["subgroup_rates"][col])
    s = group_rates(syn, col)
    s.index = s.index.astype(str)
    s = s.reindex(r.index).fillna(0)
    w = pd.Series(ref["subgroup_share"][col]).reindex(r.index)
    mae = float((w * (r - s).abs()).sum())
    return max(0.0, 1 - mae / max(ref["target_rate"], 1e-9))


# -- utility ------------------------------------------------------------
CLASSIFIERS = {
    "LogReg": lambda seed: LogisticRegression(max_iter=300),
    "HistGB": lambda seed: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed),
}


def fit_eval(Xtr, ytr, Xte, yte, seed=42):
    out = {}
    for name, mk in CLASSIFIERS.items():
        if len(np.unique(ytr)) < 2:
            out[name] = dict(auc=np.nan, ap=np.nan)
            continue
        clf = mk(seed)
        clf.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        p = clf.predict_proba(Xte)[:, 1]
        out[name] = dict(auc=roc_auc_score(yte, p), ap=average_precision_score(yte, p))
    return out


# -- privacy ------------------------------------------------------------
def exact_match_rate(real, syn) -> float:
    hr = set(pd.util.hash_pandas_object(real, index=False))
    return float(pd.util.hash_pandas_object(syn, index=False).isin(hr).mean())


def qi_hit_rate(probe, real) -> float:
    cnt = real.groupby(QI).size().rename("k").reset_index()
    uniq = cnt[cnt["k"] == 1][QI].assign(_hit=1)
    return float(probe[QI].merge(uniq, on=QI, how="left")["_hit"].notna().mean())


def privacy_audit(Xtr, Xhold, Xsyn, n, seed=42):
    rng = np.random.default_rng(seed)
    pick = lambda M: M[rng.choice(len(M), min(n, len(M)), replace=False)]
    mem, hold, probe = pick(Xtr), pick(Xhold), pick(Xsyn)
    nn_syn = NearestNeighbors(n_neighbors=1).fit(Xsyn)
    d_mem, d_non = nn_syn.kneighbors(mem)[0][:, 0], nn_syn.kneighbors(hold)[0][:, 0]
    mia = roc_auc_score(np.r_[np.ones(len(d_mem)), np.zeros(len(d_non))], -np.r_[d_mem, d_non])
    nn_tr = NearestNeighbors(n_neighbors=1).fit(Xtr)
    dcr_syn, dcr_hold = nn_tr.kneighbors(probe)[0][:, 0], nn_tr.kneighbors(hold)[0][:, 0]
    thr = np.percentile(dcr_hold, 5)
    return dict(mia_auc=float(mia), leak_rate=float((dcr_syn < thr).mean()),
                dcr_syn_median=float(np.median(dcr_syn)), dcr_hold_median=float(np.median(dcr_hold)))


def quick_evaluate(real_train, real_test, synth: dict, enc: TabularEncoder, n_probe=1500, assoc_n=6000, seed=42, progress=None):
    """Compact version of the notebook's three-pillar audit, sized for a live demo.
    Returns (metrics table incl. pillar scores, TRTR AUCs, controls table)."""
    ref = build_reference(real_train, enc, assoc_n)
    Ar = pd.DataFrame(ref["assoc"], index=ref["assoc_cols"], columns=ref["assoc_cols"])
    Xtr, Xte = enc.transform(real_train), enc.transform(real_test)
    Xtr_f, ytr = enc.split_xy(Xtr)
    Xte_f, yte = enc.split_xy(Xte)
    trtr = fit_eval(Xtr_f, ytr, Xte_f, yte, seed)
    denom = {c: max(trtr[c]["auc"] - 0.5, 0.02) for c in trtr}

    # calibration controls for the privacy audit
    rng = np.random.default_rng(seed)
    half = len(real_test) // 2
    ctrl = {"Control: memorising copy": (real_train.iloc[rng.permutation(len(real_train))].reset_index(drop=True), Xte[:half]),
            "Control: fresh real": (real_test.iloc[half:].reset_index(drop=True), Xte[:half])}
    ctrl_rows = {}
    for name, (tbl, Xhold) in ctrl.items():
        pa = privacy_audit(Xtr, Xhold, enc.transform(tbl), n_probe, seed)
        pa["exact_match"] = exact_match_rate(real_train, tbl)
        ctrl_rows[name] = pa
    ref_leak = max(0.05, ctrl_rows["Control: fresh real"]["leak_rate"])

    rows, total = {}, len(synth)
    for i, (name, syn) in enumerate(synth.items()):
        if progress: progress(i / total, f"Evaluating {name} …")
        Xs = enc.transform(syn)
        col = float(column_scores_ref(ref, syn, enc)["score"].mean())
        rel, _ = relationship_score(Ar, association_matrix(syn, enc, assoc_n))
        sub = float(np.mean([subgroup_score_ref(ref, syn, c) for c in ref["subgroup_rates"]])) if ref["subgroup_rates"] else np.nan
        res = fit_eval(*enc.split_xy(Xs), Xte_f, yte, seed)
        skill = float(np.mean([min(1.0, max(0.0, res[c]["auc"] - 0.5) / denom[c]) for c in res]))
        pa = privacy_audit(Xtr, Xte, Xs, n_probe, seed)
        p_mia = 1 - 2 * max(0.0, pa["mia_auc"] - 0.5)
        p_dcr = min(1.0, ref_leak / max(pa["leak_rate"], 1e-9))
        fid = float(np.nanmean([col, rel, sub]))
        priv = (p_mia + p_dcr) / 2
        rows[name] = {"Column shapes": col, "Relationships": rel, "Subgroups": sub, "Utility": skill,
                      "Privacy (MIA)": p_mia, "Privacy (DCR)": p_dcr, "Fidelity": fid, "Privacy": priv,
                      "Composite": (fid + skill + priv) / 3,
                      "MIA AUC": pa["mia_auc"], "DCR leak rate": pa["leak_rate"],
                      "Exact-match rate": exact_match_rate(real_train, syn)}
    if progress: progress(1.0, "Done")
    return pd.DataFrame(rows).T, trtr, pd.DataFrame(ctrl_rows).T
