"""
SynthHealth - privacy-aware synthetic clinical data (Autoencoder · VAE · WGAN-GP)
Streamlit front-end for the CAT7001 deep-learning lab project.

Run locally :  streamlit run app.py
Artifacts   :  put the notebook's `app_artifacts` export into ./artifacts  (see README.md)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

import synth_core as sc

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

st.set_page_config(page_title="SynthHealth · synthetic clinical data", page_icon="🏥",
                   layout="wide", initial_sidebar_state="expanded")

ART = Path(os.environ.get("SYNTH_ARTIFACTS", Path(__file__).parent / "artifacts"))
LOCAL_CSV = Path(os.environ.get("SYNTH_LOCAL_CSV", "diabetic_data.csv"))
REQUIRED = ["encoder.json", "meta.json", "reference.json", "ae.pt", "vae.pt", "gan.pt"]
COLORS = {"AE+GMM": "#1f77b4", "VAE": "#2ca02c", "WGAN-GP": "#d62728", "Marginals": "#9e9e9e", "Real": "#2b2b2b"}
FILTER_COLS = ["age", "gender", "race", "diag_1", "early_readmit", "insulin", "A1Cresult", "diabetesMed"]
PAGES = ["🏠 Overview", "🧬 Generate data", "📊 Compare models", "🧪 Train & test live"]


# ════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════
def artifacts_ready() -> bool:
    return all((ART / f).exists() for f in REQUIRED)


def _stamp(name: str) -> float:
    p = ART / name
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource(show_spinner="Loading trained generators …")
def get_bundle(stamp: float):
    return sc.load_bundle(ART)


@st.cache_data
def get_json(name: str, stamp: float):
    import json
    return json.loads((ART / name).read_text())


@st.cache_data
def get_csv(name: str, stamp: float, index_col=None):
    p = ART / name
    return pd.read_csv(p, index_col=index_col) if p.exists() else None


def load_ref():
    return get_json("reference.json", _stamp("reference.json"))


def load_meta():
    return get_json("meta.json", _stamp("meta.json"))


def csv(name, index_col=None):
    return get_csv(name, _stamp(name), index_col)


def missing_artifacts_notice():
    st.warning("**Trained models were not found** in the `artifacts/` folder, so this page is switched off.")
    st.markdown(
        "1. Run the notebook `Synthetic_Clinical_Data_AE_VAE_GAN.ipynb` with `FAST_DEV = False` – its last cell exports **app_artifacts.zip**.  \n"
        "2. Unzip it into this app's `artifacts/` folder and restart the app.  \n"
        "3. The **🧪 Train & test live** page works without artifacts – it trains small models on a dataset you provide."
    )


def style_fig(fig, height=380):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=height, legend_title_text="")
    return fig


def score_cols(cols):
    return {c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.2f") for c in cols}


def numeric_compare(ref, syn, col):
    pmf = pd.Series({int(k): v for k, v in ref["num_pmf"][col].items()}).sort_index()
    cap = int(pmf.index[min(np.searchsorted(pmf.cumsum().values, 0.995), len(pmf) - 1)])
    support = list(range(int(pmf.index.min()), cap + 1))
    real = pmf.reindex(support, fill_value=0.0)
    s = syn[col].value_counts(normalize=True).reindex(support, fill_value=0.0)
    fig = go.Figure()
    fig.add_bar(x=support, y=real.values, name="Real (training)", marker_color=COLORS["Real"], opacity=0.35)
    fig.add_scatter(x=support, y=s.values, name="Synthetic", mode="lines+markers", line=dict(color="#0f766e", width=2))
    fig.update_layout(title=f"{col} – share of patients per value", xaxis_title=col, yaxis_title="share")
    return style_fig(fig)


def categorical_compare(ref, syn, col):
    real = pd.Series(ref["cat_prop"][col])
    s = syn[col].astype(str).value_counts(normalize=True)
    cats = sorted(set(real.index) | set(s.index))
    long = pd.DataFrame({"category": cats * 2,
                         "share": [real.get(c, 0.0) for c in cats] + [float(s.get(c, 0.0)) for c in cats],
                         "source": ["Real (training)"] * len(cats) + ["Synthetic"] * len(cats)})
    fig = px.bar(long, x="category", y="share", color="source", barmode="group",
                 color_discrete_map={"Real (training)": COLORS["Real"], "Synthetic": "#0f766e"}, title=f"{col} – category shares")
    return style_fig(fig)


# ════════════════════════════════════════════════════════════════════
# sidebar
# ════════════════════════════════════════════════════════════════════
st.sidebar.title("🏥 SynthHealth")
st.sidebar.caption("Privacy-aware synthetic clinical data · Autoencoder · VAE · WGAN-GP")
page = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
st.sidebar.divider()
if artifacts_ready():
    m = load_meta()
    st.sidebar.success("Trained models loaded")
    st.sidebar.caption(f"Trained on {m['n_train']:,} patients · exported {m.get('exported', '')[:10]}")
    if m.get("fast_dev"):
        st.sidebar.warning("These artifacts come from a quick FAST_DEV run – re-export a full run before presenting.")
else:
    st.sidebar.info("No trained models found – see the Overview page.")
st.sidebar.divider()
st.sidebar.caption("⚠️ Synthetic records are for research and teaching only. They are not for clinical use and are "
                   "not formally differentially private.")


# ════════════════════════════════════════════════════════════════════
# 🏠 Overview
# ════════════════════════════════════════════════════════════════════
def page_overview():
    st.title("Privacy-aware synthetic clinical data")
    st.markdown("**Autoencoder · VAE · GAN** on the UCI *Diabetes 130-US Hospitals* dataset – generated, compared and audited "
                "on **fidelity**, **utility** and **privacy**.")

    if artifacts_ready():
        score = csv("scorecard.csv", 0)
        m = load_meta()
        models = [n for n in sc.MODEL_NAMES if n in score.index]
        best = score.loc[models, "Composite"].idxmax()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training patients", f"{m['n_train']:,}")
        c2.metric("Encoded width", f"{m['encoded_width']} features")
        c3.metric("Best overall model", best)
        c4.metric("Composite score", f"{score.loc[best, 'Composite']:.3f}")
    else:
        missing_artifacts_notice()

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("The problem")
        st.write("Hospital records – demographics, diagnoses, medications, insurance payer – are tightly regulated, so researchers, "
                 "students and small institutions rarely get realistic data. Synthetic data is proposed as a fix, but only if it "
                 "(a) **looks like** the real data, (b) is **still useful** for a clinical task, and (c) **does not leak** real patients.")
        st.subheader("What this app does")
        st.markdown(
            "- **🧬 Generate data** – sample synthetic patients from any of the three trained models, optionally for a *what-if cohort* "
            "(e.g. only patients aged 70–80 with a circulatory diagnosis), and download them as CSV.\n"
            "- **📊 Compare models** – interactive scorecard, utility (train-on-synthetic, test-on-real), privacy audit, "
            "relationship preservation and the β-VAE privacy–utility frontier.\n"
            "- **🧪 Train & test live** – retrain small versions of all three models on a dataset you provide and audit them on the spot.\n"
        )
    with right:
        st.subheader("Pipeline")
        st.graphviz_chart('''
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#e6f2f0", color="#0f766e", fontname="Helvetica", fontsize=11];
  edge [color="#6b7280"];
  Data [label="UCI Diabetes 130-US\\nHospitals (101,766 rows)"];
  Clean [label="Cohort cleaning\\n(first encounter, no hospice)"];
  Enc [label="Tabular encoder\\n(log+z-score | one-hot)"];
  AE [label="Autoencoder\\n+ latent GMM"]; VAE [label="β-VAE\\n(KL annealing)"]; GAN [label="WGAN-GP\\n(Gumbel-softmax)"];
  Syn [label="Synthetic patients", fillcolor="#fff4e5", color="#d97706"];
  Audit [label="Audit: fidelity · utility ·\\nprivacy (DCR, MIA) · subgroups", fillcolor="#fff4e5", color="#d97706"];
  Data -> Clean -> Enc; Enc -> AE; Enc -> VAE; Enc -> GAN;
  AE -> Syn; VAE -> Syn; GAN -> Syn; Syn -> Audit;
}''')

    st.subheader("What makes it more than a standard VAE-vs-GAN comparison")
    st.markdown(
        "1. **Autoencoder made generative** – a latent-space Gaussian mixture gives the plain AE a sampling distribution.\n"
        "2. **β-VAE as a privacy dial** – utility, fidelity and membership-leakage plotted against β.\n"
        "3. **Relationship preservation** for mixed data (Cramér's V · correlation ratio · Pearson), not just histograms.\n"
        "4. **Calibrated privacy audit** – a memorising generator and a fresh-real generator prove the audit can detect leakage.\n"
        "5. **Clinical subgroup check** – is the readmission rate preserved per age band, race and diagnosis?\n"
        "6. **Leakage-safe design** – one encounter per patient, patient-level split, aggregate-only reference statistics in this app.")


# ════════════════════════════════════════════════════════════════════
# 🧬 Generate
# ════════════════════════════════════════════════════════════════════
def page_generate():
    st.header("🧬 Generate synthetic patients")
    if not artifacts_ready():
        missing_artifacts_notice()
        return
    try:
        bundle = get_bundle(_stamp("ae.pt") + _stamp("vae.pt") + _stamp("gan.pt"))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load the trained models: {e}")
        return
    ref, enc = load_ref(), bundle.enc

    c1, c2, c3 = st.columns([1.2, 1.5, 1])
    name = c1.selectbox("Generator", sc.MODEL_NAMES, help="AE+GMM: autoencoder with a latent Gaussian mixture · VAE: sample z ~ N(0, I) · WGAN-GP: adversarial generator")
    n = c2.slider("Number of synthetic patients", 100, 50_000, 2_000, step=100)
    seed = c3.number_input("Random seed", 0, 1_000_000, 42)

    with st.expander("🎯 What-if cohort (optional filters)", expanded=False):
        st.caption("Rejection sampling: the generator keeps drawing patients until enough match your filters. Leave empty for an unfiltered sample.")
        filters, cols = {}, st.columns(3)
        usable = [c for c in FILTER_COLS if c in enc.cat_cols]
        for i, c in enumerate(usable):
            opts = [o for o in enc.cats[c] if o != enc.OTHER]
            sel = cols[i % 3].multiselect(c, opts, key=f"flt_{c}")
            if sel:
                filters[c] = sel

    if st.button("Generate", type="primary"):
        with st.spinner("Sampling synthetic patients …"):
            t0 = time.time()
            df, tried = sc.generate_filtered(bundle, name, int(n), filters, int(seed))
            st.session_state["gen"] = dict(df=df, name=name, n=int(n), tried=tried, filters=filters,
                                           seed=int(seed), secs=time.time() - t0)

    g = st.session_state.get("gen")
    if not g:
        st.info("Choose a generator and press **Generate**.")
        return
    df, filt = g["df"], g["filters"]
    if len(df) < g["n"]:
        st.warning(f"Only {len(df):,} of the requested {g['n']:,} patients matched the filters after {g['tried']:,} draws – that cohort is rare "
                   f"for this generator. Relax a filter or pick another generator.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Patients generated", f"{len(df):,}")
    m2.metric("Acceptance rate" if filt else "Generation time", f"{len(df) / max(g['tried'], 1):.1%}" if filt else f"{g['secs']:.2f}s")
    m3.metric("Early-readmission rate", f"{(df[sc.TARGET] == '1').mean():.1%}" if len(df) else "–",
              delta=f"{(df[sc.TARGET] == '1').mean() - ref['target_rate']:+.1%} vs real" if len(df) else None, delta_color="off")
    if not filt and len(df):
        m4.metric("Mean column fidelity", f"{sc.column_scores_ref(ref, df, enc)['score'].mean():.3f}")
    else:
        m4.metric("Mean column fidelity", "n/a (filtered)")

    t_prev, t_fid, t_cmp, t_dl = st.tabs(["Preview", "Fidelity vs real", "Compare a column", "Download"])
    with t_prev:
        st.dataframe(df.head(300), hide_index=True)
    with t_fid:
        if filt:
            st.info("Fidelity is measured against the *whole* training population, so it is skipped for filtered cohorts. "
                    "Use **Compare a column** to see how the cohort differs from the population.")
        elif len(df):
            cs = sc.column_scores_ref(ref, df, enc).sort_values("score")
            fig = px.bar(cs, x="score", y="column", color="type", orientation="h", range_x=[0, 1],
                         title="Per-column similarity to the real training data (1 = identical)",
                         color_discrete_map={"numeric": "#1f77b4", "categorical": "#ff7f0e"})
            st.plotly_chart(style_fig(fig, max(380, 22 * len(cs))))
            st.caption("Numeric columns: 1 − Kolmogorov–Smirnov distance · categorical columns: 1 − total-variation distance, both against aggregate statistics of the real training table.")
    with t_cmp:
        if len(df):
            col = st.selectbox("Column", enc.num_cols + enc.cat_cols, index=enc.num_cols.index("time_in_hospital") if "time_in_hospital" in enc.num_cols else 0)
            st.plotly_chart(numeric_compare(ref, df, col) if col in enc.num_cols else categorical_compare(ref, df, col))
    with t_dl:
        st.download_button("⬇️ Download as CSV", df.to_csv(index=False).encode(), file_name=f"synthetic_{g['name'].replace('+', '_')}_{len(df)}.csv", mime="text/csv")
        st.caption("Generated records are synthetic, but no formal privacy guarantee applies – see the privacy audit on the **Compare models** page.")


# ════════════════════════════════════════════════════════════════════
# 📊 Compare
# ════════════════════════════════════════════════════════════════════
def page_compare():
    st.header("📊 Compare the generators")
    if not artifacts_ready():
        missing_artifacts_notice()
        return
    score = csv("scorecard.csv", 0)
    models = [n for n in sc.MODEL_NAMES if n in score.index]
    tabs = st.tabs(["Scorecard", "Utility (TSTR)", "Privacy audit", "Relationships & subgroups", "β-VAE frontier", "Notebook figures"])

    # ---- scorecard --------------------------------------------------
    with tabs[0]:
        st.caption("Each pillar is scored 0–1 (higher is better). Change the weights to reflect your priorities – a healthcare release might weight privacy higher.")
        w1, w2, w3 = st.columns(3)
        wf = w1.slider("Fidelity weight", 0.0, 1.0, 1 / 3, 0.05)
        wu = w2.slider("Utility weight", 0.0, 1.0, 1 / 3, 0.05)
        wp = w3.slider("Privacy weight", 0.0, 1.0, 1 / 3, 0.05)
        tot = max(wf + wu + wp, 1e-9)
        sc_df = score.copy()
        sc_df["Composite"] = (wf * sc_df["Fidelity"] + wu * sc_df["Utility"] + wp * sc_df["Privacy"]) / tot
        ranked = sc_df.loc[models].sort_values("Composite", ascending=False)
        st.success(f"**Best overall with these weights: {ranked.index[0]}** (composite {ranked['Composite'].iloc[0]:.3f})")
        a, b = st.columns(2)
        axes = [c for c in ["Column shapes", "Relationships", "Subgroups", "Utility", "Privacy (MIA)", "Privacy (DCR)"] if c in sc_df.columns]
        radar = go.Figure()
        for k in models + (["Marginals"] if "Marginals" in sc_df.index else []):
            v = sc_df.loc[k, axes].astype(float).tolist()
            radar.add_trace(go.Scatterpolar(r=v + v[:1], theta=axes + axes[:1], name=k, line=dict(color=COLORS.get(k), dash="dot" if k == "Marginals" else "solid"),
                                            fill="toself" if k != "Marginals" else None, opacity=0.9))
        radar.update_layout(title="Six audit dimensions", polar=dict(radialaxis=dict(range=[0, 1])))
        a.plotly_chart(style_fig(radar, 430))
        parts = pd.DataFrame({"Fidelity": ranked["Fidelity"] * wf / tot, "Utility": ranked["Utility"] * wu / tot, "Privacy": ranked["Privacy"] * wp / tot})
        stacked = px.bar(parts, orientation="h", barmode="stack", title="Weighted composite (stacked by pillar)",
                         color_discrete_map={"Fidelity": "#2ca02c", "Utility": "#1f77b4", "Privacy": "#9467bd"})
        stacked.update_layout(xaxis_range=[0, 1], yaxis_autorange="reversed", xaxis_title="weighted composite", yaxis_title="")
        b.plotly_chart(style_fig(stacked, 430))
        show = [c for c in ["Fidelity", "Utility", "Privacy", "Composite", "Column shapes", "Relationships", "Subgroups", "Privacy (MIA)", "Privacy (DCR)"] if c in sc_df.columns]
        st.dataframe(sc_df[show + (["Train time (s)"] if "Train time (s)" in sc_df.columns else [])], column_config=score_cols(show))
        st.caption("*Marginals* = every column sampled independently: it matches each histogram but destroys relationships, so it is the sanity floor for the metrics and is not ranked.")

    # ---- utility ----------------------------------------------------
    with tabs[1]:
        st.markdown("**Train on Synthetic, Test on Real (TSTR).** A classifier trained only on synthetic data predicts early readmission "
                    "on *real* hold-out patients. **Skill retention** = (AUC_TSTR − 0.5) ÷ (AUC_TRTR − 0.5): the share of the real model's above-chance skill that survives.")
        util = csv("utility.csv")
        if util is not None:
            met = st.radio("Metric", [c for c in ["ROC_AUC", "PR_AUC", "F1"] if c in util.columns], horizontal=True)
            fig = px.bar(util, x="source", y=met, color="classifier", barmode="group", title=f"{met} on the real hold-out set")
            if met == "ROC_AUC":
                fig.add_hline(y=0.5, line_dash="dot", annotation_text="chance")
            st.plotly_chart(style_fig(fig))
        skill = csv("skill.csv", 0)
        if skill is not None:
            st.markdown("**Skill retention** (1.0 = as useful as real data)")
            st.dataframe(skill, column_config=score_cols(list(skill.columns)))

    # ---- privacy ----------------------------------------------------
    with tabs[2]:
        st.markdown("Four empirical signals: **exact-match rate**, **distance to closest record (DCR)**, **quasi-identifier hits**, and a "
                    "**membership-inference attack (MIA)**. The two *Control* rows calibrate the audit: a generator that simply copies the training "
                    "set must be flagged, and fresh real patients must look clean.")
        priv = csv("privacy.csv", 0)
        if priv is not None:
            cols = st.columns(3)
            for c, (key, ttl, ref_line) in zip(cols, [("mia_auc", "Membership-inference AUC (0.5 = safe)", 0.5), ("leak_rate", "DCR leak rate (≈0.05 = chance)", 0.05), ("qi_hit", "Hits on unique real quasi-identifier combos", None)]):
                if key in priv.columns:
                    d = priv[[key]].reset_index().rename(columns={"index": "source"})
                    fig = px.bar(d, x="source", y=key, color="source", title=ttl,
                                 color_discrete_map={**COLORS, "Control: memorising copy": "#bbbbbb", "Control: fresh real": "#dddddd"})
                    if ref_line is not None:
                        fig.add_hline(y=ref_line, line_dash="dash")
                    fig.update_layout(showlegend=False, xaxis_title="")
                    c.plotly_chart(style_fig(fig, 360))
            st.dataframe(priv.round(4))
            st.caption("These are empirical audits, not a differential-privacy guarantee. Merging rare categories and dropping identifiers lowers – but does not eliminate – risk.")

    # ---- relationships & subgroups ------------------------------------
    with tabs[3]:
        top = csv("top_relationships.csv", 0)
        if top is not None:
            t = top.reset_index().melt(id_vars=top.index.name or "pair", var_name="source", value_name="association")
            t = t.rename(columns={t.columns[0]: "pair"})
            fig = px.bar(t, x="association", y="pair", color="source", orientation="h", barmode="group", color_discrete_map=COLORS,
                         title="The strongest real dependencies and how each generator reproduces them")
            st.plotly_chart(style_fig(fig, 520))
        fid = csv("column_fidelity.csv", 0)
        if fid is not None:
            srcs = [c for c in ["Marginals"] + models if c in fid.columns]
            fig = px.imshow(fid[srcs], aspect="auto", color_continuous_scale="RdYlGn", zmin=0.5, zmax=1, title="Per-column fidelity (1 = identical)")
            st.plotly_chart(style_fig(fig, max(420, 20 * len(fid))))
        sub = csv("subgroups.csv", 0)
        if sub is not None:
            st.markdown("**Subgroup preservation** – is the readmission rate right *inside* age / race / diagnosis groups?")
            st.dataframe(sub, column_config=score_cols(list(sub.columns)))

    # ---- beta sweep ---------------------------------------------------
    with tabs[4]:
        st.markdown("β weights the KL term in the VAE: larger β → smoother latent space, less memorisation, usually lower utility. "
                    "Each point is a separately trained VAE.")
        sw = csv("beta_sweep.csv")
        if sw is not None:
            a, b = st.columns(2)
            fig = go.Figure()
            fig.add_scatter(x=sw["beta"], y=sw["fidelity"], name="fidelity", mode="lines+markers", line_color="#2ca02c")
            fig.add_scatter(x=sw["beta"], y=sw["utility"].clip(upper=1.2), name="utility (skill retention)", mode="lines+markers", line_color="#1f77b4")
            fig.update_xaxes(type="log", title="β (log scale)")
            fig.update_layout(title="Fidelity and utility vs β")
            a.plotly_chart(style_fig(fig))
            fig2 = px.scatter(sw, x="mia_auc", y=sw["utility"].clip(upper=1.2), text=[f"β={b_:g}" for b_ in sw["beta"]], size_max=14,
                              title="Privacy–utility frontier (ideal: top-left)")
            fig2.update_traces(textposition="top center", marker=dict(size=14, color="#0f766e"))
            fig2.update_layout(xaxis_title="privacy risk (MIA AUC) →", yaxis_title="utility ↑")
            b.plotly_chart(style_fig(fig2))
            st.dataframe(sw.round(3), hide_index=True)

    # ---- figures ------------------------------------------------------
    with tabs[5]:
        figs = sorted((ART / "figures").glob("*.png")) if (ART / "figures").exists() else []
        if not figs:
            st.info("No figures were exported.")
        else:
            labels = {p.name: p.stem[3:].replace("_", " ").capitalize() for p in figs}
            pick = st.selectbox("Figure from the notebook", [p.name for p in figs], format_func=lambda x: labels[x])
            st.image(str(ART / "figures" / pick), caption=labels[pick])


# ════════════════════════════════════════════════════════════════════
# 🧪 Train & test live
# ════════════════════════════════════════════════════════════════════
def _read_uploaded(file):
    return pd.read_csv(file, na_values=["?"])


def page_train():
    st.header("🧪 Train & test live")
    st.caption("Retrain small versions of all three generators on a subset of the real dataset and audit them immediately. "
               "Nothing is stored. Quick settings give rougher results than the notebook's full run.")

    options = ["Upload diabetic_data.csv", "Download from UCI (ucimlrepo)"]
    if LOCAL_CSV.exists():
        options.insert(0, f"Local file ({LOCAL_CSV.name})")
    src = st.radio("Data source", options, horizontal=True)

    raw = st.session_state.get("raw")
    if src.startswith("Upload"):
        up = st.file_uploader("diabetic_data.csv (UCI dataset 296)", type=["csv"])
        if up is not None:
            try:
                raw = _read_uploaded(up)
                st.session_state["raw"] = raw
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that file: {e}")
    elif src.startswith("Local"):
        raw = pd.read_csv(LOCAL_CSV, na_values=["?"])
        st.session_state["raw"] = raw
    else:
        if st.button("Fetch dataset from UCI"):
            with st.spinner("Downloading from the UCI repository …"):
                try:
                    from ucimlrepo import fetch_ucirepo
                    data = fetch_ucirepo(id=296).data
                    raw = getattr(data, "original", None)
                    if raw is None or "readmitted" not in raw.columns:
                        raw = pd.concat([x for x in (data.ids, data.features, data.targets) if x is not None], axis=1)
                    st.session_state["raw"] = raw.replace("?", np.nan)
                    raw = st.session_state["raw"]
                except Exception as e:  # noqa: BLE001
                    st.error(f"Download failed ({e}). Upload the CSV instead.")
    if raw is None:
        st.info("Provide the dataset to continue.")
        return
    st.success(f"Dataset loaded: {len(raw):,} encounters × {raw.shape[1]} columns")

    with st.form("train_form"):
        c1, c2, c3 = st.columns(3)
        rows = c1.slider("Patients used (subsample)", 2_000, 20_000, 5_000, step=500)
        ae_ep = c1.slider("Autoencoder epochs", 5, 60, 15)
        vae_ep = c2.slider("VAE epochs", 5, 60, 15)
        beta = c2.slider("VAE β", 0.1, 4.0, 1.0, 0.1)
        gan_steps = c3.slider("WGAN-GP steps", 50, 1_500, 300, step=50)
        seed = c3.number_input("Seed", 0, 10_000, 42)
        go_btn = st.form_submit_button("🚀 Train and audit", type="primary")

    if go_btn:
        try:
            run_live(raw, rows, ae_ep, vae_ep, beta, gan_steps, int(seed))
        except ValueError as e:
            st.error(str(e))
            return

    live = st.session_state.get("live")
    if live:
        show_live(live)


def run_live(raw, rows, ae_ep, vae_ep, beta, gan_steps, seed):
    cfg = dict(latent_dim=16, batch_size=256, lr=1e-3, ae_epochs=ae_ep, gmm_components=6, vae_epochs=vae_ep, vae_beta=beta,
               kl_anneal_frac=0.3, gan_steps=gan_steps, gan_z_dim=64, gan_n_critic=5, gan_lr=2e-4, gan_gp=10.0)
    bar, msg = st.progress(0.0), st.empty()

    def prog(f, text):
        bar.progress(min(1.0, 0.05 + 0.75 * f))
        msg.write(text)

    msg.write("Preparing the cohort …")
    df, steps, _ = sc.prepare_cohort(raw, max_rows=rows, seed=seed)
    if len(df) < 1500:
        raise ValueError("Too few patients after cleaning – use a larger subsample or the full diabetic_data.csv.")
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=0.2, stratify=df[sc.TARGET], random_state=seed)
    cat_cols = [c for c in df.columns if c not in sc.NUM_COLS]
    enc = sc.TabularEncoder(sc.NUM_COLS, cat_cols).fit(tr)
    real_train, real_test = enc.prepare(tr), enc.prepare(te)

    bundle, hist, secs = sc.train_bundle(real_train, enc, cfg, seed, prog)
    msg.write("Generating synthetic tables …")
    synth = {"Marginals": sc.marginal_baseline(real_train, len(real_train), seed)}
    for name in sc.MODEL_NAMES:
        synth[name] = sc.generate(bundle, name, len(real_train), seed)
    metrics, trtr, ctrl = sc.quick_evaluate(real_train, real_test, synth, enc, n_probe=1500, assoc_n=6000, seed=seed,
                                            progress=lambda f, t: (bar.progress(min(1.0, 0.8 + 0.2 * f)), msg.write(t)))
    bar.progress(1.0)
    msg.write("✅ Finished")
    st.session_state["live"] = dict(metrics=metrics, trtr=trtr, ctrl=ctrl, hist=hist, secs=secs, synth=synth,
                                    n_train=len(real_train), n_test=len(real_test), steps=steps, rate=float((real_train[sc.TARGET] == "1").mean()))


def show_live(L):
    st.divider()
    st.subheader("Results")
    m = L["metrics"]
    models = [n for n in sc.MODEL_NAMES if n in m.index]
    best = m.loc[models, "Composite"].astype(float).idxmax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training / test patients", f"{L['n_train']:,} / {L['n_test']:,}")
    c2.metric("Early-readmission rate", f"{L['rate']:.1%}")
    c3.metric("Real-data model AUC", f"{np.nanmax([v['auc'] for v in L['trtr'].values()]):.3f}")
    c4.metric("Best overall", best)
    if max(v["auc"] for v in L["trtr"].values()) - 0.5 < 0.03:
        st.warning("The real-data model barely beats chance with this subsample, so utility scores are unreliable – use more patients.")

    pillars = ["Fidelity", "Utility", "Privacy", "Composite"]
    detail = ["Column shapes", "Relationships", "Subgroups", "Privacy (MIA)", "Privacy (DCR)"]
    st.dataframe(m[pillars + detail].astype(float), column_config=score_cols(pillars + detail))
    a, b = st.columns(2)
    fig = px.bar(m.loc[models, pillars[:3]].astype(float).reset_index().melt(id_vars="index"), x="index", y="value", color="variable", barmode="group",
                 title="Pillar scores", labels={"index": "", "value": "score", "variable": ""},
                 color_discrete_map={"Fidelity": "#2ca02c", "Utility": "#1f77b4", "Privacy": "#9467bd"})
    a.plotly_chart(style_fig(fig, 340))
    audit = m[["MIA AUC", "DCR leak rate", "Exact-match rate"]].astype(float).round(4)
    b.markdown("**Privacy audit** (MIA 0.5 and leak ≈ 5 % are the no-leakage levels)")
    b.dataframe(pd.concat([audit, L["ctrl"].rename(columns={"mia_auc": "MIA AUC", "leak_rate": "DCR leak rate", "exact_match": "Exact-match rate"})[["MIA AUC", "DCR leak rate", "Exact-match rate"]].astype(float).round(4)]))

    st.markdown("**Training curves**")
    t1, t2, t3 = st.columns(3)
    t1.caption("Autoencoder – reconstruction loss"); t1.line_chart(pd.DataFrame({"loss": L["hist"]["AE+GMM"]}))
    rec, kl = zip(*L["hist"]["VAE"])
    t2.caption("VAE – reconstruction and KL"); t2.line_chart(pd.DataFrame({"reconstruction": rec, "KL": kl}))
    gw = np.array([h[2] for h in L["hist"]["WGAN-GP"]])
    w = max(1, len(gw) // 40)
    t3.caption("WGAN-GP – Wasserstein estimate (smoothed)"); t3.line_chart(pd.DataFrame({"Wasserstein": np.convolve(gw, np.ones(w) / w, mode="valid")}))
    st.caption("Training time: " + " · ".join(f"{k} {v:.0f}s" for k, v in L["secs"].items()))

    pick = st.selectbox("Download a synthetic table", models)
    st.download_button("⬇️ CSV", L["synth"][pick].to_csv(index=False).encode(), file_name=f"live_synthetic_{pick.replace('+', '_')}.csv", mime="text/csv")


{"🏠 Overview": page_overview, "🧬 Generate data": page_generate, "📊 Compare models": page_compare,
 "🧪 Train & test live": page_train}[page]()
