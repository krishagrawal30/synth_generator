# 🏥 SynthHealth – Streamlit app

Interactive front-end for the CAT7001 lab project **"Privacy-Aware Synthetic Clinical Data Generation using Autoencoder, VAE and GAN"**
(UCI *Diabetes 130-US Hospitals 1999–2008*).

| Page | What it does |
|---|---|
| 🏠 Overview | problem, pipeline diagram, headline numbers |
| 🧬 Generate data | sample synthetic patients from the **AE+GMM**, **VAE** or **WGAN-GP** generator, optional *what-if cohort* filters (rejection sampling), fidelity check, CSV download |
| 📊 Compare models | interactive scorecard (re-weight fidelity / utility / privacy), TSTR utility, privacy audit with calibration controls, relationship + subgroup preservation, β-VAE frontier, notebook figures |
| 🧪 Train & test live | retrain small versions of all three models on a dataset you provide, then audit them on the spot |
| 📘 About & defence | dataset facts, design decisions, limitations, viva Q&A |

**The deployed app contains no real patient rows.** It ships only model weights and *aggregate* reference statistics (histograms,
category shares, association matrix); fidelity is scored against those.

## 1 · Files

```
app.py            Streamlit UI
synth_core.py     cohort prep · encoder · AE / VAE / WGAN-GP · generation · evaluation (mirrors the notebook)
artifacts/        exported by the notebook (models + reference stats + result tables + figures)  <-- you add this
requirements.txt  CPU-only PyTorch + Streamlit + Plotly + scikit-learn
Dockerfile        container image (Render, Fly.io, Cloud Run, Hugging Face Docker Space …)
smoke_test.py     headless check that every page renders
.streamlit/       theme + server settings
```

## 2 · Get the trained models (`artifacts/`)

1. Run `Synthetic_Clinical_Data_AE_VAE_GAN.ipynb` (Colab GPU recommended) with `FAST_DEV = False`.
2. Its section **25.1** writes `outputs/app_artifacts/` and `outputs/app_artifacts.zip` (about 4–5 MB).
3. Unzip into this folder so you have `artifacts/encoder.json`, `artifacts/ae.pt`, `artifacts/figures/…` and so on.

Without `artifacts/` the app still starts; **Generate** and **Compare** show setup instructions, and **Train & test live** works if you give it the dataset.

## 3 · Run locally

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
python smoke_test.py                                      # optional: headless check of every page
```

## 4 · Deploy

### Option A – Streamlit Community Cloud (free, easiest)
1. Push this folder to a GitHub repo **including `artifacts/`** (a few MB; do **not** commit `diabetic_data.csv`).
2. On share.streamlit.io choose *New app* → pick the repo, branch and `app.py` as the main file → pick Python 3.11 or 3.12 → Deploy.
3. `requirements.txt` starts with `--extra-index-url https://download.pytorch.org/whl/cpu` so the small CPU build of PyTorch is installed.
   If the build runs out of space/time, use Option B or C instead.

### Option B – Docker (Render, Railway, Fly.io, Google Cloud Run, a VM …)
```bash
docker build -t synthhealth .
docker run -p 8501:8501 synthhealth        # http://localhost:8501
```
Point the platform at the Dockerfile; the container listens on port **8501**.

### Option C – Hugging Face Space (Docker)
Create a *Docker* Space, upload these files, and add this header to the Space's `README.md`:
```yaml
---
title: SynthHealth
emoji: 🏥
sdk: docker
app_port: 8501
---
```

## 5 · Configuration

| Environment variable | Default | Meaning |
|---|---|---|
| `SYNTH_ARTIFACTS` | `./artifacts` | folder holding the exported models and tables |
| `SYNTH_LOCAL_CSV` | `diabetic_data.csv` | if this file exists, *Train & test live* offers it as a data source |

## 6 · Troubleshooting

| Symptom | Fix |
|---|---|
| "Trained models were not found" | export `app_artifacts` from the notebook and unzip into `artifacts/` |
| Sidebar says artifacts come from a FAST_DEV run | re-run the notebook with `FAST_DEV = False` and re-export |
| *Train & test live*: "Download failed" | upload `diabetic_data.csv` instead (UCI dataset 296) |
| App is slow or runs out of memory on a small host | lower *Patients used* and *WGAN-GP steps* on the live page; the Generate page is light |
| `torch` install is huge | make sure the `--extra-index-url …/whl/cpu` line in `requirements.txt` is honoured, or deploy with Docker |

## 7 · Honest notes
* Synthetic ≠ guaranteed private. The privacy page reports empirical audits (exact matches, distance-to-closest-record,
  quasi-identifier hits, membership inference); there is **no differential-privacy guarantee**.
* Not for clinical use. The dataset is public, de-identified and from 1999–2008.
