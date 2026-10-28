"""
Smoke test: drives every page of the app headlessly and reports any exception.
    python smoke_test.py
Needs the exported `artifacts/` folder. The live-training check additionally needs diabetic_data.csv in this folder.
"""
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ok = True


def check(at, label):
    global ok
    errs = list(at.exception)
    print(f"[{'FAIL' if errs else ' ok '}] {label}")
    for e in errs:
        print("       ", str(e.value)[:300])
    ok &= not errs


at = AppTest.from_file("app.py", default_timeout=900).run()
check(at, "Overview")
go = lambda p: at.sidebar.radio(key="page").set_value(p).run()

if not (Path("artifacts") / "ae.pt").exists():
    print("No artifacts/ found - only the artifact-free pages can be checked. Export them from the notebook first.")
else:
    go("🧬 Generate data"); check(at, "Generate – page")
    gen = lambda: [b for b in at.button if b.label == "Generate"][0]
    for model in ["AE+GMM", "VAE", "WGAN-GP"]:
        at.selectbox[0].set_value(model); at.slider[0].set_value(1000)
        gen().click().run(); check(at, f"Generate – {model} ({len(at.session_state['gen']['df'])} rows)")
    ms = {m.key: m for m in at.multiselect}
    ms["flt_early_readmit"].set_value(["1"]); gen().click().run(); check(at, "Generate – what-if cohort (early readmission = 1)")
    go("📊 Compare models"); check(at, "Compare models (all tabs)")

go("📘 About & defence"); check(at, "About & defence")
go("🧪 Train & test live"); check(at, "Train & test live – page")
if Path("diabetic_data.csv").exists():
    sl = {s.label: s for s in at.slider}
    sl["Patients used (subsample)"].set_value(2000); sl["Autoencoder epochs"].set_value(5)
    sl["VAE epochs"].set_value(5); sl["WGAN-GP steps"].set_value(50)
    t0 = time.time()
    [b for b in at.button if "Train and audit" in b.label][0].click().run(timeout=1200)
    check(at, f"Train & test live – run ({time.time() - t0:.0f}s)")
else:
    print("(skipped live-training run: put diabetic_data.csv next to app.py to include it)")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
