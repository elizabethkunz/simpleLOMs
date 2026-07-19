#!/usr/bin/env python3
"""Build, execute, and install tutorials/01_fit_lc_models.ipynb.

Self-contained (matches builder_02.py): a bootstrap cell puts THIS repo's
``src`` on ``sys.path`` so ``import simpleLOMs`` resolves here and not the dead
editable install (see TUTORIAL_HANDOFF.md §6). No external PYTHONPATH needed:

    python3 tutorials/builder_01.py

Design decisions vs the old notebook (see TUTORIAL_HANDOFF.md for why):
  * single reference impedance Z0 = 50 ohm everywhere (old notebook silently
    mixed 10 / 50 / 15);
  * per-model accuracy read off with the SAME circle fit used everywhere else
    in the package (old notebook used find_resonant_frequency + a coarse-grid
    FWHM, which produced misleading 75-80% kappa "errors");
  * required load args are passed to analyze_system (old notebook had them
    commented out, so the cell could not execute);
  * dead imports, duplicate "4b" headings, the missing "section 2" and the
    stale "0.0000 GHz" S12 readout removed; typos fixed;
  * a fit_lom() one-liner added to tie tutorial 1 to tutorials 2-6.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
TUT = REPO / "tutorials"
DOCS = REPO / "docs" / "tutorials"
BASENAME = "01_fit_lc_models.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 1: Fitting LC Models to a CPW

`simpleLOMs` replaces a distributed coplanar-waveguide (CPW) resonator with a
single lumped `L`–`C` oscillator that you can drop into a circuit Hamiltonian.
This tutorial fits the three models the package ships with to one CPW device and
**ranks them against the CPW ground truth**, so you can see why the *Optimized*
model is the default.

The device here is in the strongly, asymmetrically coupled
regime ($C_{c1}=30$ fF, $C_{c2}=70$ fF) which allows us to compare the accuracy of the three lumped
       models in a realistic scenario.

"""),

    code("""
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

from pathlib import Path

from simpleLOMs import (
    CPWParams, FosterFit, OptimizedFit, AnalyticalFit,
    analyze_system, fit_lom, utils,
    plot_re_im, plot_all_models, plot_residuals,
    plot_summary, plot_transmission, plot_scan,
)
from simpleLOMs.networks.cpw import cpw_resonator_network_2port
from simpleLOMs.analysis import circle_fit_f0_kappa
from simpleLOMs.models.optimized_fit import OptimizationConfig
from simpleLOMs.schematics import lc_schematic_2port

print("All imports OK")
"""),

    md(r"""
---
## 1. Load your CPW

The CPW geometry lives in a `CPWParams` dataclass, which is defined once and passed
into every function that needs it.
"""),

    code("""
cpw_params = CPWParams(
    w=11.7e-6,              # center-conductor width (m)
    s=5.1e-6,               # gap spacing (m)
    t=0.0,                  # metal thickness — 0 = ideal thin film
    h=500e-6,               # substrate height (m)
    rho=1e-19,              # resistivity ~ 0 -> superconducting limit
    ep_r=11.45,             # relative permittivity (ultracold silicon)
    has_metal_backside=True,
    tand=0.0,               # lossless substrate
)

print(cpw_params)
"""),

    md(r"""
Next we set the circuit-level parameters: the resonator length `d`, the
reference (feedline) impedance `Z0`, and the coupling / ground capacitances. In
practice we recommend extracting `Cc1, Cc2, Ctog1, Ctog2` from the Maxwell
capacitance matrices of your EM solver of choice (e.g. $\texttt{ANSYS HFSS}$,
$\texttt{COMSOL}$).

A single reference impedance $Z_0 = 50\,\Omega$ is used throughout the notebook.
"""),

    code("""
d     = 7.0e-3    # resonator length (m) 
Z0    = 50.0      # reference / feedline impedance (ohm) — used everywhere

Cc1   = 3.0e-14   # coupling capacitor, port 1 (F)   -> 30 fF
Cc2   = 7.0e-14   # coupling capacitor, port 2 (F)   -> 70 fF  (asymmetric!)
Ctog1 = 4.0e-14   # shunt-to-ground cap, port 1 (F)
Ctog2 = 6.0e-14   # shunt-to-ground cap, port 2 (F)
"""),

    md(r"""
We may also define load resonators. These are not needed for the initial fit,
but they let us check the hybridized frequency shift the model predicts once
these loads are attached (Section 3). To approximate the case of a qubit-bus resonator, for example,
we pick two loads near a typical
qubit frequency ($3$–$5$ GHz).
"""),

    code("""
Lload1 = 2e-09    # load 1 inductance (H)
Cload1 = 6.0e-13  # load 1 capacitance (F)
Lload2 = 4e-09    # load 2 inductance (H)
Cload2 = 6.0e-13  # load 2 capacitance (F)

f_load1 = 1.0 / (2 * np.pi * np.sqrt(Lload1 * Cload1)) / 1e9
f_load2 = 1.0 / (2 * np.pi * np.sqrt(Lload2 * Cload2)) / 1e9
print(f"Load 1 bare resonance: {f_load1:.4f} GHz")
print(f"Load 2 bare resonance: {f_load2:.4f} GHz")
"""),

    md(r"""
Finally we choose a coarse frequency grid spanning the resonance. The individual fits
below refine their own windows around $f_0$.
"""),

    code("""
freq = rf.Frequency(6e9, 9e9, 400_001, unit="Hz")
"""),

    md(r"""
### 1a. Inspect the CPW network

Build the 2-port CPW network and read off its resonance with the package's
circle fit.
"""),

    code("""
cpw_net = cpw_resonator_network_2port(
    freq, d, Cc1, Cc2, Ctog1, Ctog2,
    cpw_params=cpw_params, Z0=Z0,
)

f0_s11, k_s11 = circle_fit_f0_kappa(cpw_net, 0, 0)
f0_s22, k_s22 = circle_fit_f0_kappa(cpw_net, 1, 1)
print(f"CPW  S11:  f0 = {f0_s11/1e9:.4f} GHz    kappa/2pi = {k_s11/1e6:.3f} MHz")
print(f"CPW  S22:  f0 = {f0_s22/1e9:.4f} GHz    kappa/2pi = {k_s22/1e6:.3f} MHz")
"""),

    code("""
cpw_net.plot_s_db()
plt.show()
"""),

    md("We can also look at $\\mathrm{Re}(S)$ and $\\mathrm{Im}(S)$ to get a feel for the lineshape."),

    code("""
plot_re_im(cpw_net, m=1, n=1, title=r"$S_{22}$")
plt.show()
"""),

    md(r"""
---
## 2. Fitting each model individually

`simpleLOMs` contains three lumped models:

1. **`AnalyticalFit`** — closed form (for a $\lambda/2$ resonator)
2. **`FosterFit`** — Foster synthesis from the line's admittance slope
   (Used in `PhysRevLett.108.240502`).
3. **`OptimizedFit`** — our numerical fit of $L_\mathrm{eff}, C_\mathrm{eff}$ to
   the CPW $S$-parameters, with the coupling capacitors included.

Each model follows the same three-step pattern:

```python
model = ModelClass(...)                       # 1. instantiate
model.fit(freq, ...)                          # 2. run the fit
net = model.get_network(refined_freq, ...)    # 3. build the fitted network
```

To rank them fairly we build one CPW reference and score every model
against it using `circle_fit_f0_kappa`.
"""),

    code("""
# One refined CPW reference; every model is scored against it the same way.
refined_freq = rf.Frequency(f0_s11 - 0.5e9, f0_s11 + 0.5e9, 20_001, unit="Hz")
cpw_ref = cpw_resonator_network_2port(
    refined_freq, d, Cc1, Cc2, Ctog1, Ctog2,
    cpw_params=cpw_params, Z0=Z0,
)
f0_cpw, k_cpw = circle_fit_f0_kappa(cpw_ref, 0, 0)
print(f"CPW ground truth:  f0 = {f0_cpw/1e9:.4f} GHz    kappa/2pi = {k_cpw/1e6:.3f} MHz")


def score(name, net):
    "Circle-fit a fitted LOM network and print its signed error vs the CPW."
    f0, k = circle_fit_f0_kappa(net, 0, 0)
    print(f"{name:11s} f0 = {f0/1e9:.4f} GHz  (err {(f0-f0_cpw)/f0_cpw*100:+.3f}%)"
          f"    kappa/2pi = {k/1e6:.3f} MHz  (err {(k-k_cpw)/k_cpw*100:+.2f}%)")
    return f0, k
"""),

    md(r"""
### 2a. AnalyticalFit

`AnalyticalFit` uses two closed-form expressions,

$$C_\mathrm{eff} = \frac{\pi}{2\,Z_0^{\mathrm{cpw}}\,\omega_{\lambda/2}}, \qquad
  L_\mathrm{eff} = \frac{1}{\omega_{\lambda/2}^2\, C_\mathrm{eff}},$$

so it needs an **unloaded** resonance frequency and the CPW **characteristic
impedance** $Z_0^{\mathrm{cpw}}$. Here
$Z_0^{\mathrm{cpw}} = 45.926\,\Omega$ and the unloaded $f_r = 8.58$ GHz; both are
readable from scikit-rf or TXLine.
"""),

    code("""
Z0_cpw = 45.926      # CPW characteristic impedance (geometry) — NOT the 50 ohm ref
f0_unloaded = 8.58e9 # unloaded lambda/2 resonance

analytical = AnalyticalFit(mode=2)   # mode=2 -> lambda/2 (default)
analytical.fit(refined_freq, f_r=f0_unloaded, Z0=Z0_cpw)
print(analytical)

analytical_net = analytical.get_network(
    refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0,
    reference=cpw_ref, show=True, m=0, n=0,
    lom_label="AnalyticalFit", data_label="CPW",
)
plt.show()
score("Analytical", analytical_net)
"""),

    md(r"""
### 2b. FosterFit

`FosterFit` extracts `L` and `C` from the admittance slope
$\mathrm{d}B/\mathrm{d}\omega$ at the CPW resonance. It needs only the resonator
length `d` and the CPW geometry — no measured $S$-parameters.
"""),

    code("""
foster = FosterFit(cpw_params=cpw_params)
foster.fit(freq, d=d)
print(foster)

foster_net = foster.get_network(
    refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0,
    reference=cpw_ref, show=True, m=0, n=0,
    lom_label="FosterFit", data_label="CPW",
)
plt.show()
score("Foster", foster_net)
"""),

    md(r"""
### 2c. OptimizedFit

`OptimizedFit` runs two stages of numerical optimisation against the CPW $S_{11}$
and $S_{22}$, with the coupling capacitors held fixed. It takes the CPW network
as input.

The default settings nearly always outperform `FosterFit` and `AnalyticalFit`.
When you do want to tune the optimiser, you can change settings in`OptimizationConfig`. THe settings are:

1. **`w0_window_frac`** — how tightly the Stage-1 scan brackets $\omega_0$. Use
   $\sim 0.0001$ for *localized* behaviour at resonance; relax to $\sim
   0.005$–$0.05$ for *broadband* behaviour across a wide span.
2. **`n_w0`** — number of Stage-1 scan points for the initial $\omega_0$ guess.
   $\sim 10$–$20$ is plenty (it is refined later).
3. **`n_dense`** — number of Stage-1 optimisation points; $\lesssim 100$ keeps
   runtime down.
4. **`n_widths`** — the Stage-2 window in units of linewidth, e.g.
   `n_widths=3` fits over $\pm 3\kappa$ about $\omega_r$.

For localized behaviour set `w0_window_frac` $\sim 0.0001$ and `n_widths`
$\sim 1$. For broadband behaviour, it is best to relax these both.
"""),

    code("""
config = OptimizationConfig(
    w0_window_frac=0.0001,
    n_w0=10,
    n_dense=10,
    n_widths=3.0,
    verbose=False,       # set True to watch the least-squares iterations
)

optimized = OptimizedFit(config=config)
optimized.fit(
    refined_freq, data_ntw=cpw_ref,
    Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
    d=d, cpw_params=cpw_params, Z0=Z0,
)
params = optimized.get_params()
print(optimized)
print(f"L = {params['L']:.4e} H    C = {params['C']:.4e} F")

optimized_net = optimized.get_network(
    refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0,
    reference=cpw_ref, show=True, m=0, n=0,
    lom_label="OptimizedFit", data_label="CPW",
)
plt.show()
score("Optimized", optimized_net)
"""),

    md(r"""
The optimiser minimises the  complex residuals (real and
imaginary parts) of $S_{11}$ and $S_{22}$ over a window of $\pm n_\mathrm{widths}\,\kappa$
around each resonance (here we choose `n_widths=3`). 

To confirm that the optimized fit worked,  `plot_scan` shows the Stage-1
$\omega_0$ scan found a clean minimum, and `plot_residuals` shows the leftover
$S$-parameter mismatch across the band.
"""),

    code("""
plot_scan(optimized)
plt.show()
"""),

    code("""
plot_residuals(
    data_ntw=cpw_ref,
    lom_networks={"optimized": optimized_net},
    m=0, n=0,   # S11
    save_path="figures/residuals_s11.pdf",
)
plt.show()
"""),

    md(r"""
### 2d. Side-by-side overlay

`plot_all_models` puts all four networks on one axes so you can visually rank the
methods.
"""),

    code("""
plot_all_models(
    {
        "CPW (reference)": cpw_ref,
        "Analytical LC":   analytical_net,
        "Foster LC":       foster_net,
        "Optimized LC":    optimized_net,
    },
    m=0, n=0, quantity="re",
    title=r"$\mathrm{Re}(S_{11})$",
)
plt.show()
"""),

    md(r"""
In this regime, the **Optimized** method is the clear winner. Its resonance sits within a few thousandths
of a percent of the CPW ground truth and its linewidth within about one percent,
while Foster and Analytical drift by roughly half a percent to a percent in
frequency and several percent in linewidth. This is exactly the strong,
asymmetric-coupling regime ($C_{c1}=30$ fF, $C_{c2}=70$ fF) where fitting with
the coupling capacitors included is important.
"""),

    md(r"""
---
## 3. Full system comparison in one call

`analyze_system()` runs all three fits for you, extracts $f_0$ and $\kappa$ from
$S_{11}$ and $S_{22}$ with circle fits, and,  because we pass the loads defined in the beginning, also
builds a loaded network and reports the hybridized frequency shifts against
the CPW. It returns a `results` dict holding every fitted model, network, and
error metric.
"""),

    code("""
freq_sys = rf.Frequency(5e9, 10e9, 300_001, unit="Hz")

results = analyze_system(
    freq=freq_sys,
    d=d,
    Cc1=Cc1, Cc2=Cc2,
    Ctog1=Ctog1, Ctog2=Ctog2,
    Lload1=Lload1, Cload1=Cload1,
    Lload2=Lload2, Cload2=Cload2,
    cpw_params=cpw_params,
    Z0=Z0,                 # same 50 ohm reference as above
    analytical_Z0=Z0_cpw,  # CPW characteristic impedance for the analytical formula
    verbose=False,
)

print("analyze_system() complete.")
print("Result keys:", [k for k in results
                       if not isinstance(results[k],
                                         (rf.Network, FosterFit, OptimizedFit, AnalyticalFit))][:12], "...")
"""),

    md("### 3a. Results as tables"),

    code("""
utils.make_results_table(results)
"""),

    code("""
utils.make_params_table(results)
"""),

    md(r"""
Here we can see the `f0 error` and `kappa error` columns for all three models. 
       Optimized is the best performing model, while Foster comes second and
Analytical last. 
"""),

    md(r"""
### 3b. Plotting the results

The `results` dict stores the fitted networks, so you can plot various metrics of interest without rerunning
anything.
"""),

    code("""
plot_summary(
    results, port="s11",
    title=r"Fit summary",
    save_path="figures/fit_summary.pdf",
)
plt.show()
"""),

    code("""
plot_transmission(
    networks={
        "cpw":       results["cpw_network"],
        "optimized": results["optimized_network"],
    },
    m=1, n=1,
)
plt.show()
"""),

    code("""
plot_all_models(
    {
        "CPW":        results["cpw_network"],
        "Foster":     results["foster_network"],
        "Optimized":  results["optimized_network"],
        "Analytical": results["analytical_network"],
    },
    m=0, n=0, quantity="re",
    title=r"$\mathrm{Re}(S_{11})$",
)
plt.show()
"""),

    md(r"""
### 3c. Getting the model parameters

Call `.get_params()` on any fitted model to pull out its effective
$L_\mathrm{eff}, C_\mathrm{eff}$.
"""),

    code("""
print("Optimized :", results["optimized_model"].get_params())
print("Foster    :", results["foster_model"].get_params())
print("Analytical:", results["analytical_model"].get_params())
"""),

    md(r"""
### 3d. Using `fit_lom` to get `L, C` directly

 When all you need is the
effective `L, C` for a chosen model, you can use the `fit_lom` function. This is the function 
used throughout Tutorials 2–6.
"""),

    code("""
L, C = fit_lom(
    d, model="optimized",
    Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
    cpw_params=cpw_params, freq=refined_freq,
)
f_r = 1.0 / (2 * np.pi * np.sqrt(L * C))
print(f"fit_lom(optimized):  L = {L*1e9:.4f} nH    C = {C*1e15:.2f} fF    f_r = {f_r/1e9:.4f} GHz")
"""),

    md(r"""
### 3e. Draw the fitted LC schematic

The same `L, C` drop straight into a schematic. Because `OptimizedFit` folds the
ground capacitances into its effective $(L, C)$, the lumped diagram has **no**
`Ctog` — only the coupling capacitors and the parallel LC tank
(`P1 — Cc1 — [LC] — Cc2 — P2`).
"""),

    code("""
figdir = Path("figures")
figdir.mkdir(exist_ok=True)

sch_lc = lc_schematic_2port(
    L, C, Cc1, Cc2, Z0=Z0,
    annotations={
        "L_nH": round(L * 1e9, 3),
        "C_fF": round(C * 1e15, 2),
        "f_r_GHz": round(f_r / 1e9, 4),
    },
    title="Fitted Optimized LOM",
)
sch_lc.save_svg(figdir / "fitted_lc.svg")
sch_lc.save_html(figdir / "fitted_lc.html")
print("saved figures/fitted_lc.svg|.html")
sch_lc
"""),

    md(r"""
When the coupling is strong and asymmetric,
`OptimizedFit` reproduces the distributed response where the closed-form and
Foster models cannot, so it is the default everywhere in `simpleLOMs`.
       
 ## Where to go next

- **Tutorial 2** extends this single-device ranking across the whole coupling
  plane.
- **Tutorial 3** shows how to scale up from one resonator to capacitively-coupled chains,
  reducing each resonator to its fitted LOM.
- **Tutorials 6–7** show how to use the `L, C` parameters to design a single resonator to meet design targets.

"""),
]


def build_and_execute():
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    nb = new_notebook(
        cells=CELLS,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )

    bootstrap = new_code_cell(
        "import sys, os\n"
        "os.environ.pop('MPLBACKEND', None)\n"
        f"sys.path.insert(0, r'{REPO / 'src'}')\n"
        "get_ipython().run_line_magic('matplotlib', 'inline')\n"
        "print('bootstrap ok', flush=True)\n"
    )
    nb.cells.insert(0, bootstrap)

    print("Executing notebook (includes two OptimizedFit fits — a few minutes)...", flush=True)
    client = NotebookClient(
        nb,
        timeout=1200,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(TUT)}},
    )
    try:
        client.execute()
        print("execute() finished", flush=True)
    except Exception as exc:
        print(f"execute() raised: {type(exc).__name__}: {exc}", flush=True)

    nb.cells.pop(0)

    out = TUT / BASENAME
    with out.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {out}", flush=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, DOCS / BASENAME)
    print(f"Copied to {DOCS / BASENAME}", flush=True)

    errors = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append((i, o.get("ename"), o.get("evalue")))
    if errors:
        print("ERRORS:", flush=True)
        for e in errors:
            print(" ", e, flush=True)
        sys.exit(1)
    print("OK — no cell errors.", flush=True)


if __name__ == "__main__":
    build_and_execute()
