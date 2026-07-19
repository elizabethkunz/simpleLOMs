#!/usr/bin/env python3
"""Build, execute, and install tutorials/06_readout_parameters.ipynb.

Tutorial 6: From a Lumped Oscillator Model to readout parameters. Fit a
resonator, convert L/C into readout quantities, measure loaded kappa/Q, and
size Cc for a target Q via a coupling sweep and invert_monotonic.

Self-contained: a bootstrap cell puts THIS repo's ``src`` on ``sys.path``.

    python3 tutorials/builder_06.py
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
BASENAME = "06_readout_parameters.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 6: From a Lumped Oscillator Model to readout parameters

Tutorials 1 and 2 fit a lumped `L`–`C` model to a CPW resonator and check how
accurately it reproduces the full distributed response. 

In this tutorial, you will learn how to turning that `L` and `C` into quantities 
you might actually use for designing a superconducting circuit. We will use the two helpers, `resonator_readout_params` and
`extract_f0_kappa`, to do the following:

1. Convert a fitted model into a resonance frequency $f_r$, characteristic
  impedance $Z_r$, and zero-point voltage and current fluctuations
2. Measure the loaded linewidth $\kappa$ and get the quality factor $Q$ 

3. Design a symmetric $C_c$ to match a target $Q$
"""),

    code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

import simpleLOMs as sl

sl.apply_style()
"""),

    md(r"""
## 1. Fit a resonator and read off its `L` and `C`

We use a 7 mm CPW resonator on an ultracold-silicon chip. The  `fit_lom(d, model=...)`  returns effective `L` and `C` for any of
the three models (`"optimized"`, `"foster"`, `"analytical"`). In this tutorial, we use Optimized, which is the
default and the most accurate. However, you can  switch `MODEL` below if you want a faster
Foster or Analytical fit.
"""),

    code(r"""
cpw = sl.CPWParams(ep_r=11.45)                       # ultracold-silicon substrate
freq = rf.Frequency(4e9, 12e9, 8001, unit="Hz")

MODEL = "optimized"                               # or "foster" / "analytical"
L, C = sl.fit_lom(7e-3, model=MODEL, cpw_params=cpw, freq=freq)

print("Fitted lumped model ({}):".format(MODEL))
print("  L = {:.4f} nH".format(L * 1e9))
print("  C = {:.2f} fF".format(C * 1e15))
"""),

    md(r"""
## 2. Convert fitted `L`, `C` to relevant circuit parameters.

One of the benefits of lumped oscillator models is that they can be easily used to convert to Hamiltonian parameters using standard circuit quantization techniques.
"""),

    md(r"""
### 2a. Unloaded quantities

To convert `L, C` to frequency, impedance, and zero-point fluctuations, use `resonator_readout_params(L, C)` to get these values based on the fitted `L, C` quantities.  

$$
\omega_r = \frac{1}{\sqrt{LC}}, \qquad
Z_r = \sqrt{\frac{L}{C}}, \qquad
V_\mathrm{zpf} = \sqrt{\frac{\hbar\omega_r}{2C}}, \qquad
I_\mathrm{zpf} = \sqrt{\frac{\hbar\omega_r}{2L}}.
$$

$Z_r$ and the zero-point fluctuations depend **only** on `L` and `C`, so we can
compute them before saying anything about loss or coupling.
"""),

    code(r"""
rp = sl.resonator_readout_params(L, C)
print(rp.summary())
"""),

    md(r"""

$V_\mathrm{zpf}$ is on the order of a few $\mu$V and $I_\mathrm{zpf}$ is on the order of
tens of nA, as expected for a standard coplanar readout resonator.
"""),

    md(r"""
### 2b. Loaded quantities ($\kappa$, $Q$)

The quality factor $Q$ is not a property of the bare mode. Instead it is set
by how strongly we couple the mode to the environment. 

In this geometry, the coupling capacitor $C_c$ ties the resonator to the 50 $\Omega$ feedline and
gives the mode a finite linewidth. We measure it using a circle fit via `extract_f0_kappa`, which
locates the resonance on a coarse grid, then re-simulates a narrow window on a
fine grid and reads $f_0$ and $\kappa$ from a circle fit (a single wide grid
cannot resolve a sub-MHz linewidth).
"""),

    md(r"""
Beginning with a 6 fF coupling capacitor, we'll extract $f_0$ and $\kappa$ fo the CPW system.
"""),

    code(r"""
Cc = 6e-15                                       
f0, kappa = sl.extract_f0_kappa(cpw, d=7e-3, Cc=Cc)
print("measured:  f0 = {:.4f} GHz    kappa/2pi = {:.4f} MHz".format(f0/1e9, kappa/1e6))
"""),

    md(r"""
### 2c. Summarize the results
"""),

    code(r"""
rp = sl.resonator_readout_params(L, C, kappa_Hz=kappa)
print()
print(rp.summary())
"""),

    md(r"""
Because this chip is lossless (`tand = 0`, superconducting metal), the internal
quality factor is effectively infinite and the loaded $Q$ is set entirely by the
coupling — i.e. here $Q \approx Q_c$, the **coupling quality factor**. On a real
device the loaded $Q$ combines with the internal $Q_i$ as
$Q_\ell^{-1} = Q_c^{-1} + Q_i^{-1}$.

The photon lifetime $\tau = 1/(2\pi\kappa) \approx 65$ ns is the readout
resonator's ring-down time.
"""),

    code(r"""

"""),

    md(r"""
## 3. Setting a design to match a target parameter. 

Here we will use the example of showing how the coupling capacitor $C_c$ sets $\kappa$ and $Q$ to show how you can use sweeps like this to correlate geometric features of your circuit design to actual desired parameters.
"""),

    md(r"""
### 3a. Set up the sweep
"""),

    md(r"""
We'll use 8 Cc values with all else fixed, and use `extract_f0_kappa` to get $\kappa$ an subsequently $Q$ 
"""),

    code(r"""
Cc_values = np.linspace(2e-15, 16e-15, 8)
f0s, kappas, Qs = [], [], []
for Cc in Cc_values:
    f0, kappa = sl.extract_f0_kappa(cpw, d=7e-3, Cc=Cc)
    rp = sl.resonator_readout_params(L, C, kappa_Hz=kappa)
    f0s.append(f0 / 1e9)
    kappas.append(kappa / 1e6)
    Qs.append(rp.Q)

f0s, kappas, Qs = map(np.array, (f0s, kappas, Qs))
for cc, k, q in zip(Cc_values, kappas, Qs):
    print("Cc = {:4.1f} fF    kappa/2pi = {:7.3f} MHz    Q = {:8.0f}".format(cc*1e15, k, q))
"""),

    md(r"""
### 3b. (Optional) Validate the data

We already know approximately what to expect from this data based on approximate formulas. For 
a capacitively coupled resonator we expect that expectation is $\kappa \propto C_c^2$ and
therefore $Q \propto 1/C_c^2$. To ensure that this pattern occured, we can plot the two of them in log scale:
"""),

    code(r"""
sl.plot_curve_pair(
    Cc_values * 1e15, kappas, Qs,
    xlabel=sl.LABEL_CC,
    ylabel0=sl.LABEL_KAPPA,
    ylabel1=sl.axis_label("Loaded quality factor", "Q"),
    xscale="log", yscale="log",
)
"""),

    md(r"""
Indeed the graph matches our expectations. To quantify it, we can fit a line in log-log space.
"""),

    code(r"""
slope_k = np.polyfit(np.log(Cc_values), np.log(kappas), 1)[0]
slope_Q = np.polyfit(np.log(Cc_values), np.log(Qs), 1)[0]
print("kappa  ~  Cc^{:+.2f}   (expected +2)".format(slope_k))
print("Q      ~  Cc^{:+.2f}   (expected -2)".format(slope_Q))
"""),

    md(r"""
The fitted exponents sit close to $\pm 2$, confirming that $\kappa$ and $Q$ scale with $C_c$ as expected. 

This fitted curve is now the curve we can design against. WE can pick the $C_c$ that lands the
readout resonator at the $\kappa$ (equivalently $Q$) we want. 
"""),

    md(r"""
## 3c. Reading off a coupling for a target $Q$

Because $Q$ is monotonic in $C_c$, you can interpolate the sweep to size the
coupling capacitor for a chosen quality factor. Here, let's set the target $Q$ to be $5000$.
"""),

    code(r"""
target_Q = 5000
Cc_target = sl.invert_monotonic(Cc_values, Qs, target_Q, log=True)
print("For a loaded Q of {}, choose  Cc ~ {:.2f} fF".format(target_Q, Cc_target * 1e15))
"""),

    md(r"""
Now as described in Tutorial 4, we can invert the $Cc$ graph to match the target.
"""),

    code(r"""
Cc_target = sl.invert_monotonic(Cc_values, Qs, target_Q, log=True)
print("For a loaded Q of {}, choose  Cc ~ {:.2f} fF".format(target_Q, Cc_target * 1e15))
"""),

    md(r"""
We now have a way to convert lumped parameters into design targets.

## Next up

In Tutorial 7 we will learn how to use these principles taught in this tutorial to hit a target frequency and target $Q$ simultaneously by
choosing the resonator length and coupling together.
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

    print(f"Executing notebook ({BASENAME})...", flush=True)
    client = NotebookClient(
        nb,
        timeout=600,
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
