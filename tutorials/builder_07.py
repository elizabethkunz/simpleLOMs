#!/usr/bin/env python3
"""Build, execute, and install tutorials/07_design_to_spec.ipynb.

Tutorial 7: "Designing a CPW resonator to match targets". Sequential design:
pick length d for a target frequency, then coupling Cc for a target Q, assemble
and verify against the CPW ground truth, and bundle it into a reusable
``design_resonator`` function. Uses ``fit_lom`` / ``extract_f0_kappa`` with the
Optimized model.

Self-contained (matches builder_05.py): a bootstrap cell puts THIS repo's ``src``
on ``sys.path`` so ``import simpleLOMs`` resolves here, not a stale editable
install. No external PYTHONPATH needed:

    python3 tutorials/builder_07.py
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
BASENAME = "07_design_to_spec.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 7: Designing a CPW resonator to match targets

In this tutorial you will learn how to design a circuit-level model distributed element in order to meet design targets.

For this tutorial, we will  use the design targets of readout frequency $f_r$ and quality factor $Q$.
In order to meet these targets we will vary the resonator length
$d$ and the coupling capacitor $C_c$, in the case of symmetric coupling on both sides.

This is shown for the purposes of demonstration, but the processes shown in this tutorial can be used to
perform sweeps for various circuit and geometric parameters to target certain specs. 
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
## Setup


First we define our CPW and choose what LOM model to use.
"""),

    code(r"""
cpw = sl.CPWParams(ep_r=11.45)
freq = rf.Frequency(4e9, 12e9, 8001, unit="Hz")
MODEL = "optimized"                               # or "foster" / "analytical"
"""),

    md(r"""
We will now define the target $f_0$ and $Q$ for our resonator. Here, our target $f_0 = 7 \text{GHz}$ and our target $Q=8000$.
"""),

    code(r"""
target_Q = 8000
target_f = 7.0
"""),

    md(r"""
In this tutorial we will match the design to these targets by first picking $d$ to match the target frequency, then picking $C_c$ for the
quality factor.
"""),

    md(r"""
## 1. Matching circuit parameters to design targets
"""),

    md(r"""
### 1a. Map length to frequency

First we sweep the resonator length and measure $f_0$ at a fixed reference coupling. The
relationship is smooth and monotonic, so we can interpolate it to solve for the
length that hits a target frequency.
"""),

    code(r"""
d_values = np.linspace(5.5e-3, 9.0e-3, 8)
f_of_d = np.array([sl.extract_f0_kappa(cpw, d=d, Cc=6e-15)[0] / 1e9 for d in d_values])

for d, f in zip(d_values, f_of_d):
    print("d = {:.2f} mm    f0 = {:.4f} GHz".format(d * 1e3, f))
"""),

    code(r"""
d_star = sl.invert_monotonic(d_values, f_of_d, target_f)
print("target f_r = {:.2f} GHz  ->  d = {:.3f} mm".format(target_f, d_star * 1e3))

sl.plot_curves(
    d_values * 1e3, f_of_d,
    xlabel=sl.LABEL_LENGTH,
    ylabel=sl.LABEL_F0,
    fmt="o-",
    color=sl.COLOR_PRIMARY,
    hline=target_f,
    vline=d_star * 1e3,
    point=(d_star * 1e3, target_f),
)
"""),

    md(r"""
### 1b. At that length, map coupling to quality factor

Now fix $d = d^\star$ and sweep the coupling capacitor. We need the lumped `L`, `C`
at this length (for the $Q$ bookkeeping), which one `fit_lom` call provides
(default model: optimized).

> **Note on loss:** These chips are lossless, so the achieved $Q$ is the coupling $Q_c$.
>  With a finite internal $Q_i$, design $Q_c$ to a target loaded
>  $Q_\ell^{-1} = Q_c^{-1} + Q_i^{-1}$.
"""),

    code(r"""
L_star, C_star = sl.fit_lom(d_star, model=MODEL, cpw_params=cpw, freq=freq)

Cc_values = np.linspace(2e-15, 14e-15, 8)
Q_of_Cc = []
for Cc in Cc_values:
    _, kappa = sl.extract_f0_kappa(cpw, d=d_star, Cc=Cc)
    Q_of_Cc.append(sl.resonator_readout_params(L_star, C_star, kappa_Hz=kappa).Q)
Q_of_Cc = np.array(Q_of_Cc)

for Cc, Q in zip(Cc_values, Q_of_Cc):
    print("Cc = {:4.1f} fF    Q = {:8.0f}".format(Cc * 1e15, Q))
"""),

    md(r"""
We now invert it to get the $Cc$ value for our target $Q$.
"""),

    code(r"""
Cc_star = sl.invert_monotonic(Cc_values, Q_of_Cc, target_Q, log=True)
print("target Q = {}  ->  Cc = {:.2f} fF".format(target_Q, Cc_star * 1e15))

sl.plot_curves(
    Cc_values * 1e15, Q_of_Cc,
    xlabel=sl.LABEL_CC,
    ylabel=sl.axis_label("Loaded quality factor", "Q"),
    fmt="o-",
    color=sl.COLOR_PRIMARY,
    xscale="log", yscale="log",
    hline=target_Q,
    vline=Cc_star * 1e15,
    point=(Cc_star * 1e15, target_Q),
)
"""),

    md(r"""
## 2. Assemble the design and verify it

We now have a candidate $(d^\star, C_c^\star)$. Build it, measure $f_0$ and
$\kappa$, and check both targets at once with a full readout report.
"""),

    code(r"""
f0, kappa = sl.extract_f0_kappa(cpw, d=d_star, Cc=Cc_star)
rp = sl.resonator_readout_params(L_star, C_star, kappa_Hz=kappa)

print(rp.summary())
print()
print("target  f_r = {:.2f} GHz   achieved f0 = {:.3f} GHz   ({:+.1f} %)".format(
    target_f, f0 / 1e9, 100 * (f0 / 1e9 - target_f) / target_f))
print("target  Q   = {:<6d}      achieved Q  = {:.0f}       ({:+.1f} %)".format(
    target_Q, rp.Q, 100 * (rp.Q - target_Q) / target_Q))
"""),

    md(r"""
## 3. Bundle it into a reusable design function

The two inversions compose into `sl.design_resonator`, which runs a short length
sweep and a short coupling sweep, interpolates each, and returns the geometry
plus the achieved parameters.
"""),

    code(r"""
for tf, tQ in [(9.0, 2000), (7.5, 10000)]:
    d, Cc, f0_GHz, rp = sl.design_resonator(
        tf, tQ, cpw, freq=freq, model=MODEL,
    )
    print("spec: f_r={:.1f} GHz, Q={:<6d}  ->  d={:.2f} mm, Cc={:.2f} fF"
          "   (achieved f0={:.2f} GHz, Q={:.0f})".format(
              tf, tQ, d * 1e3, Cc * 1e15, f0_GHz, rp.Q))
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
