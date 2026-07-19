#!/usr/bin/env python3
"""Build, execute, and install tutorials/04_cpw_parameter_sweeps.ipynb.

Tutorial 4: "Parameter sweeps of CPW and LOM designs" — the package's sole
sweep tutorial. ``measure_sweep`` is applied to three design questions:
resonator length → frequency, coupling → linewidth/Q, and CPW gap geometry →
frequency, using ``extract_f0_kappa`` and ``fit_lom`` (Optimized, the default LOM).

Self-contained (matches builder_05.py): a bootstrap cell puts THIS repo's ``src``
on ``sys.path`` so ``import simpleLOMs`` resolves here, not a stale editable
install. No external PYTHONPATH needed:

    python3 tutorials/builder_04.py
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
BASENAME = "04_cpw_parameter_sweeps.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 4: Parameter sweeps of CPW and LOM designs


While the primary purpose of the **simpleLOMs** pacakge is to construct lumped models of superconducting devices, another aspect of this package is its ability to test a wide variety of device geometries at the circuit level, including the length, coupling, or the CPW cross-section. 

In this tutorial, we will see how to perform parameter sweeps of both `CPW` and lumped designs using `extract_f0_kappa` , `fit_lom`, and `resonator_readout_params`. By the end you will be able to sweep any device parameter of the network and extract a circuit that fits a desired design parameter.
"""),

    code(r"""
import warnings
warnings.filterwarnings("ignore")

import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

import simpleLOMs as sl


# Shared simpleLOMs figure style — one identical look across every tutorial.
sl.apply_style()

cpw = sl.CPWParams(ep_r=11.45)                        # ultracold-silicon substrate
freq = rf.Frequency(4e9, 12e9, 8001, unit="Hz")
MODEL = "optimized"                                # best / default LOM
"""),

    md(r"""
## Parameter sweeps with `measure_sweep`

`sl.measure_sweep(values, measure)` is a function to perform a parameter sweep for an
arbitrary knob in the design of a device, including geometry fields that are not `SweepConfig`
attributes. 
"""),

    md(r"""
## Sweep 1: Resonator length sweep for frequency

The resonator length has a disproportionately large impact on the frequency of the CPW and as such it acts as the coarse frequency knob. 
"""),

    code(r"""
d_values = np.linspace(5.5e-3, 9.0e-3, 6)          # 5.5 - 9.0 mm
r1 = sl.measure_sweep(d_values, lambda d: {"f0_GHz": sl.extract_f0_kappa(cpw, d=d, Cc=6e-15)[0] / 1e9})

for d, f in zip(r1["value"], r1["f0_GHz"]):
    print("d = {:.2f} mm    f0 = {:.4f} GHz".format(d * 1e3, f))
"""),

    code(r"""
sl.plot_curves(
    r1["value"] * 1e3, r1["f0_GHz"],
    xlabel=sl.LABEL_LENGTH,
    ylabel=sl.LABEL_F0,
    fmt="o-",
    color=sl.COLOR_PRIMARY,
)
"""),

    md(r"""
## Sweep 2: Coupling sweep for linewidth and $Q$

Once an approximate desired frequency has been reached, length may be kept fixed, and the user can now tune the coupling capacitor $C_c$ to change the linewidth. 
"""),

    code(r"""
d0 = 7e-3
L, C = sl.fit_lom(d0, model=MODEL, cpw_params=cpw, freq=freq)   # one fit, reused below

def measure_coupling(Cc):
    f0, kappa = sl.extract_f0_kappa(cpw, d=d0, Cc=Cc)
    rp = sl.resonator_readout_params(L, C, kappa_Hz=kappa)
    return {"kappa_MHz": kappa / 1e6, "Q": rp.Q}

Cc_values = np.linspace(2e-15, 16e-15, 6)
r2 = sl.measure_sweep(Cc_values, measure_coupling)

for Cc, k, q in zip(r2["value"], r2["kappa_MHz"], r2["Q"]):
    print("Cc = {:4.1f} fF    kappa/2pi = {:7.3f} MHz    Q = {:8.0f}".format(Cc * 1e15, k, q))
"""),

    code(r"""
sl.plot_curve_pair(
    r2["value"] * 1e15, r2["kappa_MHz"], r2["Q"],
    xlabel=sl.LABEL_CC,
    ylabel0=sl.LABEL_KAPPA,
    ylabel1=sl.axis_label("Loaded quality factor", "Q"),
    xscale="log", yscale="log",
)
"""),

    md(r"""
## Sweep 3: CPW geometry sweep

The value to be swept in `measure_sweep` can be either a circuit value (like in the previous examples) or a physical
dimension. Here instead of sweeping a circuit value we instead sweep the CPW gap $s$ by swapping it into `CPWParams` with
`dataclasses.replace`, and measure the frequency the same way as Sweep 1.
"""),

    code(r"""
s_values = np.linspace(3e-6, 9e-6, 6)              # gap 3 - 9 um
r3 = sl.measure_sweep(s_values, lambda sg: {
    "f0_GHz": sl.extract_f0_kappa(dataclasses.replace(cpw, s=sg), d=7e-3, Cc=6e-15)[0] / 1e9})

for sg, f in zip(r3["value"], r3["f0_GHz"]):
    print("s = {:.1f} um    f0 = {:.4f} GHz".format(sg * 1e6, f))

sl.plot_curves(
    r3["value"] * 1e6, r3["f0_GHz"],
    xlabel=sl.LABEL_CPW_GAP,
    ylabel=sl.LABEL_F0,
    fmt="o-",
    color=sl.COLOR_PRIMARY,
)
"""),

    md(r"""
The gap moves $f_0$ by only tens of MHz across this range, while the length
moved it by more than a GHz in Sweep 1.

In a similar fashion, other parameters of geometry can be swept to reach target design parameters such as a desired frequency, without having to perform finite-element simulations. This concept is explored further in **Tutorial 7**.
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
