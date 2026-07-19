#!/usr/bin/env python3
"""Build, execute, and install tutorials/08_hanger_resonators.ipynb.

Tutorial 8: Fitting a Hanger Resonator. Build a CPW hanger ground truth,
contrast rewiring an inline OptimizedFit into a hanger vs a dedicated
``fit_lom_hanger`` / ``HangerOptimizedFit``, compare notches, draw the LOM
schematic, and multiplex several hangers on one feedline.

Self-contained: a bootstrap cell puts THIS repo's ``src`` on ``sys.path``.

    python3 tutorials/builder_08.py
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
BASENAME = "08_hanger_resonators.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 8: Fitting a Hanger Resonator

Tutorials 1–3 fitted series resonators where the resonance shows up as a **peak** in $|S_{21}|$. However, many readout chips
instead use a **hanger** (notch) resonators that hangs off a continuous through
feedline. Here, the resonance appears as a **dip** in
$|S_{21}|$:

In this tutorial you will learn how to use **simpleLOMs** to

1. **build** a CPW hanger and look at its notch,
2. Turn an existing or new  `OptimizedFit` into a hanger
3. Fit a hanger LOM explicitly with  with `fit_lom_hanger` and `HangerOptimizedFit` to match the notch in $|S_{21}|$.
"""),

    code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import skrf as rf

import simpleLOMs as sl


sl.apply_style()

cpw = sl.CPWParams(ep_r=11.45)
Z0 = 50.0
d = 7.0e-3          # resonator length (m)
Cc_tap = 30e-15     # coupling (F)
freq = rf.Frequency(3e9, 12e9, 8001, unit="Hz")

print("geometry: d = {:.1f} mm,  Cc_tap = {:.0f} fF".format(d * 1e3, Cc_tap * 1e15))
"""),

    md(r"""
## 1. Build the CPW hanger

`hanger_resonator_network_2port` takes a list of hanging segments and one tap
capacitance per segment. An open termination, which we use here, is the $\lambda/2$ hanger, while
 a shorted termination is the $\lambda/4$ hanger.
"""),

    code(r"""
cpw_open = sl.hanger_resonator_network_2port(
    freq,
    [{"kind": "cpw", "d": d, "termination": "open"}],
    [Cc_tap],
    cpw_params=cpw,
    Z0=Z0,
)
cpw_short = sl.hanger_resonator_network_2port(
    freq,
    [{"kind": "cpw", "d": d, "termination": "short"}],
    [Cc_tap],
    cpw_params=cpw,
    Z0=Z0,
)

f_open, dp_open = sl.notch_frequency(cpw_open)
f_short, dp_short = sl.notch_frequency(cpw_short)
print("CPW open  (lambda/2):  notch at {:.3f} GHz,  |S21| = {:.3f}".format(f_open, dp_open))
print("CPW short (lambda/4):  notch at {:.3f} GHz,  |S21| = {:.3f}".format(f_short, dp_short))
"""),

    code(r"""
sl.hanger_resonator_schematic_2port(
    [{"kind": "cpw", "d": d, "termination": "open"}],
    [Cc_tap],
)
"""),

    md(r"""
## 2. Option 1 — rewire an inline `OptimizedFit`

If you already fit a `LC` tank to your resonator against a series CPW with two-sided coupling
using the `fit_lom` / `OptimizedFit` methods of Tutorial 1, dropping those values into a single-tap hanger is the quick
path.

As an example here we can use a new `OptimizedFit` instance to calibrate $(L, C)$ for the hanger by rewiring it into a hanger.
"""),

    code(r"""
Cc1 = Cc2 = Cc_tap
freq_fit = rf.Frequency(7e9, 9e9, 3001, unit="Hz")
 
def reuse_inline(Ctog):
    "Fit an inline LOM at the given ground loading, then rewire into a hanger."
    L, C = sl.fit_lom(
        d, model="optimized",
        Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog, Ctog2=Ctog,
        cpw_params=cpw, freq=freq_fit, Z0=Z0,
    )
    net = sl.hanger_resonator_network_2port(
        freq, [{"kind": "lc", "L": L, "C": C}], [Cc_tap], Z0=Z0,
    )
    f_r, dp = sl.notch_frequency(net)
    return L, C, net, f_r, dp


# Matched reuse: drop the inline grounds to mirror the ground-free hanger tap.
L_inline, C_inline, lc_reused, f_reuse, dp_reuse = reuse_inline(0.0)
err_reuse = 100.0 * (f_reuse - f_open) / f_open


print("matched reuse (Ctog =  0 fF):  L = {:.4f} nH,  C = {:.4f} pF".format(
    L_inline * 1e9, C_inline * 1e12))
print("                               notch at {:.3f} GHz  ({:+.3f} % vs CPW open)".format(
    f_reuse, err_reuse))
"""),

    md(r"""
## 3. Option 2 — fit a hanger LOM with `fit_lom_hanger`

Instead of rewiring an inline fit, you can instead fit the hanger directly. `fit_lom_hanger`
builds a CPW hanger reference and runs `HangerOptimizedFit` against the S21
notch, instead of the Re(S11)/Im(S11) pair that is used in `OptimizedFit`.

Let's try to use `fit_lom_hanger` for our device geometry.
"""),

    code(r"""
hanger_model = sl.fit_lom_hanger(
    d,
    Cc_tap=Cc_tap,
    termination="open",
    cpw_params=cpw,
    freq=freq,
    Z0=Z0,
    return_model=True,
)
print(hanger_model)
print("L = {:.4f} nH,  C = {:.4f} pF".format(
    hanger_model.L * 1e9, hanger_model.C * 1e12))

lc_fitted = hanger_model.get_network(freq, Cc_tap=Cc_tap, Z0=Z0)
f_fit, dp_fit = sl.notch_frequency(lc_fitted)
"""),

    md(r"""
## 4. Compare the notches
"""),

    md(r"""
We can now look at the error in the notch feature between these two methods.
"""),

    code(r"""
err_fit = 100.0 * (f_fit - f_open) / f_open
print("hanger-fitted LC:     notch at {:.3f} GHz  ({:+.3f} % vs CPW open)".format(
    f_fit, err_fit))
"""),

    md(r"""
Then, we can plot $|S_{21}|$ for the two methods (using `OptimizedFit` and `HangerOptimizedFit` respectively) to see whether the resonance feature is captured.
"""),

    code(r"""
f_ghz = freq.f / 1e9
sl.plot_curves(
    f_ghz,
    (np.abs(cpw_open.s[:, 1, 0]), "CPW hanger, open (truth)",
     dict(color=sl.COLOR_PRIMARY, lw=2.2)),
    (np.abs(lc_fitted.s[:, 1, 0]), "LC, HangerOptimizedFit (Option 2)",
     dict(color=sl.COLOR_HIGHLIGHT, lw=1.8, ls="-.")),
    (np.abs(lc_reused.s[:, 1, 0]), "LC, reused inline (Option 1)",
     dict(color=sl.COLOR_SECONDARY, lw=1.5, ls="--")),
    xlabel=sl.LABEL_FREQ,
    ylabel=r"$|S_{21}|$",
    xlim=(8.25, 8.5),
)

print(sl.utils.format_comparison_table(
    [
        ("CPW open (truth)", "{:.4f}".format(f_open), "—", "{:.4f}".format(dp_open)),
        ("LC reuse, Ctog=20 fF (naive)", "{:.4f}".format(f_bad), "{:+.3f}".format(err_bad), "{:.4f}".format(dp_bad)),
        ("LC reuse, Ctog=0 (Option 1)", "{:.4f}".format(f_reuse), "{:+.3f}".format(err_reuse), "{:.4f}".format(dp_reuse)),
        ("LC HangerOptimizedFit (Option 2)", "{:.4f}".format(f_fit), "{:+.3f}".format(err_fit), "{:.4f}".format(dp_fit)),
    ],
    ["network", "notch GHz", "err %", "|S21|"],
))
"""),

    md(r"""
Although both methods reproduce the notch frequency quite well, only the dedicated fit
also reproduces the notch depth.
"""),

    md(r"""
## 5. Draw the fitted LOM hanger

As usual we can draw the fitted hanger using the schematic builder, which uses the same segment dicts as the network API. 

Let's plot the lumped model we fitted using Option 2.
"""),

    code(r"""
sl.hanger_resonator_schematic_2port(
    [{"kind": "lc", "L": hanger_model.L, "C": hanger_model.C}],
    [Cc_tap],
)
"""),

    md(r"""
This can now be saved or exported for further use.
"""),

    md(r"""
## 6. Multiple hangers on one feedline

Much line in Tutorial 3, resonator networks with multiple hangers on the same feedline can be constructed.

To do so, we will pass several segments (and one `Cc` per segment) to put several notches on the
same through line. Fit each resonator
in isolation with `fit_lom_hanger` at its own length, then reassemble.
"""),

    md(r"""
### 6a. Constructing the network
First we construct the network with our CPW from before and a CPW at a length of $8000$ mm.
"""),

    code(r"""
d2 = 8.0e-3
cpw_two = sl.hanger_resonator_network_2port(
    freq,
    [
        {"kind": "cpw", "d": d,  "termination": "open"},
        {"kind": "cpw", "d": d2, "termination": "open"},
    ],
    [Cc_tap, Cc_tap],
    cpw_params=cpw,
    Z0=Z0,
)
"""),

    md(r"""
Next we fit the LOM for each individual resonator using `fit_lom_hanger`.
"""),

    code(r"""
L1, C1 = sl.fit_lom_hanger(d,  Cc_tap=Cc_tap, termination="open",
                           cpw_params=cpw, freq=freq, Z0=Z0)
L2, C2 = sl.fit_lom_hanger(d2, Cc_tap=Cc_tap, termination="open",
                           cpw_params=cpw, freq=freq, Z0=Z0)
"""),

    md(r"""
We now reconstruct the LC-fitted network.
"""),

    code(r"""
lc_two = sl.hanger_resonator_network_2port(
    freq,
    [
        {"kind": "lc", "L": L1, "C": C1},
        {"kind": "lc", "L": L2, "C": C2},
    ],
    [Cc_tap, Cc_tap],
    Z0=Z0,
)
"""),

    md(r"""
### 6b. Comparing LC and CPW networks

Now we can plot the curved to see how the LC behaves compared to the network.
"""),

    code(r"""
sl.plot_curves(
    freq.f / 1e9,
    (np.abs(cpw_two.s[:, 1, 0]), "CPW",
     dict(color=sl.COLOR_PRIMARY, lw=2.2)),
    (np.abs(lc_two.s[:, 1, 0]), "LC, HangerOptimizedFit",
     dict(color=sl.COLOR_HIGHLIGHT, lw=1.8, ls="-.")),
    xlabel=sl.LABEL_FREQ,
    ylabel=r"$|S_{21}|$",
)
"""),

    md(r"""
The notches are very close! To be sure, we can print their frequencies and compare them numerically.
"""),

    code(r"""
print("CPW notches (GHz):", ["{:.3f}".format(x) for x in sl.notches_below(cpw_two)])
print("LC  notches (GHz):", ["{:.3f}".format(x) for x in sl.notches_below(lc_two)])
"""),

    md(r"""
Finally we can make a schematic of the fitted system and export it. As we can see, the fitted system has the same two-hanger topology as the CPW hanger system.
"""),

    code(r"""
sl.hanger_resonator_schematic_2port(
    [
        {"kind": "lc", "L": L1, "C": C1},
        {"kind": "lc", "L": L2, "C": C2},
    ],
    [Cc_tap, Cc_tap],
)
"""),

    md(r"""
## What next?

In the next tutorial, Tutorial 9, we will see how to cascade a series resonator (Tutorial 1, 3) with hanger resonators (this tutorial) to make networks that include both.
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

    print("Executing Tutorial 8 (hanger fits)...", flush=True)
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
