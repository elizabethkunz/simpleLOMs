#!/usr/bin/env python3
"""Build, execute, and install tutorials/09_hybrid_series_hanger.ipynb.

Tutorial 9: Hybrid Series + Hanger Resonators. Cascades an inline Purcell
filter with two open hangers on a through feedline: build the CPW hybrid ground
truth, draw it, fit each resonator with the matching LOM (inline Optimized /
``fit_lom_hanger``), rebuild the lumped hybrid, and compare to CPW.

Self-contained: a bootstrap cell puts THIS repo's ``src`` on ``sys.path``.

    python3 tutorials/builder_09.py
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
BASENAME = "09_hybrid_series_hanger.ipynb"


def md(source: str):
    return new_markdown_cell(source.strip("\n") + "\n")


def code(source: str):
    return new_code_cell(source.strip("\n") + "\n")


CELLS = [
    md(r"""
# Tutorial 9: Hybrid Series + Hanger Resonators

Tutorials 3 and 8 treat **inline chains** and **hanger / notch** resonators
separately. But they may also be combined into one chain and analyzed as a whole.

In this tutorial, we will learn how to:
1. Construct a chain of resonators with both inline and hanger distributed CPWs.
2. Assemble a lumped system by fitting each piece with its respective LOM (`fit_lom` for the series resonator, `fit_lom_hanger`
for each hanger)
3. Check the accuracy of the circuit model for the hybrid system.

For this tutorial, we will use the 3-resonator system: 

```
[P1] -- Cc -- Ctog -- CPW_Purcell -- Ctog -- Cc --||-- hang1 -- hang2 -- [P2]
      \_____________ series / inline _____________/   \___ hangers ___/
```

It is also drawn later during the tutorial in `hybrid_schematic_2port`. 
"""),

    md(r"""
## Setup
"""),

    code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf

import simpleLOMs as sl

sl.apply_style()
"""),

    md(r"""
Our CPWs will all share a common `CPWParams` for the sake of this tutorial, but this can be changed in systems where the hybrid network has CPWs with different width or thickness.
"""),

    code(r"""
cpw = sl.CPWParams(ep_r=11.45)
freq = rf.Frequency(5e9, 11e9, 4001, unit="Hz")
Z0 = 50.0
"""),

    md(r"""
The series CPW, which we'll refer to as the Purcell filter, uses the familiar inline topology.
"""),

    code(r"""
d_purcell = 6.5e-3
Ccs_p = [8e-15, 8e-15]
Ctogs_p = [1e-14, 1e-14]
seg_purcell_cpw = [{"kind": "cpw", "d": d_purcell}]
"""),

    md(r"""
THe inline resonator is then followed by two open hangers on a through feedline.
"""),

    code(r"""
d_hangers = [7.0e-3, 8.0e-3]         
Cc_taps = [30e-15, 30e-15]
seg_hangers_cpw = [
    {"kind": "cpw", "d": d, "termination": "open"} for d in d_hangers
]

print(f"resonators: 1 inline Purcell ({d_purcell*1e3:.1f} mm)"
      f" + {len(d_hangers)} open hangers {[d*1e3 for d in d_hangers]} mm")
"""),

    md(r"""
## 1. Build the CPW hybrid (ground truth)

Build each block with its native network builder, then cascade them into one
2-port. The inline Purcell puts a **transmission peak** near its loaded
resonance; each hanger puts a **notch** in $|S_{21}|$.
"""),

    md(r"""
## 1a. Network objects
"""),

    code(r"""
purcell_cpw = sl.resonator_chain_network_2port(
    freq, seg_purcell_cpw, Ccs_p, Ctogs_p, cpw_params=cpw, Z0=Z0,
)
hangers_cpw = sl.hanger_resonator_network_2port(
    freq, seg_hangers_cpw, Cc_taps, cpw_params=cpw, Z0=Z0,
)
hybrid_cpw = purcell_cpw ** hangers_cpw
"""),

    md(r"""
## 1b. Plot the network transmission
"""),

    code(r"""
sl.plot_transmission({"CPW hybrid": hybrid_cpw}, m=1, n=0)

peaks_cpw = np.sort(sl.resonances_from_s_max(hybrid_cpw, m=1, n=0))
notches_cpw = sl.notches_below(hybrid_cpw, thresh=0.7)
print("S21 peaks  (GHz):", np.round(peaks_cpw / 1e9, 4))
print("S21 notches (GHz):", [round(f, 4) for f in notches_cpw])
"""),

    md(r"""
As expected there are is one major peak and two notches visible on the graph.
"""),

    md(r"""
## 2. Draw the hybrid schematic

`hybrid_schematic_2port` puts the inline Purcell and both hangers on **one**
rail — the schematic counterpart of cascading the two networks. Same segment /
capacitor arguments as the network builders.
"""),

    code(r"""
from pathlib import Path

figdir = Path("figures")
figdir.mkdir(exist_ok=True)

sch_cpw = sl.hybrid_schematic_2port(
    seg_purcell_cpw, Ccs_p, seg_hangers_cpw, Cc_taps,
    inline_Ctogs=Ctogs_p, cpw_params=cpw, freq=freq, Z0=Z0,
    annotations={"block": "hybrid CPW (Purcell + hangers)"},
)
sch_cpw.save_svg(figdir / "hybrid_cpw.svg")
sch_cpw.save_html(figdir / "hybrid_cpw.html")
print("saved figures/hybrid_cpw.svg|.html")
print("groups:", [g["label"] for g in sch_cpw.groups])
print("components:", [c.id for c in sch_cpw.components])
sch_cpw
"""),

    md(r"""
## 3. Fit each resonator with the matching LOM

- **Inline Purcell:** `fit_lom(..., model="optimized")` against an isolated
  `Cc -- Ctog -- CPW -- Ctog -- Cc` subgroup (Tutorial 3).
- **Each hanger:** `fit_lom_hanger` against an isolated CPW hanger notch
  (Tutorial 8).

Fitting pieces in isolation (50 Ω ports / ideal feed) and stitching them back
is the same approximation as Tutorial 3 — the hybrid test asks whether that
still holds when the two topologies are cascaded.
"""),

    md(r"""
### 3a. Inline fit
"""),

    code(r"""
# --- Inline Purcell ---
sub_cpw = sl.cpw_resonator_network_2port(
    freq, d_purcell, Ccs_p[0], Ccs_p[1], Ctogs_p[0], Ctogs_p[1],
    cpw_params=cpw, Z0=Z0,
)
f0_sub, _ = sl.circle_fit_f0_kappa(sub_cpw, 0, 0)
fit_freq = rf.Frequency(f0_sub - 0.4e9, f0_sub + 0.4e9, 15_001, unit="Hz")

m_p = sl.fit_lom(
    d_purcell, model="optimized",
    Cc1=Ccs_p[0], Cc2=Ccs_p[1], Ctog1=Ctogs_p[0], Ctog2=Ctogs_p[1],
    cpw_params=cpw, freq=fit_freq, return_model=True,
)
print(f"Purcell (inline): d={d_purcell*1e3:.1f} mm  f0~{f0_sub/1e9:.3f} GHz  "
      f"L={m_p.L*1e9:.3f} nH  C={m_p.C*1e15:.2f} fF")
"""),

    md(r"""
### 3b. Hanger fits
"""),

    code(r"""
hanger_models = []
rows = [{
    "role": "Purcell (inline)",
    "topology": "series",
    "length_mm": d_purcell * 1e3,
    "L_nH": m_p.L * 1e9,
    "C_fF": m_p.C * 1e15,
}]
for i, (d, Cc) in enumerate(zip(d_hangers, Cc_taps), start=1):
    m = sl.fit_lom_hanger(
        d, Cc_tap=Cc, termination="open", cpw_params=cpw, return_model=True,
    )
    hanger_models.append(m)
    rows.append({
        "role": f"hanger {i}",
        "topology": "hanger",
        "length_mm": d * 1e3,
        "L_nH": m.L * 1e9,
        "C_fF": m.C * 1e15,
    })
    print(f"hanger {i}:          d={d*1e3:.1f} mm  "
          f"L={m.L*1e9:.3f} nH  C={m.C*1e15:.2f} fF")
"""),

    md(r"""
## 3c. Inspect all fits
"""),

    code(r"""
fits = pd.DataFrame(rows)
fits
"""),

    md(r"""
## 4. Rebuild the lumped hybrid and compare to CPW


We now Cascade the  inline $L$, $C$ and theHanger $L$, $C$s generated in the previous step to build the overall netowrk which we will then
 overlay on
$|S_{21}|$.
"""),

    code(r"""
seg_purcell_lc = [{"kind": "lc", "L": m_p.L, "C": m_p.C}]
seg_hangers_lc = [
    {"kind": "lc", "L": m.L, "C": m.C} for m in hanger_models
]

purcell_lc = sl.resonator_chain_network_2port(
    freq, seg_purcell_lc, Ccs_p, Ctogs=None, cpw_params=cpw, Z0=Z0,
)
hangers_lc = sl.hanger_resonator_network_2port(
    freq, seg_hangers_lc, Cc_taps, cpw_params=cpw, Z0=Z0,
)
hybrid_lc = purcell_lc ** hangers_lc
"""),

    md(r"""
Plotting the transmission shows that there is strong agreement between the network of lumped elements and the fully distributed network:
"""),

    code(r"""
sl.plot_all_models(
    {"CPW": hybrid_cpw, "Optimized": hybrid_lc},
    m=1, n=0, quantity="db",
    title=r"Hybrid $|S_{21}|$: CPW vs LOM",
)
"""),

    md(r"""
### 4b. Evaluate the accuracy

With the two networks, to go further than just a graph we can determine numerically how accurate their predicted peaks are.
"""),

    code(r"""
notches_lc = sl.notches_below(hybrid_lc, thresh=0.7)
peaks_lc = np.sort(sl.resonances_from_s_max(hybrid_lc, m=1, n=0))

rows_n = []
for nc in notches_cpw:
    nl = min(notches_lc, key=lambda x: abs(x - nc))
    rows_n.append({
        "feature": "notch",
        "CPW_GHz": nc,
        "Lumped_GHz": nl,
        "error_MHz": (nl - nc) * 1e3,
        "error_%": 100.0 * (nl - nc) / nc,
    })

rows_p = []
for i, (pc, pl) in enumerate(zip(peaks_cpw[:3], peaks_lc[:3]), start=1):
    rows_p.append({
        "feature": f"peak {i}",
        "CPW_GHz": pc / 1e9,
        "Lumped_GHz": pl / 1e9,
        "error_MHz": (pl - pc) / 1e6,
        "error_%": 100.0 * (pl - pc) / pc,
    })

compare = pd.DataFrame(rows_n + rows_p)
print("Max |error_%|:", float(np.max(np.abs(compare["error_%"]))))
compare
"""),

    md(r"""
## 5. Lumped hybrid schematic

We use the same `hybrid_schematic_2port`, now using fitted LC tanks:

"""),

    code(r"""
sch_lc = sl.hybrid_schematic_2port(
    seg_purcell_lc, Ccs_p, seg_hangers_lc, Cc_taps,
    inline_Ctogs=None, cpw_params=cpw, freq=freq, Z0=Z0,
    annotations={"block": "hybrid LOM (OptimizedFit + HangerOptimizedFit)"},
)
sch_lc.save_svg(figdir / "hybrid_lc.svg")
sch_lc.save_html(figdir / "hybrid_lc.html")
print("saved figures/hybrid_lc.svg|.html")
sch_lc
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

    print("Executing Tutorial 9 (hybrid series + hanger)...", flush=True)
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
