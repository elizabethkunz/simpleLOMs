#!/usr/bin/env python3
"""Build, execute, and install tutorials/03_resonator_networks.ipynb.

Tutorial 3: "Making Resonator Networks". Builds a chain of three
capacitively-coupled CPW resonators with staggered lengths, draws it with the
schematics feature, then fits an OptimizedFit LOM to each resonator subgroup and
validates the reassembled lumped chain against the full CPW chain. Hanger
topologies are deferred to Tutorials 8 and 9.

Runtime note: three OptimizedFit fits on the fast path (~1.5-5 s each), so the
whole notebook executes in well under a minute.

The committed notebook is authoritative: CELLS mirrors its cell sources
byte-for-byte (see _exact for the literal convention), so re-running this
builder only refreshes execution outputs. Edit the notebook first, then
mirror the change here.
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
BASENAME = "03_resonator_networks.ipynb"


def _exact(source: str) -> str:
    """Return the exact cell source from a CELLS literal.

    Literals are written with one delimiter newline right after the opening
    quotes and one right before the closing quotes; only those two newlines
    are stripped. Everything between them -- leading blank lines, trailing
    spaces, presence or absence of a final newline -- is the committed
    notebook cell's source byte-for-byte, so a rebuild reproduces it exactly.
    """
    if not (source.startswith("\n") and source.endswith("\n")):
        raise ValueError("cell literal must start and end with a delimiter newline")
    return source[1:-1]


def md(source: str):
    return new_markdown_cell(_exact(source))


def code(source: str):
    return new_code_cell(_exact(source))


CELLS = [
    md(r"""
# Tutorial 3: Making Resonator Networks

The first two tutorials modelled a **single** resonator, but real chip designs rarely stop
there. Most involve several resonators coupled in a chain. For more complicated designs such as this, the process shown in Tutorial 1 can be 
repeated for multiple resonators in a network. This tutorial shows how to

1. **build** a network of many capacitors and multiple CPW resonators and look
   at its response,
2. **draw** that network with the schematics feature so you can see what you
   built, and
3. Fit a lumped oscillator model (LOM) to each resonator in the
   chain, reassemble the pieces, and check the lumped network against the CPW
   ground truth.

The device design we will use to test this is a chain of **three capacitively-coupled CPW resonators** with
*staggered lengths*, so each rings at its own frequency:

```
[P1] -- Cc1 -- Ctog -- CPW_1 -- Ctog -- Cc2 -- Ctog -- CPW_2 -- Ctog -- Cc3 -- Ctog -- CPW_3 -- Ctog -- Cc4 -- [P2]
```

A natural
"subgroup" of this larger design is one resonator with the two couplers and two ground caps that flank
it: `Cc -- Ctog -- CPW -- Ctog -- Cc`. That subgroup is exactly the topology
`fit_lom` already knows how to fit.


More complicated designs which include hanger resonators will be covered in Tutorials 8 and 9.
"""),

    code(r"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf

from simpleLOMs import CPWParams, fit_lom
from simpleLOMs.networks.cpw import cpw_resonator_network_2port
from simpleLOMs.networks.chain import resonator_chain_network_2port
from simpleLOMs.schematics import resonator_chain_schematic_2port
from simpleLOMs.analysis import circle_fit_f0_kappa, resonances_from_s_max

from simpleLOMs import (
    plot_transmission, plot_all_models, plot_re_im,
    apply_style, axis_label,
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_HIGHLIGHT,
    FIGURE_WIDTH_1COL, FIGURE_WIDTH_2COL,
    FIGURE_HEIGHT_1COL_GR, FIGURE_HEIGHT_2COL_GR,
)


apply_style()

cpw = CPWParams(ep_r=11.45)                        # ultracold-silicon substrate
freq = rf.Frequency(4e9, 12e9, 8_001, unit="Hz")  
Z0 = 50.0

# Three staggered CPW resonators 
lengths = [8.0e-3, 7.0e-3, 6.0e-3]                 # metres

# Four coupling capacitors 
Ccs = [8e-15, 6e-15, 6e-15, 8e-15]

Ctog = 1e-14
Ctogs = [Ctog] * (2 * len(lengths))

segments_cpw = [{"kind": "cpw", "d": d} for d in lengths]
print(f"{len(lengths)} CPW resonators, lengths (mm): {[d*1e3 for d in lengths]}")

"""),

    md(r"""
## 1. Build the CPW network and look at it

Run `resonator_chain_network_2port` to wire the ladder together. Each entry
in `segments` is a resonator. Here every one is a CPW line (`{"kind": "cpw",
"d": ...}`), but in section 3 we will swap some for lumped `LC` tanks. `Ccs` holds
the coupling capacitors (one more than the number of resonators) and `Ctogs`
holds the ground capacitors (two per resonator).
"""),

    code(r"""
chain_cpw = resonator_chain_network_2port(
    freq, segments_cpw, Ccs, Ctogs, cpw_params=cpw, Z0=Z0,
)

"""),

    md(r"""

We then look at the network's transmission $|S_{21}|$ and
 reflection $|S_{11}|$. 
"""),

    code(r"""
plot_transmission(
    {"CPW chain": chain_cpw},
    m=1, n=0,
)
"""),

    md(r"""
The three modes can be read off from the S21 transmission peaks.
"""),

    code(r"""

mode_freqs = resonances_from_s_max(chain_cpw, m=1, n=0)
mode_freqs = np.sort(mode_freqs)
print("Chain modes (GHz):", np.round(mode_freqs / 1e9, 4))

"""),

    md(r"""
We have now constructed a full two-port network with the three distributed elements, which produce
three separate but coupled modes.

"""),

    md(r"""
## 2. Draw the network with the schematics feature

To view the circuit you have just simulated, the `schematics` feature, using `resonator_chain_schematic_2port`, takes the same  
arguments as the network and returns a zoomable circuit diagram with every capacitance and line length labelled.
"""),

    code(r"""
schematic_cpw = resonator_chain_schematic_2port(
    segments_cpw, Ccs, Ctogs,
    cpw_params=cpw, freq=freq, Z0=Z0,
    annotations={"resonators": len(lengths), "modes_GHz": list(np.round(mode_freqs / 1e9, 3))},
)
"""),

    md(r"""
You can choose to save portable copies next to the notebook, and render them inline.
"""),

    code(r"""

from pathlib import Path
figdir = Path("figures")
figdir.mkdir(exist_ok=True)
schematic_cpw.save_svg(figdir / "chain_cpw.svg")
schematic_cpw.save_html(figdir / "chain_cpw.html")
schematic_cpw

"""),

    md(r"""
## 3. Fit a LOM to each resonator individually


In order to use `fit_lom` to fit LC models, we can break the larger network into `Cc -- Ctog -- CPW -- Ctog -- Cc` subgroups. 
Then , we fit each subgroup in
isolation with the **Optimized** model.

The circuit parameters we will use are:
* the two coupling capacitors of subgroup $i$ which are the shared couplers
  $C_{c,i}$ and $C_{c,i+1}$,
* the two ground capacitors, which are the ones flanking that resonator.

`fit_lom` builds its own isolated CPW reference for each subgroup and returns the
effective $(L, C)$. 

Note that fitting a piece in isolation ignores that its neighbours are
*other resonators* rather than $50\,\Omega$ ports. In future versions, this will be generalized so that more accurate $Z_0$ values 
can be used.
"""),

    code(r"""
segments_lc = []
rows = []
for i, d in enumerate(lengths):
    Cc_left, Cc_right = Ccs[i], Ccs[i + 1]
    Ctog_left, Ctog_right = Ctogs[2 * i], Ctogs[2 * i + 1]

    # Locate this subgroup's resonance on its own isolated CPW reference
    sub_cpw = cpw_resonator_network_2port(
        freq, d, Cc_left, Cc_right, Ctog_left, Ctog_right, cpw_params=cpw, Z0=Z0,
    )
    f0_sub, _ = circle_fit_f0_kappa(sub_cpw, 0, 0)
    fit_freq = rf.Frequency(f0_sub - 0.4e9, f0_sub + 0.4e9, 20_001, unit="Hz")

    m = fit_lom(
        d, model="optimized",
        Cc1=Cc_left, Cc2=Cc_right, Ctog1=Ctog_left, Ctog2=Ctog_right,
        cpw_params=cpw, freq=fit_freq, return_model=True,
    )
    segments_lc.append({"kind": "lc", "L": m.L, "C": m.C})
    rows.append({
        "resonator": i + 1,
        "length_mm": d * 1e3,
        "f0_sub_GHz": f0_sub / 1e9,
        "L_nH": m.L * 1e9,
        "C_fF": m.C * 1e15,
    })
    print(f"  resonator {i+1}: d={d*1e3:.1f} mm  f0~{f0_sub/1e9:.3f} GHz  "
          f"L={m.L*1e9:.3f} nH  C={m.C*1e15:.2f} fF")

fits = pd.DataFrame(rows)
fits

"""),

    md(r"""
### 3a. Build the  `LC` chain
"""),

    md(r"""
With the three $(L, C)$ pairs extracted, we can now rebuild the chain with the **same**
coupling capacitors but each CPW line replaced by its fitted `LC` tank. 

> Note that because
> the Optimized model folds the ground capacitance into its *effective* $(L, C)$
> (the topology rule from Tutorial 2), the lumped chain no longer carries `Ctog`
> caps.
"""),

    code(r"""
chain_lc = resonator_chain_network_2port(
    freq, segments_lc, Ccs, Ctogs=None, cpw_params=cpw, Z0=Z0,
)

nets = {"CPW": chain_cpw, "Lumped (optimized)": chain_lc}

# The comparison the reduction is judged on: Re/Im of S11, lumped vs CPW.
plot_all_models(nets, m=0, n=0, quantity="re", title=r"$\mathrm{Re}\,S_{11}$")
plot_all_models(nets, m=0, n=0, quantity="im", title=r"$\mathrm{Im}\,S_{11}$")
plot_all_models(nets, m=1, n=0, quantity="db", title=r"$|S_{21}|$")

"""),

    md(r"""
### 3b. Per-mode frequency agreement

 between the lumped chain and the CPW ground truth.
Now that they are both built, we can find agreement (or lack thereof) between the lumped and CPW chains.

To do so, we locate each mode on the coarse grid, then circle-fit both chains on a dense
window around it for sub-MHz resolution.
"""),

    code(r"""

mode_guess = np.sort(resonances_from_s_max(chain_cpw, m=1, n=0))

def mode_f0(segments, Ctogs, f_guess):
    win = rf.Frequency(f_guess - 0.15e9, f_guess + 0.15e9, 40_001, unit="Hz")
    net = resonator_chain_network_2port(win, segments, Ccs, Ctogs=Ctogs,
                                        cpw_params=cpw, Z0=Z0)
    f0, _ = circle_fit_f0_kappa(net, 0, 0)
    return f0

rows_cmp = []
for j, fg in enumerate(mode_guess, start=1):
    f_cpw = mode_f0(segments_cpw, Ctogs, fg)
    f_lc = mode_f0(segments_lc, None, fg)
    rows_cmp.append({
        "mode": j,
        "CPW_GHz": f_cpw / 1e9,
        "Lumped_GHz": f_lc / 1e9,
        "error_MHz": (f_lc - f_cpw) / 1e6,
        "error_%": 100.0 * (f_lc - f_cpw) / f_cpw,
    })

compare = pd.DataFrame(rows_cmp)
compare

"""),

    md(r"""

For this network, fitting each resonator in isolation with the Optimized
LOM and stitching the pieces back together reproduces the full CPW chain
closely. The mode frequencies line up and the $S_{11}$ lineshape overlays the
ground truth. 

The main approximation we made was that each subgroup was fit against $50\,\Omega$ ports, not against its neighbouring
resonators. 

"""),

    md(r"""
### 3c. Draw the final LC chain schematic
"""),

    md(r"""
Finally, redraw the network in its lumped form — the **same schematic builder**,
now fed `LC` segments, so the CPW lines are replaced by parallel-`LC` tanks
showing the fitted $L$ and $C$.

"""),

    code(r"""
schematic_lc = resonator_chain_schematic_2port(
    segments_lc, Ccs, Ctogs=None,
    cpw_params=cpw, freq=freq, Z0=Z0,
    annotations={"model": "optimized", "resonators": len(lengths)},
)
schematic_lc.save_svg(figdir / "chain_lc.svg")
schematic_lc.save_html(figdir / "chain_lc.html")
schematic_lc

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

    print("Executing notebook (three OptimizedFit fits, fast path)...", flush=True)
    client = NotebookClient(
        nb,
        timeout=1800,
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
