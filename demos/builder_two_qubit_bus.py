#!/usr/bin/env python3
"""Build, execute, and install demos/two_qubit_bus.ipynb.

Demo: "Two qubits coupled by a bus resonator". End-to-end case study: two
lumped-LC "qubits" exchange-coupled through a lambda/2 CPW bus. Calibrates the
qubit-bus coupling g from the vacuum-Rabi-style splitting, maps the
qubit-qubit avoided crossing (2J), sweeps J against the qubit-bus detuning and
compares with the dispersive formula J = g1 g2 (1/D1 + 1/D2)/2 — first with
the distributed CPW bus (ground truth), then with its fitted Optimized LOM.

Runtime note: network builds dominate (~2 ms per frequency point for a chain
on the reference machine — see tutorials/_perf_fit_lom_final.txt). All sweeps
therefore use narrow adaptive windows (2-4k points) around the modes being
tracked; the whole notebook executes in a few minutes.

The committed notebook is authoritative: CELLS mirrors its cell sources
byte-for-byte (see _exact for the literal convention), so re-running this
builder only refreshes execution outputs. Edit the notebook first, then
mirror the change here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[1]
DEMOS = REPO / "demos"
BASENAME = "two_qubit_bus.ipynb"


def _exact(source: str) -> str:
    '''Return the exact cell source from a CELLS literal.

    Literals are written with one delimiter newline right after the opening
    quotes and one right before the closing quotes; only those two newlines
    are stripped. Everything between them -- leading blank lines, trailing
    spaces, presence or absence of a final newline -- is the committed
    notebook cell's source byte-for-byte, so a rebuild reproduces it exactly.
    '''
    if not (source.startswith("\n") and source.endswith("\n")):
        raise ValueError("cell literal must start and end with a delimiter newline")
    return source[1:-1]


def md(source: str):
    return new_markdown_cell(_exact(source))


def code(source: str):
    return new_code_cell(_exact(source))


CELLS = [
    md(r"""
# Demo: Two qubits coupled by a bus resonator

A **case study** assembling Tutorials 1 (fitting), 2 (model ranking and loaded
shifts), and 3 (resonator chains) into the canonical two-qubit coupling
element: a $\lambda/2$ CPW **bus resonator** with a lumped-LC "qubit"
capacitively tapped onto each end,

```
[P1: drive 1] -- C_d -- QUBIT 1 -- Cc -- BUS (CPW) -- Cc -- QUBIT 2 -- C_d -- [P2: drive 2]
```

Both qubits sit a detuning $\Delta = f_\mathrm{bus} - f_q$ **below** the bus,
which mediates an exchange coupling between them. At leading (dispersive)
order

$$J = \frac{g_1 g_2}{2}\left(\frac{1}{\Delta_1} + \frac{1}{\Delta_2}\right),$$

with $g_i$ the qubit-bus couplings. The plan:

1. build the mixed chain (two LC tanks + one CPW) and draw it,
2. calibrate $g$ by tuning qubit 1 through the bus and reading the minimum
   normal-mode splitting ($2g$),
3. map the **qubit-qubit avoided crossing** — tune qubit 1 through qubit 2
   and read $2J$ at closest approach,
4. sweep both qubits together to get $J(\Delta)$ and compare it with the
   dispersive formula,
5. repeat 3-4 with the bus replaced by its fitted **Optimized LOM** — the
   package's distributed-to-lumped reduction, validated on a *coupling
   strength* rather than a bare frequency.

> **Scope of the model.** The qubits are *linear* LC tanks. Mode frequencies,
> $g$, and $J$ are linear-network quantities and match the transmon results at
> leading order in the coupling; anharmonicity-dependent effects (ZZ
> interaction, $\chi$, photon-number dependence) require a nonlinear qubit
> model and are outside this package's scope. Each qubit also carries a weak
> drive-line tap ($C_d$), which is how a real device drives and probes it —
> the tap pulls every LC frequency down by the same ~$C_d/2C_q$ fraction, so
> quoted "design" frequencies are the bare LC values and plots use the
> measured (dressed) positions.
"""),

    code(r"""
import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skrf as rf
from scipy.signal import find_peaks

from simpleLOMs import (
    CPWParams, fit_lom,
    resonator_chain_network_2port,
    plot_transmission, plot_curves,
    apply_style, axis_label,
    MODEL_COLORS, COLOR_PRIMARY, COLOR_SECONDARY, COLOR_HIGHLIGHT,
)
from simpleLOMs.schematics import resonator_chain_schematic_2port

apply_style()

cpw = CPWParams(ep_r=11.45)      # ultracold-silicon chip, as in the tutorials
Z0 = 50.0

C_q = 65e-15                     # qubit shunt capacitance (both qubits)
f_q1 = 5.00e9                    # design (bare-LC) qubit frequencies
f_q2 = 5.10e9
d_bus = 8.1e-3                   # lambda/2 CPW bus, ~7.3 GHz loaded
Cc_b = 4e-15                     # qubit <-> bus couplers (both ends)
C_d = 1e-15                      # drive-line taps (probe ports)
Ctog = 1e-14                     # ground caps flanking the CPW bus

# drive taps load each qubit node by ~C_d, pulling every LC mode down by the
# same factor; used only to center the narrow sweep windows.
pull = np.sqrt(C_q / (C_q + C_d))


def L_of(f_hz: float, C: float = C_q) -> float:
    '''Inductance of an LC tank resonating at f_hz with capacitance C.'''
    return 1.0 / ((2 * np.pi * f_hz) ** 2 * C)


def bus_chain(freqgrid, f1, f2, bus_seg):
    '''The full chain with qubit design frequencies f1, f2 and a given bus.'''
    segs = [
        {"kind": "lc", "L": L_of(f1), "C": C_q},
        bus_seg,
        {"kind": "lc", "L": L_of(f2), "C": C_q},
    ]
    ctogs = [0, 0, Ctog, Ctog, 0, 0] if bus_seg["kind"] == "cpw" else None
    return resonator_chain_network_2port(
        freqgrid, segs, [C_d, Cc_b, Cc_b, C_d], ctogs, cpw_params=cpw, Z0=Z0,
    )


bus_cpw = {"kind": "cpw", "d": d_bus}
print(f"qubit 1: L = {L_of(f_q1)*1e9:.3f} nH   qubit 2: L = {L_of(f_q2)*1e9:.3f} nH")
"""),

    md(r"""
## 1. Build the mixed chain and look at it

One `resonator_chain_network_2port` call, with the bus as the only
distributed segment. The lumped qubits carry no ground caps (their $C_q$
already is the node capacitance), so their `Ctogs` entries are 0.
"""),

    code(r"""
figdir = Path("figures")
figdir.mkdir(exist_ok=True)

schematic = resonator_chain_schematic_2port(
    [{"kind": "lc", "L": L_of(f_q1), "C": C_q}, bus_cpw,
     {"kind": "lc", "L": L_of(f_q2), "C": C_q}],
    [C_d, Cc_b, Cc_b, C_d],
    [0, 0, Ctog, Ctog, 0, 0],
    cpw_params=cpw, freq=rf.Frequency(6e9, 8e9, 3, unit="Hz"), Z0=Z0,
    annotations={"P1": "drive line 1", "P2": "drive line 2",
                 "roles": "qubit 1 | bus | qubit 2"},
)
schematic.save_svg(figdir / "two_qubit_bus.svg")
schematic.save_html(figdir / "two_qubit_bus.html")
schematic
"""),

    code(r"""
t0 = time.time()
broad = rf.Frequency(4.2e9, 8.2e9, 8_001, unit="Hz")
net_broad = bus_chain(broad, f_q1, f_q2, bus_cpw)
print(f"[{time.time()-t0:.1f} s] broad-band build")

s21_db = 20 * np.log10(np.maximum(np.abs(net_broad.s[:, 1, 0]), 1e-300))
pk, _ = find_peaks(s21_db, prominence=6)
peak_f = np.sort(broad.f[pk])
print("modes in |S21| (GHz):", np.round(peak_f / 1e9, 4))
f_bus_meas = float(peak_f[-1])          # the bus is the highest mode here

plot_transmission({"drive 1 → drive 2": net_broad}, m=1, n=0)
"""),

    md(r"""
Three modes: the two qubits near 5 GHz (pulled slightly below their design
values by the drive taps) and the bus near 7.3 GHz. All the physics below is
in how the two low modes move when we tune the qubits, so every sweep works
on a **narrow window** around them — broadband rebuilds per sweep point would
be wasted time.

## 2. Calibrate the qubit-bus coupling $g$

Tune qubit 1 across the bus (qubit 2 parked far away at 4.2 GHz) and track
the two hybridized modes. For an ideal two-mode crossing the splitting obeys
$s(f_1)^2 = (2g)^2 + \delta^2$ — a parabola in the detuning $\delta$ — so we
fit $s^2$ and read $2g$ off the vertex rather than trusting the coarsest
sample to land exactly on resonance.
"""),

    code(r"""
def refine_peak(f: np.ndarray, y: np.ndarray, i: int) -> float:
    '''Sub-grid peak position: parabola through the 3 points straddling i.'''
    if not (0 < i < len(f) - 1):
        return float(f[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    return float(f[i] + np.clip(delta, -0.5, 0.5) * (f[i + 1] - f[i]))


def two_modes(net: rf.Network) -> np.ndarray:
    '''Frequencies of the two most prominent |S21| peaks, sub-grid refined.'''
    f = net.frequency.f
    db = 20 * np.log10(np.maximum(np.abs(net.s[:, 1, 0]), 1e-300))
    pk, props = find_peaks(db, prominence=3.0)
    if len(pk) < 2:
        return np.array([np.nan, np.nan])
    top = pk[np.argsort(props["prominences"])[::-1][:2]]
    return np.sort([refine_peak(f, db, int(i)) for i in top])


t0 = time.time()
f_park = 4.2e9
f1_scan = f_bus_meas / pull + np.linspace(-100e6, 100e6, 5)
splits = []
for f1 in f1_scan:
    win = rf.Frequency(f_bus_meas - 220e6, f_bus_meas + 220e6, 3_001, unit="Hz")
    lo, hi = two_modes(bus_chain(win, f1, f_park, bus_cpw))
    splits.append(hi - lo)
splits = np.array(splits)

# vertex of the s^2 parabola = (2g)^2  (centered x for conditioning)
x_c = f1_scan - f1_scan.mean()
coef = np.polyfit(x_c, splits**2, 2)
s2_min = coef[2] - coef[1] ** 2 / (4 * coef[0])
g = 0.5 * np.sqrt(s2_min)
print(f"[{time.time()-t0:.1f} s] g calibration ({len(f1_scan)} points)")
print(f"qubit-bus coupling: g = {g/1e6:.2f} MHz   (min splitting 2g = {2*g/1e6:.2f} MHz)")

plot_curves(
    f1_scan / 1e9,
    (splits / 1e6, "measured splitting"),
    (np.sqrt(np.polyval(coef, x_c)) / 1e6, "hyperbola fit", {"ls": "--"}),
    fmt="o",
    xlabel=axis_label("Qubit 1 design frequency", "f_1", "GHz"),
    ylabel=axis_label("Normal-mode splitting", "s", "MHz"),
)
"""),

    md(r"""
## 3. The qubit-qubit avoided crossing

Now park the bus far above ($\Delta \approx 2.2$ GHz) and tune qubit 1
through qubit 2. The bus is never resonantly excited — it just mediates the
exchange — and the two qubit-like modes repel with minimum splitting $2J$.
Each sweep point evaluates the chain only on a window wide enough to contain
both modes.
"""),

    code(r"""
def crossing_modes(f1_list, f2, bus_seg):
    '''(n,2) dressed mode frequencies for each qubit-1 design frequency.'''
    out = []
    for f1 in f1_list:
        e1, e2 = f1 * pull, f2 * pull
        center = 0.5 * (e1 + e2)
        half = 0.5 * abs(e1 - e2) + 20e6
        n = 2_001 if half < 60e6 else 4_001
        win = rf.Frequency(center - half, center + half, n, unit="Hz")
        out.append(two_modes(bus_chain(win, f1, f2, bus_seg)))
    return np.array(out)


t0 = time.time()
f1_cross = np.linspace(4.90e9, 5.30e9, 13)
modes_cpw = crossing_modes(f1_cross, f_q2, bus_cpw)
print(f"[{time.time()-t0:.1f} s] crossing sweep, CPW bus ({len(f1_cross)} points)")

splitting_cpw = modes_cpw[:, 1] - modes_cpw[:, 0]
J_cpw = 0.5 * np.nanmin(splitting_cpw)
print(f"closest approach (CPW bus): 2J = {2*J_cpw/1e6:.3f} MHz "
      f"-> J = {J_cpw/1e6:.3f} MHz")
"""),

    md(r"""
## 4. Swap the CPW bus for its fitted LOM

`fit_lom` reduces the bus — in its own subgroup, with its actual couplers and
ground caps — to an effective $(L, C)$, and the chain builder swaps it in as
just another `"lc"` segment (`Ctogs=None`: the Optimized model folds the
ground caps into its effective values, the topology rule from Tutorial 2).
The avoided crossing is then re-measured on the *all-lumped* chain.
"""),

    code(r"""
t0 = time.time()
L_b, C_b = fit_lom(
    d_bus, model="optimized", Cc1=Cc_b, Cc2=Cc_b, Ctog1=Ctog, Ctog2=Ctog,
    cpw_params=cpw,
)
f_bus_lom = 1.0 / (2 * np.pi * np.sqrt(L_b * C_b))
print(f"[{time.time()-t0:.1f} s] bus OptimizedFit")
print(f"bus LOM: L = {L_b*1e9:.4f} nH, C = {C_b*1e15:.2f} fF "
      f"-> f = {f_bus_lom/1e9:.4f} GHz")

bus_lom = {"kind": "lc", "L": L_b, "C": C_b}

t0 = time.time()
modes_lom = crossing_modes(f1_cross, f_q2, bus_lom)
print(f"[{time.time()-t0:.1f} s] crossing sweep, LOM bus")

splitting_lom = modes_lom[:, 1] - modes_lom[:, 0]
J_lom = 0.5 * np.nanmin(splitting_lom)
print(f"closest approach (LOM bus): 2J = {2*J_lom/1e6:.3f} MHz "
      f"-> J = {J_lom/1e6:.3f} MHz   "
      f"(error vs CPW: {100*(J_lom-J_cpw)/J_cpw:+.2f} %)")

x = f1_cross / 1e9
plot_curves(
    x,
    (modes_cpw[:, 0] / 1e9, "CPW bus", {"color": MODEL_COLORS["cpw"], "fmt": "o-"}),
    (modes_cpw[:, 1] / 1e9, None, {"color": MODEL_COLORS["cpw"], "fmt": "o-"}),
    (modes_lom[:, 0] / 1e9, "Optimized LOM bus",
     {"color": MODEL_COLORS["optimized"], "fmt": "s--", "mfc": "none"}),
    (modes_lom[:, 1] / 1e9, None,
     {"color": MODEL_COLORS["optimized"], "fmt": "s--", "mfc": "none"}),
    (f1_cross * pull / 1e9, "bare qubit 1 (dressed)", {"color": "0.6", "ls": ":", "lw": 0.8}),
    (np.full_like(x, f_q2 * pull / 1e9), "bare qubit 2 (dressed)",
     {"color": "0.6", "ls": "-.", "lw": 0.8}),
    xlabel=axis_label("Qubit 1 design frequency", "f_1", "GHz"),
    ylabel=axis_label("Mode frequency", "f", "GHz"),
)
"""),

    md(r"""
The lumped bus reproduces the avoided crossing — branch positions *and* the
minimum splitting $2J$ — to a fraction of a percent. This is a stricter test
than matching a bare resonance: $J$ depends on the bus's off-resonant
response at the qubit frequencies, 2 GHz below its own mode.

## 5. $J$ versus detuning, against the dispersive formula

Finally the design sweep: move **both** qubits together (staying degenerate,
$f_1 = f_2 = f$) from deep in the dispersive regime up toward the bus, and
compare the measured $J(\Delta) = s/2$ with

$$J_\mathrm{pred}(\Delta) = \frac{g(f)^2}{\Delta},\qquad
g(f) = g_\mathrm{cal}\sqrt{f/f_\mathrm{bus}},$$

the equal-detuning form of the dispersive expression, with the $\sqrt{f}$
scaling of a capacitive coupling to a fixed bus.
"""),

    code(r"""
t0 = time.time()
f_common = np.array([4.4e9, 4.85e9, 5.3e9, 5.75e9, 6.1e9, 6.35e9, 6.55e9])
rows = []
for f in f_common:
    win_c = f * pull
    win = rf.Frequency(win_c - 25e6, win_c + 25e6, 2_001, unit="Hz")
    lo_c, hi_c = two_modes(bus_chain(win, f, f, bus_cpw))
    lo_l, hi_l = two_modes(bus_chain(win, f, f, bus_lom))
    rows.append({
        "f (GHz)": f / 1e9,
        "Delta (GHz)": (f_bus_meas - f * pull) / 1e9,
        "J CPW (MHz)": (hi_c - lo_c) / 2e6,
        "J LOM (MHz)": (hi_l - lo_l) / 2e6,
    })
print(f"[{time.time()-t0:.1f} s] J(Delta) sweep, {len(f_common)} x 2 builds")

J_df = pd.DataFrame(rows)
J_df["J pred (MHz)"] = (
    (g * np.sqrt(f_common * pull / f_bus_meas)) ** 2
    / (f_bus_meas - f_common * pull) / 1e6
)
print(J_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

plot_curves(
    J_df["Delta (GHz)"],
    (J_df["J CPW (MHz)"], "CPW bus", {"color": MODEL_COLORS["cpw"], "fmt": "o"}),
    (J_df["J LOM (MHz)"], "Optimized LOM bus",
     {"color": MODEL_COLORS["optimized"], "fmt": "s", "mfc": "none"}),
    (J_df["J pred (MHz)"], "dispersive $g^2/\\Delta$", {"color": "0.4", "ls": "--"}),
    yscale="log",
    xlabel=axis_label("Qubit-bus detuning", r"\Delta", "GHz"),
    ylabel=axis_label("Exchange coupling", "J", "MHz"),
)
"""),

    md(r"""
Deep in the dispersive regime the three agree; as $\Delta$ shrinks the
measured points peel away from $g^2/\Delta$ — the perturbative formula is
running out of validity while the network results (CPW *and* its LOM) keep
the full hybridization. That is the practical division of labor: use the
dispersive formula for intuition, use the network for numbers.

---

**Where to go from here.** Make the qubits asymmetric ($C_q$, couplers) and
check $J = g_1 g_2(1/\Delta_1 + 1/\Delta_2)/2$; replace the single bus by the
three-resonator chain of Tutorial 3; or hang the bus off a feedline as in
Tutorials 8-9. And remember what the linear model *cannot* give you: ZZ
coupling and $\chi$ need the transmon's anharmonicity, not just this
network.
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

    print("Executing notebook (light adaptive windows; a few minutes)...",
          flush=True)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(DEMOS)}},
    )
    try:
        client.execute()
        print("execute() finished", flush=True)
    except Exception as exc:
        print(f"execute() raised: {type(exc).__name__}: {exc}", flush=True)

    nb.cells.pop(0)

    out = DEMOS / BASENAME
    with out.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {out}", flush=True)

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
