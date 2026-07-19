# Demos — end-to-end case studies

The [tutorials](../tutorials) each teach **one concept** on the smallest system
that shows it. The demos go the other way: each one assembles a **realistic
device** out of those concepts — build a large network, fit LOMs to its
distributed pieces, validate the lumped reassembly against the CPW ground
truth, and sweep a design parameter. They are meant to be copied and
re-parameterised, not worked through in order.

| Notebook | System | What it computes |
|---|---|---|
| [purcell_readout.ipynb](purcell_readout.ipynb) | qubit (LC) → readout resonator → Purcell filter → feedline | environment admittance `Re Y(ω)` seen by the qubit, Purcell `T1` with vs. without the filter at matched readout κ, LOM-vs-CPW `T1` curves, readout-κ / protection design sweep |
| [two_qubit_bus.ipynb](two_qubit_bus.ipynb) | qubit (LC) → bus resonator → qubit (LC) | vacuum-Rabi calibration of `g`, qubit–qubit avoided crossing (`2J`), `J(Δ)` sweep vs. the dispersive formula, all with a CPW bus and again with its fitted LOM |

Conventions match the tutorials:

* each notebook has a `builder_*.py` that regenerates it (`python
  builder_purcell_readout.py`); the committed notebook is authoritative and
  the builder mirrors its cells,
* figures follow the shared style in `simpleLOMs.plotting`,
* the qubits are **linear LC tanks**: frequencies, couplings (`g`, `J`), κ and
  Purcell `T1` are all linear-network quantities and carry over to transmons at
  leading order, but anharmonicity-dependent quantities (χ, ZZ, photon-number
  effects) are outside what these models can give you — each demo states this
  where it matters.

Demos are allowed to run longer than tutorials (a few minutes each; the
longest parts are parameter sweeps that rebuild the network per point).
