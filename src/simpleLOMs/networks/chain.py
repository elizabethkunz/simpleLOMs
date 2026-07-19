"""
networks/chain.py
-----------------
Network builder for a **chain of capacitively-coupled resonators** — the
"resonator network" that Tutorial 3 builds.

A single resonator in this project is the ladder

    -- Cc -- Ctog -- [segment] -- Ctog -- Cc --

where ``[segment]`` is either a CPW transmission line (the ground truth) or a
lumped parallel-LC tank (a fitted LOM).  A *chain* strings several of these
together, sharing one coupling capacitor between neighbours:

    [P1] -- Cc0 -- Ctog0 -- seg0 -- Ctog1 -- Cc1 -- Ctog2 -- seg1 -- Ctog3 -- Cc2 -- ... -- CcN -- [P2]

This one builder covers every chain Tutorial 3 needs:

* all-CPW segments  -> the distributed **ground-truth** network,
* all-LC segments   -> the reassembled **lumped** network,
* any mixture       -> swap individual resonators as you like.

Because ``OptimizedFit`` folds the ground capacitance into its *effective*
``L, C`` (see the topology note in Tutorial 2), the lumped chain is built with
**no** ``Ctog`` caps — just pass ``Ctogs=None`` (the default) and only the
coupling capacitors survive between the LC tanks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import skrf as rf

from simpleLOMs.elements import coupling_capacitor, shunt_capacitor, lc_resonator
from simpleLOMs.networks.cpw import _make_cpw_media
from simpleLOMs.params import CPWParams


def resonator_chain_network_2port(
    freq: rf.Frequency,
    segments: Sequence[Dict[str, Any]],
    Ccs: Sequence[float],
    Ctogs: Optional[Sequence[float]] = None,
    cpw_params: CPWParams = None,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port chain of capacitively-coupled resonators.

    Topology::

        [P1] -- Cc[0] -- Ctog[0] -- seg[0] -- Ctog[1] -- Cc[1] -- Ctog[2]
                -- seg[1] -- Ctog[3] -- Cc[2] -- ... -- Cc[n] -- [P2]

    Each segment is flanked by two shunt-to-ground capacitors and separated
    from its neighbours by a shared coupling capacitor.

    Parameters
    ----------
    freq : rf.Frequency
    segments : sequence of dict
        One dict per resonator.  Each is either

        * ``{"kind": "cpw", "d": <length_m>}`` — a CPW transmission line, or
        * ``{"kind": "lc", "L": <H>, "C": <F>}`` — a lumped parallel-LC tank.

    Ccs : sequence of float
        Coupling capacitances in Farads.  Length must be ``len(segments) + 1``
        (one before each segment plus one after the last).
    Ctogs : sequence of float, optional
        Shunt-to-ground capacitances in Farads, two per segment
        (``[left0, right0, left1, right1, ...]``), so length ``2 * n``.  A value
        of ``0`` (or ``None`` for the whole list) omits that ground cap — use
        ``Ctogs=None`` for a lumped chain whose ``L, C`` already absorb the
        grounds (e.g. ``OptimizedFit``).
    cpw_params : CPWParams, optional
        CPW geometry for the ``"cpw"`` segments.  Defaults to ``CPWParams()``.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port transmission network for the whole chain.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    segments = list(segments)
    n = len(segments)
    if n == 0:
        raise ValueError("segments must contain at least one resonator.")
    if len(Ccs) != n + 1:
        raise ValueError(
            f"Ccs must have len(segments)+1 = {n + 1} entries, got {len(Ccs)}."
        )
    if Ctogs is None:
        Ctogs = [0.0] * (2 * n)
    if len(Ctogs) != 2 * n:
        raise ValueError(
            f"Ctogs must have 2*len(segments) = {2 * n} entries, got {len(Ctogs)}."
        )

    cpw = _make_cpw_media(freq, cpw_params)

    # Series coupling capacitors between the ports / resonators.
    ccs = [
        coupling_capacitor(C=Ccs[i], freq=freq, name=f"cc{i}", Z0=Z0)
        for i in range(n + 1)
    ]

    def _ground_cap(value: float, kind: str, name: str):
        """A shunt-to-ground cap, using the CPW media around CPW segments."""
        if kind == "cpw":
            return cpw.shunt_capacitor(C=value, freq=freq, name=name, Z0=Z0)
        return shunt_capacitor(C=value, freq=freq, name=name, Z0=Z0)

    # Build the rail as a flat list of series-connected 2-port elements.
    rail: List[rf.Network] = [ccs[0]]
    for i, seg in enumerate(segments):
        kind = seg["kind"]
        left, right = Ctogs[2 * i], Ctogs[2 * i + 1]
        if left:
            rail.append(_ground_cap(left, kind, f"ctog{2 * i}"))
        if kind == "cpw":
            rail.append(cpw.line(d=seg["d"], unit="m", name=f"cpw{i}"))
        elif kind == "lc":
            rail.append(
                lc_resonator(L=seg["L"], C=seg["C"], freq=freq, name=f"lc{i}", Z0=Z0)
            )
        else:
            raise ValueError(
                f"segment {i}: kind must be 'cpw' or 'lc', got {kind!r}."
            )
        if right:
            rail.append(_ground_cap(right, kind, f"ctog{2 * i + 1}"))
        rail.append(ccs[i + 1])

    port1 = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2 = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [[(port1, 0), (rail[0], 0)]]
    for a, b in zip(rail[:-1], rail[1:]):
        cnx.append([(a, 1), (b, 0)])
    cnx.append([(rail[-1], 1), (port2, 0)])

    return rf.Circuit(cnx, name="resonator_chain_2port").network
