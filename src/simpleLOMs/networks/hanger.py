"""
networks/hanger.py
------------------
Network builder for **hanger / notch resonators** — resonators that hang off a
*continuous* through feedline, rather than sitting inline on the rail the way the
resonators in :mod:`simpleLOMs.networks.chain` do.

Topology::

    [P1] ───┬──────────────┬───────────── [P2]      ← through feedline
            │              │
           Cc[0]          Cc[1]                       ← tap coupling caps
            │              │
         [seg 0]        [seg 1]                       ← resonator, to ground
            │              │
           ─┴─            ─┴─   (ground)

The signal travels *past* each resonator, so each one puts a notch (dip) in the
transmission ``S21`` at its resonance instead of a peak.

Each hanging resonator is one of

* ``{"kind": "lc",  "L": <H>, "C": <F>}``
      — a lumped parallel-LC tank (the **LOM** branch).  Built with
        :func:`simpleLOMs.elements.hanger_lc_branch`.  For best accuracy, obtain
        ``L, C`` from :class:`~simpleLOMs.models.hanger_optimized_fit.HangerOptimizedFit`
        (or :func:`~simpleLOMs.system.fit_lom_hanger`), which fits against a CPW
        hanger.  Rewiring an *inline* ``OptimizedFit`` ``(L, C)`` into a hanger
        is a convenient approximation (few-percent notch error), not an identity.
* ``{"kind": "cpw", "d": <length_m>, "termination": "short" | "open"}``
      — a distributed CPW line (the **ground-truth** branch).  A short-terminated
        line is the usual λ/4 hanger; an open-terminated line is the λ/2 variant.

Electrically each branch (tap ``Cc`` in series with the grounded resonator)
collapses to a single ``ShuntAdmittance``, so a hanger drops straight into the
same ladder assembly the rest of the package already uses.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import skrf as rf

from simpleLOMs.elements import hanger_lc_branch
from simpleLOMs.networks.cpw import _make_cpw_media
from simpleLOMs.params import CPWParams


def _cpw_hanger_branch(
    freq: rf.Frequency,
    d: float,
    Cc: float,
    cpw_params: CPWParams,
    termination: str,
    Z0: float,
    name: str,
) -> rf.Network:
    """
    A CPW line terminated (``"short"`` or ``"open"``) at its far end and coupled
    to the rail through a series capacitor ``Cc``, collapsed to a shunt branch.

    The terminated line is solved for its input impedance ``Z_in``; the whole
    branch is then a single ``ShuntAdmittance`` with ``Y = 1 / (Z_Cc + Z_in)``.
    """
    cpw = _make_cpw_media(freq, cpw_params)
    line = cpw.line(d=d, unit="m", name=f"{name}_line")

    if termination == "short":
        term = rf.Circuit.Ground(freq, name=f"{name}_gnd", z0=Z0)
    elif termination == "open":
        term = rf.Circuit.Open(freq, name=f"{name}_open", z0=Z0)
    else:
        raise ValueError(
            f"termination must be 'short' or 'open', got {termination!r}."
        )

    port = rf.Circuit.Port(freq, name=f"{name}_p", z0=Z0)
    one_port = rf.Circuit(
        [[(port, 0), (line, 0)], [(line, 1), (term, 0)]],
        name=f"{name}_1port",
    ).network

    Zin = one_port.z[:, 0, 0]
    Zcc = 1 / (1j * freq.w * Cc)
    return rf.Circuit.ShuntAdmittance(
        frequency=freq, name=name, z0=Z0, Y=1 / (Zcc + Zin)
    )


def _branch(freq, seg, Cc, cpw_params, Z0, name):
    """Dispatch a single hanger segment dict to its branch builder."""
    kind = seg["kind"]
    if kind == "lc":
        return hanger_lc_branch(
            Cc=Cc, L=seg["L"], C=seg["C"], freq=freq, name=name, Z0=Z0
        )
    if kind == "cpw":
        return _cpw_hanger_branch(
            freq=freq,
            d=seg["d"],
            Cc=Cc,
            cpw_params=cpw_params,
            termination=seg.get("termination", "short"),
            Z0=Z0,
            name=name,
        )
    raise ValueError(f"segment kind must be 'lc' or 'cpw', got {kind!r}.")


def hanger_resonator_network_2port(
    freq: rf.Frequency,
    segments: Sequence[Dict[str, Any]],
    Ccs: Sequence[float],
    feed_lengths: Optional[Sequence[float]] = None,
    cpw_params: CPWParams = None,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port through feedline with one or more resonators hanging off it.

    Parameters
    ----------
    freq : rf.Frequency
    segments : sequence of dict
        One dict per hanging resonator.  Each is either

        * ``{"kind": "lc",  "L": <H>, "C": <F>}`` — a lumped LC tank, or
        * ``{"kind": "cpw", "d": <length_m>, "termination": "short"|"open"}``
          — a distributed CPW line (``termination`` defaults to ``"short"``).

    Ccs : sequence of float
        Tap coupling capacitances in Farads, **one per segment**
        (``len(Ccs) == len(segments)``).
    feed_lengths : sequence of float, optional
        CPW feedline lengths in metres, ``len(segments) + 1`` of them: one before
        the first tap, one between each pair of taps, and one after the last.  A
        value of ``0`` (or ``None`` for the whole list) makes that section an ideal
        wire — ``feed_lengths=None`` gives a fully ideal feedline (a single-hanger
        notch demo needs nothing more).
    cpw_params : CPWParams, optional
        CPW geometry for the ``"cpw"`` branches and any non-zero feedline sections.
        Defaults to ``CPWParams()``.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port transmission network for the feedline plus its hangers.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    segments = list(segments)
    n = len(segments)
    if n == 0:
        raise ValueError("segments must contain at least one resonator.")
    if len(Ccs) != n:
        raise ValueError(
            f"Ccs must have one entry per segment ({n}), got {len(Ccs)}."
        )
    if feed_lengths is None:
        feed_lengths = [0.0] * (n + 1)
    if len(feed_lengths) != n + 1:
        raise ValueError(
            f"feed_lengths must have len(segments)+1 = {n + 1} entries, "
            f"got {len(feed_lengths)}."
        )

    cpw = _make_cpw_media(freq, cpw_params)

    # Build the rail as a flat list of series-connected 2-ports: feedline
    # sections (series lines) interleaved with the hanging branches
    # (ShuntAdmittance pass-throughs).
    rail: List[rf.Network] = []

    def _add_feed(k: int):
        if feed_lengths[k]:
            rail.append(cpw.line(d=feed_lengths[k], unit="m", name=f"feed{k}"))

    for i, seg in enumerate(segments):
        _add_feed(i)
        rail.append(_branch(freq, seg, Ccs[i], cpw_params, Z0, name=f"hang{i}"))
    _add_feed(n)

    port1 = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2 = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [[(port1, 0), (rail[0], 0)]]
    for a, b in zip(rail[:-1], rail[1:]):
        cnx.append([(a, 1), (b, 0)])
    cnx.append([(rail[-1], 1), (port2, 0)])

    return rf.Circuit(cnx, name="hanger_resonator_2port").network
