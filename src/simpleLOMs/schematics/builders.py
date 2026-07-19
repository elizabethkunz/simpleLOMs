"""
schematics/builders.py
----------------------
Turn the *same parameters* that the network builders in
:mod:`simpleLOMs.networks` take into a renderable :class:`Schematic`.

These "mirror-builders" intentionally re-state each fixed topology (the same
chain documented in the ``networks/*.py`` docstrings) rather than introspecting
the returned ``rf.Network`` — the L/C values are baked into skrf's S-matrices
and cannot be recovered cleanly, so the schematic is built directly from the
physical parameters the user passed.

Convention (matches the skrf elements)
--------------------------------------
* ``coupling_capacitor`` -> ``SeriesImpedance``  -> a **series** cap on the rail
* ``shunt_capacitor``    -> ``ShuntAdmittance``  -> a **shunt** cap to ground
* ``lc_resonator``       -> ``ShuntAdmittance``  -> a **shunt** parallel LC tank
* ``cpw.line``           -> a **series** transmission line
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from simpleLOMs.schematics.netlist import Component, Net, Schematic, format_eng


def _auto_label(cid: str) -> str:
    """Derive a LaTeX-style label from a component id (e.g. ``Cc1`` -> ``C_{c1}``)."""
    if cid.startswith("Ctog"):
        return f"C_{{tog{cid[4:]}}}"
    if cid.startswith("Cc") and cid[2:].isdigit():
        return f"C_{{c{cid[2:]}}}"
    if cid.startswith("cc_port"):
        return f"C_{{p{cid[7:]}}}"
    if re.fullmatch(r"P\d+", cid):
        return f"Port {cid[1:]}"
    return cid


# --------------------------------------------------------------------------- #
# Ladder builder
# --------------------------------------------------------------------------- #
class SchematicBuilder:
    """Accumulate a ladder network left-to-right and emit a :class:`Schematic`.

    Series elements (``port``, ``series_cap``, ``tline``) advance the rail;
    shunt elements (``shunt_cap``, ``shunt_lc``) hang to ground at the current
    node.  Nets are tracked for interchange but layout only needs component
    order + ``orient``.
    """

    def __init__(self, title: str, description: str = "", source: str = "",
                 annotations: Optional[Dict[str, Any]] = None,
                 groups: Optional[list] = None):
        self.meta: Dict[str, Any] = {
            "title": title,
            "description": description,
            "source": source,
            "units": {
                "capacitance": "F", "inductance": "H",
                "length": "m", "impedance": "ohm", "frequency": "Hz",
            },
        }
        self.components: list[Component] = []
        self.nets: list[Net] = []
        self.annotations: Dict[str, Any] = dict(annotations or {})
        self.groups: list = list(groups or [])
        self._node: Optional[str] = None   # current rail node as "id.port"
        self._n_net = 0

    # -- net bookkeeping --------------------------------------------------- #
    def _net(self, a: str, b: str) -> None:
        self._n_net += 1
        self.nets.append(Net(id=f"n{self._n_net}", ports=[a, b]))

    # -- series elements --------------------------------------------------- #
    def port(self, id: str, label: Optional[str] = None, Z0: float = 50.0,
             group: Optional[str] = None) -> "SchematicBuilder":
        first = self._node is None
        pname = "out" if first else "in"
        self.components.append(Component(
            id=id, type="port", label=label or _auto_label(id),
            value=f"{Z0:.2f} \\Omega", ports=[pname], group=group,
            orient="series", props={"Z0": Z0},
        ))
        if first:
            self._node = f"{id}.out"
        else:
            self._net(self._node, f"{id}.in")
            self._node = None
        return self

    def open(self, id: str = "OPEN", label: str = "open") -> "SchematicBuilder":
        self.components.append(Component(
            id=id, type="open", label=label, ports=["in"], orient="series"))
        if self._node is not None:
            self._net(self._node, f"{id}.in")
        self._node = None
        return self

    def series_cap(self, id: str, C: float, label: Optional[str] = None,
                   group: Optional[str] = None) -> "SchematicBuilder":
        self.components.append(Component(
            id=id, type="cap", label=label or _auto_label(id), value=format_eng(C, "F"),
            ports=["a", "b"], group=group, orient="series", props={"C": C}))
        self._net(self._node, f"{id}.a")
        self._node = f"{id}.b"
        return self

    def tline(self, id: str, length: float, Z0: Optional[float] = None,
              label: str = "CPW", group: Optional[str] = None) -> "SchematicBuilder":
        # Display length only; Z0 is kept in props but deliberately not drawn.
        val = f"\\ell = {format_eng(length, 'm')}"
        self.components.append(Component(
            id=id, type="tline", label=label, value=val,
            ports=["in", "out"], group=group, orient="series",
            props={"length": length, "Z0": Z0}))
        self._net(self._node, f"{id}.in")
        self._node = f"{id}.out"
        return self

    # -- shunt elements ---------------------------------------------------- #
    def shunt_cap(self, id: str, C: float, label: Optional[str] = None,
                  group: Optional[str] = None) -> "SchematicBuilder":
        self.components.append(Component(
            id=id, type="cap", label=label or _auto_label(id), value=format_eng(C, "F"),
            ports=["a", "gnd"], group=group, orient="shunt", props={"C": C}))
        self._net(self._node, f"{id}.a")
        self._net(f"{id}.gnd", "GND")
        return self

    def shunt_lc(self, id: str, L: float, C: float, label: Optional[str] = None,
                 group: Optional[str] = None) -> "SchematicBuilder":
        val = f"L = {format_eng(L, 'H')}\nC = {format_eng(C, 'F')}"
        self.components.append(Component(
            id=id, type="lc", label=label or id, value=val,
            ports=["a", "gnd"], group=group, orient="shunt",
            props={"L": L, "C": C}))
        self._net(self._node, f"{id}.a")
        self._net(f"{id}.gnd", "GND")
        return self

    def shunt_branch(self, id: str, stack: list,
                     group: Optional[str] = None) -> "SchematicBuilder":
        """A shunt leg holding *several* stacked elements (rail -> ground).

        ``stack`` is an ordered list of element dicts (top-of-leg first), each
        ``{"type", "label", "value", ...}`` with ``type`` in
        ``{"cap", "lc", "tline"}``.  Drawn by
        :func:`simpleLOMs.schematics.render_svg._sym_branch_shunt`.  This is the
        hanger / notch resonator leg: a coupling cap stacked above the resonator.
        """
        self.components.append(Component(
            id=id, type="branch", label="", value="",
            ports=["a", "gnd"], group=group, orient="shunt",
            props={"stack": list(stack)}))
        self._net(self._node, f"{id}.a")
        self._net(f"{id}.gnd", "GND")
        return self

    # -- finish ------------------------------------------------------------ #
    def build(self) -> Schematic:
        return Schematic(
            meta=self.meta, groups=self.groups,
            components=self.components, nets=self.nets,
            annotations=self.annotations,
        )


# --------------------------------------------------------------------------- #
# CPW characteristic impedance (optional, lazy skrf)
# --------------------------------------------------------------------------- #
def _cpw_zc(freq, cpw_params) -> Optional[float]:
    """Characteristic impedance of a CPW line, or ``None`` if unavailable."""
    if freq is None or cpw_params is None:
        return None
    try:
        from skrf.media import CPW
        cpw = CPW(
            freq, w=cpw_params.w, s=cpw_params.s, t=cpw_params.t,
            h=cpw_params.h, rho=cpw_params.rho, ep_r=cpw_params.ep_r,
            has_metal_backside=cpw_params.has_metal_backside, tand=cpw_params.tand,
        )
        z0c = cpw.z0_characteristic
        return float(z0c[len(z0c) // 2].real)
    except Exception:
        return None


def _base(title, annotations, source, groups=None):
    return SchematicBuilder(
        title=title if title is not None else "",
        source=source,
        description="",
        annotations=annotations,
        groups=groups or [{"id": "resonator", "label": "resonator"}],
    )


# --------------------------------------------------------------------------- #
# CPW schematics (mirror simpleLOMs.networks.cpw)
# --------------------------------------------------------------------------- #
def cpw_schematic(d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=None, freq=None, Z0=50,
                  annotations=None, title=None) -> Schematic:
    """1-port CPW: P1 — Cc1 — Ctog1 — CPW — Ctog2 — Cc2(shunt) — open."""
    zc = _cpw_zc(freq, cpw_params)
    b = _base(title, annotations, "simpleLOMs.networks.cpw.cpw_resonator_network")
    (b.port("P1", Z0=Z0)
      .series_cap("Cc1", Cc1, group="resonator")
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .tline("TL", d, Z0=zc, group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .shunt_cap("Cc2", Cc2)
      .open())
    return b.build()


def cpw_schematic_2port(d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=None, freq=None,
                        Z0=50, annotations=None,
                        title=None) -> Schematic:
    """2-port CPW: P1 — Cc1 — Ctog1 — CPW — Ctog2 — Cc2 — P2."""
    zc = _cpw_zc(freq, cpw_params)
    b = _base(title, annotations, "simpleLOMs.networks.cpw.cpw_resonator_network_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("Cc1", Cc1)
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .tline("TL", d, Z0=zc, group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .series_cap("Cc2", Cc2)
      .port("P2", Z0=Z0))
    return b.build()


def cpw_loaded_schematic_2port(d, Cc1, Cc2, Ctog1, Ctog2,
                               Lload1, Cload1, Lload2, Cload2,
                               cpw_params=None, freq=None, Z0=50,
                               cc_port=1e-15,
                               annotations=None,
                               title=None) -> Schematic:
    """2-port loaded CPW with an LC tank on each port side."""
    zc = _cpw_zc(freq, cpw_params)
    b = _base(title, annotations, "simpleLOMs.networks.cpw.cpw_resonator_loaded_network_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("cc_port1", cc_port)
      .shunt_lc("load1", Lload1, Cload1, label="load 1", group="load")
      .series_cap("Cc1", Cc1)
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .tline("TL", d, Z0=zc, group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .series_cap("Cc2", Cc2)
      .shunt_lc("load2", Lload2, Cload2, label="load 2", group="load")
      .series_cap("cc_port2", cc_port)
      .port("P2", Z0=Z0))
    return b.build()


def cpw_single_load_schematic_2port(d, Cc1, Ctog1, Ctog2, Lload1, Cload1,
                                    cpw_params=None, freq=None, Z0=50,
                                    annotations=None,
                                    title=None) -> Schematic:
    """2-port CPW with a single LC load on the port-1 side."""
    zc = _cpw_zc(freq, cpw_params)
    b = _base(title, annotations, "simpleLOMs.networks.cpw.cpw_resonator_single_load_network_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("cc_port1", 1e-15)
      .shunt_lc("load1", Lload1, Cload1, label="load 1", group="load")
      .series_cap("Cc1", Cc1)
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .tline("TL", d, Z0=zc, group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .series_cap("cc_port2", 1e-15)
      .port("P2", Z0=Z0))
    return b.build()


# --------------------------------------------------------------------------- #
# Segment append helpers (shared by chain / hanger / hybrid mirrors)
# --------------------------------------------------------------------------- #
def _normalize_chain_args(segments, Ccs, Ctogs):
    n = len(segments)
    if len(Ccs) != n + 1:
        raise ValueError(f"Ccs must have {n + 1} entries, got {len(Ccs)}.")
    if Ctogs is None:
        Ctogs = [0.0] * (2 * n)
    if len(Ctogs) != 2 * n:
        raise ValueError(f"Ctogs must have {2 * n} entries, got {len(Ctogs)}.")
    return n, Ctogs


def _normalize_hanger_args(segments, Ccs, feed_lengths):
    n = len(segments)
    if len(Ccs) != n:
        raise ValueError(f"Ccs must have one entry per segment ({n}), got {len(Ccs)}.")
    if feed_lengths is not None and len(feed_lengths) != n + 1:
        raise ValueError(f"feed_lengths must have {n + 1} entries, got {len(feed_lengths)}.")
    return n


def _append_chain_segments(b: SchematicBuilder, segments, Ccs, Ctogs, zc=None,
                           id_prefix: str = "") -> None:
    """Append an inline capacitively-coupled chain onto ``b`` (no ports)."""
    n, Ctogs = _normalize_chain_args(segments, Ccs, Ctogs)
    p = id_prefix
    for i, seg in enumerate(segments):
        g = f"res{i + 1}"
        b.series_cap(f"{p}Cc{i + 1}", Ccs[i])
        if Ctogs[2 * i]:
            b.shunt_cap(f"{p}Ctog{2 * i + 1}", Ctogs[2 * i], group=g)
        if seg["kind"] == "cpw":
            b.tline(f"{p}TL{i + 1}", seg["d"], Z0=zc, label=f"CPW {i + 1}", group=g)
        elif seg["kind"] == "lc":
            b.shunt_lc(f"{p}LC{i + 1}", seg["L"], seg["C"],
                       label=f"LC {i + 1}", group=g)
        else:
            raise ValueError(f"segment {i}: kind must be 'cpw' or 'lc'.")
        if Ctogs[2 * i + 1]:
            b.shunt_cap(f"{p}Ctog{2 * i + 2}", Ctogs[2 * i + 1], group=g)
    b.series_cap(f"{p}Cc{n + 1}", Ccs[n])


def _append_hanger_segments(b: SchematicBuilder, segments, Ccs,
                            feed_lengths=None, zc=None,
                            id_prefix: str = "") -> None:
    """Append hanger / notch tap branches onto ``b`` (no ports)."""
    n = _normalize_hanger_args(segments, Ccs, feed_lengths)
    p = id_prefix
    for i, seg in enumerate(segments):
        g = f"hang{i + 1}"
        if feed_lengths and feed_lengths[i]:
            b.tline(f"{p}feed{i}", feed_lengths[i], Z0=zc, label="feed")
        cap_el = {"type": "cap", "label": _auto_label(f"Cc{i + 1}"),
                  "value": format_eng(Ccs[i], "F")}
        if seg["kind"] == "lc":
            res_el = {"type": "lc", "label": f"LC {i + 1}",
                      "value": f"L = {format_eng(seg['L'], 'H')}\n"
                               f"C = {format_eng(seg['C'], 'F')}"}
        elif seg["kind"] == "cpw":
            term = seg.get("termination", "short")
            res_el = {"type": "tline", "label": f"CPW {i + 1}",
                      "value": f"\\ell = {format_eng(seg['d'], 'm')} ({term})"}
        else:
            raise ValueError(f"segment {i}: kind must be 'lc' or 'cpw'.")
        b.shunt_branch(f"{p}H{i + 1}", stack=[cap_el, res_el], group=g)
    if feed_lengths and feed_lengths[n]:
        b.tline(f"{p}feed{n}", feed_lengths[n], Z0=zc, label="feed")


# --------------------------------------------------------------------------- #
# Resonator-chain schematic (mirror simpleLOMs.networks.chain)
# --------------------------------------------------------------------------- #
def resonator_chain_schematic_2port(segments, Ccs, Ctogs=None, cpw_params=None,
                                    freq=None, Z0=50, annotations=None,
                                    title=None) -> Schematic:
    """Chain of capacitively-coupled resonators (mirrors
    :func:`simpleLOMs.networks.chain.resonator_chain_network_2port`).

    ``segments`` is a list of ``{"kind": "cpw", "d": ...}`` or
    ``{"kind": "lc", "L": ..., "C": ...}`` dicts, ``Ccs`` has ``len+1`` entries
    and ``Ctogs`` (optional) has ``2*len`` entries; a ``0``/``None`` ground cap
    is omitted, so pass ``Ctogs=None`` for a lumped chain whose ``L, C`` already
    absorb the grounds.  Each resonator is drawn as its own labelled group.
    """
    n, Ctogs = _normalize_chain_args(segments, Ccs, Ctogs)
    zc = _cpw_zc(freq, cpw_params)
    groups = [{"id": f"res{i + 1}", "label": f"resonator {i + 1}"} for i in range(n)]
    b = SchematicBuilder(
        title=title if title is not None else "",
        source="simpleLOMs.networks.chain.resonator_chain_network_2port",
        annotations=annotations, groups=groups,
    )
    b.port("P1", Z0=Z0)
    _append_chain_segments(b, segments, Ccs, Ctogs, zc=zc)
    b.port("P2", Z0=Z0)
    return b.build()


# --------------------------------------------------------------------------- #
# Hanger / notch schematic (mirror simpleLOMs.networks.hanger)
# --------------------------------------------------------------------------- #
def hanger_resonator_schematic_2port(segments, Ccs, feed_lengths=None,
                                     cpw_params=None, freq=None, Z0=50,
                                     annotations=None,
                                     title=None) -> Schematic:
    """Resonators hanging off a through feedline (mirrors
    :func:`simpleLOMs.networks.hanger.hanger_resonator_network_2port`).

    ``segments`` is a list of ``{"kind": "lc", "L": ..., "C": ...}`` or
    ``{"kind": "cpw", "d": ..., "termination": "short"|"open"}`` dicts; ``Ccs``
    has one tap coupling cap per segment.  Each hanger is drawn as a shunt
    branch: the tap cap stacked above the resonator, hanging to ground.
    """
    n = _normalize_hanger_args(segments, Ccs, feed_lengths)
    zc = _cpw_zc(freq, cpw_params)
    groups = [{"id": f"hang{i + 1}", "label": f"hanger {i + 1}"} for i in range(n)]
    b = SchematicBuilder(
        title=title if title is not None else "",
        source="simpleLOMs.networks.hanger.hanger_resonator_network_2port",
        annotations=annotations, groups=groups,
    )
    b.port("P1", Z0=Z0)
    _append_hanger_segments(b, segments, Ccs, feed_lengths=feed_lengths, zc=zc)
    b.port("P2", Z0=Z0)
    return b.build()


# --------------------------------------------------------------------------- #
# Hybrid: inline chain cascaded with hangers (one continuous rail)
# --------------------------------------------------------------------------- #
def hybrid_schematic_2port(inline_segments, inline_Ccs, hanger_segments, hanger_Ccs,
                           inline_Ctogs=None, feed_lengths=None,
                           cpw_params=None, freq=None, Z0=50,
                           annotations=None, title=None) -> Schematic:
    """Inline (series) chain cascaded with hanger taps on one rail.

    Mirrors the hybrid cascade used when an inline Purcell / chain network is
    joined to a hanger feedline (``inline ** hangers`` in scikit-rf).  Draws

        [P1] -- Cc -- … -- CPW/LC chain -- Cc -- hang1 -- hang2 -- … -- [P2]

    Parameters match the corresponding pieces of
    :func:`resonator_chain_schematic_2port` and
    :func:`hanger_resonator_schematic_2port`.
    """
    n_in, inline_Ctogs = _normalize_chain_args(
        inline_segments, inline_Ccs, inline_Ctogs)
    n_h = _normalize_hanger_args(hanger_segments, hanger_Ccs, feed_lengths)
    zc = _cpw_zc(freq, cpw_params)
    groups = (
        [{"id": f"res{i + 1}", "label": f"resonator {i + 1}"} for i in range(n_in)]
        + [{"id": f"hang{i + 1}", "label": f"hanger {i + 1}"} for i in range(n_h)]
    )
    b = SchematicBuilder(
        title=title if title is not None else "",
        source="simpleLOMs.schematics.builders.hybrid_schematic_2port",
        annotations=annotations, groups=groups,
    )
    b.port("P1", Z0=Z0)
    _append_chain_segments(b, inline_segments, inline_Ccs, inline_Ctogs, zc=zc)
    _append_hanger_segments(
        b, hanger_segments, hanger_Ccs, feed_lengths=feed_lengths, zc=zc,
    )
    b.port("P2", Z0=Z0)
    return b.build()


# --------------------------------------------------------------------------- #
# LC schematics (mirror simpleLOMs.networks.lc)
# --------------------------------------------------------------------------- #
def lc_schematic(Leff, Ceff, Cc1, Cc2, Z0=50, annotations=None,
                 title=None) -> Schematic:
    """1-port LC: P1 — Cc1 — [LC] — Cc2(shunt) — open."""
    b = _base(title, annotations, "simpleLOMs.networks.lc.lc_resonator_network")
    (b.port("P1", Z0=Z0)
      .series_cap("Cc1", Cc1)
      .shunt_lc("LC", Leff, Ceff, label="LC", group="resonator")
      .shunt_cap("Cc2", Cc2)
      .open())
    return b.build()


def lc_schematic_2port(Leff, Ceff, Cc1, Cc2, Z0=50, annotations=None,
                       title=None) -> Schematic:
    """2-port LC: P1 — Cc1 — [LC] — Cc2 — P2."""
    b = _base(title, annotations, "simpleLOMs.networks.lc.lc_resonator_network_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("Cc1", Cc1)
      .shunt_lc("LC", Leff, Ceff, label="LC", group="resonator")
      .series_cap("Cc2", Cc2)
      .port("P2", Z0=Z0))
    return b.build()


def lc_with_grounds_schematic_2port(Leff, Ceff, Cc1, Cc2, Ctog1, Ctog2, Z0=50,
                                    annotations=None,
                                    title=None) -> Schematic:
    """2-port LC: P1 — Cc1 — Ctog1 — [LC] — Ctog2 — Cc2 — P2."""
    b = _base(title, annotations, "simpleLOMs.networks.lc.lc_resonator_network_with_grounds_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("Cc1", Cc1)
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .shunt_lc("LC", Leff, Ceff, label="LC", group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .series_cap("Cc2", Cc2)
      .port("P2", Z0=Z0))
    return b.build()


def lc_loaded_schematic_2port(Leff, Ceff, Cc1, Cc2, Lload1, Cload1, Lload2, Cload2,
                              Z0=50, annotations=None,
                              title=None) -> Schematic:
    """2-port loaded LC with an LC tank on each port side."""
    b = _base(title, annotations, "simpleLOMs.networks.lc.lc_resonator_loaded_network_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("cc_port1", 1e-15)
      .shunt_lc("load1", Lload1, Cload1, label="load 1", group="load")
      .series_cap("Cc1", Cc1)
      .shunt_lc("LC", Leff, Ceff, label="LC", group="resonator")
      .series_cap("Cc2", Cc2)
      .shunt_lc("load2", Lload2, Cload2, label="load 2", group="load")
      .series_cap("cc_port2", 1e-15)
      .port("P2", Z0=Z0))
    return b.build()


def lc_loaded_with_grounds_schematic_2port(Leff, Ceff, Cc1, Cc2, Ctog1, Ctog2,
                                           Lload1, Cload1, Lload2, Cload2,
                                           Z0=50, annotations=None,
                                           title=None) -> Schematic:
    """Full 2-port LC: loads, coupling caps, ground caps, and the LC tank."""
    b = _base(title, annotations, "simpleLOMs.networks.lc.lc_resonator_loaded_network_with_grounds_2port")
    (b.port("P1", Z0=Z0)
      .series_cap("cc_port1", 1e-15)
      .shunt_lc("load1", Lload1, Cload1, label="load 1", group="load")
      .series_cap("Cc1", Cc1)
      .shunt_cap("Ctog1", Ctog1, group="resonator")
      .shunt_lc("LC", Leff, Ceff, label="LC", group="resonator")
      .shunt_cap("Ctog2", Ctog2, group="resonator")
      .series_cap("Cc2", Cc2)
      .shunt_lc("load2", Lload2, Cload2, label="load 2", group="load")
      .series_cap("cc_port2", 1e-15)
      .port("P2", Z0=Z0))
    return b.build()
