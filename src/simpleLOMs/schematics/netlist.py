"""
schematics/netlist.py
----------------------
Portable, JSON-serialisable intermediate representation (IR) for circuit
schematics.

This formalises the ad-hoc ``schematic.json`` schema that the
``Quantum Metal to Circuit`` notebook already emits, so anything that reads or
writes that file keeps working.  A :class:`Schematic` is a plain data object:
the network builders in :mod:`simpleLOMs.schematics.builders` produce one, and
the renderers in :mod:`simpleLOMs.schematics.render_svg` /
:mod:`simpleLOMs.schematics.viewer` consume one.

Layout model
------------
Every topology in this project is a *ladder* network: a horizontal signal rail
carrying **series** elements (ports, coupling capacitors, transmission lines),
with **shunt** elements (ground capacitors, LC resonators) hanging down to
ground at the nodes between them.  Each :class:`Component` therefore records an
``orient`` of ``"series"`` or ``"shunt"``; the renderer walks the components in
order and needs no general graph-layout solver.  ``nets`` are still emitted for
interchange/compatibility but are not required for drawing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# --------------------------------------------------------------------------- #
# Value formatting
# --------------------------------------------------------------------------- #
_SI_PREFIXES = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m",
                0: "", 3: "k", 6: "M", 9: "G"}


def format_eng(value: float, unit: str = "", decimals: int = 2) -> str:
    """
    Format a number with an SI prefix, e.g. ``format_eng(5e-15, "F") -> "5.00 fF"``.

    Parameters
    ----------
    value : float
        The physical quantity in base SI units.
    unit : str
        Unit symbol appended after the prefix (e.g. ``"F"``, ``"H"``, ``"m"``).
    decimals : int
        Number of digits after the decimal point (default 2).
    """
    if value == 0 or not math.isfinite(value):
        return f"{0:.{decimals}f} {unit}".strip()
    exp = int(math.floor(math.log10(abs(value)) / 3) * 3)
    exp = max(min(exp, max(_SI_PREFIXES)), min(_SI_PREFIXES))
    mant = value / (10 ** exp)
    mant_str = f"{mant:.{decimals}f}"
    return f"{mant_str} {_SI_PREFIXES[exp]}{unit}".strip()


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    """A single circuit element.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. ``"Cc1"``).
    type : str
        One of ``port | cap | ind | lc | tline | ground | node | open | res``.
    label : str
        Human-readable name drawn next to the symbol.
    value : str
        Pre-formatted display string (e.g. ``"5 fF"``). May be empty.
    ports : list[str]
        Port names on this component (used to build ``nets``).
    group : str | None
        Optional group id for collapsing sub-circuits.
    orient : str
        ``"series"`` (sits on the rail) or ``"shunt"`` (hangs to ground).
    props : dict
        Optional structured values (e.g. ``{"L": 5e-10, "C": 6e-13}``) used by
        the renderer for multi-value symbols such as parallel LC tanks.
    """

    id: str
    type: str
    label: str = ""
    value: str = ""
    ports: List[str] = field(default_factory=list)
    group: Optional[str] = None
    orient: str = "series"
    props: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Net:
    """A connection between component ports."""

    id: str
    ports: List[str] = field(default_factory=list)


@dataclass
class Schematic:
    """A complete, renderable circuit netlist."""

    meta: Dict[str, Any] = field(default_factory=dict)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    components: List[Component] = field(default_factory=list)
    nets: List[Net] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": self.meta,
            "groups": self.groups,
            "components": [asdict(c) for c in self.components],
            "nets": [asdict(n) for n in self.nets],
            "annotations": self.annotations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Schematic":
        comps = []
        for c in d.get("components", []):
            comps.append(Component(
                id=c["id"],
                type=c.get("type", "node"),
                label=c.get("label", ""),
                value=c.get("value", ""),
                ports=list(c.get("ports", [])),
                group=c.get("group"),
                orient=c.get("orient", "series"),
                props=dict(c.get("props", {})),
            ))
        nets = [Net(id=n["id"], ports=list(n.get("ports", [])))
                for n in d.get("nets", [])]
        return cls(
            meta=dict(d.get("meta", {})),
            groups=list(d.get("groups", [])),
            components=comps,
            nets=nets,
            annotations=dict(d.get("annotations", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "Schematic":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "Schematic":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- rendering (lazy imports to avoid hard render deps at import time) -- #
    def to_svg(self, **kwargs) -> str:
        """Render to a standalone SVG string (white background)."""
        from simpleLOMs.schematics.render_svg import schematic_to_svg
        return schematic_to_svg(self, **kwargs)

    def save_svg(self, path: Union[str, Path], **kwargs) -> Path:
        path = Path(path)
        path.write_text(self.to_svg(**kwargs), encoding="utf-8")
        return path

    def to_html(self, **kwargs) -> str:
        """Render to a self-contained, zoomable HTML document."""
        from simpleLOMs.schematics.viewer import schematic_to_html
        return schematic_to_html(self, **kwargs)

    def save_html(self, path: Union[str, Path], **kwargs) -> Path:
        path = Path(path)
        path.write_text(self.to_html(**kwargs), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:  # noqa: N802  (Jupyter hook)
        """Zoomable inline preview in Jupyter (via a sandboxed iframe)."""
        from simpleLOMs.schematics.viewer import schematic_to_iframe
        return schematic_to_iframe(self)
