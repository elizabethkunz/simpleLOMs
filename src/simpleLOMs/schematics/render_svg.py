"""
schematics/render_svg.py
------------------------
Render a :class:`~simpleLOMs.schematics.netlist.Schematic` to a clean,
white-background SVG using hand-drawn symbols (no external dependency).

Two things worth noting:

* **LaTeX-style typesetting.**  Labels/values are written in a restricted
  LaTeX subset (``C_{c1}``, ``f_0``, ``Q_L``, ``\\ell``, ``\\Omega``,
  ``\\kappa/2\\pi`` ...) and rendered to native SVG ``<tspan>`` runs — italic
  variables, true subscripts/superscripts, and Greek letters — with no
  JavaScript math engine, so it looks identical in the static SVG and the
  zoomable HTML.
* **Ladder layout.**  Series components sit on a horizontal rail; shunt
  components hang to ground at the node following the preceding series element.
  The CPW transmission line is drawn as a 3-D cylinder / tube.
"""
from __future__ import annotations

import re
from typing import List

from simpleLOMs.schematics.netlist import Component, Schematic


# --------------------------------------------------------------------------- #
# Theme / geometry
# --------------------------------------------------------------------------- #
class Theme:
    bg = "#ffffff"
    wire = "#14181f"
    symbol = "#14181f"
    label = "#14181f"
    value = "#5b6472"
    title = "#14181f"
    subtitle = "#5b6472"
    cyl_top = "#fbfcfd"
    cyl_bot = "#dfe4ea"
    cyl_face = "#eef1f5"
    font = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
    stroke = 2.0


MARGIN = 60
RAIL_Y = 168
GROUND_Y = 286
BASE_GAP = 66
SHUNT_SPACING = 84
SHUNT_PAD = 28

_WIDTHS = {"port": 16, "cap": 30, "tline": 116, "open": 18, "node": 8}


def _bw(c: Component) -> int:
    return _WIDTHS.get(c.type, 30)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------- #
# Minimal LaTeX -> SVG tspan typesetting
# --------------------------------------------------------------------------- #
_GREEK = {
    "omega": "ω", "Omega": "Ω", "kappa": "κ", "lambda": "λ",
    "pi": "π", "mu": "µ", "ell": "ℓ", "approx": "≈",
    "cdot": "·", "times": "×", "Delta": "Δ", "delta": "δ",
    "phi": "φ", "theta": "θ", "epsilon": "ε", "gamma": "γ",
    "quad": " ", "qquad": "  ",
}


def _math_runs(s: str):
    """Parse a restricted LaTeX string into (text, italic, shift, scale) runs."""
    runs = []
    i, n = 0, len(s)

    def add(txt, italic, shift="base", scale=1.0):
        if txt != "":
            runs.append((txt, italic, shift, scale))

    def group(k):  # s[k] == '{' -> (content, index after '}')
        buf, k = "", k + 1
        while k < n and s[k] != "}":
            buf += s[k]
            k += 1
        return buf, k + 1

    def emit_sub(content, shift):
        # single letter -> italic variable; otherwise upright
        italic = content.isalpha() and len(content) == 1
        add(content, italic, shift, 0.72)

    while i < n:
        ch = s[i]
        if ch == "\\":
            j = i + 1
            cmd = ""
            while j < n and s[j].isalpha():
                cmd += s[j]
                j += 1
            if cmd == "":                      # escaped punctuation: \, \  etc.
                nxt = s[j] if j < n else ""
                add(" " if nxt == "," else nxt, False)
                i = j + 1
                continue
            if cmd == "mathrm" and j < n and s[j] == "{":
                buf, j = group(j)
                add(buf, False)
                i = j
                continue
            if cmd in _GREEK:
                add(_GREEK[cmd], False)
                i = j
                continue
            add(cmd, False)                    # unknown command -> upright text
            i = j
            continue
        if ch in "_^":
            shift = "sub" if ch == "_" else "sup"
            if i + 1 < n and s[i + 1] == "{":
                buf, j = group(i + 1)
                emit_sub(buf, shift)
                i = j
            else:
                emit_sub(s[i + 1] if i + 1 < n else "", shift)
                i += 2
            continue
        if ch in "{}":
            i += 1
            continue
        if ch.isalpha():                       # maximal letter run
            j = i
            while j < n and s[j].isalpha():
                j += 1
            word = s[i:j]
            add(word, len(word) == 1)          # single letter -> italic variable
            i = j
            continue
        add(ch, False)
        i += 1
    return runs


def _mathtext(x, y, s, size=12.5, color=Theme.label, anchor="middle", weight="400"):
    spans = []
    for txt, italic, shift, scale in _math_runs(str(s)):
        fs = size * scale
        if shift == "sub":
            bs = f' baseline-shift="{-size * 0.30:.1f}"'
        elif shift == "sup":
            bs = f' baseline-shift="{size * 0.42:.1f}"'
        else:
            bs = ""
        style = "italic" if italic else "normal"
        spans.append(
            f'<tspan{bs} font-size="{fs:.1f}" font-style="{style}">{_esc(txt)}</tspan>')
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{Theme.font}" '
            f'fill="{color}" text-anchor="{anchor}" font-weight="{weight}">'
            f'{"".join(spans)}</text>')


# --------------------------------------------------------------------------- #
# Primitive SVG helpers
# --------------------------------------------------------------------------- #
def _line(x1, y1, x2, y2, w=Theme.stroke, color=Theme.wire):
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{w}" stroke-linecap="round"/>')


def _ground(x, y):
    p = [_line(x, y, x, y + 8)]
    for half, yy in [(20, y + 8), (12, y + 13), (5, y + 18)]:
        p.append(_line(x - half / 2, yy, x + half / 2, yy))
    return "".join(p)


# --------------------------------------------------------------------------- #
# Series symbols (centred on the rail at x=cx)
# --------------------------------------------------------------------------- #
def _sym_port(c: Component, cx: float) -> str:
    p = [f'<circle cx="{cx:.1f}" cy="{RAIL_Y}" r="5.5" fill="{Theme.bg}" '
         f'stroke="{Theme.symbol}" stroke-width="{Theme.stroke}"/>']
    p.append(_mathtext(cx, RAIL_Y - 18, c.label, size=13, weight="600"))
    if c.value:
        p.append(_mathtext(cx, RAIL_Y + 28, c.value, size=11.5, color=Theme.value))
    return "".join(p)


def _sym_cap_series(c: Component, cx: float) -> str:
    gap, plate = 6, 13
    p = [
        _line(cx - _bw(c) / 2, RAIL_Y, cx - gap, RAIL_Y),
        _line(cx + gap, RAIL_Y, cx + _bw(c) / 2, RAIL_Y),
        _line(cx - gap, RAIL_Y - plate, cx - gap, RAIL_Y + plate),
        _line(cx + gap, RAIL_Y - plate, cx + gap, RAIL_Y + plate),
        _mathtext(cx, RAIL_Y - 24, c.label, size=13, weight="600"),
    ]
    if c.value:
        p.append(_mathtext(cx, RAIL_Y + 34, c.value, size=11.5, color=Theme.value))
    return "".join(p)


def _cylinder_h(cx: float, cy: float, half_len: float,
                ry: float = 16, rx: float = 8) -> str:
    """Horizontal 3-D cylinder / tube (open left face, rounded right cap)."""
    xL, xR = cx - half_len, cx + half_len
    yt, yb = cy - ry, cy + ry
    body = (f'<path d="M {xL:.1f} {yt:.1f} L {xR:.1f} {yt:.1f} '
            f'A {rx} {ry} 0 0 1 {xR:.1f} {yb:.1f} L {xL:.1f} {yb:.1f} Z" '
            f'fill="url(#cylBody)" stroke="{Theme.symbol}" '
            f'stroke-width="{Theme.stroke}" stroke-linejoin="round"/>')
    face = (f'<ellipse cx="{xL:.1f}" cy="{cy:.1f}" rx="{rx}" ry="{ry}" '
            f'fill="{Theme.cyl_face}" stroke="{Theme.symbol}" '
            f'stroke-width="{Theme.stroke}"/>')
    return body + face


def _cylinder_v(cx: float, y_top: float, y_bot: float,
                rx: float = 16, ry: float = 8) -> str:
    """Vertical 3-D cylinder / tube (open top face, rounded bottom cap).

    Same styling as the horizontal CPW symbol in :func:`_cylinder_h` /
    :func:`_sym_tline`, oriented for shunt / hanger legs.
    """
    xl, xr = cx - rx, cx + rx
    body = (f'<path d="M {xl:.1f} {y_top:.1f} L {xr:.1f} {y_top:.1f} '
            f'L {xr:.1f} {y_bot:.1f} '
            f'A {rx:.1f} {ry:.1f} 0 0 1 {xl:.1f} {y_bot:.1f} Z" '
            f'fill="url(#cylBodyV)" stroke="{Theme.symbol}" '
            f'stroke-width="{Theme.stroke}" stroke-linejoin="round"/>')
    face = (f'<ellipse cx="{cx:.1f}" cy="{y_top:.1f}" '
            f'rx="{rx:.1f}" ry="{ry:.1f}" '
            f'fill="{Theme.cyl_face}" stroke="{Theme.symbol}" '
            f'stroke-width="{Theme.stroke}"/>')
    return body + face


def _sym_tline(c: Component, cx: float) -> str:
    """A CPW transmission line drawn as a horizontal 3-D cylinder / tube."""
    HW = _WIDTHS["tline"] / 2 - 8
    ry, rx = 16, 8
    xL, xR = cx - HW, cx + HW
    yt, yb = RAIL_Y - ry, RAIL_Y + ry
    leads = _line(cx - _WIDTHS["tline"] / 2, RAIL_Y, xL, RAIL_Y) + \
        _line(xR, RAIL_Y, cx + _WIDTHS["tline"] / 2, RAIL_Y)
    lbl = _mathtext(cx + 4, RAIL_Y - ry - 12, c.label, size=12.5, weight="600")
    val = _mathtext(cx + 4, yb + 22, c.value, size=11.5, color=Theme.value) if c.value else ""
    return leads + _cylinder_h(cx, RAIL_Y, HW, ry=ry, rx=rx) + lbl + val


def _sym_open(c: Component, cx: float) -> str:
    r = 4
    return (_line(cx - _bw(c) / 2, RAIL_Y, cx - r, RAIL_Y)
            + f'<circle cx="{cx:.1f}" cy="{RAIL_Y}" r="{r}" fill="{Theme.bg}" '
              f'stroke="{Theme.symbol}" stroke-width="{Theme.stroke}"/>'
            + _mathtext(cx, RAIL_Y - 16, c.label, size=11.5, color=Theme.value))


# --------------------------------------------------------------------------- #
# Shunt symbols (hang from rail node at x=nx down to ground)
# --------------------------------------------------------------------------- #
def _node_dot(nx):
    return f'<circle cx="{nx:.1f}" cy="{RAIL_Y}" r="2.6" fill="{Theme.wire}"/>'


def _value_lines(x, y, value):
    out = []
    for k, line in enumerate(str(value).split("\n")):
        out.append(_mathtext(x, y + k * 15, line, size=11.5,
                             color=Theme.value, anchor="start"))
    return "".join(out)


def _sym_cap_shunt(c: Component, nx: float) -> str:
    y_top, y_bot = RAIL_Y + 48, RAIL_Y + 60
    p = [
        _node_dot(nx),
        _line(nx, RAIL_Y, nx, y_top),
        _line(nx - 11, y_top, nx + 11, y_top),
        _line(nx - 11, y_bot, nx + 11, y_bot),
        _line(nx, y_bot, nx, GROUND_Y),
        _ground(nx, GROUND_Y),
        _mathtext(nx + 18, y_top - 1, c.label, size=12.5, anchor="start", weight="600"),
    ]
    if c.value:
        p.append(_value_lines(nx + 18, y_top + 15, c.value))
    return "".join(p)


def _vertical_inductor(x, ya, yb) -> str:
    n = 4
    seg = (yb - ya) / n
    d = [f"M {x:.1f} {ya:.1f}"]
    for _ in range(n):
        d.append(f"a {seg/2:.1f} {seg/2:.1f} 0 0 1 0 {seg:.1f}")
    return (f'<path d="{" ".join(d)}" fill="none" stroke="{Theme.symbol}" '
            f'stroke-width="{Theme.stroke}"/>')


def _sym_lc_shunt(c: Component, nx: float) -> str:
    y1, y2 = RAIL_Y + 24, GROUND_Y - 30
    xl, xr = nx - 15, nx + 15
    pmid = (y1 + y2) / 2
    p = [
        _node_dot(nx),
        _line(nx, RAIL_Y, nx, y1),
        _line(xl, y1, xr, y1),
        _line(xl, y2, xr, y2),
        _line(nx, y2, nx, GROUND_Y),
        _ground(nx, GROUND_Y),
        _vertical_inductor(xl, y1, y2),
        _line(xr, y1, xr, pmid - 4),
        _line(xr - 9, pmid - 4, xr + 9, pmid - 4),
        _line(xr - 9, pmid + 4, xr + 9, pmid + 4),
        _line(xr, pmid + 4, xr, y2),
        _mathtext(xr + 18, pmid - 8, c.label, size=12.5, anchor="start", weight="600"),
    ]
    if c.value:
        p.append(_value_lines(xr + 18, pmid + 8, c.value))
    return "".join(p)


def _sym_branch_shunt(c: Component, nx: float, ground_y: float) -> str:
    """A *multi-element* shunt leg hanging from the rail node down to ground.

    Used for hanger / notch resonators, whose leg is a coupling cap stacked
    above the resonator (an LC tank or a CPW line).  ``c.props["stack"]`` is an
    ordered list (rail -> ground) of element dicts, each ``{"type", "label",
    "value", ...}`` with ``type`` in ``{"cap", "lc", "tline"}``.  A ``tline``
    in the stack is drawn as a vertical 3-D cylinder matching the inline CPW
    symbol.  Single-element shunts still go through :func:`_sym_cap_shunt` /
    :func:`_sym_lc_shunt`; this function only draws the stacked case, so
    existing schematics are unchanged.
    """
    stack = c.props.get("stack", [])
    if not stack:
        return _sym_cap_shunt(c, nx)

    y0 = RAIL_Y + 18
    slot = (ground_y - y0) / len(stack)
    p = [_node_dot(nx), _line(nx, RAIL_Y, nx, y0)]

    y = y0
    for elem in stack:
        etype = elem.get("type", "cap")
        yt, yb = y, y + slot
        ymid = (yt + yb) / 2
        label, value = elem.get("label", ""), elem.get("value", "")

        if etype == "lc":
            ry_t, ry_b = yt + 12, yb - 12
            xl, xr = nx - 15, nx + 15
            pmid = (ry_t + ry_b) / 2
            p += [
                _line(nx, yt, nx, ry_t),
                _line(xl, ry_t, xr, ry_t),
                _line(xl, ry_b, xr, ry_b),
                _line(nx, ry_b, nx, yb),
                _vertical_inductor(xl, ry_t, ry_b),
                _line(xr, ry_t, xr, pmid - 4),
                _line(xr - 9, pmid - 4, xr + 9, pmid - 4),
                _line(xr - 9, pmid + 4, xr + 9, pmid + 4),
                _line(xr, pmid + 4, xr, ry_b),
            ]
            lx = xr + 18
        elif etype == "tline":
            # Vertical cylinder matching the inline CPW tube (_sym_tline).
            box_t, box_b = yt + 12, yb - 12
            rx_cyl, ry_cap = 16, 8
            y_top = box_t + ry_cap
            y_bot = box_b - ry_cap
            if y_bot < y_top + 8:
                y_top, y_bot = box_t + 4, box_b - 4
                ry_cap = max(4.0, min(ry_cap, y_top - box_t))
            p += [
                _line(nx, yt, nx, y_top),
                _cylinder_v(nx, y_top, y_bot, rx=rx_cyl, ry=ry_cap),
                _line(nx, y_bot, nx, yb),
            ]
            lx = nx + rx_cyl + 8
        else:  # series cap sitting in the leg
            gap = 6
            p += [
                _line(nx, yt, nx, ymid - gap),
                _line(nx - 11, ymid - gap, nx + 11, ymid - gap),
                _line(nx - 11, ymid + gap, nx + 11, ymid + gap),
                _line(nx, ymid + gap, nx, yb),
            ]
            lx = nx + 18

        if label:
            p.append(_mathtext(lx, ymid - 4, label, size=12.5,
                               anchor="start", weight="600"))
        if value:
            p.append(_value_lines(lx, ymid + 11, value))
        y = yb

    p.append(_ground(nx, ground_y))
    return "".join(p)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def _segment(components: List[Component]):
    segs, cur = [], None
    for c in components:
        if c.orient == "shunt":
            if cur is None:
                cur = {"series": None, "shunts": []}
                segs.append(cur)
            cur["shunts"].append(c)
        else:
            cur = {"series": c, "shunts": []}
            segs.append(cur)
    return segs


def _defs() -> str:
    return (
        '<defs>'
        '<linearGradient id="cylBody" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{Theme.cyl_top}"/>'
        f'<stop offset="1" stop-color="{Theme.cyl_bot}"/>'
        '</linearGradient>'
        '<linearGradient id="cylBodyV" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{Theme.cyl_top}"/>'
        f'<stop offset="1" stop-color="{Theme.cyl_bot}"/>'
        '</linearGradient>'
        '</defs>'
    )


def schematic_to_svg(sch: Schematic, scale: float = 1.0,
                     standalone: bool = True) -> str:
    """Render ``sch`` to an SVG string with a white background."""
    segs = _segment(sch.components)

    # Multi-element shunt legs (hanger branches) need a taller canvas; single
    # shunts keep the original GROUND_Y so existing schematics are unchanged.
    has_branch = any(c.orient == "shunt" and c.type == "branch"
                     for c in sch.components)
    branch_gnd = GROUND_Y + (96 if has_branch else 0)

    # x-position of each series body (widen the gap where shunts sit)
    x = MARGIN
    for seg in segs:
        c = seg["series"]
        bw = _bw(c) if c is not None else 0
        seg["cx"] = x + bw / 2
        seg["right"] = seg["cx"] + bw / 2
        n = len(seg["shunts"])
        # LC tanks and hanger branches carry wide right-hand labels: extra room.
        base = (168 if any(s.type in ("lc", "branch") for s in seg["shunts"])
                else BASE_GAP)
        seg["gap"] = max(base, n * SHUNT_SPACING + 20)
        x = seg["right"] + seg["gap"]
    total_w = (x - segs[-1]["gap"] + _bw(segs[-1]["series"]) / 2 + MARGIN
               if segs else 2 * MARGIN)

    # distribute shunts within each gap
    for i, seg in enumerate(segs):
        shunts = seg["shunts"]
        if not shunts:
            continue
        left = seg["right"] + SHUNT_PAD
        right = (segs[i + 1]["cx"] - _bw(segs[i + 1]["series"]) / 2 - SHUNT_PAD
                 if i + 1 < len(segs) else seg["right"] + BASE_GAP)
        if len(shunts) == 1:
            seg["shunt_x"] = [(left + right) / 2]
        else:
            step = (right - left) / (len(shunts) - 1)
            seg["shunt_x"] = [left + k * step for k in range(len(shunts))]

    parts: List[str] = []

    # rail wires between consecutive series bodies
    series_segs = [s for s in segs if s["series"] is not None]
    for a, b in zip(series_segs, series_segs[1:]):
        parts.append(_line(a["cx"] + _bw(a["series"]) / 2, RAIL_Y,
                           b["cx"] - _bw(b["series"]) / 2, RAIL_Y))

    # series symbols
    for seg in segs:
        c = seg["series"]
        if c is None:
            continue
        draw = {"port": _sym_port, "cap": _sym_cap_series,
                "tline": _sym_tline, "open": _sym_open}.get(c.type, _sym_cap_series)
        parts.append(draw(c, seg["cx"]))

    # shunt symbols
    for seg in segs:
        for c, nx in zip(seg["shunts"], seg.get("shunt_x", [])):
            if c.type == "branch":
                parts.append(_sym_branch_shunt(c, nx, branch_gnd))
            elif c.type == "lc":
                parts.append(_sym_lc_shunt(c, nx))
            else:
                parts.append(_sym_cap_shunt(c, nx))

    body = "".join(parts)

    # header (title only when the user supplied one)
    title = sch.meta.get("title") or ""
    head = []
    sub = _annotation_line(sch)
    if title:
        head.append(
            f'<text x="{MARGIN - 10}" y="36" font-family="{Theme.font}" '
            f'font-size="18" font-weight="700" fill="{Theme.title}" '
            f'text-anchor="start">{_esc(title)}</text>'
        )
        if sub:
            head.append(_mathtext(MARGIN - 10, 58, sub, size=12.5,
                                 color=Theme.subtitle, anchor="start"))
    elif sub:
        head.append(_mathtext(MARGIN - 10, 36, sub, size=12.5,
                             color=Theme.subtitle, anchor="start"))

    width = max(total_w, 440)
    height = branch_gnd + 44
    xmlns = 'xmlns="http://www.w3.org/2000/svg"' if standalone else ""
    return (
        f'<svg {xmlns} width="{width * scale:.0f}" height="{height * scale:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="{Theme.font}">'
        f'{_defs()}'
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{Theme.bg}"/>'
        f'{"".join(head)}{body}</svg>'
    )


def _annotation_line(sch: Schematic) -> str:
    """LaTeX annotation string: f_0 (3 dp), Q_L (1 dp), kappa/2pi."""
    ann = sch.annotations or {}
    bits = []
    f0 = ann.get("f0_GHz")
    if f0 is None and ann.get("f0") is not None:
        f0 = ann["f0"] / 1e9
    if f0 is not None:
        bits.append(f"f_0 = {f0:.3f}\\ \\mathrm{{GHz}}")
    if ann.get("QL") is not None:
        bits.append(f"Q_L = {ann['QL']:.1f}")
    if ann.get("kappa") is not None:
        bits.append(f"\\kappa/2\\pi = {ann['kappa'] / 1e6:.2f}\\ \\mathrm{{MHz}}")
    if bits:
        return "\\quad ".join(bits)
    return sch.meta.get("description", "")
