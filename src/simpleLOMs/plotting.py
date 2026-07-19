"""
plotting.py
-----------

Plotting functions for notebooks or scripts after running fits.

Axis labels follow the convention ``Title, $variable$ (Unit)`` when a
named physical quantity is plotted (e.g. ``Resonant frequency, $f_0$ (GHz)``).
Residual / expression-style labels keep bare math with no title prefix.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skrf as rf
from scipy.signal import find_peaks


# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for figure style
# ─────────────────────────────────────────────────────────────────────────────
# Every plot in simpleLOMs (this module, ``figure_graphs``, the docs-figure
# generator, and the tutorial notebooks) uses the constants, palette, labels,
# and helpers defined here so that all figures share one look. There is no
# longer a separate "fancy" vs. "normal" style.

FIGURE_WIDTH_1COL = 3.375
FIGURE_WIDTH_2COL = 7.057
FIGURE_HEIGHT_1COL_GR = FIGURE_WIDTH_1COL * 2 / (1 + np.sqrt(5))
FIGURE_HEIGHT_2COL_GR = FIGURE_WIDTH_2COL * 2 / (1 + np.sqrt(5))

_PLT_RC = {
    "font.size":           8,
    "figure.titlesize":    "medium",
    "figure.dpi":          300,
    "figure.figsize":      (FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR),
    "axes.titlesize":      "medium",
    "axes.axisbelow":      False,
    "xtick.direction":     "in",
    "xtick.labelsize":     "small",
    "ytick.direction":     "in",
    "ytick.labelsize":     "small",
    "image.interpolation": "none",
}
plt.rcParams.update(_PLT_RC)


def apply_style() -> None:
    """(Re)apply the shared simpleLOMs figure style to ``plt.rcParams``.

    Importing this module already applies the style once. Call this if some
    other code (or a notebook cell) has since clobbered ``rcParams`` and you
    want the standard look back.
    """
    plt.rcParams.update(_PLT_RC)


# ── Canonical method colours / labels / line styles ─────────────────────────
# CPW is the ground-truth reference (not one of the lumped models); the three
# lumped models rank Optimized > Foster > Analytical.
MODEL_COLORS = {
    "cpw":        "#2c7bb6",
    "optimized":  "#d7191c",
    "foster":     "#1a9641",
    "analytical": "#ff7f00",
}

MODEL_LABELS = {
    "cpw":        "CPW",
    "optimized":  "Optimized",
    "foster":     "Foster",
    "analytical": "Analytical",
}

MODEL_LINESTYLES = {
    "cpw":        "-",
    "optimized":  "-",
    "foster":     "--",
    "analytical": ":",
}

# Accent colours (drawn from the model palette) for single-series plots that
# are not tied to a specific method — e.g. an f0/kappa/Q parameter sweep.
COLOR_PRIMARY   = MODEL_COLORS["cpw"]         # blue
COLOR_SECONDARY = MODEL_COLORS["analytical"]  # orange
COLOR_HIGHLIGHT = MODEL_COLORS["optimized"]   # red — target / design points

# Backward-compatible private aliases (older internal references).
_MODEL_COLORS = MODEL_COLORS
_MODEL_LABELS = MODEL_LABELS


def _model_key(label) -> str | None:
    """Resolve a free-form series label to a canonical method key.

    Matches case-insensitively on substring, so ``"Optimized LC"``,
    ``"optimized"`` and ``"CPW (reference)"`` all map correctly. Returns
    ``None`` if no method name is found.
    """
    text = str(label).lower()
    for key in MODEL_COLORS:
        if key in text:
            return key
    return None


def _model_color(label):
    key = _model_key(label)
    return MODEL_COLORS[key] if key else None


def _model_linestyle(label, default: str = "-"):
    key = _model_key(label)
    return MODEL_LINESTYLES[key] if key else default


# ── Label helpers ───────────────────────────────────────────────────────────

def axis_label(title: str, symbol: str | None = None, unit: str | None = None) -> str:
    """
    Format an axis label as ``Title, $symbol$ (Unit)``.

    This is the house convention for every named physical quantity. If
    ``symbol`` is None, returns ``Title (Unit)`` or just ``Title``. Pure
    expression-style labels (e.g. ``Re(S11)``) keep bare math and do not use
    this helper.
    """
    if symbol and unit:
        return f"{title}, ${symbol}$ ({unit})"
    if symbol:
        return f"{title}, ${symbol}$"
    if unit:
        return f"{title} ({unit})"
    return title


# Backward-compatible private alias.
_axlabel = axis_label

# Reusable label constants (all follow the ``Thing, $var$ (Unit)`` convention).
LABEL_FREQ = axis_label("Frequency", "f", "GHz")
LABEL_F0 = axis_label("Resonant frequency", r"f_0", "GHz")
LABEL_OMEGA0 = axis_label("Resonant frequency", r"\omega_0/2\pi", "GHz")
LABEL_KAPPA = axis_label("Linewidth", r"\kappa/2\pi", "MHz")
LABEL_Q = axis_label("Quality factor", "Q")
LABEL_LENGTH = axis_label("Resonator length", "d", "mm")
LABEL_CC = axis_label("Coupling capacitance", r"C_c", "fF")
LABEL_CPW_GAP = axis_label("CPW gap", "s", r"\mu\mathrm{m}")
LABEL_PCT_ERROR = "Percent error (%)"


def _finish(fig, show: bool = True, save_path: str | None = None, tight: bool = True):
    """Tight layout, optional save, optional show."""
    if tight:
        try:
            fig.tight_layout()
        except Exception:
            pass
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig


def _normalize_series(ys, colors=None, default_x=None):
    """Normalize ``*ys`` entries to ``(x, y, label, style_dict)`` tuples.

    Each entry may be:
    - ``y``
    - ``(y, label)``
    - ``(y, label, style_dict)``
    - ``(x, y, label, style_dict)``  — per-series independent ``x``
    """
    default_colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_HIGHLIGHT,
                      MODEL_COLORS["foster"], "0.45"]
    out = []
    for i, entry in enumerate(ys):
        x_i = default_x
        if isinstance(entry, tuple):
            if len(entry) == 2:
                y, label = entry
                style = {}
            elif len(entry) == 3:
                y, label, style = entry
                style = dict(style or {})
            elif len(entry) == 4:
                x_i, y, label, style = entry
                style = dict(style or {})
            else:
                raise ValueError(
                    "Each series must be y, (y, label), (y, label, style_dict), "
                    "or (x, y, label, style_dict)."
                )
        else:
            y, label, style = entry, None, {}
        if x_i is None:
            raise ValueError(
                "plot_curves needs a shared x=... or per-series (x, y, label, style)."
            )
        if "color" not in style:
            if colors is not None and i < len(colors):
                style["color"] = colors[i]
            else:
                style.setdefault("color", default_colors[i % len(default_colors)])
        out.append((np.asarray(x_i), np.asarray(y), label, style))
    return out


def _draw_design_markers(ax, hline=None, vline=None, point=None):
    """Optional target / design-point markers in COLOR_HIGHLIGHT."""
    if hline is not None:
        for h in np.atleast_1d(hline):
            ax.axhline(float(h), color=COLOR_HIGHLIGHT, ls="--", lw=1)
    if vline is not None:
        for v in np.atleast_1d(vline):
            ax.axvline(float(v), color=COLOR_HIGHLIGHT, ls="--", lw=1)
    if point is not None:
        px, py = point
        ax.plot([px], [py], "s", color=COLOR_HIGHLIGHT, ms=9, label="design point",
                zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# General curve helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(
    x=None,
    *ys,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    color=None,
    colors=None,
    fmt: str | None = None,
    ax=None,
    grid: bool = True,
    xlim=None,
    ylim=None,
    hline=None,
    vline=None,
    point=None,
    show: bool = True,
    save_path: str | None = None,
    **plot_kwargs,
):
    """
    Plot one or more ``y`` series.

    ``x`` is a shared abscissa.  Each positional series may be an array,
    ``(y, label)``, ``(y, label, style_dict)``, or ``(x, y, label, style_dict)``
    for an independent frequency / parameter axis.  Optional ``hline`` /
    ``vline`` / ``point`` mark a design target in ``COLOR_HIGHLIGHT``
    (``hline``/``vline`` accept a scalar or a sequence).

    Returns
    -------
    fig, ax
    """
    if x is not None:
        x = np.asarray(x)
    if color is not None and colors is None:
        colors = [color]
    series = _normalize_series(ys, colors=colors, default_x=x)

    own_ax = ax is None
    if own_ax:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR))
    else:
        fig = ax.figure

    for x_i, y, label, style in series:
        kw = {**plot_kwargs, **style}
        series_fmt = kw.pop("fmt", fmt)
        if series_fmt is not None:
            ax.plot(x_i, y, series_fmt, label=label, **kw)
        else:
            ax.plot(x_i, y, label=label, **kw)

    _draw_design_markers(ax, hline=hline, vline=vline, point=point)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if grid:
        which = "both" if xscale == "log" or yscale == "log" else "major"
        ax.grid(True, which=which, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if any(labels):
        ax.legend(framealpha=0.9, edgecolor="0.8")

    if own_ax:
        _finish(fig, show=show, save_path=save_path)
    return fig, ax


def plot_curve_pair(
    x,
    y0,
    y1,
    *,
    xlabel: str | None = None,
    ylabel0: str | None = None,
    ylabel1: str | None = None,
    title0: str | None = None,
    title1: str | None = None,
    title: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    yscale0: str | None = None,
    yscale1: str | None = None,
    color0=None,
    color1=None,
    marker: str = "o-",
    grid: bool = True,
    show: bool = True,
    save_path: str | None = None,
):
    """
    Plot two ``y`` series against a shared ``x`` in a 1×2 figure.

    Per-panel titles default to ``None`` (no title).  ``title`` is applied as
    ``fig.suptitle`` only when given.

    Returns
    -------
    fig, axes
    """
    x = np.asarray(x)
    y0 = np.asarray(y0)
    y1 = np.asarray(y1)
    ys0 = yscale0 if yscale0 is not None else yscale
    ys1 = yscale1 if yscale1 is not None else yscale
    c0 = color0 if color0 is not None else COLOR_PRIMARY
    c1 = color1 if color1 is not None else COLOR_SECONDARY

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
    axes[0].plot(x, y0, marker, color=c0)
    axes[1].plot(x, y1, marker, color=c1)

    for ax, ylab, ttl, ysc in (
        (axes[0], ylabel0, title0, ys0),
        (axes[1], ylabel1, title1, ys1),
    ):
        ax.set_xscale(xscale)
        ax.set_yscale(ysc)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylab is not None:
            ax.set_ylabel(ylab)
        if ttl is not None:
            ax.set_title(ttl)
        if grid:
            which = "both" if xscale == "log" or ysc == "log" else "major"
            ax.grid(True, which=which, alpha=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.01)
    _finish(fig, show=show, save_path=save_path)
    return fig, axes


def plot_heatmaps(
    x,
    y,
    grids: dict,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    title: str | None = None,
    titles: dict | None = None,
    norm: str = "log",
    cmap: str = "viridis",
    xscale: str = "log",
    yscale: str = "log",
    show: bool = True,
    save_path: str | None = None,
):
    """
    Side-by-side heatmaps of 2-D grids that share ``x`` / ``y`` and a color scale.

    Parameters
    ----------
    x, y : array-like
        Bin *centers* along each axis (e.g. coupling capacitances in fF).
        Cell edges are inferred from midpoints (log or linear spacing).
    grids : dict[str, ndarray]
        Mapping of panel label → 2-D array with shape ``(len(y), len(x))``.
    xlabel, ylabel, cbar_label : str, optional
        Axis / colorbar labels.
    title : str, optional
        Figure super-title (default: none).
    titles : dict, optional
        Optional per-panel title overrides keyed like ``grids``.
    norm : {"log", "linear"}
        Shared color normalization across panels.
    cmap, xscale, yscale
        Matplotlib colormap and axis scales.

    Returns
    -------
    fig, axes
    """
    from matplotlib.colors import LogNorm, Normalize

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keys = list(grids.keys())
    if not keys:
        raise ValueError("grids must contain at least one array")

    def _edges(centers, log_space: bool):
        c = np.asarray(centers, dtype=float)
        if log_space:
            lc = np.log10(c)
            step = lc[1] - lc[0] if len(lc) > 1 else 0.5
            return 10 ** np.concatenate([lc - step / 2, [lc[-1] + step / 2]])
        step = c[1] - c[0] if len(c) > 1 else 1.0
        return np.concatenate([c - step / 2, [c[-1] + step / 2]])

    x_edges = _edges(x, xscale == "log")
    y_edges = _edges(y, yscale == "log")

    allv = np.concatenate([np.asarray(grids[k], dtype=float).ravel() for k in keys])
    allv = allv[np.isfinite(allv)]
    if norm == "log":
        allv = allv[allv > 0]
        if allv.size == 0:
            raise ValueError("no positive finite values for log-normalized heatmaps")
        cnorm = LogNorm(vmin=float(allv.min()), vmax=float(allv.max()))
    elif norm == "linear":
        cnorm = Normalize(vmin=float(allv.min()), vmax=float(allv.max()))
    else:
        raise ValueError("norm must be 'log' or 'linear'")

    n = len(keys)
    fig, axes = plt.subplots(
        1, n,
        figsize=(FIGURE_WIDTH_2COL, FIGURE_WIDTH_2COL * 0.4),
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    im = None
    for ax, key in zip(axes, keys):
        z = np.asarray(grids[key], dtype=float)
        im = ax.pcolormesh(x_edges, y_edges, z, norm=cnorm, cmap=cmap)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_aspect("equal")
        panel_title = None
        if titles is not None and key in titles:
            panel_title = titles[key]
        elif titles is None:
            panel_title = key
        if panel_title is not None:
            ax.set_title(panel_title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
    if ylabel is not None:
        axes[0].set_ylabel(ylabel)
    if im is not None:
        fig.colorbar(im, ax=axes, label=cbar_label, fraction=0.046, pad=0.02)
    if title is not None:
        fig.suptitle(title, fontsize=10, y=1.02)
    _finish(fig, show=show, save_path=save_path, tight=False)
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Core comparison / overlay plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_re_im(
    network: rf.Network,
    m: int = 0,
    n: int = 0,
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Plot Re(S[m,n]) and Im(S[m,n]) side by side.

    Parameters
    ----------
    network : rf.Network
    m, n : int
        S-parameter indices (0-based).
    title : str, optional
        Super-title for the figure.
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig, axes
    """
    f = network.frequency.f / 1e9
    s = network.s[:, m, n]
    port = f"S_{{{m + 1}{n + 1}}}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
    ax1.plot(f, np.real(s), color=COLOR_PRIMARY, label=rf"$\mathrm{{Re}}({port})$")
    ax1.set_xlabel(LABEL_FREQ)
    ax1.set_ylabel(rf"$\mathrm{{Re}}({port})$")
    ax1.legend()

    ax2.plot(f, np.imag(s), color=COLOR_PRIMARY, label=rf"$\mathrm{{Im}}({port})$")
    ax2.set_xlabel(LABEL_FREQ)
    ax2.set_ylabel(rf"$\mathrm{{Im}}({port})$")
    ax2.legend()

    if title:
        fig.suptitle(title)
    _finish(fig, show=show, save_path=save_path)
    return fig, (ax1, ax2)


def plot_lom_vs_data(
    lom_network: rf.Network,
    data_network: rf.Network,
    m: int = 0,
    n: int = 0,
    lom_label: str = "LOM",
    data_label: str = "Data",
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Overlay Re and Im of a fitted LOM network against a reference (data/CPW).

    Parameters
    ----------
    lom_network : rf.Network
        The fitted lumped-element model.
    data_network : rf.Network
        The reference (measured or CPW simulation).
    m, n : int
        S-parameter indices (0-based).
    lom_label, data_label : str
        Legend labels.
    title : str, optional
        Super-title for the figure.
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    fig, axes
    """
    f_lom = lom_network.frequency.f / 1e9
    f_data = data_network.frequency.f / 1e9
    s_lom = lom_network.s[:, m, n]
    s_data = data_network.s[:, m, n]
    port = f"S_{{{m + 1}{n + 1}}}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))

    ax1.plot(f_data, np.real(s_data), color=COLOR_PRIMARY, linestyle="--", label=data_label)
    ax1.plot(f_lom, np.real(s_lom), color=COLOR_HIGHLIGHT, label=lom_label)
    ax1.set_xlabel(LABEL_FREQ)
    ax1.set_ylabel(rf"$\mathrm{{Re}}({port})$")
    ax1.legend(framealpha=0.9, edgecolor="0.8")

    ax2.plot(f_data, np.imag(s_data), color=COLOR_PRIMARY, linestyle="--", label=data_label)
    ax2.plot(f_lom, np.imag(s_lom), color=COLOR_HIGHLIGHT, label=lom_label)
    ax2.set_xlabel(LABEL_FREQ)
    ax2.set_ylabel(rf"$\mathrm{{Im}}({port})$")
    ax2.legend(framealpha=0.9, edgecolor="0.8")

    if title:
        fig.suptitle(title)
    _finish(fig, show=show, save_path=save_path)
    return fig, (ax1, ax2)


def plot_all_models(
    networks: dict[str, rf.Network],
    m: int = 0,
    n: int = 0,
    quantity: str = "re",
    title: str = None,
    xlim: tuple = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Plot one S-parameter quantity for multiple networks on a single axes.

    Useful for comparing CPW, FosterFit, OptimizedFit, and AnalyticalFit
    in one call.

    Parameters
    ----------
    networks : dict[str, rf.Network]
        Mapping of label → network.
    m, n : int
        S-parameter indices (0-based).
    quantity : {"re", "im", "db", "abs"}
        Which quantity to plot.
    title : str, optional
        Plot title.
    xlim : tuple, optional
        ``(fmin, fmax)`` in GHz to zoom the frequency axis around a resonance.
        When omitted the full frequency span of the networks is shown.
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    _extractors = {
        "re":  lambda s: np.real(s),
        "im":  lambda s: np.imag(s),
        "db":  lambda s: 20 * np.log10(np.maximum(np.abs(s), 1e-300)),
        "abs": lambda s: np.abs(s),
    }
    port = f"S_{{{m + 1}{n + 1}}}"
    _ylabels = {
        "re":  rf"$\mathrm{{Re}}({port})$",
        "im":  rf"$\mathrm{{Im}}({port})$",
        "db":  rf"$|{port}|$ (dB)",
        "abs": rf"$|{port}|$",
    }

    if quantity not in _extractors:
        raise ValueError("quantity must be one of: {}".format(list(_extractors)))

    extract = _extractors[quantity]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR))
    for label, net in networks.items():
        f = net.frequency.f / 1e9
        s = net.s[:, m, n]
        ax.plot(f, extract(s), label=label,
                color=_model_color(label), linestyle=_model_linestyle(label))

    ax.set_xlabel(LABEL_FREQ)
    ax.set_ylabel(_ylabels[quantity])
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    if title:
        ax.set_title(title)
    _finish(fig, show=show, save_path=save_path)
    return fig, ax


def plot_scan(
    model_or_scan,
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Plot OptimizedFit Stage-1 ω₀ scan (SSE vs resonant frequency).

    Parameters
    ----------
    model_or_scan : OptimizedFit or array-like
        An ``OptimizedFit`` instance (reads ``get_params()["scan_results"]``)
        or a raw scan array of shape ``(n_w0, 4)`` with columns
        ``[ω0, Ceff, Leff, sse]``.
    title : str, optional
        Axis title.  Default is a short technical Stage-1 label.
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    if hasattr(model_or_scan, "get_params"):
        scan = model_or_scan.get_params().get("scan_results")
        if scan is None:
            raise ValueError(
                "OptimizedFit has no scan_results. "
                "Call fit() before plot_scan()."
            )
    else:
        scan = np.asarray(model_or_scan)

    if scan.ndim != 2 or scan.shape[1] < 4:
        raise ValueError(
            "scan_results must have shape (n_w0, 4) with columns "
            "[ω0, Ceff, Leff, sse]."
        )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR))
    ax.plot(scan[:, 0] / (2 * np.pi * 1e9), scan[:, 3], "o-", markersize=4)
    ax.set_xlabel(LABEL_OMEGA0)
    ax.set_ylabel("SSE (Stage 1 objective)")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(r"Stage 1 $\omega_0$ scan")
    _finish(fig, show=show, save_path=save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Fit diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def plot_residuals(
    data_ntw,
    lom_networks: dict,
    m: int = 0,
    n: int = 0,
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Plot the complex residual S_lom - S_data for one or more LOM networks.

    Residuals are shown as Re and Im separately.  A perfect fit is a flat
    zero line.

    Parameters
    ----------
    data_ntw : rf.Network
        Reference (CPW or measured) network.
    lom_networks : dict[str, rf.Network]
        Mapping of label → fitted LOM network.
    m, n : int
        S-parameter indices (0-based).
    title : str, optional
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH_1COL, FIGURE_WIDTH_1COL), sharex=True)

    f_data = data_ntw.frequency.f / 1e9
    s_data = data_ntw.s[:, m, n]

    for label, lom_ntw in lom_networks.items():
        f_lom = lom_ntw.frequency.f / 1e9
        s_lom = lom_ntw.s[:, m, n]
        if not np.allclose(f_lom, f_data):
            re_interp = np.interp(f_data, f_lom, np.real(s_lom))
            im_interp = np.interp(f_data, f_lom, np.imag(s_lom))
            s_lom = re_interp + 1j * im_interp

        residual = s_lom - s_data
        color = _model_color(label)
        key = _model_key(label)
        disp_label = MODEL_LABELS[key] if key else label

        axes[0].plot(f_data, np.real(residual), color=color, label=disp_label)
        axes[1].plot(f_data, np.imag(residual), color=color, label=disp_label)

    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())

    axes[0].set_ylabel(r"$\mathrm{Re}(S_\mathrm{LOM} - S_\mathrm{ref})$")
    axes[1].set_ylabel(r"$\mathrm{Im}(S_\mathrm{LOM} - S_\mathrm{ref})$")
    axes[1].set_xlabel(LABEL_FREQ)
    axes[0].legend(framealpha=0.9, edgecolor="0.8")

    port_str = f"S{m + 1}{n + 1}"
    fig_title = title or f"Fit residuals — {port_str}"
    axes[0].set_title(fig_title)

    _finish(fig, show=show, save_path=save_path)
    return fig, axes


def plot_summary(
    results: dict,
    port: str = "s11",
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """
    Three-panel summary of fitting quality for all three LC models vs CPW.

    Panels:
      Left   — Overlay of Re(S) for CPW + all three models
      Centre — f₀ and κ percent errors as a grouped bar chart
      Right  — Complex residual magnitude |S_lom - S_cpw| across the window

    Parameters
    ----------
    results : dict
        Output dict from analyze_system().
    port : {"s11", "s22"}
        Which reflection port to summarise.
    title : str, optional
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional

    Returns
    -------
    fig, axes
    """
    port = port.lower()
    m, n = (0, 0) if port == "s11" else (1, 1)

    cpw_net = results["cpw_network"]
    networks = {
        "optimized":  results["optimized_network"],
        "foster":     results["foster_network"],
        "analytical": results["analytical_network"],
    }

    f = cpw_net.frequency.f / 1e9
    s_cpw = cpw_net.s[:, m, n]
    port_tex = f"S_{{{m + 1}{n + 1}}}"

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))

    # Panel 1: Re(S) overlay
    ax = axes[0]
    ax.plot(f, np.real(s_cpw), color=_MODEL_COLORS["cpw"],
            lw=2.0, label="CPW", zorder=5)
    for label, net in networks.items():
        f_lom = net.frequency.f / 1e9
        s_lom = net.s[:, m, n]
        ax.plot(f_lom, np.real(s_lom),
                color=_MODEL_COLORS[label], lw=1.2, linestyle="--",
                label=_MODEL_LABELS[label])
    ax.set_xlabel(LABEL_FREQ)
    ax.set_ylabel(rf"$\mathrm{{Re}}({port_tex})$")
    ax.set_title(rf"$\mathrm{{Re}}({port_tex})$ overlay")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Panel 2: f₀ and κ error bar chart
    ax = axes[1]
    PORT = port.upper()
    model_keys = ["optimized", "foster", "analytical"]
    f0_errors = [
        results.get(f"{_MODEL_LABELS[k]} f0 error {PORT} (%)", np.nan)
        for k in model_keys
    ]
    k_errors = [
        results.get(f"{_MODEL_LABELS[k]} kappa error {PORT} (%)", np.nan)
        for k in model_keys
    ]

    x = np.arange(len(model_keys))
    width = 0.32
    ax.bar(x - width / 2, f0_errors, width,
           color=[_MODEL_COLORS[k] for k in model_keys],
           alpha=0.85, label=r"$f_0$ error")
    ax.bar(x + width / 2, k_errors, width,
           color=[_MODEL_COLORS[k] for k in model_keys],
           alpha=0.45, edgecolor=[_MODEL_COLORS[k] for k in model_keys],
           linewidth=1.2, label=r"$\kappa$ error")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_MODEL_LABELS[k] for k in model_keys])
    ax.set_ylabel(LABEL_PCT_ERROR)
    ax.set_title(rf"$f_0$ and $\kappa$ errors ({PORT})")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="grey", alpha=0.85, label=r"$f_0$ error"),
        Patch(facecolor="grey", alpha=0.45, edgecolor="grey", label=r"$\kappa$ error"),
    ], framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    # Panel 3: residual magnitude
    ax = axes[2]
    for label, net in networks.items():
        f_lom = net.frequency.f / 1e9
        s_lom = net.s[:, m, n]
        if not np.allclose(f_lom, f):
            re_i = np.interp(f, f_lom, np.real(s_lom))
            im_i = np.interp(f, f_lom, np.imag(s_lom))
            s_lom = re_i + 1j * im_i
        ax.plot(f, np.abs(s_lom - s_cpw),
                color=_MODEL_COLORS[label], label=_MODEL_LABELS[label])

    ax.set_xlabel(LABEL_FREQ)
    ax.set_ylabel(r"$|S_\mathrm{LOM} - S_\mathrm{CPW}|$")
    ax.set_title(f"|Residual| magnitude ({PORT})")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    fig.suptitle(title or "Fit summary", fontsize=10, y=1.01)
    _finish(fig, show=show, save_path=save_path)
    return fig, axes


def plot_transmission(
    networks: dict,
    m: int = 0,
    n: int = 1,
    annotate_resonances: bool = False,
    prominence: float = 3.0,
    title: str = None,
    quantity: str = "db",
    ylabel: str = None,
    xlim=None,
    panels=None,
    center: float = None,
    half_width: float = 0.35,
    show: bool = True,
    save_path: str = None,
):
    """
    Plot |S_mn| for one or more networks.

    Parameters
    ----------
    networks : dict[str, rf.Network]
        Mapping of label → network.
    m, n : int
        S-parameter indices (0-based).  Default S21 (0,1).
    annotate_resonances : bool
        If True, marks each resonance with a vertical line and frequency label.
        Default False — annotations are opt-in because automatic dip detection
        is unreliable for multi-mode / chain spectra.
    prominence : float
        Minimum prominence in dB for peak/dip detection (used only when
        ``annotate_resonances`` and ``quantity=\"db\"``).
    title : str, optional
        Axis / figure title.  Default is a short ``$|S_{mn}|$`` label when
        ``panels`` is None; with a zoom panel, titles are omitted unless given
        (applied as ``fig.suptitle``).
    quantity : {"db", "abs"}
        Plot ``20 log10 |S|`` (default) or linear magnitude.
    ylabel : str, optional
        Override the default y-axis label.
    xlim : tuple, optional
        ``(fmin, fmax)`` in GHz for a single-panel plot.
    panels : {None, "zoom"}
        If ``\"zoom\"``, draw full-band and zoomed panels side by side.
        Zoom uses ``center`` ± ``half_width`` (GHz).
    center, half_width : float
        Zoom window for ``panels=\"zoom\"``.  ``center`` defaults to the
        mid-band frequency of the first network when omitted.
    show : bool
        If True (default), call ``plt.show()``.
    save_path : str, optional

    Returns
    -------
    fig, ax  or  fig, axes
    """
    if quantity not in ("db", "abs"):
        raise ValueError("quantity must be 'db' or 'abs'")
    if panels not in (None, "zoom"):
        raise ValueError("panels must be None or 'zoom'")

    port = f"S_{{{m + 1}{n + 1}}}"
    default_ylabel = {
        "db":  rf"$|{port}|$ (dB)",
        "abs": rf"$|{port}|$",
    }[quantity]
    y_label = ylabel if ylabel is not None else default_ylabel

    def _trace(net):
        f_ghz = net.frequency.f / 1e9
        s_mn = net.s[:, m, n]
        mag = np.abs(s_mn)
        if quantity == "db":
            y = 20 * np.log10(np.maximum(mag, 1e-300))
        else:
            y = mag
        return f_ghz, y

    def _plot_on(ax, with_legend=True, do_annotate=False):
        all_annotations = []
        for label, net in networks.items():
            f_ghz, y = _trace(net)
            color = _model_color(label)
            key = _model_key(label)
            disp = MODEL_LABELS[key] if key else label
            style = {}
            # Preserve common tutorial styles for non-model labels.
            if color is None and label.lower().startswith("hfss"):
                color, style = "k", {"lw": 2.4, "alpha": 0.8}
            elif key is not None:
                style["linestyle"] = _model_linestyle(label)
            ax.plot(f_ghz, y, color=color, label=disp, zorder=3, **style)

            if do_annotate and quantity == "db":
                dip_idx, _ = find_peaks(-y, prominence=prominence)
                for idx in dip_idx:
                    all_annotations.append((f_ghz[idx], y[idx], color or "0.3"))

        if do_annotate and all_annotations:
            all_annotations.sort(key=lambda x: x[0])
            kept = [all_annotations[0]]
            for ann in all_annotations[1:]:
                if ann[0] - kept[-1][0] > 0.05:
                    kept.append(ann)
            for f_ann, db_ann, color in kept:
                ax.axvline(f_ann, color=color, linewidth=0.7,
                           linestyle=":", alpha=0.7, zorder=2)
                ax.annotate(
                    f"{f_ann:.3f} GHz",
                    xy=(f_ann, db_ann),
                    xytext=(4, 8),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=color,
                    rotation=90,
                    va="bottom",
                )

        ax.set_xlabel(LABEL_FREQ)
        ax.set_ylabel(y_label)
        if with_legend:
            ax.legend(framealpha=0.9, edgecolor="0.8")
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5)
        ax.grid(True, which="minor", linestyle="--", linewidth=0.2, alpha=0.3)

    if panels == "zoom":
        fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
        _plot_on(axes[0], with_legend=True, do_annotate=annotate_resonances)
        _plot_on(axes[1], with_legend=False, do_annotate=False)
        if center is None:
            first = next(iter(networks.values()))
            f0 = first.frequency.f / 1e9
            center = 0.5 * (f0[0] + f0[-1])
        axes[1].set_xlim(center - half_width, center + half_width)
        if title is not None:
            fig.suptitle(title, fontsize=10, y=1.01)
        _finish(fig, show=show, save_path=save_path)
        return fig, axes

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
    _plot_on(ax, with_legend=True, do_annotate=annotate_resonances)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(rf"$|{port}|$")
    _finish(fig, show=show, save_path=save_path)
    return fig, ax


def plot_s_residual(
    networks: dict,
    ref: rf.Network,
    m: int = 1,
    n: int = 0,
    title: str = None,
    ylabel: str = None,
    xlim=None,
    quantity: str = "db",
    show: bool = True,
    save_path: str = None,
):
    """
    Plot Δ|S_mn| of each network relative to a reference network.

    Parameters
    ----------
    networks : dict[str, rf.Network]
        Mapping of label → network to compare against ``ref``.
    ref : rf.Network
        Reference network (e.g. HFSS or CPW ground truth).
    m, n : int
        S-parameter indices (0-based).  Default S21 as ``s[:, 1, 0]``.
    title : str, optional
        Axis title (default: none).
    ylabel : str, optional
        Override default residual ylabel.
    xlim : tuple, optional
        ``(fmin, fmax)`` in GHz.
    quantity : {"db", "abs"}
        Residual of dB magnitudes or linear magnitudes.
    show, save_path
        Standard finish options.

    Returns
    -------
    fig, ax
    """
    if quantity not in ("db", "abs"):
        raise ValueError("quantity must be 'db' or 'abs'")

    port = f"S_{{{m + 1}{n + 1}}}"
    f_ref = ref.frequency.f / 1e9
    s_ref = ref.s[:, m, n]
    if quantity == "db":
        y_ref = 20 * np.log10(np.maximum(np.abs(s_ref), 1e-300))
        y_default = rf"$\Delta|{port}|$ (dB)"
    else:
        y_ref = np.abs(s_ref)
        y_default = rf"$\Delta|{port}|$"

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR))
    for label, net in networks.items():
        f = net.frequency.f / 1e9
        s = net.s[:, m, n]
        if quantity == "db":
            y = 20 * np.log10(np.maximum(np.abs(s), 1e-300))
        else:
            y = np.abs(s)
        if not np.allclose(f, f_ref):
            y = np.interp(f_ref, f, y)
            f = f_ref
        color = _model_color(label) or COLOR_PRIMARY
        ax.plot(f, y - y_ref, color=color, label=label)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(LABEL_FREQ)
    ax.set_ylabel(ylabel if ylabel is not None else y_default)
    if title is not None:
        ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(framealpha=0.9, edgecolor="0.8")
    _finish(fig, show=show, save_path=save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Error heatmaps
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_LABELS = {
    "f0_s11":       r"$f_0$ error S11 (%)",
    "f0_s22":       r"$f_0$ error S22 (%)",
    "kappa_s11":    r"$\kappa$ error S11 (%)",
    "kappa_s22":    r"$\kappa$ error S22 (%)",
    "shift_mode1":  "Mode 1 shift error (%)",
    "shift_mode2":  "Mode 2 shift error (%)",
    "shift_mode3":  "Mode 3 shift error (%)",
    "shift_max":    "Max shift error (%)",
}


def _extract_error(result: dict, model: str, metric: str) -> float:
    """Pull a signed percent error from an analyze_system() result dict."""
    label = _MODEL_LABELS.get(model, model)

    if metric in ("f0_s11", "f0_s22"):
        port = metric.split("_")[1].upper()
        cpw_val = result.get(f"CPW f0 {port} (GHz)", np.nan)
        mod_val = result.get(f"{label} f0 {port} (GHz)", np.nan)

    elif metric in ("kappa_s11", "kappa_s22"):
        port = metric.split("_")[1].upper()
        cpw_val = result.get(f"CPW kappa {port} (MHz)", np.nan)
        mod_val = result.get(f"{label} kappa {port} (MHz)", np.nan)

    elif metric in ("shift_mode1", "shift_mode2", "shift_mode3"):
        idx = int(metric[-1]) - 1
        cpw_shifts = result.get("CPW all shifts (GHz)", np.array([np.nan] * 3))
        mod_shifts = result.get(f"{label} all shifts (GHz)", np.array([np.nan] * 3))
        cpw_val = cpw_shifts[idx] if len(cpw_shifts) > idx else np.nan
        mod_val = mod_shifts[idx] if len(mod_shifts) > idx else np.nan

    elif metric == "shift_max":
        cpw_shifts = result.get("CPW shifted freqs S11 (GHz)", [np.nan] * 3)
        mod_shifts = result.get(f"{label} shifted freqs S11 (GHz)", [np.nan] * 3)
        if np.all(np.isnan(cpw_shifts)):
            return np.nan
        idx = np.nanargmax(np.abs(cpw_shifts))
        cpw_val = cpw_shifts[idx]
        mod_val = mod_shifts[idx]

    else:
        return np.nan

    if np.isnan(cpw_val) or np.isnan(mod_val) or cpw_val == 0:
        return np.nan

    return (mod_val - cpw_val) / abs(cpw_val) * 100


def plot_error_heatmap(
    results_grid: list[list[dict]],
    param1_values,
    param2_values,
    param1_label: str,
    param2_label: str,
    model: str,
    metric: str,
    param1_scale: float = 1.0,
    param2_scale: float = 1.0,
    param1_unit: str = "",
    param2_unit: str = "",
    vmin: float = None,
    vmax: float = None,
    cmap: str = "RdBu_r",
    title: str = None,
    ax=None,
    show: bool = True,
    save_path: str = None,
):
    """
    Heatmap of signed percent error from a 2D grid of analyze_system() runs.

    Returns
    -------
    matplotlib.image.AxesImage
    """
    p1 = np.asarray(param1_values) * param1_scale
    p2 = np.asarray(param2_values) * param2_scale

    error_matrix = np.full((len(p1), len(p2)), np.nan)
    for i, row in enumerate(results_grid):
        for j, result in enumerate(row):
            error_matrix[i, j] = _extract_error(result, model, metric)

    if vmin is None and vmax is None:
        abs_max = np.nanmax(np.abs(error_matrix))
        vmin, vmax = -abs_max, abs_max
    elif vmin is None:
        vmin = -vmax
    elif vmax is None:
        vmax = -vmin

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_WIDTH_1COL))

    im = ax.pcolormesh(
        p2, p1, error_matrix,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto",
    )

    p1_ax_label = f"{param1_label} ({param1_unit})" if param1_unit else param1_label
    p2_ax_label = f"{param2_label} ({param2_unit})" if param2_unit else param2_label
    ax.set_xlabel(p2_ax_label)
    ax.set_ylabel(p1_ax_label)

    metric_label = _METRIC_LABELS.get(metric, metric)
    model_label = _MODEL_LABELS.get(model, model)
    ax.set_title(title or f"{model_label} — {metric_label}")

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    if standalone:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(metric_label, fontsize=8)
        cb.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        _finish(plt.gcf(), show=show, save_path=save_path)

    return im


def plot_error_heatmap_trio(
    results_grid: list[list[dict | None]],
    param1_values,
    param2_values,
    param1_label: str,
    param2_label: str,
    metric: str,
    param1_scale: float = 1.0,
    param2_scale: float = 1.0,
    param1_unit: str = "",
    param2_unit: str = "",
    cmap: str = "RdBu_r",
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """Three-panel error heatmap for optimized / foster / analytical."""
    models = ["optimized", "foster", "analytical"]

    p1 = np.asarray(param1_values) * param1_scale
    p2 = np.asarray(param2_values) * param2_scale

    all_errors = {}
    for model in models:
        mat = np.full((len(p1), len(p2)), np.nan)
        for i, row in enumerate(results_grid):
            for j, result in enumerate(row):
                if result is None:
                    continue
                try:
                    mat[i, j] = _extract_error(result, model, metric)
                except (KeyError, TypeError, ValueError):
                    mat[i, j] = np.nan
        all_errors[model] = mat

    combined = np.concatenate([m.ravel() for m in all_errors.values()])

    if np.all(np.isnan(combined)):
        vmin, vmax = -1.0, 1.0
    else:
        abs_max = np.nanmax(np.abs(combined))
        vmin, vmax = -abs_max, abs_max

    fig, axes = plt.subplots(
        1, 3,
        figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR),
        gridspec_kw={"wspace": 0.35},
    )

    images = []
    for ax, model in zip(axes, models):
        im = ax.pcolormesh(
            p2, p1, all_errors[model],
            cmap=cmap, vmin=vmin, vmax=vmax,
            shading="auto",
        )
        images.append(im)

        p1_ax_label = f"{param1_label} ({param1_unit})" if param1_unit else param1_label
        p2_ax_label = f"{param2_label} ({param2_unit})" if param2_unit else param2_label
        ax.set_xlabel(p2_ax_label)
        ax.set_ylabel(p1_ax_label)
        ax.set_title(_MODEL_LABELS[model], fontsize=9)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    metric_label = _METRIC_LABELS.get(metric, metric)
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.018, 0.75])
    cb = fig.colorbar(images[-1], cax=cbar_ax)
    cb.set_label(metric_label, fontsize=8)
    cb.ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    sup_title = title or f"Signed percent error: {metric_label}"
    fig.suptitle(sup_title, fontsize=10, y=1.01)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Pole-fit / complex comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = {
    "data": "#1f77b4",
    "fit":  "#f86d9b",
    "f0":   "#d62728",
    "grid": "#e0e0e0",
}


def _apply_style(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=_COLORS["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_edgecolor(_COLORS["grid"])


def plot_pole_fit(network: rf.Network, pole: complex, title: str = None,
                  show: bool = True, save_path: str = None):
    """Plot magnitude, phase, and IQ circle for a pole fit of S21."""
    freqs = network.f
    S21 = network.s[:, 1, 0]
    s = 1j * 2 * np.pi * freqs
    alpha = -pole.real
    omega_0 = pole.imag

    poles = np.array([
        complex(-alpha, omega_0),
        complex(-alpha, -omega_0),
    ])

    phi = np.column_stack([
        1 / (s - poles[0]),
        1 / (s - poles[1]),
        np.ones(len(s)),
    ])
    A = np.vstack([phi.real, phi.imag])
    b = np.hstack([S21.real, S21.imag])
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    S21_fit = phi @ coeffs

    f_GHz = freqs / 1e9
    f0_GHz = omega_0 / (2 * np.pi) / 1e9

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
    default_title = (
        rf"Pole fit — $f_0={f0_GHz:.4f}$ GHz,  "
        rf"$Q={omega_0 / (2 * alpha):.1f}$,  "
        rf"$\kappa/2\pi={alpha / np.pi / 1e6:.1f}$ MHz"
    )
    fig.suptitle(title if title is not None else default_title)

    ax = axes[0]
    ax.plot(f_GHz, 20 * np.log10(np.abs(S21)), color=_COLORS["data"], lw=2, label="Data")
    ax.plot(f_GHz, 20 * np.log10(np.abs(S21_fit)), color=_COLORS["fit"], lw=1.5, ls="--", label="Fit")
    ax.axvline(f0_GHz, color=_COLORS["f0"], ls=":", label=r"$f_0$")
    _apply_style(ax, title=r"$|S_{21}|$", xlabel=LABEL_FREQ, ylabel=r"$|S_{21}|$ (dB)")

    ax = axes[1]
    ax.plot(f_GHz, np.angle(S21, deg=True), color=_COLORS["data"], lw=2, label="Data")
    ax.plot(f_GHz, np.angle(S21_fit, deg=True), color=_COLORS["fit"], lw=1.5, ls="--", label="Fit")
    ax.axvline(f0_GHz, color=_COLORS["f0"], ls=":", label=r"$f_0$")
    _apply_style(ax, title=r"Phase of $S_{21}$", xlabel=LABEL_FREQ, ylabel="Phase (deg)")

    ax = axes[2]
    ax.plot(S21.real, S21.imag, color=_COLORS["data"], lw=2, label="Data")
    ax.plot(S21_fit.real, S21_fit.imag, color=_COLORS["fit"], lw=1.5, ls="--", label="Fit")
    pk = np.argmin(np.abs(freqs - omega_0 / (2 * np.pi)))
    ax.plot(S21_fit[pk].real, S21_fit[pk].imag, "o", color=_COLORS["f0"], ms=8, label=r"$f_0$")
    ax.set_aspect("equal")
    _apply_style(ax, title="IQ Plane",
                 xlabel=r"$\mathrm{Re}(S_{21})$", ylabel=r"$\mathrm{Im}(S_{21})$")

    _finish(fig, show=show, save_path=save_path)
    print(f"f0 = {f0_GHz:.6f} GHz")
    return fig, axes


def plot_complex_network_comparison(
    network1: rf.Network,
    network2: rf.Network,
    label1: str = "Network 1",
    label2: str = "Network 2",
    color1: str = "#1f77b4",
    color2: str = "#d62728",
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """Compare two networks in magnitude, phase, and IQ plane (S21)."""
    S21_1 = network1.s[:, 1, 0]
    S21_2 = network2.s[:, 1, 0]
    f_GHz1 = network1.f / 1e9
    f_GHz2 = network2.f / 1e9

    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR))
    if title is not None:
        fig.suptitle(title)

    ax = axes[0]
    ax.plot(f_GHz1, 20 * np.log10(np.abs(S21_1)), color=color1, lw=2, label=label1)
    ax.plot(f_GHz2, 20 * np.log10(np.abs(S21_2)), color=color2, lw=1.5, ls="--", label=label2)
    _apply_style(ax, title=r"$|S_{21}|$", xlabel=LABEL_FREQ, ylabel=r"$|S_{21}|$ (dB)")

    ax = axes[1]
    ax.plot(f_GHz1, np.angle(S21_1, deg=True), color=color1, lw=2, label=label1)
    ax.plot(f_GHz2, np.angle(S21_2, deg=True), color=color2, lw=1.5, ls="--", label=label2)
    _apply_style(ax, title=r"Phase of $S_{21}$", xlabel=LABEL_FREQ, ylabel="Phase (deg)")

    ax = axes[2]
    ax.plot(S21_1.real, S21_1.imag, color=color1, lw=2, label=label1)
    ax.plot(S21_2.real, S21_2.imag, color=color2, lw=1.5, ls="--", label=label2)
    ax.set_aspect("equal")
    _apply_style(ax, title="IQ Plane",
                 xlabel=r"$\mathrm{Re}(S_{21})$", ylabel=r"$\mathrm{Im}(S_{21})$")

    _finish(fig, show=show, save_path=save_path)
    return fig, axes


def plot_shift_errors(
    results: dict,
    title: str = None,
    show: bool = True,
    save_path: str = None,
):
    """Grouped bar chart of hybridized mode shift errors vs CPW reference."""
    models = ["Foster", "Optimized", "Analytical"]
    errors = {m: results.get(f"{m} shift errors (%)", [float("nan")] * 3) for m in models}

    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_1COL, FIGURE_HEIGHT_1COL_GR))

    for i, model in enumerate(models):
        offset = (i - 1) * width
        ax.bar(
            x + offset, errors[model], width,
            color=_MODEL_COLORS[model.lower()],
            label=model, alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Mode 1", "Mode 2", "Mode 3"])
    ax.set_ylabel("Frequency shift error (%)")
    ax.set_title(title or "Hybridized mode shift errors vs CPW reference")
    ax.legend(framealpha=0.9, edgecolor="0.8")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    _finish(fig, show=show, save_path=save_path)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible aliases
# ─────────────────────────────────────────────────────────────────────────────

plot_fit_residuals = plot_residuals
plot_fit_summary = plot_summary
plot_transmission_spectrum = plot_transmission
fancy_plot = plot_lom_vs_data
plot_lom_vs_data_re_im = plot_lom_vs_data
upgraded_plot_lom_vs_data_re_im = plot_lom_vs_data
fancy_plot_all_models = plot_all_models
