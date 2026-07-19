from __future__ import annotations

import os
import re
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# Shared figure style: single source of truth lives in ``plotting``.
from simpleLOMs.plotting import (
    apply_style,
    MODEL_COLORS as _MODEL_COLORS_LOWER,
    FIGURE_WIDTH_2COL,
    FIGURE_HEIGHT_2COL_GR,
)

# Title-cased view of the canonical palette (this module keys on "Optimized",
# "Foster", ... rather than the lowercase keys used elsewhere).
MODEL_COLORS = {
    "CPW": _MODEL_COLORS_LOWER["cpw"],
    "Optimized": _MODEL_COLORS_LOWER["optimized"],
    "Foster": _MODEL_COLORS_LOWER["foster"],
    "Analytical": _MODEL_COLORS_LOWER["analytical"],
}

MODELS = ["Optimized", "Foster", "Analytical"]
MODE_MARKERS = ["o", "^", "s"]
_MODEL_SHORT = {"Optimized": "opt", "Foster": "foster", "Analytical": "analytical"}


def apply_plot_style() -> None:
    """Apply the shared simpleLOMs figure style (kept for backward compat)."""
    apply_style()


def infer_tag_from_data_path(data_path: str) -> str:
    m = re.search(r"cc_sweep_(.+)\.json$", str(data_path))
    return m.group(1) if m else "sweep"


def figure_dir(base_dir: str = "figures_APR11", data_path: str | None = None) -> str:
    if data_path:
        tag = infer_tag_from_data_path(data_path)
        out = os.path.join(base_dir, tag)
    else:
        out = base_dir
    os.makedirs(out, exist_ok=True)
    return out


def save_fig(name: str, fig=None, *, base_dir: str = "figures_APR11", data_path: str | None = None) -> str:
    if fig is None:
        fig = plt.gcf()
    out_dir = figure_dir(base_dir=base_dir, data_path=data_path)
    out_path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def build_grid(results_flat: list[dict]) -> tuple[np.ndarray, np.ndarray, list[list[dict | None]]]:
    cc1_vals = sorted(set(r["_sweep_point"]["Cc1"] for r in results_flat if r))
    cc2_vals = sorted(set(r["_sweep_point"]["Cc2"] for r in results_flat if r))
    c1_index = {v: i for i, v in enumerate(cc1_vals)}
    c2_index = {v: j for j, v in enumerate(cc2_vals)}
    grid = [[None for _ in cc2_vals] for __ in cc1_vals]
    for r in results_flat:
        if not r:
            continue
        sp = r.get("_sweep_point", {})
        i = c1_index.get(sp.get("Cc1"))
        j = c2_index.get(sp.get("Cc2"))
        if i is not None and j is not None:
            grid[i][j] = r
    return np.array(cc1_vals, dtype=float), np.array(cc2_vals, dtype=float), grid


def get(r: dict | None, key: str, default=np.nan):
    if not r:
        return default
    return r.get(key, default)


def _cf_shift_errs_from_result(r: dict | None, model: str) -> np.ndarray:
    if not r:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    # New/full schema: top-level array
    vals = r.get(f"{model} shift errors CF (%)", None)
    if vals is not None:
        return np.asarray(vals, dtype=float)

    # Recovery schema: per-load-combo nested key, e.g. "opt_shift_errors_cf"
    short = _MODEL_SHORT.get(model, model.lower())
    nested_key = f"{short}_shift_errors_cf"
    combos = r.get("load_sweep_results", [])
    rows = []
    for c in combos:
        if isinstance(c, dict) and nested_key in c:
            rows.append(np.asarray(c[nested_key], dtype=float))
    if not rows:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    stacked = np.vstack(rows)
    return np.nanmean(stacked, axis=0)


def _cf_shift_ghz_from_result(r: dict | None, model: str) -> np.ndarray:
    if not r:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    # New/full schema: top-level array
    vals = r.get(f"{model} all shifts CF (GHz)", None)
    if vals is not None:
        return np.asarray(vals, dtype=float)

    # Recovery schema: per-load-combo nested key, e.g. "opt_all_shifts_cf"
    short = _MODEL_SHORT.get(model, model.lower())
    nested_key = f"{short}_all_shifts_cf"
    combos = r.get("load_sweep_results", [])
    rows = []
    for c in combos:
        if isinstance(c, dict) and nested_key in c:
            rows.append(np.asarray(c[nested_key], dtype=float))
    if not rows:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    stacked = np.vstack(rows)
    return np.nanmean(stacked, axis=0)


def matrix_scalar(grid: list[list[dict | None]], key: str) -> np.ndarray:
    n1, n2 = len(grid), len(grid[0])
    out = np.full((n1, n2), np.nan, dtype=float)
    for i in range(n1):
        for j in range(n2):
            out[i, j] = get(grid[i][j], key, np.nan)
    return out


def matrix_cf_shift_err(grid: list[list[dict | None]], model: str, mode_idx: int) -> np.ndarray:
    n1, n2 = len(grid), len(grid[0])
    out = np.full((n1, n2), np.nan, dtype=float)
    for i in range(n1):
        for j in range(n2):
            errs = _cf_shift_errs_from_result(grid[i][j], model)
            out[i, j] = errs[mode_idx] if mode_idx < len(errs) else np.nan
    return out


def matrix_cf_shift_ghz(grid: list[list[dict | None]], model: str, mode_idx: int) -> np.ndarray:
    n1, n2 = len(grid), len(grid[0])
    out = np.full((n1, n2), np.nan, dtype=float)
    for i in range(n1):
        for j in range(n2):
            shifts = _cf_shift_ghz_from_result(grid[i][j], model)
            out[i, j] = shifts[mode_idx] if mode_idx < len(shifts) else np.nan
    return out


def matrix_cf_max_shift(grid: list[list[dict | None]], model: str, signed: bool = False) -> np.ndarray:
    n1, n2 = len(grid), len(grid[0])
    out = np.full((n1, n2), np.nan, dtype=float)
    for i in range(n1):
        for j in range(n2):
            shifts = _cf_shift_ghz_from_result(grid[i][j], model)
            if np.all(np.isnan(shifts)):
                continue
            out[i, j] = np.nanmax(shifts) if signed else np.nanmax(np.abs(shifts))
    return out


def matrix_cf_rms_shift(grid: list[list[dict | None]], model: str) -> np.ndarray:
    n1, n2 = len(grid), len(grid[0])
    out = np.full((n1, n2), np.nan, dtype=float)
    for i in range(n1):
        for j in range(n2):
            shifts = _cf_shift_ghz_from_result(grid[i][j], model)
            if np.all(np.isnan(shifts)):
                continue
            out[i, j] = float(np.sqrt(np.nanmean(np.square(shifts))))
    return out


def heatmap(ax, data: np.ndarray, *, cmap: str = "RdBu_r", vmin=None, vmax=None, title: str = "", annotate: bool = True):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(title)
    if annotate:
        for (i, j), val in np.ndenumerate(data):
            if np.isnan(val):
                continue
            col = "white" if abs(val) > 0.65 * np.nanmax(np.abs(data)) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=col)
    return im


def trio_heatmap(
    matrices: list[np.ndarray],
    *,
    suptitle: str,
    cbar_label: str,
    titles: list[str] | None = None,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    save_name: str | None = None,
    save_func: Callable[[str, object], str] | None = None,
):
    if titles is None:
        titles = MODELS
    vals = np.concatenate([m[np.isfinite(m)] for m in matrices]) if matrices else np.array([])
    if len(vals) == 0:
        vmin, vmax = -1, 1
    else:
        if symmetric:
            vmax = float(np.nanmax(np.abs(vals)))
            vmin = -vmax
        else:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
    fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_2COL, FIGURE_HEIGHT_2COL_GR),
                             gridspec_kw={"wspace": 0.38})
    for ax, mat, title in zip(axes, matrices, titles):
        im = heatmap(ax, mat, cmap=cmap, vmin=vmin, vmax=vmax, title=title, annotate=True)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9, pad=0.02)
    cbar.set_label(cbar_label)
    fig.suptitle(suptitle)
    if save_name and save_func:
        save_func(save_name, fig)
    return fig, axes
