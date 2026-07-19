"""
system.py
---------
Single home for device-level orchestration. Two clearly separated concerns:

  §A  DESIGN API (user-facing)
        fit_lom(d, model=...) -> (L, C)
        fit_lom_hanger(d, Cc_tap=...) -> (L, C)
      Pick a model, get its effective L, C from the CPW. No loads involved.
      Use fit_lom for inline resonators; fit_lom_hanger for notch / hanger LOMs.
      This is what user-facing tutorials should use.

  §C  ACCURACY-TESTING HARNESS (author-facing)
        analyze_system / analyze_system_load_grid / run_accuracy_sweep*
      Compares Foster/Optimized/Analytical against the CPW ground truth using
      reflection circle fits, including loaded-network hybridization shifts.
      This is the validation machinery — end users doing design do NOT need it.

§B in between holds the shared circle-fit / metric / residual helpers used by
both. (Formerly split across system.py + load_sweeps.py; merged here so there is
one system module. The generic parameter-sweep engine lives in sweeps.py.)
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import skrf as rf

from simpleLOMs.analysis import (
    circle_fit_f0_kappa,
    circle_fit_modes,
    resonance_from_s_max,
    resonances_from_s_max,
    stitch_shifted_freqs,
)
from simpleLOMs.models.analytical_fit import AnalyticalFit
from simpleLOMs.models.foster_fit import FosterFit
from simpleLOMs.models.hanger_optimized_fit import HangerOptimizedFit
from simpleLOMs.models.optimized_fit import OptimizedFit, OptimizationConfig
from simpleLOMs.networks.cpw import (
    cpw_resonator_network_2port,
    cpw_resonator_loaded_network_2port,
    bare_cpw_resonance_hz,
    cpw_line_impedance,
)
from simpleLOMs.networks.hanger import hanger_resonator_network_2port
from simpleLOMs.networks.lc import (
    lc_load_dressed_network_2,
    lc_resonator_loaded_network_2port,
    lc_resonator_loaded_network_2port_shifted,
    lc_resonator_loaded_network_with_grounds_2port,
)
from simpleLOMs.params import CPWParams
from simpleLOMs.readout import _resolve_port_loads

logger = logging.getLogger(__name__)

DEFAULT_FREQ_LOADED = rf.Frequency(1e9, 12e9, 800_001, unit="Hz")

# Minimum half-width and point count of the dense window ``fit_lom`` builds
# around the located resonance before running the optimised fit. A wide user
# grid (or a coarse one, e.g. design_resonator's 8001-pt span) does not resolve
# a sub-MHz linewidth, so the optimiser's stage-2 window would contain too few
# points and silently return an unphysical (L, C). Refining to ~8 kHz/pt around
# f0 makes the fit robust to whatever grid the caller supplies.
#
# The half-width is a *floor*: stage 2 brackets the FWHM of both reflection
# dips and windows the residuals at f0 ± n_widths·κ, so the window must hold
# several linewidths. At strong coupling (Cc ~ 150 fF) κ approaches a GHz and a
# fixed 0.5 GHz half-width cannot bracket the dip — the actual window is
# max(floor, KAPPA_MULT·κ_coarse), doubled on a bracketing failure up to
# MAX_TRIES attempts, and capped at MAX_FRAC_F0·f0 so it stays clear of DC and
# of the next mode family. The coarse-grid κ only ever *widens* the window
# (it may be smoothing-inflated or a grid-resolution floor, never trusted to
# narrow), which preserves the tuned narrow-linewidth behaviour exactly.
_FIT_LOM_REFINE_SPAN_HZ = 0.5e9
_FIT_LOM_REFINE_NPTS = 120_001
_FIT_LOM_REFINE_KAPPA_MULT = 5.0
_FIT_LOM_REFINE_MAX_TRIES = 4
_FIT_LOM_REFINE_MAX_FRAC_F0 = 0.75

# Substrings identifying the ValueErrors that mean "the refine window failed to
# bracket the resonance" (from analysis.fwhm_from_trace_db and the stage-2
# channel-mask validation) — the only failures worth retrying with a wider
# window. Config errors (bad s_params etc.) must propagate immediately.
_REFINE_WINDOW_ERRORS = (
    "crossings for FWHM",
    "bracket resonance",
    "mask is empty",
    "fewer than 2 zero crossings",
)


def _is_refine_window_error(err: ValueError) -> bool:
    msg = str(err)
    return any(s in msg for s in _REFINE_WINDOW_ERRORS)

DEFAULT_SERIALISABLE_KEYS = [
    "_sweep_point",
    "Z0 (Ohms)", "Cc1 (F)", "Cc2 (F)", "Ctog1 (F)", "Ctog2 (F)",
    "Optimized L (H)", "Optimized C (F)",
    "Foster L (H)", "Foster C (F)",
    "Analytical L (H)", "Analytical C (F)",
    "CPW f0 S11 (GHz)", "CPW kappa S11 (MHz)",
    "Foster f0 S11 (GHz)", "Foster kappa S11 (MHz)",
    "Foster f0 error S11 (%)", "Foster kappa error S11 (%)",
    "Optimized f0 S11 (GHz)", "Optimized kappa S11 (MHz)",
    "Optimized f0 error S11 (%)", "Optimized kappa error S11 (%)",
    "Analytical f0 S11 (GHz)", "Analytical kappa S11 (MHz)",
    "Analytical f0 error S11 (%)", "Analytical kappa error S11 (%)",
    "CPW f0 S22 (GHz)", "CPW kappa S22 (MHz)",
    "Foster f0 S22 (GHz)", "Foster kappa S22 (MHz)",
    "Foster f0 error S22 (%)", "Foster kappa error S22 (%)",
    "Optimized f0 S22 (GHz)", "Optimized kappa S22 (MHz)",
    "Optimized f0 error S22 (%)", "Optimized kappa error S22 (%)",
    "Analytical f0 S22 (GHz)", "Analytical kappa S22 (MHz)",
    "Analytical f0 error S22 (%)", "Analytical kappa error S22 (%)",
    "CPW shifted freqs CF (GHz)",
    "Optimized shifted freqs CF (GHz)",
    "Foster shifted freqs CF (GHz)",
    "Analytical shifted freqs CF (GHz)",
    "CPW all shifts (S12) (GHz)",
    "Optimized all shifts (S12) (GHz)",
    "Foster all shifts (S12) (GHz)",
    "Analytical all shifts (S12) (GHz)",
    "Optimized shift errors (S12) (%)",
    "Foster shift errors (S12) (%)",
    "Analytical shift errors (S12) (%)",
    "CPW all shifts CF (GHz)",
    "Optimized all shifts CF (GHz)",
    "Foster all shifts CF (GHz)",
    "Analytical all shifts CF (GHz)",
    "Optimized shift errors CF (%)",
    "Foster shift errors CF (%)",
    "Analytical shift errors CF (%)",
    "Foster S residuals",
    "Optimized S residuals",
    "Analytical S residuals",
    "Foster S residuals (integrated)",
    "Optimized S residuals (integrated)",
    "Analytical S residuals (integrated)",
]


# =============================================================================
# §A  DESIGN API  (user-facing) — pick a model, get L, C from the CPW. No loads.
# =============================================================================

def fit_lom(
    d: float,
    model: str = "optimized",
    *,
    Cc: float = 6e-15,
    Ctog: float = 1e-14,
    Cc1: float | None = None,
    Cc2: float | None = None,
    Ctog1: float | None = None,
    Ctog2: float | None = None,
    Z0: float = 50.0,
    cpw_params: CPWParams | None = None,
    freq: rf.Frequency | None = None,
    opt_config: OptimizationConfig | None = None,
    analytical_f_r: float | None = None,
    analytical_Z0: float | None = None,
    return_model: bool = False,
):
    """
    Fit the chosen lumped-oscillator model at resonator length ``d`` and return
    its effective ``(L, C)``.

    This is the design-time entry point for a normal user: pick a model, get
    ``L, C`` from the CPW. **No loads are involved** — the load/sweep machinery in
    this module is a separate accuracy-testing path.

    Parameters
    ----------
    d : float
        Resonator length in metres.
    model : {"optimized", "foster", "analytical"}
        Which LOM to fit. ``"optimized"`` (default) is the most accurate;
        ``"foster"`` is a fast closed form from geometry; ``"analytical"`` is a
        closed form anchored to the CPW's resonance frequency.
    Cc : float
        Coupling capacitance in Farads applied symmetrically to both ports. Used
        by the "optimized" and "analytical" references; "foster" is
        coupling-independent. For asymmetric loading, use ``Cc1``/``Cc2`` instead.
    Ctog : float
        Shunt-to-ground capacitance in Farads applied to both ports. Overridden
        per port by ``Ctog1``/``Ctog2``.
    Cc1, Cc2 : float, optional
        Per-port coupling capacitances in Farads for an asymmetrically loaded
        resonator. Each defaults to ``Cc`` when not given, so leaving both unset
        reproduces the symmetric case. "foster" ignores these (it is
        coupling-independent).
    Ctog1, Ctog2 : float, optional
        Per-port shunt-to-ground capacitances in Farads. Each defaults to
        ``Ctog`` when not given.
    Z0 : float
        Reference / feedline impedance in Ohms.
    cpw_params : CPWParams, optional
        CPW geometry / material. Defaults to ``CPWParams()``.
    freq : rf.Frequency, optional
        Frequency grid used to locate the resonance (must span the loaded
        fundamental). Defaults to a 4-12 GHz, 20,001-point grid. "foster"
        synthesises on it directly. "optimized" uses it only to find f0, then
        rebuilds the CPW reference on a dense window around f0 (~8 kHz/pt) for
        the least-squares fit, so a coarse or wide input grid still yields a
        physical ``(L, C)``. "analytical" ignores the grid for its anchor and
        instead uses the bare (unloaded) λ/2 resonance — see ``analytical_f_r``.
        Narrow the span to isolate a specific mode.
    opt_config : OptimizationConfig, optional
        Optimiser settings for ``model="optimized"``. Defaults to a quiet
        ``n_widths=4`` config.
    analytical_f_r : float, optional
        Bare (unloaded) resonance frequency in Hz used to anchor
        ``model="analytical"``. If omitted, it is computed from the CPW geometry
        with :func:`simpleLOMs.networks.cpw.bare_cpw_resonance_hz`. Anchoring to
        the *bare* line (not the loaded resonance) is what keeps the closed form
        from double-counting the Cc/Ctog loading that ``get_network`` re-applies.
        Ignored for other models.
    analytical_Z0 : float, optional
        Characteristic impedance in Ohms of the *line geometry* used in the
        closed form for ``model="analytical"`` — not the 50 Ω port reference.
        If omitted, it is computed from the CPW geometry with
        :func:`simpleLOMs.networks.cpw.cpw_line_impedance`. Ignored for other
        models.
    return_model : bool
        If True, return the fitted model instance instead of ``(L, C)``.

    Returns
    -------
    tuple[float, float] or BaseFit
        ``(L, C)`` in Henries/Farads, or the fitted model if ``return_model``.

    Notes
    -----
    For the "optimized" and "analytical" models the effective ``L, C`` absorb the
    coupling/ground capacitors, so they do depend on ``Cc``/``Ctog`` (the
    couplers pull the effective values); refit if the loading changes materially.
    Reusing a single fit across a coupling sweep is fine for the resonance
    frequency and loaded ``Q`` (which only need ``f_r`` and a measured linewidth)
    but not for ``Z_r``/``V_zpf``. To get the readout linewidth, pair this with
    :func:`simpleLOMs.extract_f0_kappa` and
    :func:`simpleLOMs.resonator_readout_params`.
    """
    if cpw_params is None:
        cpw_params = CPWParams()
    if freq is None:
        freq = rf.Frequency(4e9, 12e9, 20_001, unit="Hz")

    Cc1, Cc2, Ctog1, Ctog2 = _resolve_port_loads(Cc, Ctog, Cc1, Cc2, Ctog1, Ctog2)

    model = model.lower()
    if model not in ("optimized", "foster", "analytical"):
        raise ValueError(
            "model must be 'optimized', 'foster', or 'analytical', got {!r}".format(model)
        )

    if model == "foster":
        m = FosterFit(cpw_params=cpw_params)
        m.fit(freq, d=d)
        return m if return_model else (m.L, m.C)

    if model == "analytical":
        # Anchor to the *bare* (unloaded) λ/2 resonance of the CPW line, not to
        # the loaded circle-fit f0. The loaded resonance already contains the
        # Cc/Ctog pull, and get_network re-applies those capacitors, so anchoring
        # there double-counts the loading (a several-percent error). The bare
        # frequency is a geometry property; the caller may pass it via
        # analytical_f_r when it is already known.
        f_r = analytical_f_r if analytical_f_r is not None else bare_cpw_resonance_hz(
            d, cpw_params=cpw_params
        )
        # The closed form wants the impedance of the line geometry (~46 Ω for
        # the default CPW), not the 50 Ω port reference; its λ/2 effective
        # impedance is 2 Z_line / π.
        Z_line = analytical_Z0 if analytical_Z0 is not None else cpw_line_impedance(
            cpw_params, f_hz=f_r
        )
        m = AnalyticalFit()
        m.fit(freq, f_r=f_r, Z0=Z_line)
        return m if return_model else (m.L, m.C)

    # "optimized": locate (f0, kappa) on the caller's grid, then refine onto a
    # dense window so the stage-2 windowed least-squares resolves the linewidth.
    # A coarse or very wide input grid otherwise leaves the (L, C) fit
    # under-determined and can return an unphysical impedance. The window is
    # sized from the coarse linewidth (see the _FIT_LOM_REFINE_* constants):
    # broad, strongly coupled resonances need far more than the 0.5 GHz floor,
    # and the coarse f0 of such a dip can itself be off-centre by a sizeable
    # fraction of kappa, so bracketing failures retry with a doubled window.
    #
    # On a *pre-windowed* asymmetric grid the S11 circle-fit angular-velocity
    # peak can lock onto a spurious feature while |S11| still has a clear dip.
    # Prefer the dip whenever it disagrees with the circle fit by more than a
    # linewidth (or 100 MHz floor).
    coarse_net = cpw_resonator_network_2port(
        freq, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
    )
    f0_circle, kappa_coarse = circle_fit_f0_kappa(coarse_net, m=0, n=0)
    f0_dip = float(freq.f[int(np.argmin(np.abs(coarse_net.s[:, 0, 0])))])
    tol = max(
        float(kappa_coarse) if np.isfinite(kappa_coarse) and kappa_coarse > 0 else 0.0,
        0.1e9,
    )
    if abs(f0_circle - f0_dip) > tol:
        f0_coarse = f0_dip
        # Circle-fit kappa is not trustworthy if f0 itself was wrong; fall back
        # to the default refine span and let retries widen if needed.
        kappa_coarse = float("nan")
    else:
        f0_coarse = f0_circle

    half = _FIT_LOM_REFINE_SPAN_HZ
    if np.isfinite(kappa_coarse) and kappa_coarse > 0:
        half = max(half, _FIT_LOM_REFINE_KAPPA_MULT * kappa_coarse)
    half_cap = _FIT_LOM_REFINE_MAX_FRAC_F0 * f0_coarse
    half = min(half, half_cap)

    cfg = opt_config if opt_config is not None else OptimizationConfig(
        verbose=False, n_widths=4
    )

    for attempt in range(_FIT_LOM_REFINE_MAX_TRIES):
        refined = rf.Frequency(
            f0_coarse - half, f0_coarse + half, _FIT_LOM_REFINE_NPTS, unit="Hz"
        )
        net = cpw_resonator_network_2port(
            refined, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
        )
        m = OptimizedFit(config=cfg)
        try:
            m.fit(
                refined, data_ntw=net, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
                d=d, cpw_params=cpw_params, Z0=Z0,
            )
            return m if return_model else (m.L, m.C)
        except ValueError as err:
            if not _is_refine_window_error(err):
                raise
            if attempt == _FIT_LOM_REFINE_MAX_TRIES - 1 or half >= half_cap:
                raise ValueError(
                    "fit_lom could not bracket the resonance on a refine window "
                    f"of ±{half / 1e9:.2f} GHz around the coarse f0 = "
                    f"{f0_coarse / 1e9:.3f} GHz (coarse kappa = "
                    f"{kappa_coarse / 1e6:.1f} MHz). The loaded resonance is "
                    "either extremely broad or mislocated on the supplied grid; "
                    "pass a `freq` grid centred on the resonance and spanning "
                    "at least ~10 linewidths."
                ) from err
            logger.debug(
                "fit_lom refine window ±%.2f GHz failed to bracket the "
                "resonance (%s); retrying with ±%.2f GHz.",
                half / 1e9, err, min(2.0 * half, half_cap) / 1e9,
            )
            half = min(2.0 * half, half_cap)


def fit_lom_hanger(
    d: float,
    *,
    Cc_tap: float = 30e-15,
    termination: str = "open",
    Z0: float = 50.0,
    cpw_params: CPWParams | None = None,
    freq: rf.Frequency | None = None,
    opt_config: OptimizationConfig | None = None,
    return_model: bool = False,
):
    """
    Fit an effective ``(L, C)`` for a *hanger / notch* resonator against a CPW
    hanger ground truth (S21 dip).

    Unlike :func:`fit_lom`, which optimises against an **inline** CPW topology,
    this builds a hanger CPW reference and runs :class:`HangerOptimizedFit` so
    the lumped tank is calibrated for single-tap coupling — not rewired from an
    inline fit.

    Parameters
    ----------
    d : float
        Resonator length in metres.
    Cc_tap : float
        Tap coupling capacitance in Farads between the through feedline and the
        hanging resonator.
    termination : {"open", "short"}
        Far-end termination of the CPW hanger branch.  ``"open"`` is λ/2 (default,
        matching the usual open–open inline mode family); ``"short"`` is λ/4.
    Z0 : float
        Reference / feedline impedance in Ohms.
    cpw_params : CPWParams, optional
        CPW geometry / material. Defaults to ``CPWParams()``.
    freq : rf.Frequency, optional
        Frequency grid spanning the notch. Defaults to 1–12 GHz, 40_001 points
        (wider than the inline default so λ/4 short notches are also covered).
    opt_config : OptimizationConfig, optional
        Optimiser settings. Defaults to a quiet S21-oriented config.
    return_model : bool
        If True, return the fitted :class:`HangerOptimizedFit` instead of
        ``(L, C)``.

    Returns
    -------
    tuple[float, float] or HangerOptimizedFit
        ``(L, C)`` in Henries/Farads, or the fitted model if ``return_model``.
    """
    if cpw_params is None:
        cpw_params = CPWParams()
    if freq is None:
        freq = rf.Frequency(1e9, 12e9, 40_001, unit="Hz")
    if termination not in ("open", "short"):
        raise ValueError(
            "termination must be 'open' or 'short', got {!r}".format(termination)
        )

    net = hanger_resonator_network_2port(
        freq,
        [{"kind": "cpw", "d": d, "termination": termination}],
        [Cc_tap],
        cpw_params=cpw_params,
        Z0=Z0,
    )

    cfg = opt_config if opt_config is not None else OptimizationConfig(
        s_params=["S21"],
        verbose=False,
        n_widths=4.0,
        max_nfev=150,
    )
    m = HangerOptimizedFit(config=cfg)
    m.fit(
        freq,
        data_ntw=net,
        Cc_tap=Cc_tap,
        d=d,
        cpw_params=cpw_params,
        termination=termination,
        Z0=Z0,
    )
    return m if return_model else (m.L, m.C)


# =============================================================================
# §B  SHARED INTERNALS — circle-fit / metric / residual / serialisation helpers
#     used by both the single-device and load-grid comparison paths below.
# =============================================================================

def _to_jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {kk: _to_jsonable(vv) for kk, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def _make_serialisable(r, keys=None):
    if r is None:
        return None
    if keys is None:
        keys = DEFAULT_SERIALISABLE_KEYS
    out = {}
    for k in keys:
        if k in r:
            out[k] = _to_jsonable(r[k])
    return out


def _s_residual_sum(ref_net: rf.Network, lom_net: rf.Network) -> dict:
    if not np.allclose(lom_net.f, ref_net.f):
        lom_interp = lom_net.interpolate(ref_net.frequency)
    else:
        lom_interp = lom_net
    lom_s = lom_interp.s.copy()
    lom_s[:, 1, 0] *= -1
    lom_s[:, 0, 1] *= -1

    diff = np.abs(lom_s - ref_net.s)
    n_ports = ref_net.nports
    port_labels = {
        (m, n): f"S{m+1}{n+1}"
        for m in range(n_ports)
        for n in range(n_ports)
    }
    result = {}
    for (m, n), label in port_labels.items():
        result[label] = float(np.sum(diff[:, m, n]))
    result["total"] = float(np.sum(diff))
    return result


def _metrics(net: rf.Network) -> dict:
    f0_s11_hz, k_s11_hz = circle_fit_f0_kappa(net, m=0, n=0)
    f0_s22_hz, k_s22_hz = circle_fit_f0_kappa(net, m=1, n=1)
    try:
        f0_s12 = resonance_from_s_max(net, 0, 1) / 1e9
    except Exception:
        f0_s12 = float("nan")
    return {
        "f0_s11": f0_s11_hz / 1e9,
        "f0_s22": f0_s22_hz / 1e9,
        "k_s11": k_s11_hz,
        "k_s22": k_s22_hz,
        "f0_s12": f0_s12,
    }


def _shifted_freqs_cf(net: rf.Network, n_modes: int = 3) -> tuple[np.ndarray, np.ndarray]:
    f0s_s11, _ = circle_fit_modes(net, m=0, n=0, n_modes=n_modes)
    f0s_s22, _ = circle_fit_modes(net, m=1, n=1, n_modes=n_modes)
    stitched, sources = stitch_shifted_freqs(
        f0s_s11 / 1e9,
        f0s_s22 / 1e9,
        dedup_tol_ghz=0.05,
        return_sources=True,
    )
    if len(stitched) < n_modes:
        stitched = np.concatenate([stitched, np.full(n_modes - len(stitched), np.nan)])
        sources = np.concatenate([sources, np.full(n_modes - len(sources), None, dtype=object)])
    return stitched[:n_modes], sources[:n_modes]


def _ref_freqs_cf(
    sources: np.ndarray,
    f0_s11: float,
    f0_s22: float,
    f0_load1: float,
    f0_load2: float,
) -> np.ndarray:
    f0_mid = f0_s11 if (len(sources) > 1 and sources[1] == "s11") else f0_s22
    return np.sort([f0_load1, f0_mid, f0_load2])


def _pct_err(ref, val):
    return (ref - val) / ref * 100 if ref != 0 else float("nan")


def _shift_errors(ref, val):
    return (ref - val) / ref * 100


# =============================================================================
# §C  ACCURACY-TESTING HARNESS  (author-facing) — CPW-vs-LOM comparison with
#     loaded-network hybridization. End users doing design do NOT need this.
# =============================================================================

def analyze_system(
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    Lload1: float,
    Cload1: float,
    Lload2: float,
    Cload2: float,
    cpw_params: CPWParams | None = None,
    Z0: float = 50.0,
    analytical_f_r: float | None = None,
    analytical_Z0: float | None = None,
    opt_config: OptimizationConfig | None = None,
    verbose: bool = False,
    shifted: bool = False,
) -> dict:
    """
    Compare FosterFit, OptimizedFit, and AnalyticalFit against a CPW reference.

    Uses reflection circle fits on S11 and S22 for f0 and kappa extraction.

    ``analytical_f_r`` sets the bare (unloaded) resonance in Hz that anchors the
    analytical closed form; when omitted it is computed from the CPW geometry via
    :func:`simpleLOMs.networks.cpw.bare_cpw_resonance_hz` (no longer hardcoded to
    the 7 mm value). ``analytical_Z0`` optionally overrides the impedance used in
    that closed form.
    """
    if cpw_params is None:
        cpw_params = CPWParams()
    if opt_config is None:
        opt_config = OptimizationConfig(verbose=verbose, n_widths=4)

    cpw_coarse = cpw_resonator_network_2port(
        freq, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
    )
    f0_coarse, kappa_coarse = circle_fit_f0_kappa(cpw_coarse, m=0, n=0)

    if Cc1 == Cc2:
        refined_freq = rf.Frequency(
            f0_coarse - 0.5e9, f0_coarse + 0.5e9, 500_001, unit="Hz"
        )
    else:
        refined_freq = rf.Frequency(
            f0_coarse - 5 * kappa_coarse,
            f0_coarse + 5 * kappa_coarse,
            200_001,
            unit="Hz",
        )

    cpw_network = cpw_resonator_network_2port(
        refined_freq, d, Cc1, Cc2, Ctog1, Ctog2,
        cpw_params=cpw_params, Z0=Z0,
    )

    f0_cpw_s11, k_cpw_s11 = circle_fit_f0_kappa(cpw_network, m=0, n=0)
    f0_cpw_s22, k_cpw_s22 = circle_fit_f0_kappa(cpw_network, m=1, n=1)
    f0_cpw_s11 /= 1e9
    f0_cpw_s22 /= 1e9

    try:
        f0_cpw_s12 = resonance_from_s_max(cpw_network, m=0, n=1) / 1e9
    except Exception:
        f0_cpw_s12 = float("nan")

    logger.debug(
        "CPW S11 (circle fit): f0=%.4f GHz, κ=%.4f MHz",
        f0_cpw_s11, k_cpw_s11 / 1e6,
    )

    foster_model = FosterFit(cpw_params=cpw_params)
    foster_model.fit(freq, d=d)
    foster_network = foster_model.get_network(
        refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0,
    )

    optimized_model = OptimizedFit(config=opt_config)
    optimized_model.fit(
        refined_freq,
        data_ntw=cpw_network,
        Cc1=Cc1, Cc2=Cc2,
        Ctog1=Ctog1, Ctog2=Ctog2,
        d=d,
        cpw_params=cpw_params,
        Z0=Z0,
    )
    if shifted:
        optimized_network = optimized_model.get_network(
            refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0, shifted=True,
        )
    else:
        optimized_network = optimized_model.get_network(
            refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0,
        )

    Z0_for_analytical = analytical_Z0 if analytical_Z0 is not None else Z0
    f_r_for_analytical = (
        analytical_f_r if analytical_f_r is not None
        else bare_cpw_resonance_hz(d, cpw_params=cpw_params)
    )
    analytical_model = AnalyticalFit()
    analytical_model.fit(refined_freq, f_r=f_r_for_analytical, Z0=Z0_for_analytical)
    analytical_network = analytical_model.get_network(
        refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0,
    )

    m_foster = _metrics(foster_network)
    m_opt = _metrics(optimized_network)
    m_analytical = _metrics(analytical_network)

    freq_loaded = DEFAULT_FREQ_LOADED
    load_kw = dict(
        Lload1=Lload1, Cload1=Cload1,
        Lload2=Lload2, Cload2=Cload2,
    )

    cpw_loaded = cpw_resonator_loaded_network_2port(
        freq_loaded, d, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
        cpw_params=cpw_params, Z0=Z0, **load_kw,
    )

    if shifted:
        opt_loaded = lc_resonator_loaded_network_2port_shifted(
            freq=freq_loaded,
            Leff=optimized_model.L, Ceff=optimized_model.C,
            Cc1=Cc1, Cc2=Cc2, Z0=Z0, **load_kw,
        )
    else:
        opt_loaded = lc_resonator_loaded_network_2port(
            freq=freq_loaded,
            Leff=optimized_model.L, Ceff=optimized_model.C,
            Cc1=Cc1, Cc2=Cc2, Z0=Z0, **load_kw,
        )

    foster_loaded = lc_resonator_loaded_network_with_grounds_2port(
        freq=freq_loaded,
        Leff=foster_model.L, Ceff=foster_model.C,
        Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
        Z0=Z0, **load_kw,
    )

    analytical_loaded = lc_resonator_loaded_network_with_grounds_2port(
        freq=freq_loaded,
        Leff=analytical_model.L, Ceff=analytical_model.C,
        Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2,
        Z0=Z0, **load_kw,
    )

    cpw_shifted_s12 = np.array(resonances_from_s_max(cpw_loaded, m=0, n=1), dtype=float) / 1e9
    opt_shifted_s12 = np.array(resonances_from_s_max(opt_loaded, m=0, n=1), dtype=float) / 1e9
    foster_shifted_s12 = np.array(resonances_from_s_max(foster_loaded, m=0, n=1), dtype=float) / 1e9
    analytical_shifted_s12 = np.array(resonances_from_s_max(analytical_loaded, m=0, n=1), dtype=float) / 1e9

    net1 = lc_load_dressed_network_2(Lload1, Cload1, 1e-15, Cc1, freq=freq, Z0=Z0)
    net2 = lc_load_dressed_network_2(Lload2, Cload2, 1e-15, Cc2, freq=freq, Z0=Z0)
    f0_1, _ = circle_fit_modes(net1, m=0, n=0, n_modes=1)
    f0_2, _ = circle_fit_modes(net2, m=0, n=0, n_modes=1)
    f0_load1 = f0_1[0] / 1e9
    f0_load2 = f0_2[0] / 1e9

    cpw_ref_freqs_s12 = np.sort([f0_load1, f0_cpw_s12, f0_load2])
    opt_ref_freqs_s12 = np.sort([f0_load1, m_opt["f0_s12"], f0_load2])
    foster_ref_freqs_s12 = np.sort([f0_load1, m_foster["f0_s12"], f0_load2])
    analytical_ref_freqs_s12 = np.sort([f0_load1, m_analytical["f0_s12"], f0_load2])

    cpw_all_shifts_s12 = cpw_ref_freqs_s12 - cpw_shifted_s12
    opt_all_shifts_s12 = opt_ref_freqs_s12 - opt_shifted_s12
    foster_all_shifts_s12 = foster_ref_freqs_s12 - foster_shifted_s12
    analytical_all_shifts_s12 = analytical_ref_freqs_s12 - analytical_shifted_s12

    cpw_shifted_cf, cpw_sources_cf = _shifted_freqs_cf(cpw_loaded)
    opt_shifted_cf, opt_sources_cf = _shifted_freqs_cf(opt_loaded)
    foster_shifted_cf, foster_sources_cf = _shifted_freqs_cf(foster_loaded)
    analytical_shifted_cf, analytical_sources_cf = _shifted_freqs_cf(analytical_loaded)

    cpw_ref_freqs_cf = _ref_freqs_cf(cpw_sources_cf, f0_cpw_s11, f0_cpw_s22, f0_load1, f0_load2)
    opt_ref_freqs_cf = _ref_freqs_cf(opt_sources_cf, m_opt["f0_s11"], m_opt["f0_s22"], f0_load1, f0_load2)
    foster_ref_freqs_cf = _ref_freqs_cf(foster_sources_cf, m_foster["f0_s11"], m_foster["f0_s22"], f0_load1, f0_load2)
    analytical_ref_freqs_cf = _ref_freqs_cf(
        analytical_sources_cf, m_analytical["f0_s11"], m_analytical["f0_s22"], f0_load1, f0_load2
    )

    cpw_all_shifts_cf = cpw_ref_freqs_cf - cpw_shifted_cf
    opt_all_shifts_cf = opt_ref_freqs_cf - opt_shifted_cf
    foster_all_shifts_cf = foster_ref_freqs_cf - foster_shifted_cf
    analytical_all_shifts_cf = analytical_ref_freqs_cf - analytical_shifted_cf

    def _s_residual_integral(net_a, net_b):
        freqs = net_a.f
        diff = np.abs(net_a.s - net_b.s) ** 2
        integrand = diff.sum(axis=(1, 2))
        return np.trapz(integrand, freqs)

    foster_s_residuals_integral = _s_residual_integral(cpw_network, foster_network)
    opt_s_residuals_integral = _s_residual_integral(cpw_network, optimized_network)
    analytical_s_residuals_integral = _s_residual_integral(cpw_network, analytical_network)

    foster_s_residuals = _s_residual_sum(cpw_network, foster_network)
    opt_s_residuals = _s_residual_sum(cpw_network, optimized_network)
    analytical_s_residuals = _s_residual_sum(cpw_network, analytical_network)

    return {
        "cpw_network": cpw_network,
        "foster_model": foster_model,
        "optimized_model": optimized_model,
        "analytical_model": analytical_model,
        "foster_network": foster_network,
        "optimized_network": optimized_network,
        "analytical_network": analytical_network,
        "Z0 (Ohms)": Z0,
        "Cc1 (F)": Cc1,
        "Cc2 (F)": Cc2,
        "Ctog1 (F)": Ctog1,
        "Ctog2 (F)": Ctog2,
        "Optimized L (H)": optimized_model.L,
        "Optimized C (F)": optimized_model.C,
        "Foster L (H)": foster_model.L,
        "Foster C (F)": foster_model.C,
        "Analytical L (H)": analytical_model.L,
        "Analytical C (F)": analytical_model.C,
        "CPW f0 S11 (GHz)": f0_cpw_s11,
        "Optimized f0 S11 (GHz)": m_opt["f0_s11"],
        "Foster f0 S11 (GHz)": m_foster["f0_s11"],
        "Analytical f0 S11 (GHz)": m_analytical["f0_s11"],
        "Optimized f0 error S11 (%)": _pct_err(f0_cpw_s11, m_opt["f0_s11"]),
        "Foster f0 error S11 (%)": _pct_err(f0_cpw_s11, m_foster["f0_s11"]),
        "Analytical f0 error S11 (%)": _pct_err(f0_cpw_s11, m_analytical["f0_s11"]),
        "CPW kappa S11 (MHz)": k_cpw_s11 / 1e6,
        "Optimized kappa S11 (MHz)": m_opt["k_s11"] / 1e6,
        "Foster kappa S11 (MHz)": m_foster["k_s11"] / 1e6,
        "Analytical kappa S11 (MHz)": m_analytical["k_s11"] / 1e6,
        "Optimized kappa error S11 (%)": _pct_err(k_cpw_s11, m_opt["k_s11"]),
        "Foster kappa error S11 (%)": _pct_err(k_cpw_s11, m_foster["k_s11"]),
        "Analytical kappa error S11 (%)": _pct_err(k_cpw_s11, m_analytical["k_s11"]),
        "CPW f0 S22 (GHz)": f0_cpw_s22,
        "Optimized f0 S22 (GHz)": m_opt["f0_s22"],
        "Foster f0 S22 (GHz)": m_foster["f0_s22"],
        "Analytical f0 S22 (GHz)": m_analytical["f0_s22"],
        "Optimized f0 error S22 (%)": _pct_err(f0_cpw_s22, m_opt["f0_s22"]),
        "Foster f0 error S22 (%)": _pct_err(f0_cpw_s22, m_foster["f0_s22"]),
        "Analytical f0 error S22 (%)": _pct_err(f0_cpw_s22, m_analytical["f0_s22"]),
        "CPW kappa S22 (MHz)": k_cpw_s22 / 1e6,
        "Optimized kappa S22 (MHz)": m_opt["k_s22"] / 1e6,
        "Foster kappa S22 (MHz)": m_foster["k_s22"] / 1e6,
        "Analytical kappa S22 (MHz)": m_analytical["k_s22"] / 1e6,
        "Optimized kappa error S22 (%)": _pct_err(k_cpw_s22, m_opt["k_s22"]),
        "Foster kappa error S22 (%)": _pct_err(k_cpw_s22, m_foster["k_s22"]),
        "Analytical kappa error S22 (%)": _pct_err(k_cpw_s22, m_analytical["k_s22"]),
        "Foster S residuals": foster_s_residuals,
        "Optimized S residuals": opt_s_residuals,
        "Analytical S residuals": analytical_s_residuals,
        "Foster S residuals (integrated)": foster_s_residuals_integral,
        "Optimized S residuals (integrated)": opt_s_residuals_integral,
        "Analytical S residuals (integrated)": analytical_s_residuals_integral,
        "CPW shifted freqs CF (GHz)": cpw_shifted_cf,
        "Optimized shifted freqs CF (GHz)": opt_shifted_cf,
        "Foster shifted freqs CF (GHz)": foster_shifted_cf,
        "Analytical shifted freqs CF (GHz)": analytical_shifted_cf,
        "CPW all shifts (S12) (GHz)": cpw_all_shifts_s12,
        "Optimized all shifts (S12) (GHz)": opt_all_shifts_s12,
        "Foster all shifts (S12) (GHz)": foster_all_shifts_s12,
        "Analytical all shifts (S12) (GHz)": analytical_all_shifts_s12,
        "Optimized shift errors (S12) (%)": _shift_errors(cpw_all_shifts_s12, opt_all_shifts_s12),
        "Foster shift errors (S12) (%)": _shift_errors(cpw_all_shifts_s12, foster_all_shifts_s12),
        "Analytical shift errors (S12) (%)": _shift_errors(cpw_all_shifts_s12, analytical_all_shifts_s12),
        "CPW all shifts CF (GHz)": cpw_all_shifts_cf,
        "Optimized all shifts CF (GHz)": opt_all_shifts_cf,
        "Foster all shifts CF (GHz)": foster_all_shifts_cf,
        "Analytical all shifts CF (GHz)": analytical_all_shifts_cf,
        "Optimized shift errors CF (%)": _shift_errors(cpw_all_shifts_cf, opt_all_shifts_cf),
        "Foster shift errors CF (%)": _shift_errors(cpw_all_shifts_cf, foster_all_shifts_cf),
        "Analytical shift errors CF (%)": _shift_errors(cpw_all_shifts_cf, analytical_all_shifts_cf),
    }


analyze_system_cf = analyze_system


# ---------------------------------------------------------------------------
# Load-grid variant: reuse one model prefit across many load-frequency pairs.
# ---------------------------------------------------------------------------

@dataclass
class PrefitContext:
    refined_freq: rf.Frequency
    cpw_network: rf.Network
    optimized_model: OptimizedFit
    foster_model: FosterFit
    analytical_model: AnalyticalFit
    f0_cpw_s11: float
    f0_cpw_s22: float
    f0_cpw_s12: float
    m_opt: dict
    m_foster: dict
    m_analytical: dict


def _safe_resonances_from_s_max(
    network: rf.Network,
    m: int = 0,
    n: int = 1,
    n_modes: int = 3,
) -> np.ndarray:
    """Best-effort S-peak extraction (GHz); returns NaNs instead of raising."""
    try:
        vals = np.array(resonances_from_s_max(network, m=m, n=n), dtype=float) / 1e9
        if len(vals) < n_modes:
            vals = np.concatenate([vals, np.full(n_modes - len(vals), np.nan)])
        return vals[:n_modes]
    except Exception:
        return np.full(n_modes, np.nan, dtype=float)


def load_from_freq(f_ghz: float, C: float = 6e-13) -> tuple[float, float]:
    f_hz = f_ghz * 1e9
    L = 1.0 / ((2.0 * np.pi * f_hz) ** 2 * C)
    return L, C


def fit_models_once(
    *,
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    cpw_params: CPWParams | None = None,
    Z0: float = 50.0,
    analytical_Z0: float | None = None,
    analytical_f_r: float | None = None,
    opt_config: OptimizationConfig | None = None,
    verbose: bool = False,
    shifted: bool = False,
) -> PrefitContext:
    """
    Fit all three models once against the (unloaded) CPW and return a
    ``PrefitContext`` with their L/C, circle-fit f0/kappa metrics, and networks.

    This is the **load-free** comparison core: it never touches loads, so it can
    be used on its own to compare LOMs against the CPW ground truth. The load
    grid below reuses one prefit across many load-frequency pairs.
    """
    if cpw_params is None:
        cpw_params = CPWParams()
    if opt_config is None:
        opt_config = OptimizationConfig(verbose=verbose, n_widths=4)

    cpw_coarse = cpw_resonator_network_2port(
        freq, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
    )
    f0_coarse, kappa_coarse = circle_fit_f0_kappa(cpw_coarse, m=0, n=0)

    if Cc1 == Cc2:
        refined_freq = rf.Frequency(f0_coarse - 0.5e9, f0_coarse + 0.5e9, 500_001, unit="Hz")
    else:
        refined_freq = rf.Frequency(
            f0_coarse - 5 * kappa_coarse,
            f0_coarse + 5 * kappa_coarse,
            200_001,
            unit="Hz",
        )

    cpw_network = cpw_resonator_network_2port(
        refined_freq, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
    )
    f0_cpw_s11, _ = circle_fit_f0_kappa(cpw_network, m=0, n=0)
    f0_cpw_s22, _ = circle_fit_f0_kappa(cpw_network, m=1, n=1)
    try:
        f0_cpw_s12 = resonance_from_s_max(cpw_network, m=0, n=1) / 1e9
    except Exception:
        f0_cpw_s12 = float("nan")

    foster_model = FosterFit(cpw_params=cpw_params)
    foster_model.fit(freq, d=d)
    foster_network = foster_model.get_network(
        refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0
    )

    optimized_model = OptimizedFit(config=opt_config)
    optimized_model.fit(
        refined_freq,
        data_ntw=cpw_network,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        d=d,
        cpw_params=cpw_params,
        Z0=Z0,
    )
    optimized_network = optimized_model.get_network(
        refined_freq,
        Cc1=Cc1,
        Cc2=Cc2,
        Z0=Z0,
        shifted=shifted,
    ) if shifted else optimized_model.get_network(refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0)

    f_r_for_analytical = (
        analytical_f_r if analytical_f_r is not None
        else bare_cpw_resonance_hz(d, cpw_params=cpw_params)
    )
    analytical_model = AnalyticalFit()
    analytical_model.fit(refined_freq, f_r=f_r_for_analytical, Z0=(analytical_Z0 or Z0))
    analytical_network = analytical_model.get_network(
        refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0
    )

    return PrefitContext(
        refined_freq=refined_freq,
        cpw_network=cpw_network,
        optimized_model=optimized_model,
        foster_model=foster_model,
        analytical_model=analytical_model,
        f0_cpw_s11=f0_cpw_s11 / 1e9,
        f0_cpw_s22=f0_cpw_s22 / 1e9,
        f0_cpw_s12=f0_cpw_s12,
        m_opt=_metrics(optimized_network),
        m_foster=_metrics(foster_network),
        m_analytical=_metrics(analytical_network),
    )


def _compute_combo_result(
    *,
    freq: rf.Frequency,
    freq_loaded: rf.Frequency,
    prefit: PrefitContext,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    cpw_params: CPWParams,
    Z0: float,
    Lload1: float,
    Cload1: float,
    Lload2: float,
    Cload2: float,
    target_f1_ghz: float,
    target_f2_ghz: float,
    shifted: bool = False,
) -> dict:
    net1 = lc_load_dressed_network_2(Lload1, Cload1, 1e-15, Cc1, freq=freq, Z0=Z0)
    net2 = lc_load_dressed_network_2(Lload2, Cload2, 1e-15, Cc2, freq=freq, Z0=Z0)
    f0_1_cf, _ = circle_fit_modes(net1, m=0, n=0, n_modes=1)
    f0_2_cf, _ = circle_fit_modes(net2, m=0, n=0, n_modes=1)
    f0_load1_ghz = target_f1_ghz if np.isfinite(target_f1_ghz) else (float(f0_1_cf[0]) / 1e9)
    f0_load2_ghz = target_f2_ghz if np.isfinite(target_f2_ghz) else (float(f0_2_cf[0]) / 1e9)

    load_kw = dict(Lload1=Lload1, Cload1=Cload1, Lload2=Lload2, Cload2=Cload2)
    cpw_loaded = cpw_resonator_loaded_network_2port(
        freq_loaded,
        d,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        cpw_params=cpw_params,
        Z0=Z0,
        **load_kw,
    )
    if shifted:
        opt_loaded = lc_resonator_loaded_network_2port_shifted(
            freq=freq_loaded,
            Leff=prefit.optimized_model.L,
            Ceff=prefit.optimized_model.C,
            Cc1=Cc1,
            Cc2=Cc2,
            Z0=Z0,
            **load_kw,
        )
    else:
        opt_loaded = lc_resonator_loaded_network_2port(
            freq=freq_loaded,
            Leff=prefit.optimized_model.L,
            Ceff=prefit.optimized_model.C,
            Cc1=Cc1,
            Cc2=Cc2,
            Z0=Z0,
            **load_kw,
        )
    foster_loaded = lc_resonator_loaded_network_with_grounds_2port(
        freq=freq_loaded,
        Leff=prefit.foster_model.L,
        Ceff=prefit.foster_model.C,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        Z0=Z0,
        **load_kw,
    )
    analytical_loaded = lc_resonator_loaded_network_with_grounds_2port(
        freq=freq_loaded,
        Leff=prefit.analytical_model.L,
        Ceff=prefit.analytical_model.C,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        Z0=Z0,
        **load_kw,
    )

    cpw_sf_cf, cpw_src = _shifted_freqs_cf(cpw_loaded)
    opt_sf_cf, opt_src = _shifted_freqs_cf(opt_loaded)
    fos_sf_cf, fos_src = _shifted_freqs_cf(foster_loaded)
    ana_sf_cf, ana_src = _shifted_freqs_cf(analytical_loaded)

    cpw_ref_cf = _ref_freqs_cf(cpw_src, prefit.f0_cpw_s11, prefit.f0_cpw_s22, f0_load1_ghz, f0_load2_ghz)
    opt_ref_cf = _ref_freqs_cf(opt_src, prefit.m_opt["f0_s11"], prefit.m_opt["f0_s22"], f0_load1_ghz, f0_load2_ghz)
    fos_ref_cf = _ref_freqs_cf(fos_src, prefit.m_foster["f0_s11"], prefit.m_foster["f0_s22"], f0_load1_ghz, f0_load2_ghz)
    ana_ref_cf = _ref_freqs_cf(ana_src, prefit.m_analytical["f0_s11"], prefit.m_analytical["f0_s22"], f0_load1_ghz, f0_load2_ghz)

    cpw_shifts_cf = cpw_ref_cf - cpw_sf_cf
    opt_shifts_cf = opt_ref_cf - opt_sf_cf
    fos_shifts_cf = fos_ref_cf - fos_sf_cf
    ana_shifts_cf = ana_ref_cf - ana_sf_cf

    cpw_sf_s12 = _safe_resonances_from_s_max(cpw_loaded, m=0, n=1, n_modes=3)
    opt_sf_s12 = _safe_resonances_from_s_max(opt_loaded, m=0, n=1, n_modes=3)
    fos_sf_s12 = _safe_resonances_from_s_max(foster_loaded, m=0, n=1, n_modes=3)
    ana_sf_s12 = _safe_resonances_from_s_max(analytical_loaded, m=0, n=1, n_modes=3)

    cpw_ref_s12 = np.sort([f0_load1_ghz, prefit.f0_cpw_s12, f0_load2_ghz])
    opt_ref_s12 = np.sort([f0_load1_ghz, prefit.m_opt["f0_s12"], f0_load2_ghz])
    fos_ref_s12 = np.sort([f0_load1_ghz, prefit.m_foster["f0_s12"], f0_load2_ghz])
    ana_ref_s12 = np.sort([f0_load1_ghz, prefit.m_analytical["f0_s12"], f0_load2_ghz])

    return {
        "_load_combo": {"f_load1_ghz": f0_load1_ghz, "f_load2_ghz": f0_load2_ghz},
        "Lload1": Lload1,
        "Cload1": Cload1,
        "Lload2": Lload2,
        "Cload2": Cload2,
        "cpw_shifted_freqs_cf": cpw_sf_cf,
        "cpw_all_shifts_cf": cpw_shifts_cf,
        "opt_shifted_freqs_cf": opt_sf_cf,
        "opt_all_shifts_cf": opt_shifts_cf,
        "opt_shift_errors_cf": _shift_errors(cpw_shifts_cf, opt_shifts_cf),
        "foster_shifted_freqs_cf": fos_sf_cf,
        "foster_all_shifts_cf": fos_shifts_cf,
        "foster_shift_errors_cf": _shift_errors(cpw_shifts_cf, fos_shifts_cf),
        "analytical_shifted_freqs_cf": ana_sf_cf,
        "analytical_all_shifts_cf": ana_shifts_cf,
        "analytical_shift_errors_cf": _shift_errors(cpw_shifts_cf, ana_shifts_cf),
        "cpw_shifted_freqs_s12": cpw_sf_s12,
        "cpw_all_shifts_s12": cpw_ref_s12 - cpw_sf_s12,
        "opt_shifted_freqs_s12": opt_sf_s12,
        "opt_all_shifts_s12": opt_ref_s12 - opt_sf_s12,
        "opt_shift_errors_s12": _shift_errors(cpw_ref_s12 - cpw_sf_s12, opt_ref_s12 - opt_sf_s12),
        "foster_shifted_freqs_s12": fos_sf_s12,
        "foster_all_shifts_s12": fos_ref_s12 - fos_sf_s12,
        "foster_shift_errors_s12": _shift_errors(cpw_ref_s12 - cpw_sf_s12, fos_ref_s12 - fos_sf_s12),
        "analytical_shifted_freqs_s12": ana_sf_s12,
        "analytical_all_shifts_s12": ana_ref_s12 - ana_sf_s12,
        "analytical_shift_errors_s12": _shift_errors(cpw_ref_s12 - cpw_sf_s12, ana_ref_s12 - ana_sf_s12),
    }


def evaluate_load_grid(
    *,
    freq: rf.Frequency,
    prefit: PrefitContext,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    load1_freqs_hz: np.ndarray,
    load2_freqs_hz: np.ndarray,
    Cload1_fixed: float = 6e-13,
    Cload2_fixed: float = 6e-13,
    freq_loaded: rf.Frequency | None = None,
    cpw_params: CPWParams | None = None,
    Z0: float = 50.0,
    shifted: bool = False,
) -> list[dict]:
    if cpw_params is None:
        cpw_params = CPWParams()
    if freq_loaded is None:
        freq_loaded = DEFAULT_FREQ_LOADED

    results = []
    for f1_hz, f2_hz in itertools.product(np.atleast_1d(load1_freqs_hz), np.atleast_1d(load2_freqs_hz)):
        f1_ghz = float(f1_hz) / 1e9
        f2_ghz = float(f2_hz) / 1e9
        L1, C1 = load_from_freq(f1_ghz, C=Cload1_fixed)
        L2, C2 = load_from_freq(f2_ghz, C=Cload2_fixed)
        results.append(
            _compute_combo_result(
                freq=freq,
                freq_loaded=freq_loaded,
                prefit=prefit,
                d=d,
                Cc1=Cc1,
                Cc2=Cc2,
                Ctog1=Ctog1,
                Ctog2=Ctog2,
                cpw_params=cpw_params,
                Z0=Z0,
                Lload1=L1,
                Cload1=C1,
                Lload2=L2,
                Cload2=C2,
                target_f1_ghz=f1_ghz,
                target_f2_ghz=f2_ghz,
                shifted=shifted,
            )
        )
    return results


def analyze_system_load_grid(
    *,
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    load1_freqs_hz: np.ndarray,
    load2_freqs_hz: np.ndarray,
    cpw_params: CPWParams | None = None,
    Z0: float = 50.0,
    analytical_Z0: float | None = None,
    analytical_f_r: float | None = None,
    opt_config: OptimizationConfig | None = None,
    verbose: bool = False,
    shifted: bool = False,
    Cload1_fixed: float = 6e-13,
    Cload2_fixed: float = 6e-13,
) -> dict:
    prefit = fit_models_once(
        freq=freq,
        d=d,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        cpw_params=cpw_params,
        Z0=Z0,
        analytical_Z0=analytical_Z0,
        analytical_f_r=analytical_f_r,
        opt_config=opt_config,
        verbose=verbose,
        shifted=shifted,
    )
    load_sweep_results = evaluate_load_grid(
        freq=freq,
        prefit=prefit,
        d=d,
        Cc1=Cc1,
        Cc2=Cc2,
        Ctog1=Ctog1,
        Ctog2=Ctog2,
        load1_freqs_hz=load1_freqs_hz,
        load2_freqs_hz=load2_freqs_hz,
        Cload1_fixed=Cload1_fixed,
        Cload2_fixed=Cload2_fixed,
        cpw_params=cpw_params,
        Z0=Z0,
        shifted=shifted,
    )

    foster_network = prefit.foster_model.get_network(
        prefit.refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0
    )
    optimized_network = prefit.optimized_model.get_network(
        prefit.refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0, shifted=shifted
    ) if shifted else prefit.optimized_model.get_network(
        prefit.refined_freq, Cc1=Cc1, Cc2=Cc2, Z0=Z0
    )
    analytical_network = prefit.analytical_model.get_network(
        prefit.refined_freq, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, Z0=Z0
    )
    cpw_metrics = _metrics(prefit.cpw_network)

    def _s_residual_integral(net_a: rf.Network, net_b: rf.Network) -> float:
        freqs = net_a.f
        diff = np.abs(net_a.s - net_b.s) ** 2
        return float(np.trapz(diff.sum(axis=(1, 2)), freqs))

    foster_s_residuals = _s_residual_sum(prefit.cpw_network, foster_network)
    opt_s_residuals = _s_residual_sum(prefit.cpw_network, optimized_network)
    analytical_s_residuals = _s_residual_sum(prefit.cpw_network, analytical_network)
    foster_s_residuals_integral = _s_residual_integral(prefit.cpw_network, foster_network)
    opt_s_residuals_integral = _s_residual_integral(prefit.cpw_network, optimized_network)
    analytical_s_residuals_integral = _s_residual_integral(prefit.cpw_network, analytical_network)

    return {
        "Z0 (Ohms)": Z0,
        "Cc1 (F)": Cc1,
        "Cc2 (F)": Cc2,
        "Ctog1 (F)": Ctog1,
        "Ctog2 (F)": Ctog2,
        "Optimized L (H)": prefit.optimized_model.L,
        "Optimized C (F)": prefit.optimized_model.C,
        "Foster L (H)": prefit.foster_model.L,
        "Foster C (F)": prefit.foster_model.C,
        "Analytical L (H)": prefit.analytical_model.L,
        "Analytical C (F)": prefit.analytical_model.C,
        "CPW f0 S11 (GHz)": prefit.f0_cpw_s11,
        "CPW f0 S22 (GHz)": prefit.f0_cpw_s22,
        "CPW kappa S11 (MHz)": cpw_metrics["k_s11"] / 1e6,
        "CPW kappa S22 (MHz)": cpw_metrics["k_s22"] / 1e6,
        "Optimized f0 S11 (GHz)": prefit.m_opt["f0_s11"],
        "Optimized kappa S11 (MHz)": prefit.m_opt["k_s11"] / 1e6,
        "Foster f0 S11 (GHz)": prefit.m_foster["f0_s11"],
        "Foster kappa S11 (MHz)": prefit.m_foster["k_s11"] / 1e6,
        "Analytical f0 S11 (GHz)": prefit.m_analytical["f0_s11"],
        "Analytical kappa S11 (MHz)": prefit.m_analytical["k_s11"] / 1e6,
        "Optimized f0 S22 (GHz)": prefit.m_opt["f0_s22"],
        "Optimized kappa S22 (MHz)": prefit.m_opt["k_s22"] / 1e6,
        "Foster f0 S22 (GHz)": prefit.m_foster["f0_s22"],
        "Foster kappa S22 (MHz)": prefit.m_foster["k_s22"] / 1e6,
        "Analytical f0 S22 (GHz)": prefit.m_analytical["f0_s22"],
        "Analytical kappa S22 (MHz)": prefit.m_analytical["k_s22"] / 1e6,
        "Optimized f0 error S11 (%)": _pct_err(prefit.f0_cpw_s11, prefit.m_opt["f0_s11"]),
        "Foster f0 error S11 (%)": _pct_err(prefit.f0_cpw_s11, prefit.m_foster["f0_s11"]),
        "Analytical f0 error S11 (%)": _pct_err(prefit.f0_cpw_s11, prefit.m_analytical["f0_s11"]),
        "Optimized f0 error S22 (%)": _pct_err(prefit.f0_cpw_s22, prefit.m_opt["f0_s22"]),
        "Foster f0 error S22 (%)": _pct_err(prefit.f0_cpw_s22, prefit.m_foster["f0_s22"]),
        "Analytical f0 error S22 (%)": _pct_err(prefit.f0_cpw_s22, prefit.m_analytical["f0_s22"]),
        "Optimized kappa error S11 (%)": _pct_err(cpw_metrics["k_s11"], prefit.m_opt["k_s11"]),
        "Foster kappa error S11 (%)": _pct_err(cpw_metrics["k_s11"], prefit.m_foster["k_s11"]),
        "Analytical kappa error S11 (%)": _pct_err(cpw_metrics["k_s11"], prefit.m_analytical["k_s11"]),
        "Optimized kappa error S22 (%)": _pct_err(cpw_metrics["k_s22"], prefit.m_opt["k_s22"]),
        "Foster kappa error S22 (%)": _pct_err(cpw_metrics["k_s22"], prefit.m_foster["k_s22"]),
        "Analytical kappa error S22 (%)": _pct_err(cpw_metrics["k_s22"], prefit.m_analytical["k_s22"]),
        "Foster S residuals": foster_s_residuals,
        "Optimized S residuals": opt_s_residuals,
        "Analytical S residuals": analytical_s_residuals,
        "Foster S residuals (integrated)": foster_s_residuals_integral,
        "Optimized S residuals (integrated)": opt_s_residuals_integral,
        "Analytical S residuals (integrated)": analytical_s_residuals_integral,
        "load_sweep_results": load_sweep_results,
    }


# ---------------------------------------------------------------------------
# Sweep runners over a grid of device parameters (Cc, etc.)
# ---------------------------------------------------------------------------

def run_accuracy_sweep(
    sweep_params: dict,
    fixed_params: dict,
    save_path: str | None = None,
    skip_errors: bool = True,
    verbose: bool = True,
) -> list:
    """Run analyze_system() over a grid of parameter combinations."""
    try:
        from tqdm import tqdm
        _tqdm_available = True
    except ImportError:
        _tqdm_available = False

    def _dummy_tqdm(iterable, **kwargs):
        print(f"Running ({len(list(iterable))} points)...")
        return iter(iterable)

    param_names = list(sweep_params.keys())
    param_values = [np.asarray(v) for v in sweep_params.values()]
    grid_points = list(itertools.product(*param_values))
    n_total = len(grid_points)
    n_failed = 0

    if verbose:
        print(
            f"Sweep grid: "
            f"{' × '.join(f'{k}({len(v)})' for k, v in sweep_params.items())} "
            f"= {n_total} points"
        )

    flat_results = []
    _tqdm_func = tqdm if _tqdm_available else _dummy_tqdm

    json_file = None
    first_entry = [True]
    if save_path and save_path.endswith(".json"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        json_file = open(save_path, "w")
        json_file.write("[\n")

    pbar = _tqdm_func(grid_points, desc="Sweeping", unit="pt", disable=not verbose)
    for point in pbar:
        combo = dict(zip(param_names, point))
        if verbose and _tqdm_available:
            pbar.set_postfix_str(
                "  ".join(f"{k}={v:.3g}" for k, v in combo.items()),
                refresh=False,
            )
        call_kwargs = {**fixed_params, **combo}
        try:
            result = analyze_system(**call_kwargs)
            result["_sweep_point"] = combo
            flat_results.append(result)
            if json_file is not None:
                entry = _make_serialisable(result)
                prefix = "" if first_entry[0] else ","
                json_file.write(prefix + json.dumps(entry, indent=2, default=str) + "\n")
                json_file.flush()
                first_entry[0] = False
        except Exception as exc:
            if skip_errors:
                n_failed += 1
                logger.warning("analyze_system failed at %s: %s", combo, exc)
                flat_results.append(None)
            else:
                raise

    if json_file is not None:
        json_file.write("]\n")
        json_file.close()

    if verbose and n_failed > 0:
        print(f"  {n_failed}/{n_total} points failed and were skipped.")

    def _reshape(flat, shapes):
        if len(shapes) == 1:
            return flat
        stride = int(np.prod(shapes[1:]))
        return [
            _reshape(flat[i * stride:(i + 1) * stride], shapes[1:])
            for i in range(shapes[0])
        ]

    shapes = [len(v) for v in param_values]
    results_grid = _reshape(flat_results, shapes)

    if save_path and not save_path.endswith(".json"):
        serialisable_flat = [_make_serialisable(r) for r in flat_results]
        if save_path.endswith(".csv"):
            rows = []
            for r in serialisable_flat:
                if r is None:
                    rows.append({k: np.nan for k in param_names})
                    continue
                row = {**r.get("_sweep_point", {})}
                for k, v in r.items():
                    if k == "_sweep_point":
                        continue
                    if isinstance(v, list):
                        for idx, val in enumerate(v):
                            row[f"{k}[{idx}]"] = val
                    elif isinstance(v, (int, float, str, type(None))):
                        row[k] = v
                rows.append(row)
            pd.DataFrame(rows).to_csv(save_path, index=False)
            if verbose:
                print(f"Saved CSV: {save_path}")

    return results_grid


run_accuracy_sweep_cf = run_accuracy_sweep


def run_accuracy_sweep_load_grid(
    *,
    sweep_params: dict,
    fixed_params: dict,
    load1_freqs_hz: np.ndarray,
    load2_freqs_hz: np.ndarray,
    save_path: str | None = None,
    skip_errors: bool = True,
    verbose: bool = True,
    n_jobs: int = 1,
) -> list:
    param_names = list(sweep_params.keys())
    param_values = [np.asarray(v) for v in sweep_params.values()]
    grid_points = list(itertools.product(*param_values))

    flat_results = []
    json_file = None
    first_entry = True
    if save_path and save_path.endswith(".json"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        json_file = open(save_path, "w")
        json_file.write("[\n")

    combos = [dict(zip(param_names, point)) for point in grid_points]
    t_start = time.perf_counter()

    def _run_combo(combo: dict) -> tuple[dict, dict]:
        call_kwargs = {**fixed_params, **combo}
        result = analyze_system_load_grid(
            **call_kwargs,
            load1_freqs_hz=load1_freqs_hz,
            load2_freqs_hz=load2_freqs_hz,
        )
        result["_sweep_point"] = combo
        return combo, result

    def _error_result(combo: dict, exc: Exception) -> dict:
        return {
            "_sweep_point": combo,
            "_error": str(exc),
            "Cc1 (F)": combo.get("Cc1", np.nan),
            "Cc2 (F)": combo.get("Cc2", np.nan),
        }

    if n_jobs <= 1:
        for combo in combos:
            try:
                _, result = _run_combo(combo)
                flat_results.append(result)
                if json_file is not None:
                    entry = _to_jsonable(result)
                    prefix = "" if first_entry else ","
                    json_file.write(prefix + json.dumps(entry, indent=2, default=str) + "\n")
                    json_file.flush()
                    first_entry = False
            except Exception as exc:
                if skip_errors:
                    logger.warning("analyze_system_load_grid failed at %s: %s", combo, exc)
                    err = _error_result(combo, exc)
                    flat_results.append(err)
                    if json_file is not None:
                        entry = _to_jsonable(err)
                        prefix = "" if first_entry else ","
                        json_file.write(prefix + json.dumps(entry, indent=2, default=str) + "\n")
                        json_file.flush()
                        first_entry = False
                else:
                    raise
    else:
        ordered_results: list[dict | None] = [None] * len(combos)
        combo_to_index = {tuple(combo[k] for k in param_names): i for i, combo in enumerate(combos)}
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(_run_combo, combo): combo for combo in combos}
            for fut in as_completed(futures):
                combo = futures[fut]
                idx_key = tuple(combo[k] for k in param_names)
                idx = combo_to_index[idx_key]
                try:
                    _, result = fut.result()
                    ordered_results[idx] = result
                except Exception as exc:
                    if skip_errors:
                        logger.warning("analyze_system_load_grid failed at %s: %s", combo, exc)
                        ordered_results[idx] = _error_result(combo, exc)
                    else:
                        raise
        flat_results = ordered_results
        if json_file is not None:
            for result in flat_results:
                entry = _to_jsonable(result)
                prefix = "" if first_entry else ","
                json_file.write(prefix + json.dumps(entry, indent=2, default=str) + "\n")
                json_file.flush()
                first_entry = False

    if json_file is not None:
        json_file.write("]\n")
        json_file.close()

    def _reshape(flat: list, shapes: list[int]) -> list:
        if len(shapes) == 1:
            return flat
        stride = int(np.prod(shapes[1:]))
        return [_reshape(flat[i * stride:(i + 1) * stride], shapes[1:]) for i in range(shapes[0])]

    shapes = [len(v) for v in param_values]
    results_grid = _reshape(flat_results, shapes)

    if save_path and save_path.endswith(".csv"):
        serialisable_flat = [_to_jsonable(r) for r in flat_results]
        rows = []
        for r in serialisable_flat:
            if r is None:
                rows.append({k: np.nan for k in param_names})
                continue
            if "_error" in r:
                row = {**r.get("_sweep_point", {})}
                row["_error"] = r["_error"]
                rows.append(row)
                continue
            row = {**r.get("_sweep_point", {})}
            row["n_load_combos"] = len(r.get("load_sweep_results", []))
            rows.append(row)
        pd.DataFrame(rows).to_csv(save_path, index=False)

    if verbose:
        print(f"Completed {len(grid_points)} Cc points.")
        elapsed = time.perf_counter() - t_start
        print(f"n_jobs={n_jobs}, elapsed={elapsed:.1f}s")
    return results_grid


def run_accuracy_sweep_load_grid_apr22(
    *,
    sweep_params: dict,
    fixed_params: dict,
    load1_freqs_hz: np.ndarray,
    load2_freqs_hz: np.ndarray,
    save_path: str | None = None,
    skip_errors: bool = True,
    verbose: bool = True,
    n_jobs: int = 1,
) -> list:
    """APR22 compatibility wrapper for the load-grid sweep runner."""
    return run_accuracy_sweep_load_grid(
        sweep_params=sweep_params,
        fixed_params=fixed_params,
        load1_freqs_hz=load1_freqs_hz,
        load2_freqs_hz=load2_freqs_hz,
        save_path=save_path,
        skip_errors=skip_errors,
        verbose=verbose,
        n_jobs=n_jobs,
    )


# Backward-compatible aliases (previously in load_sweeps.py / the dead
# system_alternate_new.py). Kept so existing callers keep importing cleanly.
analyze_system_cf_new = analyze_system_load_grid
run_accuracy_sweep_cf_new = run_accuracy_sweep_load_grid
run_accuracy_sweep_cf_new_APR22 = run_accuracy_sweep_load_grid_apr22
