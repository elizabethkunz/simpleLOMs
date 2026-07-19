from __future__ import annotations

"""
sweeps.py
---------
Parameter sweep infrastructure for simpleLOMs.

The central idea is a SweepConfig dataclass that holds a complete snapshot
of a device (geometry, circuit parameters, frequency sweep).  A sweep then
replaces one field at a time using dataclasses.replace(), which returns a
new config with everything else held fixed — no mutation, no side effects.

Typical usage
-------------
    from simpleLOMs.sweeps import SweepConfig, sweep_length, sweep_coupling

    base = SweepConfig(
        cpw_params=CPWParams(),
        freq=rf.Frequency(4e9, 12e9, 10_001, unit="Hz"),
        d=7e-3,
        Cc1=5e-15, Cc2=5e-15,
        Ctog1=1e-14, Ctog2=1e-14,
        Lload1=5e-10, Cload1=6e-13,
        Lload2=5e-10, Cload2=6e-13,
    )

    # Sweep resonator length, extract f0 and kappa from CPW (default)
    df = sweep_length(np.linspace(5e-3, 10e-3, 20), base,
                      extract=["f0_s11", "kappa_s11"])

    # Sweep coupling cap using the already-fitted Foster model
    df = sweep_coupling(np.linspace(1e-15, 20e-15, 30), base,
                        model="foster", extract=["f0_s11", "kappa_s11", "Q_s11"])

Default model options
-------------
"cpw"        Built from scratch at every sweep point.  Ground truth.
"foster"     FosterFit.  Re-fit when d changes; reuses L/C otherwise (fast).
"optimized"  OptimizedFit.  Re-fit when d changes; reuses L/C otherwise.
"analytical" AnalyticalFit.  Re-fit at every point (formula is instant).


"""

import dataclasses
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
import skrf as rf

from simpleLOMs.params import CPWParams
from simpleLOMs.analysis import (
    resonance,
    resonances_from_s,
    fwhm_from_trace_db,
)
from simpleLOMs.networks.cpw import (
    cpw_resonator_network_2port,
    cpw_resonator_loaded_network_2port,
)
from simpleLOMs.networks.lc import (
    lc_resonator_network_with_grounds_2port,
    lc_resonator_loaded_network_with_grounds_2port,
)
from simpleLOMs.models.foster_fit import FosterFit
from simpleLOMs.models.optimized_fit import OptimizedFit, OptimizationConfig
from simpleLOMs.models.analytical_fit import AnalyticalFit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight measure sweep (arbitrary knobs)
# ---------------------------------------------------------------------------

def measure_sweep(values, measure) -> dict:
    """
    Apply ``measure(v) -> dict`` at each value and return column arrays.

    This is the small reusable helper used by the tutorials for knobs that are
    not ``SweepConfig`` fields (e.g. a CPW gap via ``dataclasses.replace``).

    Parameters
    ----------
    values : array-like
        Sweep points.
    measure : callable
        ``measure(v)`` must return a dict of scalar results.

    Returns
    -------
    dict[str, np.ndarray]
        One array per measured key, plus ``\"value\"`` holding the swept inputs.
    """
    values = np.asarray(values)
    rows = [measure(v) for v in values]
    if not rows:
        return {"value": values}
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    cols["value"] = values
    return cols


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """
    A complete snapshot of a device and its operating conditions.

    Every parameter that any sweep might want to vary is a field here.
    The sweep machinery calls dataclasses.replace(config, field=new_value)
    to create a modified copy at each sweep point — the original is never
    mutated.

    Parameters
    ----------
    cpw_params : CPWParams
        Physical geometry of the CPW line.
    freq : rf.Frequency
        Frequency sweep used for network construction.
        Should be wide enough to contain all resonances of interest.
    d : float
        Resonator length in metres.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    Lload1, Cload1 : float
        Inductance (H) and capacitance (F) of the load on port 1 side.
    Lload2, Cload2 : float
        Inductance (H) and capacitance (F) of the load on port 2 side.
    Z0 : float
        Reference impedance in Ohms (default 50 Ω).
    analytical_Z0 : float or None
        Characteristic impedance used in the AnalyticalFit formula.
        If None, Z0 is used.
    opt_config : OptimizationConfig or None
        Hyperparameters for OptimizedFit.  Uses defaults if None.

    Examples
    --------
    Creating a base config::

        base = SweepConfig(
            cpw_params=CPWParams(),
            freq=rf.Frequency(4e9, 12e9, 10_001, unit="Hz"),
            d=7e-3,
            Cc1=5e-15, Cc2=5e-15,
            Ctog1=1e-14, Ctog2=1e-14,
            Lload1=5e-10, Cload1=6e-13,
            Lload2=5e-10, Cload2=6e-13,
        )

    Manually overriding one field::

        longer = dataclasses.replace(base, d=9e-3)
    """
    cpw_params:    CPWParams
    freq:          rf.Frequency
    d:             float
    Cc1:           float
    Cc2:           float
    Ctog1:         float
    Ctog2:         float
    Lload1:        float
    Cload1:        float
    Lload2:        float
    Cload2:        float
    Z0:            float = 50.0
    analytical_Z0: float | None = None
    opt_config:    OptimizationConfig | None = None


# ---------------------------------------------------------------------------
# Parameters that require a model refit when they change
# ---------------------------------------------------------------------------

# These are the fields that change the resonator's L and C.
# Any sweep over these parameters must re-fit the model at every point.
# Sweeps over the remaining parameters (Cc, Ctog, loads) can reuse L/C.
_REFIT_PARAMS = {"d", "cpw_params"}


# ---------------------------------------------------------------------------
# Internal: build a network from a config + fitted L/C (or CPW directly)
# ---------------------------------------------------------------------------

def _build_2port_network(
    cfg: SweepConfig,
    model: str,
    L: float | None = None,
    C: float | None = None,
) -> rf.Network:
    """
    Build a 2-port network from cfg.

    For "cpw", builds directly from the CPW transmission line model.
    For the three LC models, uses the supplied L and C values with the
    "with_grounds" topology (Ctog1/Ctog2 included).

    Parameters
    ----------
    cfg : SweepConfig
    model : str
        One of "cpw", "foster", "optimized", "analytical".
    L, C : float or None
        Pre-fitted inductance and capacitance.  Ignored for "cpw".

    Returns
    -------
    rf.Network
        2-port network evaluated on cfg.freq.
    """
    if model == "cpw":
        return cpw_resonator_network_2port(
            cfg.freq, cfg.d, cfg.Cc1, cfg.Cc2, cfg.Ctog1, cfg.Ctog2,
            cpw_params=cfg.cpw_params, Z0=cfg.Z0,
        )
    if L is None or C is None:
        raise ValueError("L and C must be provided for LC model '{}'.".format(model))
    return lc_resonator_network_with_grounds_2port(
        Leff=L, Ceff=C,
        Cc1=cfg.Cc1, Cc2=cfg.Cc2,
        Ctog1=cfg.Ctog1, Ctog2=cfg.Ctog2,
        freq=cfg.freq, Z0=cfg.Z0,
    )


def _build_loaded_2port_network(
    cfg: SweepConfig,
    model: str,
    L: float | None = None,
    C: float | None = None,
) -> rf.Network:
    """
    Build a 2-port loaded network (with LC loads on each port side).

    """
    freq_loaded = rf.Frequency(1e9, 15e9, 400_001, unit="Hz")

    if model == "cpw":
        return cpw_resonator_loaded_network_2port(
            freq_loaded, cfg.d,
            Cc1=cfg.Cc1, Cc2=cfg.Cc2,
            Ctog1=cfg.Ctog1, Ctog2=cfg.Ctog2,
            Lload1=cfg.Lload1, Cload1=cfg.Cload1,
            Lload2=cfg.Lload2, Cload2=cfg.Cload2,
            cpw_params=cfg.cpw_params, Z0=cfg.Z0,
        )
    if L is None or C is None:
        raise ValueError("L and C must be provided for LC model '{}'.".format(model))
    return lc_resonator_loaded_network_with_grounds_2port(
        Leff=L, Ceff=C,
        Cc1=cfg.Cc1, Cc2=cfg.Cc2,
        Ctog1=cfg.Ctog1, Ctog2=cfg.Ctog2,
        Lload1=cfg.Lload1, Cload1=cfg.Cload1,
        Lload2=cfg.Lload2, Cload2=cfg.Cload2,
        freq=freq_loaded, Z0=cfg.Z0,
    )


# ---------------------------------------------------------------------------
# Internal: fit a model and return (L, C)
# ---------------------------------------------------------------------------

def _fit_model(cfg: SweepConfig, model: str) -> tuple[float, float]:
    """
    Fit the chosen model on cfg and return (L, C).

    For "cpw" this is never called — CPW has no L/C to fit.
    """
    if model == "foster":
        m = FosterFit(cpw_params=cfg.cpw_params)
        m.fit(cfg.freq, d=cfg.d)
        return m.L, m.C

    if model == "optimized":
        cpw_net = cpw_resonator_network_2port(
            cfg.freq, cfg.d, cfg.Cc1, cfg.Cc2, cfg.Ctog1, cfg.Ctog2,
            cpw_params=cfg.cpw_params, Z0=cfg.Z0,
        )
        opt_cfg = cfg.opt_config if cfg.opt_config is not None else OptimizationConfig()
        m = OptimizedFit(config=opt_cfg)
        m.fit(
            cfg.freq,
            data_ntw=cpw_net,
            Cc1=cfg.Cc1, Cc2=cfg.Cc2,
            Ctog1=cfg.Ctog1, Ctog2=cfg.Ctog2,
            d=cfg.d, cpw_params=cfg.cpw_params,
            Z0=cfg.Z0,
        )
        return m.L, m.C

    if model == "analytical":
        cpw_net = cpw_resonator_network_2port(
            cfg.freq, cfg.d, cfg.Cc1, cfg.Cc2, cfg.Ctog1, cfg.Ctog2,
            cpw_params=cfg.cpw_params, Z0=cfg.Z0,
        )
        f0 = resonance(cpw_net, m=0, n=0)
        Z0_analytical = cfg.analytical_Z0 if cfg.analytical_Z0 is not None else cfg.Z0
        m = AnalyticalFit()
        m.fit(cfg.freq, f_r=f0, Z0=Z0_analytical)
        return m.L, m.C

    raise ValueError("Unknown model '{}'. Choose: cpw, foster, optimized, analytical.".format(model))


# ---------------------------------------------------------------------------
# Internal: extract quantities from a network
# ---------------------------------------------------------------------------

def _extract(
    net_2port: rf.Network,
    net_loaded: rf.Network | None,
    quantities: list[str],
    cfg: SweepConfig,
) -> dict[str, float]:
    """
    Extract a set of named scalar quantities from a pair of networks.

    Parameters
    ----------
    net_2port : rf.Network
        The unloaded 2-port network (for f0, kappa, Q).
    net_loaded : rf.Network or None
        The loaded 2-port network (for shift_modeN).
        Built lazily — only if a shift quantity is requested.
    quantities : list of str
        Names of quantities to extract.  See the table in sweep() docstring.
    cfg : SweepConfig
        Used to compute bare load resonance frequencies for shift calculations.

    Returns
    -------
    dict mapping quantity name → float value
    """
    result = {}

    # Bare load frequencies — needed for shift calculations
    f0_load1 = 1.0 / (2 * np.pi * np.sqrt(cfg.Lload1 * cfg.Cload1))
    f0_load2 = 1.0 / (2 * np.pi * np.sqrt(cfg.Lload2 * cfg.Cload2))

    for q in quantities:
        try:
            if q == "f0_s11":
                result[q] = resonance(net_2port, 0, 0) / 1e9

            elif q == "f0_s22":
                result[q] = resonance(net_2port, 1, 1) / 1e9

            elif q == "kappa_s11":
                result[q] = fwhm_from_trace_db(net_2port, 0, 0, kind="dip") / 1e6

            elif q == "kappa_s22":
                result[q] = fwhm_from_trace_db(net_2port, 1, 1, kind="dip") / 1e6

            elif q == "Q_s11":
                f0  = resonance(net_2port, 0, 0)
                bw  = fwhm_from_trace_db(net_2port, 0, 0, kind="dip")
                result[q] = f0 / bw if bw > 0 else float("nan")

            elif q == "Q_s22":
                f0  = resonance(net_2port, 1, 1)
                bw  = fwhm_from_trace_db(net_2port, 1, 1, kind="dip")
                result[q] = f0 / bw if bw > 0 else float("nan")

            elif q in ("shift_mode1", "shift_mode2", "shift_mode3"):
                if net_loaded is None:
                    result[q] = float("nan")
                    continue
                mode_idx = int(q[-1]) - 1   # 0, 1, or 2
                shifted = np.sort(
                    np.array(resonances_from_s(net_loaded, m=0, n=0), dtype=float) / 1e9
                )
                f0_res = resonance(net_2port, 0, 0) / 1e9
                bare   = np.sort([f0_load1 / 1e9, f0_res, f0_load2 / 1e9])
                if mode_idx < len(shifted) and mode_idx < len(bare):
                    result[q] = float(abs(bare[mode_idx] - shifted[mode_idx]))
                else:
                    result[q] = float("nan")

            else:
                warnings.warn("Unknown extract quantity '{}' — skipped.".format(q))
                result[q] = float("nan")

        except Exception as exc:
            logger.warning("Failed to extract '%s': %s", q, exc)
            result[q] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Core sweep() function
# ---------------------------------------------------------------------------

_VALID_MODELS = {"cpw", "foster", "optimized", "analytical"}

_SHIFT_QUANTITIES = {"shift_mode1", "shift_mode2", "shift_mode3"}


def sweep(
    param: str,
    values: np.ndarray,
    base_config: SweepConfig,
    extract: list[str],
    model: str = "cpw",
    refit: bool | None = None,
) -> pd.DataFrame:
    """
    Sweep a single parameter across a range of values and extract quantities.

    At each sweep point, the chosen model is either rebuilt from scratch
    (for CPW or when the swept parameter changes L/C) or reuses the L/C
    fitted at the first sweep point (for Cc, Ctog, and load parameters).

    Parameters
    ----------
    param : str
        Name of the SweepConfig field to vary.  Must be an attribute of
        SweepConfig, e.g. "d", "Cc1", "Cc2", "Ctog1", "Lload1", "Cload1".
        Use "Lload1_Lload2" or "Cload1_Cload2" to sweep both loads together.
    values : np.ndarray
        Array of values for the swept parameter.
    base_config : SweepConfig
        Fixed device state.  Not mutated.
    extract : list of str
        Quantities to compute at each sweep point.  Supported values:

        ============== ================================================
        "f0_s11"       S11 resonance frequency (GHz)
        "f0_s22"       S22 resonance frequency (GHz)
        "kappa_s11"    S11 linewidth / FWHM (MHz)
        "kappa_s22"    S22 linewidth / FWHM (MHz)
        "Q_s11"        Quality factor from S11 = f0 / kappa
        "Q_s22"        Quality factor from S22
        ============== ================================================

    model : str
        Which model to use: "cpw" (default), "foster", "optimized",
        or "analytical".
    refit : bool or None
        Whether to re-fit the LC model at every sweep point.
        None (default) means automatic: refit when sweeping a parameter
        in {d, cpw_params}, reuse L/C otherwise.
        Setting refit=True forces refitting at every point (slower but
        safer when sweeping Cc1/Cc2 with OptimizedFit).
        Setting refit=False always reuses the first-point L/C (fastest).

    Returns
    -------
    pd.DataFrame
        One row per sweep point.  Columns are the swept parameter value
        plus all requested extract quantities.

    Examples
    --------
    Sweep resonator length with CPW ground truth::

        df = sweep("d", np.linspace(5e-3, 10e-3, 20), base,
                   extract=["f0_s11", "kappa_s11"])

    Sweep coupling cap with FosterFit (reuses L/C — fast)::

        df = sweep("Cc1", np.linspace(1e-15, 20e-15, 30), base,
                   extract=["f0_s11", "kappa_s11", "Q_s11"],
                   model="foster")

    """
    if model not in _VALID_MODELS:
        raise ValueError("model must be one of {}".format(_VALID_MODELS))

    # Determine whether to refit at each point
    _needs_refit = param in _REFIT_PARAMS if refit is None else refit
    # CPW always rebuilds from scratch — no concept of "refit"
    if model == "cpw":
        _needs_refit = True

    needs_loaded = bool(_SHIFT_QUANTITIES & set(extract))

    # --- Fit once at the first sweep point if we can reuse L/C ---
    L_cached: float | None = None
    C_cached: float | None = None

    if model != "cpw" and not _needs_refit:
        logger.debug("sweep: fitting %s once on base_config (refit=False)", model)
        L_cached, C_cached = _fit_model(base_config, model)

    rows = []
    n = len(values)

    for i, val in enumerate(values):
        logger.debug("sweep point %d/%d: %s = %s", i + 1, n, param, val)

        # Build the modified config for this sweep point
        # Handle the "Lload1_Lload2" and "Cload1_Cload2" shorthand
        if param == "Lload1_Lload2":
            cfg = dataclasses.replace(base_config, Lload1=val, Lload2=val)
        elif param == "Cload1_Cload2":
            cfg = dataclasses.replace(base_config, Cload1=val, Cload2=val)
        else:
            cfg = dataclasses.replace(base_config, **{param: val})

        # Fit or reuse
        if model == "cpw":
            L, C = None, None
        elif _needs_refit:
            try:
                L, C = _fit_model(cfg, model)
            except Exception as exc:
                logger.warning("sweep point %d: fit failed (%s), skipping.", i, exc)
                row = {param: val}
                row.update({q: float("nan") for q in extract})
                rows.append(row)
                continue
        else:
            L, C = L_cached, C_cached

        # Build networks
        try:
            net_2port = _build_2port_network(cfg, model, L, C)
            net_loaded = _build_loaded_2port_network(cfg, model, L, C) if needs_loaded else None
        except Exception as exc:
            logger.warning("sweep point %d: network build failed (%s), skipping.", i, exc)
            row = {param: val}
            row.update({q: float("nan") for q in extract})
            rows.append(row)
            continue

        # Extract quantities
        extracted = _extract(net_2port, net_loaded, extract, cfg)
        row = {param: val, **extracted}
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def sweep_length(
    d_values: np.ndarray,
    base_config: SweepConfig,
    extract: list[str] = None,
    model: str = "cpw",
) -> pd.DataFrame:
    """
    Sweep resonator length d.

    Always refits the model at every point because d directly sets L and C.

    Parameters
    ----------
    d_values : np.ndarray
        Array of resonator lengths in metres.
    base_config : SweepConfig
    extract : list of str, optional
        Defaults to ["f0_s11", "kappa_s11"] if not provided.
    model : str
        "cpw" (default), "foster", "optimized", or "analytical".

    Returns
    -------
    pd.DataFrame
        Columns: "d", plus all extract quantities.
    """
    if extract is None:
        extract = ["f0_s11", "kappa_s11"]
    return sweep("d", d_values, base_config, extract=extract, model=model)


def sweep_coupling(
    Cc_values: np.ndarray,
    base_config: SweepConfig,
    extract: list[str] = None,
    model: str = "cpw",
    symmetric: bool = True,
    refit: bool = False,
) -> pd.DataFrame:
    """
    Sweep coupling capacitor(s).

    Parameters
    ----------
    Cc_values : np.ndarray
        Array of coupling capacitance values in Farads.
    base_config : SweepConfig
    extract : list of str, optional
        Defaults to ["f0_s11", "kappa_s11", "Q_s11"] if not provided.
    model : str
        "cpw" (default), "foster", "optimized", or "analytical".
    symmetric : bool
        If True (default), sweeps Cc1 and Cc2 together.
        If False, sweeps only Cc1 and leaves Cc2 fixed.
    refit : bool
        Whether to re-fit the LC model at every point.  Default False
        because Cc does not change L or C — reusing is valid and fast.
        Set to True if you suspect OptimizedFit is sensitive to Cc during
        Stage 1 (unlikely but possible for very asymmetric coupling).

    Returns
    -------
    pd.DataFrame
        Columns: "Cc" (the swept value), plus all extract quantities.
    """
    if extract is None:
        extract = ["f0_s11", "kappa_s11", "Q_s11"]

    param = "Cc1" if not symmetric else "Cc1"   # handled below via manual loop for symmetric case

    if symmetric:
        # Both Cc1 and Cc2 move together — use the manual loop path
        rows = []
        L_cached, C_cached = (None, None)

        if model != "cpw" and not refit:
            L_cached, C_cached = _fit_model(base_config, model)

        for val in Cc_values:
            cfg = dataclasses.replace(base_config, Cc1=val, Cc2=val)
            if model == "cpw":
                L, C = None, None
            elif refit:
                L, C = _fit_model(cfg, model)
            else:
                L, C = L_cached, C_cached

            needs_loaded = bool(_SHIFT_QUANTITIES & set(extract))
            try:
                net_2port  = _build_2port_network(cfg, model, L, C)
                net_loaded = _build_loaded_2port_network(cfg, model, L, C) if needs_loaded else None
            except Exception as exc:
                logger.warning("sweep_coupling: build failed at Cc=%.2e (%s)", val, exc)
                row = {"Cc": val, **{q: float("nan") for q in extract}}
                rows.append(row)
                continue

            extracted = _extract(net_2port, net_loaded, extract, cfg)
            rows.append({"Cc": val, **extracted})

        return pd.DataFrame(rows)

    else:
        df = sweep("Cc1", Cc_values, base_config, extract=extract, model=model, refit=refit)
        df = df.rename(columns={"Cc1": "Cc"})
        return df


def sweep_load_frequency(
    f_load_values: np.ndarray,
    base_config: SweepConfig,
    extract: list[str] = None,
    model: str = "cpw",
    side: str = "both",
    load_impedance: float = 50.0,
    refit: bool = False,
) -> pd.DataFrame:
    """
    Sweep the bare load resonance frequency.

    Converts each target frequency to a (L, C) pair by fixing the load
    characteristic impedance Z_load = sqrt(L/C), then sweeping C = 1/(ω²L).

    Parameters
    ----------
    f_load_values : np.ndarray
        Array of target load resonance frequencies in Hz.
    base_config : SweepConfig
    extract : list of str, optional
        Defaults to ["f0_s11", "shift_mode1", "shift_mode2", "shift_mode3"].
    model : str
        "cpw" (default), "foster", "optimized", or "analytical".
    side : {"both", "1", "2"}
        Which load to sweep.  "both" sweeps load 1 and load 2 together
        (useful when the two loads are identical, e.g. symmetric qubits).
    load_impedance : float
        Characteristic impedance of the load resonator in Ohms.
        Used to fix L from Z_load = sqrt(L/C) → L = Z_load / ω₀.
        Default 50 Ω.
    refit : bool
        Whether to re-fit the LC model at every point.  Default False.

    Returns
    -------
    pd.DataFrame
        Columns: "f_load_GHz" (the swept load frequency in GHz),
        "Lload" (H), "Cload" (F), plus all extract quantities.
    """
    if extract is None:
        extract = ["f0_s11", "shift_mode1", "shift_mode2", "shift_mode3"]

    if side not in ("both", "1", "2"):
        raise ValueError("side must be 'both', '1', or '2'.")

    L_cached, C_cached = None, None
    if model != "cpw" and not refit:
        L_cached, C_cached = _fit_model(base_config, model)

    needs_loaded = bool(_SHIFT_QUANTITIES & set(extract))
    rows = []

    for f_load in f_load_values:
        w_load = 2 * np.pi * f_load
        # Fix Z_load = sqrt(L/C) → L = Z_load / w_load, C = 1 / (w_load * Z_load)
        L_load = load_impedance / w_load
        C_load = 1.0 / (w_load * load_impedance)

        if side == "both":
            cfg = dataclasses.replace(
                base_config,
                Lload1=L_load, Cload1=C_load,
                Lload2=L_load, Cload2=C_load,
            )
        elif side == "1":
            cfg = dataclasses.replace(base_config, Lload1=L_load, Cload1=C_load)
        else:
            cfg = dataclasses.replace(base_config, Lload2=L_load, Cload2=C_load)

        if model == "cpw":
            L, C = None, None
        elif refit:
            L, C = _fit_model(cfg, model)
        else:
            L, C = L_cached, C_cached

        try:
            net_2port  = _build_2port_network(cfg, model, L, C)
            net_loaded = _build_loaded_2port_network(cfg, model, L, C) if needs_loaded else None
        except Exception as exc:
            logger.warning("sweep_load_frequency: build failed at f=%.4f GHz (%s)", f_load / 1e9, exc)
            row = {
                "f_load_GHz": f_load / 1e9,
                "Lload": L_load, "Cload": C_load,
                **{q: float("nan") for q in extract},
            }
            rows.append(row)
            continue

        extracted = _extract(net_2port, net_loaded, extract, cfg)
        rows.append({
            "f_load_GHz": f_load / 1e9,
            "Lload": L_load,
            "Cload": C_load,
            **extracted,
        })

    return pd.DataFrame(rows)
