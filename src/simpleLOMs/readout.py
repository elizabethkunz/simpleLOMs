"""
readout.py
----------
Translate a fitted lumped oscillator model (LOM) into the readout / circuit-QED
figures of merit that superconducting-qubit designers actually work with.

Every model in simpleLOMs (FosterFit, OptimizedFit, AnalyticalFit) reduces a
distributed CPW resonator to an effective inductance ``L`` and capacitance ``C``.
Those two numbers, plus the loaded linewidth ``kappa`` extracted from an
S-parameter fit, are enough to compute the quantities a designer targets:

    * resonance frequency        f_r  = 1 / (2 pi sqrt(L C))
    * characteristic impedance   Z_r  = sqrt(L / C)
    * loaded quality factor      Q    = f_r / kappa
    * photon lifetime            tau  = 1 / (2 pi kappa)
    * zero-point voltage         V_zpf = sqrt(hbar omega_r / (2 C))
    * zero-point current         I_zpf = sqrt(hbar omega_r / (2 L))

This module is a thin, well-tested layer on top of what the models already
return -- it does not do any fitting itself.  Use :func:`resonator_readout_params`
when you already have ``L`` and ``C`` (e.g. from ``model.get_params()``), and
:func:`extract_f0_kappa` when you want to measure ``f_r`` and ``kappa`` directly
from a device geometry via a robust circle fit.

Examples
--------
From a fitted model::

    from simpleLOMs import FosterFit, CPWParams, resonator_readout_params

    model = FosterFit(cpw_params=CPWParams())
    model.fit(freq, d=7e-3)
    rp = resonator_readout_params(model.L, model.C, kappa_Hz=2.4e6)
    print(rp)                      # human-readable summary
    print(rp.Z_r_ohm, rp.Q)       # individual quantities

Measuring f_r and kappa straight from geometry::

    from simpleLOMs import CPWParams, extract_f0_kappa

    f0, kappa = extract_f0_kappa(CPWParams(), d=7e-3, Cc=6e-15)
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import skrf as rf

from .params import CPWParams
from .networks.cpw import cpw_resonator_network_2port
from .analysis import circle_fit_f0_kappa, circle_fit_f0_kappa_windowed

#: Reduced Planck constant (J s).
HBAR = 1.054_571_817e-34


def _resolve_port_loads(
    Cc: float | None,
    Ctog: float | None,
    Cc1: float | None,
    Cc2: float | None,
    Ctog1: float | None,
    Ctog2: float | None,
) -> tuple[float, float, float, float]:
    """
    Resolve symmetric shorthand (``Cc``/``Ctog``) and per-port overrides
    (``Cc1``/``Cc2``/``Ctog1``/``Ctog2``) into a concrete per-port
    ``(Cc1, Cc2, Ctog1, Ctog2)`` tuple.

    Each per-port value falls back to the corresponding symmetric value when not
    given, so symmetric loading (only ``Cc``/``Ctog`` supplied) and fully
    asymmetric loading are both expressible.  Raises if a coupling capacitance is
    left completely unspecified for a port.
    """
    cc1 = Cc1 if Cc1 is not None else Cc
    cc2 = Cc2 if Cc2 is not None else Cc
    if cc1 is None or cc2 is None:
        raise ValueError(
            "Coupling capacitance is required: pass Cc for symmetric loading, "
            "or Cc1 and Cc2 for asymmetric loading."
        )
    ctog1 = Ctog1 if Ctog1 is not None else Ctog
    ctog2 = Ctog2 if Ctog2 is not None else Ctog
    return cc1, cc2, ctog1, ctog2


@dataclass
class ReadoutParams:
    """
    Readout / circuit-QED figures of merit for a single lumped resonator mode.

    All stored quantities are in SI base units.  Use the derived properties
    (``f_r_GHz``, ``Z_r_ohm`` ...) or :meth:`summary` for human-friendly values.

    Attributes
    ----------
    L : float
        Effective inductance (H).
    C : float
        Effective capacitance (F).
    kappa_Hz : float or None
        Loaded linewidth (FWHM) in Hz.  ``None`` if no linewidth was supplied,
        in which case the loss-dependent quantities (Q, tau_photon) are ``None``.
    """

    L: float
    C: float
    kappa_Hz: float | None = None

    # -- frequency / impedance (depend only on L, C) --------------------------

    @property
    def omega_r(self) -> float:
        """Angular resonance frequency omega_r = 1/sqrt(L C) (rad/s)."""
        return 1.0 / np.sqrt(self.L * self.C)

    @property
    def f_r_Hz(self) -> float:
        """Resonance frequency f_r = omega_r / 2 pi (Hz)."""
        return self.omega_r / (2.0 * np.pi)

    @property
    def f_r_GHz(self) -> float:
        """Resonance frequency in GHz."""
        return self.f_r_Hz / 1e9

    @property
    def Z_r_ohm(self) -> float:
        """Characteristic impedance Z_r = sqrt(L / C) (Ohms)."""
        return np.sqrt(self.L / self.C)

    # -- zero-point fluctuations (depend only on L, C) ------------------------

    @property
    def V_zpf(self) -> float:
        """Zero-point voltage fluctuation sqrt(hbar omega_r / 2C) (V)."""
        return np.sqrt(HBAR * self.omega_r / (2.0 * self.C))

    @property
    def I_zpf(self) -> float:
        """Zero-point current fluctuation sqrt(hbar omega_r / 2L) (A)."""
        return np.sqrt(HBAR * self.omega_r / (2.0 * self.L))

    # -- loss-dependent quantities (need kappa) -------------------------------

    @property
    def Q(self) -> float | None:
        """Loaded quality factor Q = f_r / kappa, or None if kappa is unknown."""
        if self.kappa_Hz is None or self.kappa_Hz <= 0:
            return None
        return self.f_r_Hz / self.kappa_Hz

    @property
    def tau_photon_s(self) -> float | None:
        """
        Photon lifetime tau = 1 / (2 pi kappa) (s), where kappa is the FWHM
        linewidth.  This is the 1/e energy-decay time of the mode.
        Returns None if kappa is unknown.
        """
        if self.kappa_Hz is None or self.kappa_Hz <= 0:
            return None
        return 1.0 / (2.0 * np.pi * self.kappa_Hz)

    def summary(self) -> str:
        """Return a multi-line, unit-annotated summary string."""
        lines = [
            "Resonator readout parameters",
            "----------------------------",
            "  f_r        = {:10.4f} GHz".format(self.f_r_GHz),
            "  Z_r        = {:10.2f} Ohm".format(self.Z_r_ohm),
            "  L          = {:10.4f} nH".format(self.L * 1e9),
            "  C          = {:10.4f} fF".format(self.C * 1e15),
            "  V_zpf      = {:10.3f} uV".format(self.V_zpf * 1e6),
            "  I_zpf      = {:10.3f} nA".format(self.I_zpf * 1e9),
        ]
        if self.kappa_Hz is not None:
            lines += [
                "  kappa/2pi  = {:10.4f} MHz".format(self.kappa_Hz / 1e6),
                "  Q (loaded) = {:10.0f}".format(self.Q),
                "  tau_photon = {:10.1f} ns".format(self.tau_photon_s * 1e9),
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def resonator_readout_params(
    L: float,
    C: float,
    kappa_Hz: float | None = None,
) -> ReadoutParams:
    """
    Build :class:`ReadoutParams` from an effective inductance and capacitance.

    This is the "last mile" from a fitted LOM to circuit-QED quantities: pass
    the ``L`` and ``C`` any simpleLOMs model returns (``model.L``, ``model.C``,
    or ``model.get_params()``), optionally with a measured linewidth.

    Parameters
    ----------
    L : float
        Effective inductance in Henries.
    C : float
        Effective capacitance in Farads.
    kappa_Hz : float, optional
        Loaded linewidth (FWHM) in Hz.  If given, the returned object also
        exposes ``Q`` and ``tau_photon_s``.

    Returns
    -------
    ReadoutParams

    Examples
    --------
    >>> rp = resonator_readout_params(5.5e-10, 6.2e-13, kappa_Hz=2.4e6)
    >>> round(rp.Z_r_ohm, 1)
    29.8
    >>> round(rp.Q)
    3591
    """
    if L <= 0 or C <= 0:
        raise ValueError("L and C must be positive.")
    return ReadoutParams(L=L, C=C, kappa_Hz=kappa_Hz)


def extract_f0_kappa(
    cpw_params: CPWParams,
    d: float,
    Cc: float | None = None,
    Ctog: float = 1e-14,
    Z0: float = 50.0,
    f_lo: float = 4e9,
    f_hi: float = 12e9,
    n_coarse: int = 20_001,
    span: float = 0.4e9,
    n_fine: int = 120_001,
    *,
    Cc1: float | None = None,
    Cc2: float | None = None,
    Ctog1: float | None = None,
    Ctog2: float | None = None,
) -> tuple[float, float]:
    """
    Measure the loaded resonance frequency and linewidth of a CPW resonator
    directly from its geometry, using a robust two-stage circle fit.

    A single wide frequency grid usually cannot resolve a sub-MHz linewidth, so
    this function first locates the resonance on a coarse grid, then re-simulates
    a narrow window around it on a fine grid before extracting ``kappa``.  This
    is the same coarse-then-refine strategy used internally by ``analyze_system``.

    Loading may be symmetric or asymmetric.  Pass ``Cc``/``Ctog`` for the common
    symmetric case (the same value on both ports), or give per-port overrides via
    ``Cc1``/``Cc2`` and ``Ctog1``/``Ctog2`` for an asymmetric device.  The
    reflection circle fit probes the more strongly coupled port: both ports see
    the same loaded ``(f0, kappa)``, and the stronger port's larger resonance
    circle is far more robust against the coupling background at extreme
    asymmetry.

    Parameters
    ----------
    cpw_params : CPWParams
        CPW geometry / material parameters.
    d : float
        Resonator length in metres.
    Cc : float, optional
        Coupling capacitance in Farads applied symmetrically to both ports.
        Provide this or the per-port ``Cc1``/``Cc2`` pair.
    Ctog : float
        Shunt-to-ground capacitance in Farads applied to both ports.  Default
        1e-14.  Overridden per port by ``Ctog1``/``Ctog2``.
    Z0 : float
        Reference / feedline impedance in Ohms.  Default 50.
    f_lo, f_hi : float
        Bounds of the coarse search grid in Hz.  Default 4-12 GHz.
    n_coarse : int
        Number of points in the coarse grid.
    span : float
        *Minimum* half-width of the refined window around the coarse f0, in Hz.
        The actual window is ``max(span, 5 * kappa_coarse)`` — a strongly
        coupled resonance can be ~1 GHz wide, and a fixed narrow window then
        clips the dip and yields ``kappa = NaN``. If the refined extraction
        still fails, the window is doubled (a few attempts, capped at
        ``0.75 * f0``) before giving up. Narrow-linewidth results are
        unaffected: the window never shrinks below ``span``.
    n_fine : int
        Number of points in the refined grid.
    Cc1, Cc2 : float, optional
        Per-port coupling capacitances in Farads.  Each defaults to ``Cc`` when
        not given, so passing only ``Cc`` reproduces symmetric loading.
    Ctog1, Ctog2 : float, optional
        Per-port shunt-to-ground capacitances in Farads.  Each defaults to
        ``Ctog`` when not given.

    Returns
    -------
    f0 : float
        Loaded resonance frequency in Hz.
    kappa : float
        Loaded linewidth (FWHM) in Hz.

    Examples
    --------
    >>> f0, kappa = extract_f0_kappa(CPWParams(), d=7e-3, Cc=6e-15)
    >>> 7e9 < f0 < 9e9
    True

    Asymmetric coupling (a stronger tap on port 2)::

    >>> f0, kappa = extract_f0_kappa(CPWParams(), d=7e-3, Cc1=6e-15, Cc2=1.2e-14)
    """
    Cc1, Cc2, Ctog1, Ctog2 = _resolve_port_loads(Cc, Ctog, Cc1, Cc2, Ctog1, Ctog2)

    # Measure reflection at the more strongly coupled port. The resonance
    # circle seen from port i has diameter ~ kappa_i / kappa_total, so probing
    # the weak port of a strongly asymmetric device leaves a small circle
    # riding on a comparable frequency-dependent background — the Kasa fit
    # then centres on the background arc and inflates kappa (about +18 % at
    # Cc1 = 15, Cc2 = 150 fF, where reciprocity demands the same result as the
    # mirrored device). Both ports see the same loaded (f0, kappa); the strong
    # port just measures it more robustly. Cc is the proxy for coupling
    # strength (exact for equal Ctog, the common case).
    port = 0 if Cc1 >= Cc2 else 1

    coarse = rf.Frequency(f_lo, f_hi, n_coarse, unit="Hz")
    net_c = cpw_resonator_network_2port(
        coarse, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
    )
    f0_coarse, kappa_coarse = circle_fit_f0_kappa(net_c, m=port, n=port)

    # The fine window must contain the whole dip: size it from the coarse
    # linewidth (which may be crude — smoothing-inflated or a grid-resolution
    # floor — so it only ever *widens* the window beyond `span`, never narrows
    # it), and on a failed extraction (kappa NaN from a clipped, off-centre
    # dip) retry with a doubled window, capped at 0.75*f0 to stay clear of DC
    # and the next mode family.
    half = span
    if np.isfinite(kappa_coarse) and kappa_coarse > 0:
        half = max(half, 5.0 * kappa_coarse)
    half_cap = 0.75 * f0_coarse
    half = min(half, half_cap)

    for _ in range(4):
        fine = rf.Frequency(f0_coarse - half, f0_coarse + half, n_fine, unit="Hz")
        net_f = cpw_resonator_network_2port(
            fine, d, Cc1, Cc2, Ctog1, Ctog2, cpw_params=cpw_params, Z0=Z0
        )
        # Use the windowed extractor: the fine grid (~7 kHz/pt) is far denser
        # than a sub-MHz linewidth, so the fixed 51-pt smoothing window in the
        # plain circle fit acts as a ~0.3 MHz filter and inflates kappa at weak
        # coupling (~+11 % at Cc=2 fF). The windowed variant smooths only to
        # locate the resonance, then re-extracts kappa from an unsmoothed
        # narrow slice of the same network — no extra simulation, negligible
        # cost.
        f0, kappa = circle_fit_f0_kappa_windowed(net_f, m=port, n=port)
        if (np.isfinite(kappa) and kappa > 0) or half >= half_cap:
            break
        half = min(2.0 * half, half_cap)
    return f0, kappa


def invert_monotonic(
    x,
    y,
    target: float,
    *,
    log: bool = False,
) -> float:
    """
    Invert a monotonic ``y(x)`` curve by interpolation.

    Parameters
    ----------
    x, y : array-like
        Sampled independent and dependent values (need not be sorted).
    target : float
        Desired ``y`` value.
    log : bool
        If True, interpolate in ``log(y)`` / ``log(x)`` space (appropriate for
        power-law curves such as ``Q(C_c)``).

    Returns
    -------
    float
        Interpolated ``x`` that yields ``target``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(y)
    ys = y[order]
    xs = x[order]
    if log:
        return float(np.exp(np.interp(np.log(target), np.log(ys), np.log(xs))))
    return float(np.interp(target, ys, xs))


def design_resonator(
    target_f_GHz: float,
    target_Q: float,
    cpw: CPWParams,
    *,
    freq: rf.Frequency | None = None,
    model: str = "optimized",
    d_range: tuple[float, float] = (5.5e-3, 9.0e-3),
    Cc_range: tuple[float, float] = (2e-15, 14e-15),
    n: int = 6,
    Cc_ref: float = 6e-15,
    Ctog: float = 1e-14,
    Z0: float = 50.0,
) -> tuple[float, float, float, ReadoutParams]:
    """
    Choose resonator length and coupling to hit a target frequency and loaded Q.

    Sequentially: (1) interpolate length from an ``f_0(d)`` sweep at fixed
    ``Cc_ref``, (2) fit ``L, C`` at that length, (3) interpolate ``C_c`` from a
    ``Q(C_c)`` sweep, (4) verify against ``extract_f0_kappa``.

    Parameters
    ----------
    target_f_GHz : float
        Desired resonant frequency in GHz.
    target_Q : float
        Desired loaded quality factor.
    cpw : CPWParams
        CPW geometry / material.
    freq : rf.Frequency, optional
        Frequency grid for ``fit_lom``.  Defaults to 4–12 GHz, 8001 points.
    model : str
        LOM used for ``L, C`` (default ``\"optimized\"``).
    d_range, Cc_range : tuple
        Sweep bounds for length (m) and coupling (F).
    n : int
        Number of sweep points per axis.
    Cc_ref : float
        Reference coupling used during the length → frequency sweep.
    Ctog, Z0
        Passed through to ``extract_f0_kappa`` / ``fit_lom``.

    Returns
    -------
    d, Cc, f0_GHz, ReadoutParams
        Designed length (m), coupling (F), achieved ``f_0`` in GHz, and readout
        report at the designed point.
    """
    # Local import avoids a circular dependency (system → readout).
    from simpleLOMs.system import fit_lom

    if freq is None:
        freq = rf.Frequency(4e9, 12e9, 8001, unit="Hz")

    ds = np.linspace(*d_range, n)
    fs = np.array([
        extract_f0_kappa(cpw, d=d, Cc=Cc_ref, Ctog=Ctog, Z0=Z0)[0] / 1e9
        for d in ds
    ])
    d = invert_monotonic(ds, fs, target_f_GHz)

    L, C = fit_lom(d, model=model, cpw_params=cpw, freq=freq,
                   Cc=Cc_ref, Ctog=Ctog, Z0=Z0)
    ccs = np.linspace(*Cc_range, n)
    Qs = np.array([
        resonator_readout_params(
            L, C,
            kappa_Hz=extract_f0_kappa(cpw, d=d, Cc=cc, Ctog=Ctog, Z0=Z0)[1],
        ).Q
        for cc in ccs
    ])
    Cc = invert_monotonic(ccs, Qs, target_Q, log=True)

    f0, kappa = extract_f0_kappa(cpw, d=d, Cc=Cc, Ctog=Ctog, Z0=Z0)
    return d, Cc, f0 / 1e9, resonator_readout_params(L, C, kappa_Hz=kappa)
