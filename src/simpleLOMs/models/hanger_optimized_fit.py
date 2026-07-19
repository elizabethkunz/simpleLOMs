"""
models/hanger_optimized_fit.py
------------------------------
HangerOptimizedFit: numerical optimisation of effective L, C against a
*hanger / notch* CPW ground-truth network (S21 dip), rather than an inline
transmission resonator.

This is the hanger counterpart to :class:`~simpleLOMs.models.optimized_fit.OptimizedFit`.
Inline ``OptimizedFit`` remains unchanged; do not rewire its ``(L, C)`` into a
hanger when accuracy matters — fit against the hanger topology instead.
"""
from __future__ import annotations

import logging

import numpy as np
import skrf as rf
from scipy.optimize import least_squares, minimize_scalar

from simpleLOMs.analysis import fwhm_from_trace_db
from simpleLOMs.models.base import BaseFit
from simpleLOMs.models.foster_fit import FosterFit
from simpleLOMs.models.optimized_fit import OptimizationConfig
from simpleLOMs.networks.hanger import hanger_resonator_network_2port

logger = logging.getLogger(__name__)


class HangerOptimizedFit(BaseFit):
    """
    Lumped LC hanger model via least-squares fit to a CPW hanger S21 notch.

    Algorithm
    ---------
    1. Seed ``(L, C)`` from Foster synthesis on the bare CPW length ``d``.
    2. Stage 1 — 1-D notch match: hold the Foster tank impedance
       ``Z = sqrt(L/C)`` fixed and scan the bare tank frequency so the LC
       hanger notch lands on the CPW notch.
    3. Stage 2 — windowed least_squares on ``|S21|`` (and optionally a global
       phase) with a wide notch window, freeing ``L`` and ``C``.

    Parameters
    ----------
    config : OptimizationConfig, optional
        Reuses the same knobs as inline ``OptimizedFit`` (``n_widths``,
        ``max_nfev``, ``fit_phase``, …).  Channel selection is ignored —
        residuals always target S21.
    """

    def __init__(self, config: OptimizationConfig | None = None):
        self.config: OptimizationConfig = (
            config if config is not None else OptimizationConfig(
                s_params=["S21"],
                n_widths=4.0,
                verbose=False,
            )
        )
        self.L: float | None = None
        self.C: float | None = None
        self.phase: float | None = None
        self._data_ntw: rf.Network | None = None
        self._Cc_tap: float | None = None
        self._Z0: float = 50.0
        self._f0_hz: float | None = None
        self._kappa_hz: float | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _notch_f0(ntw: rf.Network) -> float:
        """Frequency of the deepest |S21| dip (Hz)."""
        mag = np.abs(ntw.s[:, 1, 0])
        return float(ntw.frequency.f[np.argmin(mag)])

    def _lc_hanger(
        self,
        freq: rf.Frequency,
        L: float,
        C: float,
        Cc_tap: float,
        Z0: float,
    ) -> rf.Network:
        return hanger_resonator_network_2port(
            freq,
            [{"kind": "lc", "L": float(L), "C": float(C)}],
            [Cc_tap],
            Z0=Z0,
        )

    def _align_network(self, data_ntw: rf.Network, freq: rf.Frequency) -> rf.Network:
        f_old = data_ntw.frequency.f
        f_new = freq.f
        if len(f_old) == len(f_new) and np.allclose(f_old, f_new):
            return data_ntw
        S_old = data_ntw.s
        S_new = np.zeros((len(f_new), 2, 2), dtype=complex)
        for i in range(2):
            for j in range(2):
                S_new[:, i, j] = (
                    np.interp(f_new, f_old, S_old[:, i, j].real)
                    + 1j * np.interp(f_new, f_old, S_old[:, i, j].imag)
                )
        z0_old = data_ntw.z0
        if z0_old.ndim == 1:
            z0_new = np.tile(z0_old[0], (len(f_new), 2))
        else:
            z0_new = np.zeros((len(f_new), 2))
            for p in range(2):
                z0_new[:, p] = np.interp(f_new, f_old, z0_old[:, p])
        return rf.Network(frequency=freq, s=S_new, z0=z0_new)

    def _seed_from_foster(
        self, freq: rf.Frequency, d: float, cpw_params
    ) -> tuple[float, float]:
        foster = FosterFit(cpw_params=cpw_params)
        foster.fit(freq, d=d)
        return float(foster.L), float(foster.C)

    def _stage1_match_notch(
        self,
        freq: rf.Frequency,
        f0_target: float,
        L_seed: float,
        C_seed: float,
        Cc_tap: float,
        Z0: float,
    ) -> tuple[float, float]:
        """
        Hold tank impedance Z=sqrt(L/C) fixed; vary bare ω0 so the LC hanger
        notch matches ``f0_target``.
        """
        Z_tank = np.sqrt(L_seed / C_seed)

        # Dense local grid around the target notch for cheap f0 evaluations.
        half = max(0.08 * f0_target, 200e6)
        f_lo = max(float(freq.f[0]), f0_target - half)
        f_hi = min(float(freq.f[-1]), f0_target + half)
        n_pts = min(4001, max(501, int((f_hi - f_lo) / 1e6) + 1))
        local = rf.Frequency(f_lo, f_hi, n_pts, unit="Hz")

        def notch_of_w0(w0: float) -> float:
            L = Z_tank / w0
            C = 1.0 / (Z_tank * w0)
            return self._notch_f0(self._lc_hanger(local, L, C, Cc_tap, Z0))

        def obj(w0: float) -> float:
            return (notch_of_w0(w0) - f0_target) ** 2

        # Bare ω0 must sit *above* the loaded notch (Cc pulls it down).
        w0_guess = 2.0 * np.pi * f0_target
        w0_lo = w0_guess * 0.95
        w0_hi = w0_guess * 1.25
        res = minimize_scalar(obj, bounds=(w0_lo, w0_hi), method="bounded",
                              options=dict(xatol=1e-6 * w0_guess))
        w0_best = float(res.x)
        L1 = Z_tank / w0_best
        C1 = 1.0 / (Z_tank * w0_best)
        f1 = notch_of_w0(w0_best)
        logger.debug(
            "Stage 1: L=%.4e H, C=%.4e F, notch=%.4e Hz (target %.4e)",
            L1, C1, f1, f0_target,
        )
        return L1, C1

    def _window_half_hz(self, f0: float, kappa: float) -> float:
        """
        Half-width of the Stage-2 residual window.

        Must be wide enough that a seed notch a few percent off still falls
        inside the window (otherwise the |S21| residual is flat / degenerate).
        """
        cfg = self.config
        # n_widths is in units of kappa for inline fits; for hangers we also
        # enforce a fractional floor so a ~3% seed offset is visible.
        return max(float(cfg.n_widths) * kappa, 0.05 * f0, 50 * kappa, 100e6)

    def _make_s21_residuals(
        self,
        freq: rf.Frequency,
        data_ntw: rf.Network,
        Cc_tap: float,
        f0: float,
        kappa: float,
        Z0: float,
        eps: float = 1e-18,
    ):
        cfg = self.config
        data = self._align_network(data_ntw, freq)
        f = freq.f
        half = self._window_half_hz(f0, kappa)
        mask = (f > f0 - half) & (f < f0 + half)
        if mask.sum() < 10:
            raise ValueError(
                "S21 notch window too small — check f0 / kappa / frequency grid."
            )

        win_freq = rf.Frequency.from_f(f[mask], unit="Hz")
        s21_data = data.s[mask, 1, 0]
        depth = max(float(np.max(np.abs(s21_data)) - np.min(np.abs(s21_data))), eps)

        def _lc_s21(Le: float, Ce: float) -> np.ndarray:
            return self._lc_hanger(win_freq, Le, Ce, Cc_tap, Z0).s[:, 1, 0]

        if cfg.fit_phase:
            def residuals(x):
                Le, Ce, phi = float(x[0]), float(x[1]), float(x[2])
                pred = _lc_s21(Le, Ce) * np.exp(1j * phi)
                r = pred - s21_data
                return np.concatenate([r.real, r.imag]) / depth
        else:
            def residuals(x):
                Le, Ce = float(x[0]), float(x[1])
                pred = _lc_s21(Le, Ce)
                # Magnitude residual + soft notch-frequency penalty so L↔C
                # scaling (constant ω0) is not a flat direction.
                mag_r = (np.abs(pred) - np.abs(s21_data)) / depth
                f0_m = float(win_freq.f[np.argmin(np.abs(pred))])
                f_pen = np.array([(f0_m - f0) / max(kappa, eps)])
                return np.concatenate([mag_r, f_pen])

        return residuals

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        freq: rf.Frequency,
        data_ntw: rf.Network,
        Cc_tap: float,
        d: float,
        cpw_params,
        termination: str = "open",
        Z0: float = 50.0,
        **kwargs,
    ) -> None:
        """
        Fit ``L, C`` so an LC hanger reproduces the CPW hanger S21 notch.

        Parameters
        ----------
        freq : rf.Frequency
            Optimisation grid (should span the notch with enough points).
        data_ntw : rf.Network
            2-port CPW hanger reference (typically from
            ``hanger_resonator_network_2port`` with ``kind="cpw"``).
        Cc_tap : float
            Tap coupling capacitance in Farads (same value used in the GT).
        d : float
            Resonator length in metres (used to seed Foster ``L, C``).
        cpw_params : CPWParams
            CPW geometry for the Foster seed.
        termination : {"open", "short"}
            Far-end termination of the reference hanger (API / documentation;
            already encoded in ``data_ntw``).
        Z0 : float
            Reference impedance in Ohms.
        """
        if termination not in ("open", "short"):
            raise ValueError(
                f"termination must be 'open' or 'short', got {termination!r}."
            )

        cfg = self.config
        self._data_ntw = data_ntw
        self._Cc_tap = float(Cc_tap)
        self._Z0 = float(Z0)

        f0 = self._notch_f0(data_ntw)
        try:
            kappa = fwhm_from_trace_db(data_ntw, m=1, n=0, kind="dip")
        except ValueError:
            kappa = 0.01 * f0
            logger.warning(
                "Could not measure S21 FWHM; using kappa = 0.01 * f0 = %.3e Hz",
                kappa,
            )
        self._f0_hz = f0
        self._kappa_hz = kappa
        logger.debug("Hanger notch: f0=%.4e Hz, kappa=%.4e Hz", f0, kappa)

        L0, C0 = self._seed_from_foster(freq, d, cpw_params)
        L1, C1 = self._stage1_match_notch(freq, f0, L0, C0, Cc_tap, Z0)

        residuals_fn = self._make_s21_residuals(
            freq=freq,
            data_ntw=data_ntw,
            Cc_tap=Cc_tap,
            f0=f0,
            kappa=kappa,
            Z0=Z0,
        )

        # Relative bounds around the Stage-1 seed (not a flat L∝C ray).
        L_lo, L_hi = L1 * 0.5, L1 * 1.5
        C_lo, C_hi = C1 * 0.5, C1 * 1.5
        x0 = [L1, C1]
        lo = [L_lo, C_lo]
        hi = [L_hi, C_hi]
        if cfg.fit_phase:
            x0.append(0.0)
            lo.append(-np.pi)
            hi.append(np.pi)

        res = least_squares(
            residuals_fn,
            x0=x0,
            bounds=(lo, hi),
            method="trf",
            jac="3-point",
            diff_step=1e-4,
            x_scale=[L1, C1] + ([1.0] if cfg.fit_phase else []),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=cfg.max_nfev,
            verbose=2 if cfg.verbose else 0,
        )

        self.L = float(res.x[0])
        self.C = float(res.x[1])
        self.phase = float(res.x[2]) if cfg.fit_phase else None
        logger.debug(
            "HangerOptimizedFit: L=%.4e H, C=%.4e F, phase=%s",
            self.L,
            self.C,
            f"{self.phase:.4f} rad" if self.phase is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # get_network / get_params
    # ------------------------------------------------------------------

    def get_network(
        self,
        freq: rf.Frequency,
        Cc_tap: float | None = None,
        Z0: float | None = None,
        show: bool = False,
        reference: rf.Network | None = None,
        m: int = 1,
        n: int = 0,
        lom_label: str = "HangerOptimizedFit",
        data_label: str = "CPW hanger",
        save_path: str | None = None,
        **kwargs,
    ) -> rf.Network:
        """Build the fitted LC hanger 2-port network."""
        self._require_fitted()
        Cc = float(Cc_tap if Cc_tap is not None else self._Cc_tap)
        z0 = float(Z0 if Z0 is not None else self._Z0)
        net = self._lc_hanger(freq, self.L, self.C, Cc, z0)
        ref = reference if reference is not None else self._data_ntw
        self._maybe_plot(
            net,
            show=show,
            reference=ref,
            m=m,
            n=n,
            lom_label=lom_label,
            data_label=data_label,
            save_path=save_path,
        )
        return net

    def get_params(self) -> dict:
        self._require_fitted()
        return {
            "L": self.L,
            "C": self.C,
            "phase": self.phase,
            "f0_Hz": self._f0_hz,
            "kappa_Hz": self._kappa_hz,
            "Cc_tap": self._Cc_tap,
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            phase_str = (
                f", phase={self.phase:.4f} rad" if self.phase is not None else ""
            )
            return (
                f"HangerOptimizedFit(L={self.L:.4e} H, C={self.C:.4e} F"
                + phase_str
                + ")"
            )
        return "HangerOptimizedFit(unfitted)"
