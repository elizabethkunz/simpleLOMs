"""
models/analytical.py
--------------------
AnalyticalFit: closed-form approximation for the effective L and C of a
λ/2 CPW resonator.

Algorithm summary
-----------------
For a half-wave (λ/2) CPW resonator the distributed line has a well-known
lumped-element equivalent valid near the fundamental resonance:

    C_r = π / (2 · ω_r · Z0)        [effective capacitance]
    L_r = 1 / (ω_r² · C_r)          [effective inductance]

where ω_r = 2π f_r is the angular resonance frequency and Z0 is the
characteristic impedance of the CPW line.

This is the simplest of the three methods — no optimisation, no network
simulation, just two formulas.  It is fast and useful as a sanity check
or initial guess, but it ignores the loading from coupling capacitors and
ground capacitors.

Reference: standard microwave engineering result, see e.g. Pozar §6.
"""
from __future__ import annotations
import logging

import numpy as np
import skrf as rf

from simpleLOMs.models.base import BaseFit
from simpleLOMs.networks.lc import (
    lc_resonator_network_with_grounds_2port,
    lc_resonator_network_2port,
)

logger = logging.getLogger(__name__)


class AnalyticalFit(BaseFit):
    """
    Lumped LC model via the closed-form λ/2 CPW approximation.

    Parameters
    ----------
    mode : int, optional
        Wavelength-fraction selector for the effective capacitance, **not** a
        harmonic index: 2 for a half-wave (λ/2) resonator (default), 4 for a
        quarter-wave (λ/4). It enters as C_r = π / (mode · ω_r · Z0), so the
        λ/2 value C_r = π / (2·ω_r·Z0) is the ``mode=2`` case.

    Attributes
    ----------
    L : float or None
        Effective inductance in Henries.  None before fit().
    C : float or None
        Effective capacitance in Farads.  None before fit().

    Examples
    --------
    ::

        from simpleLOMs.models.analytical import AnalyticalFit
        import skrf as rf

        freq  = rf.Frequency(4e9, 10e9, 10_001, unit="Hz")
        model = AnalyticalFit()

        # Provide f_r and Z0 directly — no network needed
        model.fit(freq, f_r=8.581e9, Z0=45.926)

        print(model.get_params())
        # {'L': 5.42e-10, 'C': 6.34e-13, 'f_r_Hz': 8.581e9, 'Z0': 45.926}

        net = model.get_network(freq, Cc1=5e-15, Cc2=5e-15,
                                Ctog1=1e-14, Ctog2=1e-14)
    """

    def __init__(self, mode: int = 2):
        self.mode: int           = mode
        self.L:    float | None  = None
        self.C:    float | None  = None
        self._f_r_hz: float | None = None
        self._Z0:     float | None = None

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        freq: rf.Frequency,
        f_r: float,
        Z0: float,
        **kwargs,
    ) -> None:
        """
        Compute L and C from the closed-form λ/2 approximation.

        Parameters
        ----------
        freq : rf.Frequency
            Frequency sweep (not used in the calculation itself, but kept
            for a consistent BaseFit interface).
        f_r : float
            Resonance frequency in Hz.
        Z0 : float
            Characteristic impedance of the CPW line in Ohms.
        **kwargs
            Ignored.
        """
        w_r  = 2.0 * np.pi * f_r
        C_r  = (np.pi) / (self.mode * w_r * Z0)
        L_r  = 1.0 / (w_r ** 2 * C_r)
        logger.debug(
            "AnalyticalFit: f_r=%.4f GHz, Z0=%.3f Ω → C=%.4e F, L=%.4e H",
            f_r / 1e9, Z0, C_r, L_r,
        )

        self.C      = C_r
        self.L      = L_r
        self._f_r_hz = f_r
        self._Z0     = Z0

    # ------------------------------------------------------------------
    # get_network()
    # ------------------------------------------------------------------

    def get_network(
        self,
        freq: rf.Frequency,
        Cc1: float,
        Cc2: float,
        Z0: float = 50.0,
        with_grounds: bool = True,
        Ctog1: float = None,
        Ctog2: float = None,
        show: bool = False,
        reference: rf.Network = None,
        m: int = 0,
        n: int = 0,
        lom_label: str = "AnalyticalFit",
        data_label: str = "CPW",
        save_path: str = None,
        **kwargs,
    ) -> rf.Network:
        """
        Build the 2-port LC LOM network from the analytical L and C.

        Parameters
        ----------
        freq : rf.Frequency
        Cc1, Cc2 : float
            Coupling capacitances in Farads.
        Z0 : float
            Reference impedance in Ohms.
        with_grounds : bool
            If True (default), include Ctog shunt capacitors.  The
            analytical model is typically compared alongside the full
            CPW topology so including the grounds is more representative.
        Ctog1, Ctog2 : float or None
            Required when with_grounds=True.
        show : bool
            If True, automatically plot the built network (default False).
        reference : rf.Network, optional
            Overlay reference used when ``show=True``.
        m, n : int
            S-parameter indices for the auto-plot.
        lom_label, data_label : str
            Legend labels for the auto-plot.
        save_path : str, optional
            Save path used when ``show=True``.

        Returns
        -------
        rf.Network
        """
        self._require_fitted()

        if with_grounds:
            if Ctog1 is None or Ctog2 is None:
                raise ValueError("Ctog1 and Ctog2 must be provided when with_grounds=True.")
            net = lc_resonator_network_with_grounds_2port(
                Leff=self.L, Ceff=self.C,
                Cc1=Cc1, Cc2=Cc2,
                Ctog1=Ctog1, Ctog2=Ctog2,
                freq=freq, Z0=Z0,
            )
        else:
            net = lc_resonator_network_2port(
                Leff=self.L, Ceff=self.C,
                Cc1=Cc1, Cc2=Cc2,
                freq=freq, Z0=Z0,
            )

        self._maybe_plot(
            net,
            show=show,
            reference=reference,
            m=m, n=n,
            lom_label=lom_label,
            data_label=data_label,
            save_path=save_path,
        )
        return net

    # ------------------------------------------------------------------
    # get_params()
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """
        Return the analytically computed parameters.

        Returns
        -------
        dict
            Keys: "L" (H), "C" (F), "f_r_Hz", "Z0".
        """
        self._require_fitted()
        return {
            "L":      self.L,
            "C":      self.C,
            "f_r_Hz": self._f_r_hz,
            "Z0":     self._Z0,
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            return "AnalyticalFit(L={:.4e} H, C={:.4e} F, f_r={:.4f} GHz)".format(
                self.L, self.C, self._f_r_hz / 1e9
            )
        return "AnalyticalFit(unfitted)"