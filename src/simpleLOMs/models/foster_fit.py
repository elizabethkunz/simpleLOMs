"""
models/foster.py
----------------
FosterFit: admittance-slope (Foster synthesis) method for extracting
the effective L and C of a CPW resonator.

Algorithm summary
-----------------
1. Build a 2-port CPW transmission line network (no coupling caps, just
   the bare line terminated at both ends with a high impedance).
2. Compute the input admittance Yin(ω) seen at port 1 with the port 2
   load in place:
       Yin = Y11 - Y12*Y21 / (Y22 + YL)
3. Find the resonance frequency ω0 from the peak in Re(S22).
4. At ω0, the imaginary part of Yin passes through zero with a positive
   slope (for a series-like resonance).  The Foster synthesis theorem
   relates that slope to an equivalent LC:
       C_eq = 0.5 * dB/dω |_{ω=ω0}    where B = Im(Yin)
       L_eq = 1 / (ω0² * C_eq)
5. Store C_eq → self.C, L_eq → self.L.

"""
from __future__ import annotations
import logging

import numpy as np
import skrf as rf
from scipy.signal import find_peaks
from skrf.media import CPW

from simpleLOMs.models.base import BaseFit
from simpleLOMs.params import CPWParams
from simpleLOMs.elements import lc_resonator
from simpleLOMs.networks.lc import lc_resonator_network_with_grounds_2port

logger = logging.getLogger(__name__)


class FosterFit(BaseFit):
    """
    Lumped LC model via Foster admittance synthesis on a bare CPW line.

    Parameters
    ----------
    cpw_params : CPWParams, optional
        Physical geometry and material parameters for the CPW.
        Uses default CPWParams() (11.7 µm / 5.1 µm gap, ultracold Si)
        if not provided.
    port_z0 : float, optional
        Port impedance used when building the bare CPW network for synthesis.
        Default 500 Ω (high impedance → weakly loaded line, closer to
        the unloaded resonator condition).

    Attributes
    ----------
    L : float or None
        Fitted effective inductance in Henries.  None before fit().
    C : float or None
        Fitted effective capacitance in Farads.  None before fit().
    yin : np.ndarray or None
        Complex input admittance array computed during fit().  Stored
        for diagnostics / plotting without recomputing.

    Examples
    --------
    Basic usage::

        from simpleLOMs.params import CPWParams
        from simpleLOMs.models.foster import FosterFit
        import skrf as rf

        freq = rf.Frequency(4e9, 10e9, 10_001, unit="Hz")
        cpw  = CPWParams(ep_r=11.45)

        model = FosterFit(cpw_params=cpw)
        model.fit(freq, d=7e-3)

        print(model.get_params())
        # {'L': 5.53e-10, 'C': 6.22e-13, 'f0_Hz': 8.58e9}

        net = model.get_network(freq, Cc1=5e-15, Cc2=5e-15,
                                Ctog1=1e-14, Ctog2=1e-14)
    """

    def __init__(self, cpw_params: CPWParams = None, port_z0: float = 500.0):
        self.cpw_params: CPWParams = cpw_params if cpw_params is not None else CPWParams()
        self.port_z0:   float      = port_z0

        # Set after fit()
        self.L:   float | None       = None
        self.C:   float | None       = None
        self.yin: np.ndarray | None  = None
        self._f0_hz: float | None    = None   # resonance freq found during fit

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, freq: rf.Frequency, d: float, **kwargs) -> None:
        """
        Run Foster synthesis on a bare CPW line of length d.

        Computes the input admittance Yin(ω), finds ω0 from a peak in
        Re(S22), then extracts C_eq from the admittance slope and L_eq
        from the resonance constraint ω0² L C = 1.

        Parameters
        ----------
        freq : rf.Frequency
            Frequency sweep.  Should span at least one resonance of the
            bare CPW line at length d.
        d : float
            Physical resonator length in metres.
        **kwargs
            Ignored — included for a consistent BaseFit interface.
        """
        p = self.cpw_params

        # 1. Build bare CPW media and a simple 2-port transmission line
        cpw_media = CPW(
            freq,
            w=p.w,
            s=p.s,
            t=p.t,
            h=p.h,
            rho=p.rho,
            ep_r=p.ep_r,
            has_metal_backside=p.has_metal_backside,
            tand=p.tand,
        )
        line  = cpw_media.line(d=d, unit="m", name="cpw_line")
        port1 = rf.Circuit.Port(freq, name="P1", z0=self.port_z0)
        port2 = rf.Circuit.Port(freq, name="P2", z0=self.port_z0)

        cnx = [
            [(port1, 0), (line,  0)],
            [(line,  1), (port2, 0)],
        ]
        ntw = rf.Circuit(cnx, name="bare_cpw").network

        # 2. Extract resonance from peak in Re(S22)
        #    (S22 has a reflection peak at resonance for this topology)
        f_hz  = ntw.frequency.f
        s22   = ntw.s[:, 1, 1]
        peaks, _ = find_peaks(np.real(s22))

        if len(peaks) == 0:
            raise ValueError(
                "FosterFit.fit(): no resonance peak found in Re(S22). "
                "Check that freq spans a resonance of the CPW line at d={} m.".format(d)
            )

        # Use the first (lowest-frequency) resonance
        f0_hz = float(f_hz[peaks[0]])
        w0    = 2 * np.pi * f0_hz
        logger.debug("FosterFit: resonance found at %.4f GHz", f0_hz / 1e9)

        # 3. Compute Yin = Y11 - Y12*Y21 / (Y22 + YL)
        Y  = ntw.y
        YL = 1.0 / self.port_z0
        yin = Y[:, 0, 0] - (Y[:, 0, 1] * Y[:, 1, 0]) / (Y[:, 1, 1] + YL)
        self.yin = yin  # store for diagnostics

        # 4. Foster synthesis: C_eq = 0.5 * dB/dω at ω0
        w_rad  = 2 * np.pi * f_hz
        B      = np.imag(yin)
        dB_dw  = np.gradient(B, w_rad)

        i0           = np.argmin(np.abs(w_rad - w0))
        slope_at_w0  = float(dB_dw[i0])
        C_eq         = 0.5 * slope_at_w0
        L_eq         = 1.0 / (w0 ** 2 * C_eq)

        logger.debug("FosterFit: C_eq = %.4e F, L_eq = %.4e H", C_eq, L_eq)

        self.C     = C_eq
        self.L     = L_eq
        self._f0_hz = f0_hz

    # ------------------------------------------------------------------
    # get_network()
    # ------------------------------------------------------------------

    def get_network(
        self,
        freq: rf.Frequency,
        Cc1: float,
        Cc2: float,
        Ctog1: float,
        Ctog2: float,
        Z0: float = 50.0,
        show: bool = False,
        reference: rf.Network = None,
        m: int = 0,
        n: int = 0,
        lom_label: str = "FosterFit",
        data_label: str = "CPW",
        save_path: str = None,
        **kwargs,
    ) -> rf.Network:
        """
        Build the 2-port LC LOM network using the Foster-synthesised L and C.

        Uses the "with grounds" topology so that Ctog1/Ctog2 are included,
        matching the structure of the CPW model that was synthesised.

        Parameters
        ----------
        freq : rf.Frequency
            May differ from the freq used in fit().
        Cc1, Cc2 : float
            Coupling capacitances in Farads.
        Ctog1, Ctog2 : float
            Shunt-to-ground capacitances in Farads.
        Z0 : float
            Reference impedance in Ohms.
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
            2-port LC network with ground capacitors.
        """
        self._require_fitted()
        net = lc_resonator_network_with_grounds_2port(
            Leff=self.L,
            Ceff=self.C,
            Cc1=Cc1,
            Cc2=Cc2,
            Ctog1=Ctog1,
            Ctog2=Ctog2,
            freq=freq,
            Z0=Z0,
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
        Return fitted parameters.

        Returns
        -------
        dict
            Keys: "L" (H), "C" (F), "f0_Hz" (resonance used for synthesis).
        """
        self._require_fitted()
        return {
            "L":     self.L,
            "C":     self.C,
            "f0_Hz": self._f0_hz,
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            return "FosterFit(L={:.4e} H, C={:.4e} F, f0={:.4f} GHz)".format(
                self.L, self.C, self._f0_hz / 1e9
            )
        return "FosterFit(unfitted)"