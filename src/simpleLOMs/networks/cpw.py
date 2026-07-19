"""
networks/cpw.py
---------------
Network builders for CPW (Coplanar Waveguide) resonator topologies.

Each function takes a CPWParams dataclass for device geometry plus
circuit-level parameters (coupling caps, frequency, etc.) and returns
an rf.Network.

Topology overview
-----------------
All networks follow the same chain:

    [Port1] -- Cc1 -- Ctog1 -- [CPW line] -- Ctog2 -- Cc2 -- [Port2 or Open]

The "loaded" variants insert an LC resonator between the port and the
coupling capacitor on each side:

    [Port1] -- cc_port1 -- load1 -- Cc1 -- Ctog1 -- [CPW line] -- ...
"""
from __future__ import annotations
import numpy as np
import skrf as rf
from scipy.signal import find_peaks
from skrf.media import CPW

from simpleLOMs.elements import coupling_capacitor, shunt_capacitor, lc_resonator
from simpleLOMs.params import CPWParams


def _make_cpw_media(freq: rf.Frequency, params: CPWParams) -> CPW:
    """Build a skrf CPW media object from a CPWParams dataclass."""
    return CPW(
        freq,
        w=params.w,
        s=params.s,
        t=params.t,
        h=params.h,
        rho=params.rho,
        ep_r=params.ep_r,
        has_metal_backside=params.has_metal_backside,
        tand=params.tand,
    )


def bare_cpw_resonance_hz(
    d: float,
    cpw_params: CPWParams = None,
    *,
    f_lo: float = 1e9,
    f_hi: float = 20e9,
    n: int = 40_001,
    port_z0: float = 500.0,
) -> float:
    """
    Fundamental (λ/2) resonance frequency of a *bare*, unloaded CPW line.

    This is the intrinsic open–open resonance of the transmission line itself,
    with **no coupling or ground capacitors** — the natural anchor frequency for
    the closed-form :class:`~simpleLOMs.models.analytical_fit.AnalyticalFit`.
    Anchoring the analytical model here (rather than to the *loaded* circle-fit
    resonance, which already contains the Cc/Ctog pull) avoids double-counting
    the loading when ``get_network`` re-applies those capacitors.

    A bare line is built from the CPW geometry, terminated in high-impedance
    ports, and the fundamental is located from the first peak in ``Re(S22)`` with
    a parabolic sub-sample refinement.

    Parameters
    ----------
    d : float
        Resonator length in metres.
    cpw_params : CPWParams, optional
        CPW geometry / material. Defaults to ``CPWParams()``.
    f_lo, f_hi : float
        Search band in Hz (default 1–20 GHz, wide enough to contain the
        fundamental for the usual few-mm lengths).
    n : int
        Number of points in the search grid.
    port_z0 : float
        High port impedance used to keep the line weakly loaded (default 500 Ω),
        matching the convention used by :class:`FosterFit`.

    Returns
    -------
    float
        Bare fundamental resonance frequency in Hz.
    """
    p = cpw_params if cpw_params is not None else CPWParams()
    freq = rf.Frequency(f_lo, f_hi, n, unit="Hz")
    media = _make_cpw_media(freq, p)
    line = media.line(d=d, unit="m", name="cpw_line")
    port1 = rf.Circuit.Port(freq, name="P1", z0=port_z0)
    port2 = rf.Circuit.Port(freq, name="P2", z0=port_z0)
    ntw = rf.Circuit(
        [[(port1, 0), (line, 0)], [(line, 1), (port2, 0)]], name="bare_cpw"
    ).network

    f = ntw.frequency.f
    s22 = np.real(ntw.s[:, 1, 1])
    peaks, _ = find_peaks(s22)
    if len(peaks) == 0:
        raise ValueError(
            "bare_cpw_resonance_hz: no resonance peak found in Re(S22); "
            "widen [f_lo, f_hi] to bracket the λ/2 fundamental for d={} m.".format(d)
        )

    i = int(peaks[0])  # lowest-frequency (fundamental) mode
    if 0 < i < len(f) - 1:
        y0, y1, y2 = s22[i - 1], s22[i], s22[i + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        return float(f[i] + delta * (f[1] - f[0]))
    return float(f[i])


def cpw_line_impedance(
    cpw_params: CPWParams = None,
    f_hz: float = 8e9,
) -> float:
    """
    Characteristic impedance of the CPW line (Ohms) at frequency ``f_hz``.

    This is the impedance of the line *geometry* — distinct from the 50 Ω
    port/reference impedance — and is the impedance the closed-form
    :class:`~simpleLOMs.models.analytical_fit.AnalyticalFit` expects (its λ/2
    effective impedance is ``2 Z_line / π``). Dispersion is weak, so a single
    frequency near the resonance is sufficient.

    Parameters
    ----------
    cpw_params : CPWParams, optional
        CPW geometry / material. Defaults to ``CPWParams()``.
    f_hz : float
        Evaluation frequency in Hz (default 8 GHz).

    Returns
    -------
    float
        Real part of the characteristic impedance in Ohms.
    """
    p = cpw_params if cpw_params is not None else CPWParams()
    freq = rf.Frequency(f_hz, f_hz * 1.001, 3, unit="Hz")
    media = _make_cpw_media(freq, p)
    return float(np.real(media.z0[0]))


def cpw_resonator_network(
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    cpw_params: CPWParams = None,
    Z0: float = 50,
) -> rf.Network:
    """
    Single-port CPW resonator (reflection measurement, port + open termination).

    Topology:
        [Port1] -- Cc1 (series) -- Ctog1 (shunt) -- [CPW line] -- Ctog2 (shunt) -- Cc2 (shunt) -- [Open]

    Parameters
    ----------
    freq : rf.Frequency
    d : float
        Physical resonator length in metres.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    cpw_params : CPWParams, optional
        CPW geometry. Uses default CPWParams() if not provided.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        1-port reflection network.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    cpw = _make_cpw_media(freq, cpw_params)
    line   = cpw.line(d=d, unit="m", name="cpw_line")
    cc1    = coupling_capacitor(C=Cc1,  freq=freq, name="cc1",   Z0=Z0)
    cc2    = shunt_capacitor(   C=Cc2,  freq=freq, name="cc2",   Z0=Z0)
    ctog1  = cpw.shunt_capacitor(C=Ctog1, freq=freq, name="ctog1", Z0=Z0)
    ctog2  = cpw.shunt_capacitor(C=Ctog2, freq=freq, name="ctog2", Z0=Z0)
    open_  = rf.Circuit.Open(freq, name="open")
    port1  = rf.Circuit.Port(freq, name="P1", z0=Z0)

    cnx = [
        [(port1, 0), (cc1,   0)],
        [(cc1,   1), (ctog1, 0)],
        [(ctog1, 1), (line,  0)],
        [(line,  1), (ctog2, 0)],
        [(ctog2, 1), (cc2,   0)],
        [(cc2,   1), (open_, 0)],
    ]
    return rf.Circuit(cnx, name="CPW_1port").network


def cpw_resonator_network_2port(
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    cpw_params: CPWParams = None,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port CPW resonator (transmission measurement, port on each side).

    Topology:
        [Port1] -- Cc1 (series) -- Ctog1 (shunt) -- [CPW line] -- Ctog2 (shunt) -- Cc2 (series) -- [Port2]

    Parameters
    ----------
    freq : rf.Frequency
    d : float
        Physical resonator length in metres.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    cpw_params : CPWParams, optional
        CPW geometry. Uses default CPWParams() if not provided.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port transmission network.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    cpw = _make_cpw_media(freq, cpw_params)
    line   = cpw.line(d=d, unit="m", name="cpw_line")
    cc1    = coupling_capacitor(C=Cc1,  freq=freq, name="cc1",   Z0=Z0)
    cc2    = coupling_capacitor(C=Cc2,  freq=freq, name="cc2",   Z0=Z0)
    ctog1  = cpw.shunt_capacitor(C=Ctog1, freq=freq, name="ctog1", Z0=Z0)
    ctog2  = cpw.shunt_capacitor(C=Ctog2, freq=freq, name="ctog2", Z0=Z0)
    port1  = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2  = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1, 0), (cc1,   0)],
        [(cc1,   1), (ctog1, 0)],
        [(ctog1, 1), (line,  0)],
        [(line,  1), (ctog2, 0)],
        [(ctog2, 1), (cc2,   0)],
        [(cc2,   1), (port2, 0)],
    ]
    return rf.Circuit(cnx, name="CPW_2port").network


def cpw_resonator_loaded_network_2port(
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
    cpw_params: CPWParams = None,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port CPW resonator with an LC load on each port side.

    The loads model external resonators (e.g. qubits or readout resonators)
    weakly coupled to each port via small series capacitors. The measurement
    ports probe the load nodes through weak ``cc_port`` capacitors (1 fF by
    default), so the resonance is set by the resonator + coupling + load, not
    by the 50 Ω terminations. Note this differs from the *bare* builder
    :func:`cpw_resonator_network_2port`, whose ports connect through the full
    ``Cc1``/``Cc2`` — compare loaded vs bare only under the same probe
    convention (e.g. this builder with a negligible load as the bare
    reference).

    Topology:
        [Port1] -- cc_port1 -- load1 (shunt LC) -- Cc1 -- Ctog1 -- [CPW line]
                                                                      -- Ctog2 -- Cc2 -- load2 (shunt LC) -- cc_port2 -- [Port2]

    Parameters
    ----------
    freq : rf.Frequency
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
    cpw_params : CPWParams, optional
        CPW geometry. Uses default CPWParams() if not provided.
    Z0 : float
        Reference impedance in Ohms.
    cc_port : float
        Weak probe capacitance in Farads between each port and its load node
        (default 1 fF).

    Returns
    -------
    rf.Network
        2-port loaded transmission network.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    cpw = _make_cpw_media(freq, cpw_params)
    line      = cpw.line(d=d, unit="m", name="cpw_line")
    cc1       = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    cc2       = coupling_capacitor(C=Cc2,   freq=freq, name="cc2",      Z0=Z0)
    cc_port1  = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2  = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    ctog1     = cpw.shunt_capacitor(C=Ctog1, freq=freq, name="ctog1", Z0=Z0)
    ctog2     = cpw.shunt_capacitor(C=Ctog2, freq=freq, name="ctog2", Z0=Z0)
    load1     = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    load2     = lc_resonator(L=Lload2, C=Cload2, freq=freq, name="load2", Z0=Z0)
    port1     = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2     = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (ctog1,    0)],
        [(ctog1,    1), (line,     0)],
        [(line,     1), (ctog2,    0)],
        [(ctog2,    1), (cc2,      0)],
        [(cc2,      1), (load2,    0)],
        [(load2,    1), (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="CPW_loaded_2port").network



def cpw_resonator_single_load_network_2port(
    freq: rf.Frequency,
    d: float,
    Cc1: float,
    Ctog1: float,
    Ctog2: float,
    Lload1: float,
    Cload1: float,
    cpw_params: CPWParams = None,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port CPW resonator with a single LC load on the port-1 side only.

    The load models an external resonator (e.g. a qubit or readout resonator)
    weakly coupled to port 1 via a small series capacitor. Port 2 taps the line
    through a weak probe capacitor with no load and no coupling capacitor on its
    path.

    Topology:
        [Port1] -- cc_port1 -- load1 (shunt LC) -- Cc1 -- Ctog1 -- [CPW line]
                                                            -- Ctog2 -- cc_port2 -- [Port2]

    Parameters
    ----------
    freq : rf.Frequency
    d : float
        Resonator length in metres.
    Cc1 : float
        Coupling capacitance on the port-1 side in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    Lload1, Cload1 : float
        Inductance (H) and capacitance (F) of the load on the port-1 side.
    cpw_params : CPWParams, optional
        CPW geometry. Uses default CPWParams() if not provided.
    Z0 : float
        Reference impedance in Ohms.
    cc_port : float
        Weak probe capacitance in Farads between each port and the network
        (default 1 fF).

    Returns
    -------
    rf.Network
        2-port loaded transmission network.
    """
    if cpw_params is None:
        cpw_params = CPWParams()

    cpw = _make_cpw_media(freq, cpw_params)
    line      = cpw.line(d=d, unit="m", name="cpw_line")
    cc1       = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    cc_port1  = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2  = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    ctog1     = cpw.shunt_capacitor(C=Ctog1, freq=freq, name="ctog1", Z0=Z0)
    ctog2     = cpw.shunt_capacitor(C=Ctog2, freq=freq, name="ctog2", Z0=Z0)
    load1     = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    port1     = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2     = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (ctog1,    0)],
        [(ctog1,    1), (line,     0)],
        [(line,     1), (ctog2,    0)],
        [(ctog2,    1),  (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="CPW_loaded_2port").network
