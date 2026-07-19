"""
elements.py
-----------
Primitive lumped-element circuit blocks built on top of scikit-rf. Each function returns an rf.Network (a 2-port element) that can
be wired into an rf.Circuit connection list.

For netowrks built using these eleemnts see simpleLOMs/networks/.
"""
from __future__ import annotations
import skrf as rf


def coupling_capacitor(C: float, freq: rf.Frequency, name: str = "cc", Z0: float = 50) -> rf.Network:
    """
    Series coupling capacitor.

    Implemented as a SeriesImpedance with Z = 1 / (jωC).

    Parameters
    ----------
    C : float
        Capacitance in Farads.
    freq : rf.Frequency
        Frequency sweep object.
    name : str
        Label used inside rf.Circuit.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port series capacitor network.
    """
    return rf.Circuit.SeriesImpedance(
        frequency=freq,
        name=name,
        z0=Z0,
        Z=1 / (1j * freq.w * C),
    )


def shunt_capacitor(C: float, freq: rf.Frequency, name: str = "ctog", Z0: float = 50) -> rf.Network:
    """
    Shunt (to-ground) capacitor.

    Implemented as a ShuntAdmittance with Y = jωC.

    Parameters
    ----------
    C : float
        Capacitance in Farads.
    freq : rf.Frequency
        Frequency sweep object.
    name : str
        Label used inside rf.Circuit.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port shunt capacitor network.
    """
    return rf.Circuit.ShuntAdmittance(
        frequency=freq,
        name=name,
        z0=Z0,
        Y=1j * freq.w * C,
    )


def lc_resonator(L: float, C: float, freq: rf.Frequency, name: str = "lc", Z0: float = 50) -> rf.Network:
    """
    Parallel LC resonator shunted to ground.

    Implemented as a ShuntAdmittance with Y = jωC + 1/(jωL), the admittance of a parallel LC tank.

    Parameters
    ----------
    L : float
        Inductance in Henries.
    C : float
        Capacitance in Farads.
    freq : rf.Frequency
        Frequency sweep object.
    name : str
        Label used inside rf.Circuit.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port shunt parallel-LC network.
    """
    return rf.Circuit.ShuntAdmittance(
        frequency=freq,
        name=name,
        z0=Z0,
        Y=1j * freq.w * C + 1 / (1j * freq.w * L),
    )


def hanger_lc_branch(
    Cc: float, L: float, C: float, freq: rf.Frequency, name: str = "hang", Z0: float = 50
) -> rf.Network:
    """
    A parallel-LC tank coupled to the signal rail through a series capacitor,
    hanging to ground — the lumped ("LOM") version of a hanger / notch resonator.

    The branch is a *coupling capacitor* ``Cc`` in series with a parallel-LC tank
    whose far end is grounded::

        rail ── Cc ──┬── L ──┬── ground
                     └── C ──┘

    The whole branch collapses to a single ``ShuntAdmittance`` with

        Y = 1 / (Z_Cc + Z_LC),
        Z_Cc = 1/(jωCc),   Z_LC = 1 / (jωC + 1/(jωL)),

    so it drops straight into the existing ladder machinery exactly like
    :func:`lc_resonator` (both are shunt 2-ports that pass the rail through while
    presenting an admittance to ground).  Prefer ``L, C`` from
    :class:`~simpleLOMs.models.hanger_optimized_fit.HangerOptimizedFit` when the
    target topology is a hanger.  Reusing an inline LOM fit is only approximate
    unless the inline ground loading is matched to the hanger: a hanger tap has
    no shunt-to-ground caps, so an inline fit must be taken at ``Ctog = 0`` to
    land on the notch (the ~3% "reuse gap" is that mismatched ``Ctog`` loading,
    not the hanger-vs-inline boundary condition).

    Parameters
    ----------
    Cc : float
        Coupling (tap) capacitance in Farads between the rail and the tank.
    L : float
        Tank inductance in Henries.
    C : float
        Tank capacitance in Farads.
    freq : rf.Frequency
        Frequency sweep object.
    name : str
        Label used inside rf.Circuit.
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port shunt network for the hanging LC branch.
    """
    w = freq.w
    Zcc = 1 / (1j * w * Cc)
    Zlc = 1 / (1j * w * C + 1 / (1j * w * L))
    return rf.Circuit.ShuntAdmittance(
        frequency=freq,
        name=name,
        z0=Z0,
        Y=1 / (Zcc + Zlc),
    )
