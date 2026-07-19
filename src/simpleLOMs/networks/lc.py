"""
networks/lc.py
--------------
Network builders for lumped-element LC resonator topologies.

Topology families
-----------------
Basic 2-port:
    [Port1] -- Cc1 -- [LC shunt] -- Cc2 -- [Port2]

With grounds (mirrors CPW Ctog capacitors):
    [Port1] -- Cc1 -- Ctog1 -- [LC shunt] -- Ctog2 -- Cc2 -- [Port2]

Loaded variants insert an LC load (e.g. a qubit) on each port side:
    [Port1] -- cc_port -- load -- Cc1 -- ... -- Cc2 -- load -- cc_port -- [Port2]
"""
from __future__ import annotations
import skrf as rf

from simpleLOMs.elements import coupling_capacitor, shunt_capacitor, lc_resonator


def lc_resonator_network(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Single-port LC resonator (reflection measurement).

    Topology:
        [Port1] -- Cc1 (series) -- [LC shunt] -- Cc2 (shunt to open) -- [Open]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the LC resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        1-port reflection network.
    """
    cc1   = coupling_capacitor(C=Cc1,  freq=freq, name="cc1", Z0=Z0)
    lc    = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc", Z0=Z0)
    cc2   = shunt_capacitor(C=Cc2, freq=freq, name="cc2", Z0=Z0)
    open_ = rf.Circuit.Open(freq, name="open")
    port1 = rf.Circuit.Port(freq, name="P1", z0=Z0)

    cnx = [
        [(port1, 0), (cc1,   0)],
        [(cc1,   1), (lc,    0)],
        [(lc,    1), (cc2,   0)],
        [(cc2,   1), (open_, 0)],
    ]
 
    return rf.Circuit(cnx, name="LC_1port").network


def lc_resonator_network_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port LC resonator (transmission measurement).

    Topology:
        [Port1] -- Cc1 (series) -- [LC shunt] -- Cc2 (series) -- [Port2]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the LC resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port transmission network.
    """
    cc1   = coupling_capacitor(C=Cc1,  freq=freq, name="cc1", Z0=Z0)
    lc    = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc", Z0=Z0)
    cc2   = coupling_capacitor(C=Cc2,  freq=freq, name="cc2", Z0=Z0)
    port1 = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2 = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1, 0), (cc1, 0)],
        [(cc1,   1), (lc,  0)],
        [(lc,    1), (cc2, 0)],
        [(cc2,   1), (port2, 0)],
    ]
    return rf.Circuit(cnx, name="LC_2port").network


def lc_resonator_network_with_grounds_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port LC resonator with shunt-to-ground capacitors on each side.

    These Ctog capacitors mirror the geometry of the CPW model, where
    the ground capacitance at each gap is an important feature of the
    distributed line.

    Topology:
        [Port1] -- Cc1 -- Ctog1 (shunt) -- [LC shunt] -- Ctog2 (shunt) -- Cc2 -- [Port2]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the LC resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port transmission network with ground capacitances.
    """
    cc1   = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",   Z0=Z0)
    lc    = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc")
    cc2   = coupling_capacitor(C=Cc2,   freq=freq, name="cc2",   Z0=Z0)
    ctog1 = shunt_capacitor(   C=Ctog1, freq=freq, name="ctog1", Z0=Z0)
    ctog2 = shunt_capacitor(   C=Ctog2, freq=freq, name="ctog2", Z0=Z0)
    port1 = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2 = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1, 0), (cc1,   0)],
        [(cc1,   1), (ctog1, 0)],
        [(ctog1, 1), (lc,    0)],
        [(lc,    1), (ctog2, 0)],
        [(ctog2, 1), (cc2,   0)],
        [(cc2,   1), (port2, 0)],
    ]
    return rf.Circuit(cnx, name="LC_with_grounds_2port").network


def lc_resonator_loaded_network_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    Lload1: float,
    Cload1: float,
    Lload2: float,
    Cload2: float,
    freq: rf.Frequency,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port LC resonator with LC loads on each port side.

    Topology:
        [Port1] -- cc_port1 -- load1 -- Cc1 -- [LC shunt] -- Cc2 -- load2 -- cc_port2 -- [Port2]

    The ports probe the load nodes through weak ``cc_port`` capacitors (1 fF by
    default), matching the loaded CPW builders — not through the full
    ``Cc1``/``Cc2`` as in the bare builders.

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the main resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Lload1, Cload1 : float
        Load resonator on port 1 side.
    Lload2, Cload2 : float
        Load resonator on port 2 side.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    ------- 
    rf.Network
        2-port loaded transmission network.
    """
    cc1      = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    lc       = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc",      Z0=Z0)
    cc2      = coupling_capacitor(C=Cc2,   freq=freq, name="cc2",      Z0=Z0)
    cc_port1 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    load1    = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    load2    = lc_resonator(L=Lload2, C=Cload2, freq=freq, name="load2", Z0=Z0)
    port1    = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2    = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (lc,       0)],
        [(lc,       1), (cc2,      0)],
        [(cc2,      1), (load2,    0)],
        [(load2,    1), (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="LC_loaded_2port").network


def lc_resonator_loaded_network_with_grounds_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    Ctog1: float,
    Ctog2: float,
    Lload1: float,
    Cload1: float,
    Lload2: float,
    Cload2: float,
    freq: rf.Frequency,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port LC resonator with both ground capacitors and LC loads.

    This is the most complete LC LOM topology: it includes the shunt Ctog
    capacitors that model CPW ground geometry AND external load resonators
    on each port side.

    Topology:
        [Port1] -- cc_port1 -- load1 -- Cc1 -- Ctog1 -- [LC shunt] -- Ctog2 -- Cc2 -- load2 -- cc_port2 -- [Port2]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the main resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    Lload1, Cload1 : float
        Load resonator on port 1 side.
    Lload2, Cload2 : float
        Load resonator on port 2 side.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port fully-loaded transmission network with ground caps.
    """
    cc1      = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    lc       = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc")
    cc2      = coupling_capacitor(C=Cc2,   freq=freq, name="cc2",      Z0=Z0)
    cc_port1 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    ctog1    = shunt_capacitor(   C=Ctog1, freq=freq, name="ctog1",    Z0=Z0)
    ctog2    = shunt_capacitor(   C=Ctog2, freq=freq, name="ctog2",    Z0=Z0)
    load1    = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    load2    = lc_resonator(L=Lload2, C=Cload2, freq=freq, name="load2", Z0=Z0)
    port1    = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2    = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (ctog1,    0)],
        [(ctog1,    1), (lc,       0)],
        [(lc,       1), (ctog2,    0)],
        [(ctog2,    1), (cc2,      0)],
        [(cc2,      1), (load2,    0)],
        [(load2,    1), (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="LC_loaded_with_grounds_2port").network



def lc_resonator_network_2port_shifted(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port LC resonator with π-phase correction on S21/S12.

    Identical to lc_resonator_network_2port but negates the off-diagonal
    S-matrix elements to account for the π electrical phase that a half-wave
    CPW resonator accumulates at resonance.  A lumped LC shunt has no
    transmission line length so it does not pick up this phase naturally;
    the correction restores the correct sign convention for comparison
    against CPW reference networks.

    Topology:
        [Port1] -- Cc1 (series) -- [LC shunt] -- Cc2 (series) -- [Port2]
        (then S21, S12 negated)
    """
    net = lc_resonator_network_2port(Leff, Ceff, Cc1, Cc2, freq, Z0)
    s = net.s.copy()
    s[:, 1, 0] *= -1
    s[:, 0, 1] *= -1
    return rf.Network(frequency=freq, s=s, z0=net.z0,
                      name="LC_2port_shifted")


def lc_resonator_loaded_network_2port_shifted(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Cc2: float,
    Lload1: float,
    Cload1: float,
    Lload2: float,
    Cload2: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Two-port loaded LC resonator with π-phase correction on S21/S12.

    Identical to lc_resonator_loaded_network_2port but negates S21/S12
    to account for the missing half-wave CPW phase.  See
    lc_resonator_network_2port_shifted for full explanation.

    Topology:
        [Port1] -- cc_port1 -- load1 -- Cc1 -- [LC shunt] -- Cc2 -- load2 -- cc_port2 -- [Port2]
        (then S21, S12 negated)
    """
    net = lc_resonator_loaded_network_2port(
        Leff, Ceff, Cc1, Cc2,
        Lload1, Cload1, Lload2, Cload2,
        freq, Z0,
    )
    s = net.s.copy()
    s[:, 1, 0] *= -1
    s[:, 0, 1] *= -1
    return rf.Network(frequency=freq, s=s, z0=net.z0,
                      name="LC_loaded_2port_shifted")   




def lc_resonator_single_load_network_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Lload1: float,
    Cload1: float,
    freq: rf.Frequency,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port LC resonator with LC loads on each port side.

    Topology:
        [Port1] -- cc_port1 -- load1 -- Cc1 -- [LC shunt] -- Cc2 -- load2 -- cc_port2 -- [Port2]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the main resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Lload1, Cload1 : float
        Load resonator on port 1 side.
    Lload2, Cload2 : float
        Load resonator on port 2 side.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.
    cc_port : float
        Weak probe capacitance in Farads on each port (default 1 fF).

    Returns
    -------
    rf.Network
        2-port loaded transmission network.
    """
    cc1      = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    lc       = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc",      Z0=Z0)
    cc_port1 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    load1    = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    port1    = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2    = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (lc,       0)],
        [(lc,       1), (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="LC_loaded_2port").network


def lc_resonator_single_load_network_with_grounds_2port(
    Leff: float,
    Ceff: float,
    Cc1: float,
    Ctog1: float,
    Ctog2: float,
    Lload1: float,
    Cload1: float,
    freq: rf.Frequency,
    Z0: float = 50,
    cc_port: float = 1e-15,
) -> rf.Network:
    """
    Two-port LC resonator with both ground capacitors and LC loads.

    This is the most complete LC LOM topology: it includes the shunt Ctog
    capacitors that model CPW ground geometry AND external load resonators
    on each port side.

    Topology:
        [Port1] -- cc_port1 -- load1 -- Cc1 -- Ctog1 -- [LC shunt] -- Ctog2 -- Cc2 -- load2 -- cc_port2 -- [Port2]

    Parameters
    ----------
    Leff, Ceff : float
        Effective inductance (H) and capacitance (F) of the main resonator.
    Cc1, Cc2 : float
        Coupling capacitances in Farads.
    Ctog1, Ctog2 : float
        Shunt-to-ground capacitances in Farads.
    Lload1, Cload1 : float
        Load resonator on port 1 side.
    Lload2, Cload2 : float
        Load resonator on port 2 side.
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        2-port fully-loaded transmission network with ground caps.
    """
    cc1      = coupling_capacitor(C=Cc1,   freq=freq, name="cc1",      Z0=Z0)
    lc       = lc_resonator(L=Leff, C=Ceff, freq=freq, name="lc")
    cc_port1 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port1", Z0=Z0)
    cc_port2 = coupling_capacitor(C=cc_port, freq=freq, name="cc_port2", Z0=Z0)
    ctog1    = shunt_capacitor(   C=Ctog1, freq=freq, name="ctog1",    Z0=Z0)
    ctog2    = shunt_capacitor(   C=Ctog2, freq=freq, name="ctog2",    Z0=Z0)
    load1    = lc_resonator(L=Lload1, C=Cload1, freq=freq, name="load1", Z0=Z0)
    port1    = rf.Circuit.Port(freq, name="P1", z0=Z0)
    port2    = rf.Circuit.Port(freq, name="P2", z0=Z0)

    cnx = [
        [(port1,    0), (cc_port1, 0)],
        [(cc_port1, 1), (load1,    0)],
        [(load1,    1), (cc1,      0)],
        [(cc1,      1), (ctog1,    0)],
        [(ctog1,    1), (lc,       0)],
        [(lc,       1), (ctog2,    0)],
        [(ctog2,    1), (cc_port2, 0)],
        [(cc_port2, 1), (port2,    0)],
    ]
    return rf.Circuit(cnx, name="LC_loaded_with_grounds_2port").network
def lc_load_bare_network(
    Lload: float,
    Cload: float,
    Cc_port: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Single isolated LC load resonator coupled to one port via Cc_port.
    Used to extract the 'true' bare frequency of a load as seen from
    the port, accounting for the capacitive loading of the coupler.

    Topology:
        [Port1] -- Cc_port (series) -- [LC shunt] -- [Open]

    Parameters
    ----------
    Lload, Cload : float
        Inductance (H) and capacitance (F) of the load resonator.
    Cc_port : float
        Coupling capacitance to the port (F).
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        1-port reflection network.
    """
    cc_port = coupling_capacitor(C=Cc_port, freq=freq, name="cc_port", Z0=Z0)
    load    = lc_resonator(L=Lload, C=Cload, freq=freq, name="load",   Z0=Z0)
    open_   = rf.Circuit.Open(freq, name="open")
    port1   = rf.Circuit.Port(freq, name="P1", z0=Z0)

    cnx = [
        [(port1,   0), (cc_port, 0)],
        [(cc_port, 1), (load,    0)],
        [(load,    1), (open_,   0)],
    ]
    return rf.Circuit(cnx, name="LC_load_bare").network

def lc_load_dressed_network_2(
    Lload: float,
    Cload: float,
    Cc_port: float,
    Cc1: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    """
    Isolated LC load with both port coupler and CPW coupler attached,
    to extract the 'dressed' bare frequency accounting for capacitive
    loading from both sides.

    Topology:
        [Port1] -- Cc_port (series) -- [LC shunt] -- Cc1 (series) -- [Open]

    Parameters
    ----------
    Lload, Cload : float
        Inductance (H) and capacitance (F) of the load resonator.
    Cc_port : float
        Port coupling capacitance (F) — hardcoded as 1e-15 in your topology.
    Cc1 : float
        CPW-side coupling capacitance (F).
    freq : rf.Frequency
    Z0 : float
        Reference impedance in Ohms.

    Returns
    -------
    rf.Network
        1-port reflection network.
    """
    cc_port = coupling_capacitor(C=Cc_port, freq=freq, name="cc_port", Z0=Z0)
    load    = lc_resonator(L=Lload, C=Cload, freq=freq, name="load",   Z0=Z0)
    cc1     = coupling_capacitor(C=Cc1,     freq=freq, name="cc1",     Z0=Z0)
    open_   = rf.Circuit.Open(freq, name="open")
    port1   = rf.Circuit.Port(freq, name="P1", z0=Z0)

    cnx = [
        [(port1,   0), (cc_port, 0)],
        [(cc_port, 1), (load,    0)],
        [(load,    1), (cc1,     0)],
        [(cc1,     1), (open_,   0)],
    ]
    return rf.Circuit(cnx, name="LC_load_dressed").network



# def lc_load_dressed_network(
#     Lload: float,
#     Cload: float,
#     Cc_port: float,
#     Cc1: float,
#     freq: rf.Frequency,
#     Z0: float = 50,
# ) -> rf.Network:
#     """
#     Isolated LC load with both port coupler and CPW coupler attached,
#     to extract the 'dressed' bare frequency accounting for capacitive
#     loading from both sides.

#     Topology:
#         [Port1] -- Cc_port (series) -- [LC shunt] -- Cc1 (series) -- [Open]

#     Parameters
#     ----------
#     Lload, Cload : float
#         Inductance (H) and capacitance (F) of the load resonator.
#     Cc_port : float
#         Port coupling capacitance (F) — hardcoded as 1e-15 in your topology.
#     Cc1 : float
#         CPW-side coupling capacitance (F).
#     freq : rf.Frequency
#     Z0 : float
#         Reference impedance in Ohms.

#     Returns
#     -------
#     rf.Network
#         1-port reflection network.
#     """
#     cc_port = coupling_capacitor(C=Cc_port, freq=freq, name="cc_port", Z0=Z0)
#     load    = lc_resonator(L=Lload, C=Cload, freq=freq, name="load",   Z0=Z0)
#     cc1     = coupling_capacitor(C=Cc1,     freq=freq, name="cc1",     Z0=Z0)
#     open_   = rf.Circuit.Open(freq, name="open")
#     port1   = rf.Circuit.Port(freq, name="P1", z0=Z0)

#     cnx = [
#         [(port1,   0), (cc_port, 0)],
#         [(cc_port, 1), (load,    0)],
#         [(load,    1), (cc1,     0)],
#         [(cc1,     1), (open_,   0)],
#     ]
#     return rf.Circuit(cnx, name="LC_load_dressed").network


def lc_load_dressed_network(
    Lload: float,
    Cload: float,
    Cc_port: float,
    Cc1: float,
    freq: rf.Frequency,
    Z0: float = 50,
) -> rf.Network:
    cc_port = coupling_capacitor(C=Cc_port, freq=freq, name="cc_port", Z0=Z0)
    load    = lc_resonator(L=Lload, C=Cload, freq=freq, name="load",   Z0=Z0)
    cc1     = coupling_capacitor(C=Cc1,      freq=freq, name="cc1",    Z0=Z0)
    term    = rf.Circuit.Ground(freq, name="term")  # Z0 termination
    port1   = rf.Circuit.Port(freq, name="P1", z0=Z0)
    cnx = [
        [(port1,   0), (cc_port, 0)],
        [(cc_port, 1), (load,    0)],
        [(load,    1), (cc1,     0)],
        [(cc1,     1), (term,    0)],
    ]
    return rf.Circuit(cnx, name="LC_load_dressed").network