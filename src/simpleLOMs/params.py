"""
params.py
---------
Shared parameter dataclasses used across simpleLOMs.

Here we define CPWParams once.  Every function that needs CPW geometry
accepts a single CPWParams.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CPWParams:
    """
    Physical geometry and material parameters for a Coplanar Waveguide (CPW).

    Attributes
    ----------
    w : float
        Center conductor width (m).  Default 11.7 µm.
    s : float
        Gap spacing between center conductor and ground plane (m).
        Default 5.1 µm.
    t : float
        Metal thickness (m).  Set to 0 to ignore (for an ideal thin film). This quantity is generally on the order of 100-200 nm.
    h : float
        Substrate height (m).  Default 500 µm.
    rho : float
        Metal resistivity (Ω·m).  Set near 0 for superconducting limit.
    ep_r : float
        Substrate relative permittivity.  Defaults to 11.45 (ultracold silicon).
    has_metal_backside : bool
        Whether the substrate has a metal ground plane on its back side.
    tand : float
        Loss tangent of the substrate.  Set to 0 for a lossless superconducting circuit.

    Examples
    --------
    Use the defaults (ultracold silicon chip):
        >>> cpw = CPWParams()

    Override just the substrate:
        >>> cpw = CPWParams(ep_r=11.9, has_metal_backside=False)

    Pass into a network builder:
        >>> net = cpw_resonator_network_2port(freq, d=7e-3, cpw_params=cpw, ...)
    """

    w:                float = 11.7e-6   # center conductor width
    s:                float = 5.1e-6    # gap spacing
    t:                float = 0.0       # metal thickness (0 = ideal)
    h:                float = 500e-6    # substrate height
    rho:              float = 1e-19     # resistivity (near-zero = superconducting)
    ep_r:             float = 11.45     # relative permittivity (ultracold Si)
    has_metal_backside: bool = False     # metallic backside ground plane
    tand:             float = 0.0       # loss tangent
