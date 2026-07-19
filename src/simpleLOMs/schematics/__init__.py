"""
simpleLOMs.schematics
=====================
Optional circuit-diagram generation for the networks built in
:mod:`simpleLOMs.networks`.

Produces clean, **white-background** schematics that show the capacitance and
inductance values, rendered as a **zoomable** interactive object (scroll to
zoom, drag to pan) either as a standalone HTML file or inline in a Jupyter
notebook.

Quick start
-----------
    from simpleLOMs.schematics import cpw_schematic_2port
    from simpleLOMs import CPWParams
    import skrf as rf

    freq = rf.Frequency(4, 10, 4001, "GHz")
    sch = cpw_schematic_2port(
        d=7e-3, Cc1=5e-15, Cc2=5e-15, Ctog1=1e-14, Ctog2=1e-14,
        cpw_params=CPWParams(ep_r=11.45), freq=freq,
        annotations={"f0_GHz": 7.918, "QL": 52},
    )
    sch.save_html("circuit.html")   # zoomable, white background
    sch.save_svg("circuit.svg")     # static vector
    sch.save_json("schematic.json") # portable netlist
    sch                             # renders inline & zoomable in Jupyter

The intermediate :class:`Schematic` is JSON-serialisable and round-trips with
the ``schematic.json`` schema already used in the notebooks.
"""
from simpleLOMs.schematics.netlist import (
    Component,
    Net,
    Schematic,
    format_eng,
)
from simpleLOMs.schematics.builders import (
    SchematicBuilder,
    cpw_schematic,
    cpw_schematic_2port,
    cpw_loaded_schematic_2port,
    cpw_single_load_schematic_2port,
    lc_schematic,
    lc_schematic_2port,
    lc_with_grounds_schematic_2port,
    lc_loaded_schematic_2port,
    lc_loaded_with_grounds_schematic_2port,
    resonator_chain_schematic_2port,
    hanger_resonator_schematic_2port,
    hybrid_schematic_2port,
)
from simpleLOMs.schematics.render_svg import schematic_to_svg
from simpleLOMs.schematics.viewer import (
    schematic_to_html,
    schematic_to_iframe,
)

__all__ = [
    "Component",
    "Net",
    "Schematic",
    "format_eng",
    "SchematicBuilder",
    "cpw_schematic",
    "cpw_schematic_2port",
    "cpw_loaded_schematic_2port",
    "cpw_single_load_schematic_2port",
    "lc_schematic",
    "lc_schematic_2port",
    "lc_with_grounds_schematic_2port",
    "lc_loaded_schematic_2port",
    "lc_loaded_with_grounds_schematic_2port",
    "resonator_chain_schematic_2port",
    "hanger_resonator_schematic_2port",
    "hybrid_schematic_2port",
    "schematic_to_svg",
    "schematic_to_html",
    "schematic_to_iframe",
]
