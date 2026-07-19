"""Simple Lumped Oscillator Models for superconducting quantum device design."""

__version__ = "0.1.0"


"""
simpleLOMs
--------
Lumped-element LC model fitting for CPW microwave resonators.

Quick start
-----------
    import skrf as rf
    from simpleLOMs import analyze_system, CPWParams

    freq = rf.Frequency(4e9, 10e9, 10_001, unit="Hz")
    cpw  = CPWParams(ep_r=11.45)

    results = analyze_system(
        freq=freq,
        d=7e-3,
        Cc1=5e-15, Cc2=5e-15,
        Ctog1=1e-14, Ctog2=1e-14,
        Lload1=5e-10, Cload1=6e-13,
        Lload2=5e-10, Cload2=6e-13,
        cpw_params=cpw,
    )

Or use the model classes directly:

    from simpleLOMs.models.foster_fit import FosterFit
    from simpleLOMs.models.optimized_fit import OptimizedFit
    from simpleLOMs.models.analytical_fit import AnalyticalFit
"""

from simpleLOMs.params import CPWParams
from simpleLOMs.system import (
    fit_lom,
    fit_lom_hanger,
    analyze_system,
    run_accuracy_sweep,
    analyze_system_load_grid,
    run_accuracy_sweep_load_grid,
)
from simpleLOMs.models.foster_fit import FosterFit
from simpleLOMs.models.optimized_fit import OptimizedFit, OptimizationConfig
from simpleLOMs.models.hanger_optimized_fit import HangerOptimizedFit
from simpleLOMs.models.analytical_fit import AnalyticalFit
from simpleLOMs.readout import (
    ReadoutParams,
    resonator_readout_params,
    extract_f0_kappa,
    invert_monotonic,
    design_resonator,
)
from simpleLOMs.sweeps import (
    SweepConfig,
    sweep,
    sweep_length,
    sweep_coupling,
    sweep_load_frequency,
    measure_sweep,
)
from simpleLOMs.plotting import (
    plot_lom_vs_data,
    plot_residuals,
    plot_summary,
    plot_transmission,
    plot_s_residual,
    plot_all_models,
    plot_scan,
    plot_re_im,
    plot_curves,
    plot_curve_pair,
    plot_heatmaps,
    plot_error_heatmap,
    plot_error_heatmap_trio,
    # Shared figure style (single source of truth for every plot)
    apply_style,
    axis_label,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_HIGHLIGHT,
    FIGURE_WIDTH_1COL,
    FIGURE_WIDTH_2COL,
    FIGURE_HEIGHT_1COL_GR,
    FIGURE_HEIGHT_2COL_GR,
    LABEL_FREQ,
    LABEL_F0,
    LABEL_KAPPA,
    LABEL_Q,
    LABEL_LENGTH,
    LABEL_CC,
    LABEL_CPW_GAP,
)

# Network / analysis / schematic builders, promoted to the top level so a plain
# `import simpleLOMs as sl` reaches everything (no deep submodule imports needed).
from simpleLOMs.analysis import (
    circle_fit_f0_kappa,
    circle_fit_f0_kappa_windowed,
    resonances_from_s_max,
    s_db,
    notch_frequency,
    notches_below,
)
from simpleLOMs.networks.chain import resonator_chain_network_2port
from simpleLOMs.networks.cpw import (
    cpw_resonator_network_2port,
    cpw_resonator_loaded_network_2port,
    bare_cpw_resonance_hz,
    cpw_line_impedance,
)
from simpleLOMs.networks.hanger import hanger_resonator_network_2port
from simpleLOMs.networks.lc import (
    lc_resonator_loaded_network_2port,
    lc_resonator_loaded_network_with_grounds_2port,
)
from simpleLOMs.schematics import (
    cpw_schematic_2port,
    cpw_loaded_schematic_2port,
    resonator_chain_schematic_2port,
    hanger_resonator_schematic_2port,
    hybrid_schematic_2port,
)
from simpleLOMs import utils

__all__ = [
    "CPWParams",
    "fit_lom",
    "fit_lom_hanger",
    "analyze_system",
    "run_accuracy_sweep",
    "analyze_system_load_grid",
    "run_accuracy_sweep_load_grid",
    "FosterFit",
    "OptimizedFit",
    "OptimizationConfig",
    "HangerOptimizedFit",
    "AnalyticalFit",
    "ReadoutParams",
    "resonator_readout_params",
    "extract_f0_kappa",
    "invert_monotonic",
    "design_resonator",
    "SweepConfig",
    "sweep",
    "sweep_length",
    "sweep_coupling",
    "sweep_load_frequency",
    "measure_sweep",
    "plot_lom_vs_data",
    "plot_residuals",
    "plot_summary",
    "plot_transmission",
    "plot_s_residual",
    "plot_all_models",
    "plot_scan",
    "plot_re_im",
    "plot_curves",
    "plot_curve_pair",
    "plot_heatmaps",
    "plot_error_heatmap",
    "plot_error_heatmap_trio",
    "apply_style",
    "axis_label",
    "MODEL_COLORS",
    "MODEL_LABELS",
    "MODEL_LINESTYLES",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_HIGHLIGHT",
    "FIGURE_WIDTH_1COL",
    "FIGURE_WIDTH_2COL",
    "FIGURE_HEIGHT_1COL_GR",
    "FIGURE_HEIGHT_2COL_GR",
    "LABEL_FREQ",
    "LABEL_F0",
    "LABEL_KAPPA",
    "LABEL_Q",
    "LABEL_LENGTH",
    "LABEL_CC",
    "LABEL_CPW_GAP",
    "circle_fit_f0_kappa",
    "circle_fit_f0_kappa_windowed",
    "resonances_from_s_max",
    "s_db",
    "notch_frequency",
    "notches_below",
    "resonator_chain_network_2port",
    "cpw_resonator_network_2port",
    "cpw_resonator_loaded_network_2port",
    "bare_cpw_resonance_hz",
    "cpw_line_impedance",
    "hanger_resonator_network_2port",
    "lc_resonator_loaded_network_2port",
    "lc_resonator_loaded_network_with_grounds_2port",
    "cpw_schematic_2port",
    "cpw_loaded_schematic_2port",
    "resonator_chain_schematic_2port",
    "hanger_resonator_schematic_2port",
    "hybrid_schematic_2port",
    "utils",
]
