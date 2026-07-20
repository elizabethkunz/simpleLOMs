Tutorials
=========

Step-by-step guides for using **simpleLOMs**.

.. toctree::
   :maxdepth: 1

   01_fit_lc_models
   02_ranking_and_loaded_shifts
   03_resonator_networks
   04_cpw_parameter_sweeps
   05_compare_to_simulations
   06_readout_parameters
   07_design_to_spec
   08_hanger_resonators
   09_hybrid_series_hanger


.. Overview
.. --------

.. 1. **Fitting LC models** — single-device workflow with ``analyze_system()`` (circle-fit S11/S22 metrics and hybridization shifts).
.. 2. **Ranking LOMs and loaded shifts** — compare Foster / Analytical / Optimized against the CPW ground truth on an asymmetric coupling grid, predict the frequency shift when shunt loads are attached (bare and loaded measured under the same weak-probe convention), and predict the avoided crossing when the load is a resonant LC tank (rebuild via each model's own ``get_network()``).
.. 3. **Making resonator networks** — build a chain of capacitively-coupled CPW resonators (``resonator_chain_network_2port``), draw it with the schematics feature, fit an Optimized LOM to each resonator subgroup, rebuild the LC chain, and check per-mode agreement against the CPW ground truth (hangers in Tutorials 8–9).
.. 4. **Parameter sweeps** — sweep length, coupling, and CPW geometry with ``measure_sweep`` / ``extract_f0_kappa`` / ``fit_lom`` to map design knobs to circuit targets (bridge to Tutorial 7).
.. 5. **EM comparison** — overlay a lumped model against a Touchstone / FEM reference and quantify the residual.
.. 6. **Readout parameters** — convert fitted ``L``, ``C`` into circuit / readout quantities, measure loaded ``\kappa`` and ``Q``, then size ``C_c`` for a target ``Q`` via a coupling sweep.
.. 7. **Designing a CPW resonator to match targets** — pick length ``d`` for a target frequency, then coupling ``C_c`` for a target ``Q``, verify against the CPW ground truth, and bundle the recipe in ``design_resonator``.
.. 8. **Hanger resonators** — build a CPW hanger / notch, contrast rewiring an inline ``OptimizedFit`` with ``fit_lom_hanger``, compare notches, draw the LOM schematic, and multiplex several hangers on one feedline.
.. 9. **Hybrid series + hanger** — build a CPW hybrid (inline Purcell + open hangers), draw it with ``hybrid_schematic_2port``, fit each piece with the matching LOM (``fit_lom`` / ``fit_lom_hanger``), rebuild the lumped hybrid, and check transmission / notch accuracy against the CPW ground truth.
