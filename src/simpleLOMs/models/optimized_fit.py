# from __future__ import annotations

# """
# models/optimized.py
# -------------------
# OptimizedFit: numerical optimization over S11 + S22 residuals to find
# the best-fit effective L and C for a CPW resonator.

# Algorithm summary
# -----------------
# Stage 1 — coarse scan (fit_Ceff_Leff):
#     Scan ω0 in a window around the measured resonance.  For each ω0,
#     use Nelder-Mead to optimise Leff with Ceff forced by the resonance
#     constraint ω0² * L * (C + Cc1 + Cc2) = 1.  Keep the best (L, C).

# Stage 2 — windowed refinement (least_squares):
#     Build separate frequency windows around the S11 and S22 resonances.
#     Minimise the complex residuals of the 2-port LC network against the
#     CPW reference network, weighted by resonance depth to prevent one
#     shallow trace from dominating.
# """

# import logging
# from dataclasses import dataclass, field

# import numpy as np
# import skrf as rf
# from scipy.optimize import minimize, least_squares

# from simpleLOMs.models.base import BaseFit
# from simpleLOMs.analysis import (
#     resonance_from_res11,
#     fwhm_from_res11,
#     fwhm_from_trace_db,
#     resonance,
#     resonance_from_s_max,
# )
# from simpleLOMs.networks.lc import (
#     lc_resonator_network,
#     lc_resonator_network_2port,
#     lc_resonator_network_with_grounds_2port,
# )

# logger = logging.getLogger(__name__)


# # ---------------------------------------------------------------------------
# # Helper dataclass for optimizer hyperparameters
# # ---------------------------------------------------------------------------

# @dataclass
# class OptimizationConfig:
#     """
#     Tuning knobs for the two-stage OptimizedFit algorithm.

#     You rarely need to change these — the defaults work well for typical
#     CPW resonators in the 4–10 GHz range.  They are collected here so
#     that a caller can pass a single object instead of many keyword args.

#     Attributes
#     ----------
#     w0_window_frac : float
#         Half-width of the ω0 scan window as a fraction of ω0_guess.
#         e.g. 0.005 → scan ω0 ± 0.5 %.
#     n_w0 : int
#         Number of ω0 grid points in the coarse scan.
#     n_dense : int
#         Number of data points sampled around resonance in Stage 1.
#     n_kappa : float
#         Half-width of the dense sampling window in units of κ (linewidth).
#     n_widths : float
#         Half-width of each residual window in Stage 2, in units of κ.
#     max_nfev : int
#         Maximum function evaluations for the Stage 2 least_squares call.
#     verbose : bool
#         If True, least_squares prints iteration progress.
#     """
#     w0_window_frac: float = 0.00001
#     n_w0:          int   = 20
#     n_dense:       int   = 100
#     n_kappa:       float = 0.75
#     n_widths:      float = 1.0
#     max_nfev:      int   = 100
#     verbose:       bool  = False


# # ---------------------------------------------------------------------------
# # OptimizedFit class
# # ---------------------------------------------------------------------------

# class OptimizedFit(BaseFit):
#     """
#     Lumped LC model via two-stage numerical optimisation.

#     Parameters
#     ----------
#     config : OptimizationConfig, optional
#         Hyperparameters for the fitting algorithm.
#         Uses sensible defaults if not provided.

#     Attributes
#     ----------
#     L : float or None
#         Fitted effective inductance in Henries.  None before fit().
#     C : float or None
#         Fitted effective capacitance in Farads.  None before fit().
#     scan_results : np.ndarray or None
#         Diagnostic array from the Stage 1 ω0 scan, shape (n_w0, 4):
#         columns are [ω0, Ceff, Leff, sse].

#     Examples
#     --------
#     ::

#         from simpleLOMs.models.optimized import OptimizedFit, OptimizationConfig
#         import skrf as rf

#         freq       = rf.Frequency(4e9, 10e9, 10_001, unit="Hz")
#         data_ntw   = ...  # rf.Network of the CPW S11 measurement
#         config     = OptimizationConfig(n_w0=30, w0_window_frac=0.01)

#         model = OptimizedFit(config=config)
#         model.fit(freq, data_ntw=data_ntw, Cc1=5e-15, Cc2=5e-15, Z0=50)

#         net = model.get_network(freq, Cc1=5e-15, Cc2=5e-15, Z0=50)
#     """

#     def __init__(self, config: OptimizationConfig = None):
#         self.config: OptimizationConfig    = config if config is not None else OptimizationConfig()
#         self.L:            float | None    = None
#         self.C:            float | None    = None
#         self.scan_results: np.ndarray | None = None

#     # ------------------------------------------------------------------
#     # Internal helpers
#     # ------------------------------------------------------------------

#     def _build_sparse_data_points(
#         self,
#         data_ntw: rf.Network,
#         n_dense: int,
#         n_kappa: float,
#     ) -> list[tuple[float, float, float]]:
#         """
#         Build a sparse, physically-motivated set of (ω, Re(S11), Im(S11)) tuples.

#         Always includes the minimum of Re(S11) (resonance point).
#         Optionally adds a dense cluster of points around the resonance.

#         Parameters
#         ----------
#         data_ntw : rf.Network
#             Single-port S11 measurement or model.
#         n_dense : int
#             How many extra points to add around resonance. 0 disables.
#         n_kappa : float
#             Half-width of the dense window in units of linewidth κ.

#         Returns
#         -------
#         list of (omega, ReS11, ImS11)
#         """
#         f_hz  = data_ntw.frequency.f
#         w     = 2 * np.pi * f_hz
#         S11   = data_ntw.s[:, 0, 0]
#         re_s11 = np.real(S11)
#         im_s11 = np.imag(S11)

#         # Resonance point (minimum of Re(S11))
#         idx_re_min = np.argmin(re_s11)
#         data_points = [(w[idx_re_min], re_s11[idx_re_min], im_s11[idx_re_min])]

#         if n_dense > 0:
#             f0    = f_hz[idx_re_min]
#             kappa = fwhm_from_res11(data_ntw)

#             f_dense = np.linspace(
#                 f0 - n_kappa * kappa,
#                 f0 + (n_kappa / 2) * kappa,
#                 n_dense + 2,
#             )[1:-1]

#             re_dense = np.interp(f_dense, f_hz, re_s11)
#             im_dense = np.interp(f_dense, f_hz, im_s11)

#             for f_pt, r, i in zip(f_dense, re_dense, im_dense):
#                 data_points.append((2 * np.pi * f_pt, r, i))

#         return data_points

#     def _fit_Ceff_Leff(
#         self,
#         w0_guess: float,
#         Cc1: float,
#         Cc2: float,
#         Z0: float,
#         data_points: list,
#         freq: rf.Frequency,
#     ) -> tuple[float, float, float, np.ndarray]:
#         """
#         Stage 1: scan ω0 and fit Leff by Nelder-Mead with Ceff constrained.

#         Returns (Ceff_best, Leff_best, w0_best, scan_results).
#         """
#         cfg = self.config
#         dp  = np.asarray(data_points, dtype=float)
#         freqs  = dp[:, 0]
#         target = dp[:, 1] + 1j * dp[:, 2]

#         def ZCc1(w):   return 1.0 / (1j * w * Cc1)
#         def ZCc2(w):   return 1.0 / (1j * w * Cc2)
#         def ZL(w, Le): return 1j * w * Le
#         def ZC(w, Ce): return 1.0 / (1j * w * Ce)
#         def ZLC(w, Le, Ce):
#             return 1.0 / (1.0 / ZL(w, Le) + 1.0 / ZC(w, Ce))
#         def Zshunt(w, Le, Ce):
#             return 1.0 / (1.0 / ZLC(w, Le, Ce) + 1.0 / ZCc2(w))
#         def Zin(w, Le, Ce):
#             return ZCc1(w) + Zshunt(w, Le, Ce)
#         def S11(w, Le, Ce):
#             z = Zin(w, Le, Ce)
#             return (z - Z0) / (z + Z0)

#         w0_min  = w0_guess * (1.0 - cfg.w0_window_frac)
#         w0_max  = w0_guess * (1.0 + cfg.w0_window_frac)
#         w0_grid = np.linspace(w0_min, w0_max, int(cfg.n_w0))

#         best    = None
#         results = []

#         for w0 in w0_grid:
#             def ceff_of_L(Le):
#                 return 1.0 / (Le * (w0 ** 2)) - (Cc1 + Cc2)

#             Le_max   = 1.0 / ((Cc1 + Cc2) * (w0 ** 2))
#             Le_upper = 0.999 * Le_max
#             x0_local = 0.5 * Le_upper

#             def obj(Le_arr):
#                 Le = float(Le_arr[0])
#                 if not (0.0 < Le < Le_upper):
#                     dist = (0.0 - Le) if Le <= 0.0 else (Le - Le_upper)
#                     return 1e6 * (1.0 + dist ** 2)
#                 Ce = ceff_of_L(Le)
#                 if Ce <= 0.0 or not np.isfinite(Ce):
#                     return 1e6
#                 pred  = np.array([S11(w, Le, Ce) for w in freqs])
#                 resid = pred - target
#                 return float(np.sum(resid.real ** 2 + resid.imag ** 2))

#             fit = minimize(
#                 obj,
#                 x0=np.array([x0_local]),
#                 method="Nelder-Mead",
#                 options=dict(maxiter=3000, xatol=1e-30, fatol=1e-30, disp=False),
#             )

#             Le_hat = float(np.clip(fit.x[0], np.nextafter(0.0, 1.0), Le_upper))
#             Ce_hat = ceff_of_L(Le_hat)
#             sse    = float(fit.fun)
#             results.append((w0, Ce_hat, Le_hat, sse))

#             if best is None or sse < best[-1]:
#                 best = (w0, Ce_hat, Le_hat, sse)

#         w0_best, Ceff_best, Leff_best, _ = best
#         return Ceff_best, Leff_best, w0_best, np.array(results)

#     def _make_windowed_residuals(
#         self,
#         freq: rf.Frequency,
#         data_ntw: rf.Network,
#         Cc1: float,
#         Cc2: float,
#         f0_s11: float,
#         kappa_s11: float,
#         f0_s22: float,
#         kappa_s22: float,
#         Z0: float,
#         eps: float = 1e-18,
#     ):
#         """
#         Stage 2: build the windowed residual function for least_squares.

#         Returns a callable residuals(x) where x = [Leff, Ceff].
#         """
#         cfg = self.config

#         # Interpolate data onto the optimisation frequency grid
#         f_old = data_ntw.frequency.f
#         f_new = freq.f

#         if len(f_old) == len(f_new) and np.allclose(f_old, f_new):
#             data_aligned = data_ntw
#         else:
#             S_old = data_ntw.s
#             S_new = np.zeros((len(f_new), 2, 2), dtype=complex)
#             for i in range(2):
#                 for j in range(2):
#                     yr = np.interp(f_new, f_old, np.real(S_old[:, i, j]))
#                     yi = np.interp(f_new, f_old, np.imag(S_old[:, i, j]))
#                     S_new[:, i, j] = yr + 1j * yi
#             z0_old = data_ntw.z0
#             if z0_old.ndim == 1:
#                 z0_new = np.tile(z0_old[0], (len(f_new), 2))
#             else:
#                 z0_new = np.zeros((len(f_new), 2))
#                 for p in range(2):
#                     z0_new[:, p] = np.interp(f_new, f_old, z0_old[:, p])
#             data_aligned = rf.Network(frequency=freq, s=S_new, z0=z0_new)

#         f = freq.f
#         n = cfg.n_widths
#         mask11 = (f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)
#         mask22 = (f > f0_s22 - n * kappa_s22) & (f < f0_s22 + n * kappa_s22)

#         if not np.any(mask11):
#             raise ValueError("S11 mask is empty: check f0_s11/kappa_s11/n_widths.")
#         if not np.any(mask22):
#             raise ValueError("S22 mask is empty: check f0_s22/kappa_s22/n_widths.")

#         freq11   = rf.Frequency.from_f(f[mask11], unit="Hz")
#         freq22   = rf.Frequency.from_f(f[mask22], unit="Hz")
#         s11_data = data_aligned.s[mask11, 0, 0]
#         s22_data = data_aligned.s[mask22, 1, 1]

#         def _depth(s_complex):
#             m = np.real(s_complex)
#             return max(float(np.max(m) - np.min(m)), eps)

#         S11_scale = _depth(s11_data)
#         S22_scale = _depth(s22_data)
#         depth_floor = max(S11_scale, S22_scale)
#         S11_scale = max(S11_scale, depth_floor)
#         S22_scale = max(S22_scale, depth_floor)

#         logger.debug("S11 depth scale=%.4e, N11=%d", S11_scale, mask11.sum())
#         logger.debug("S22 depth scale=%.4e, N22=%d", S22_scale, mask22.sum())

#         def residuals(x):
#             Le, Ce = float(x[0]), float(x[1])
#             lom11  = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq11, Z0)
#             lom22  = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq22, Z0)
#             r11    = lom11.s[:, 0, 0] - s11_data
#             r22    = lom22.s[:, 1, 1] - s22_data
#             r11_sc = np.concatenate([r11.real, r11.imag]) / S11_scale
#             r22_sc = np.concatenate([r22.real, r22.imag]) / S22_scale
#             return np.concatenate([r11_sc, r22_sc])

#         return residuals

#     def _make_windowed_residuals2(
#         self,
#         freq: rf.Frequency,
#         data_ntw: rf.Network,
#         Cc1: float,
#         Cc2: float,
#         f0_s11: float,
#         kappa_s11: float,
#         f0_s22: float,
#         kappa_s22: float,
#         Z0: float,
#         eps: float = 1e-18,
#         phase=None,
#     ):
#         cfg = self.config
#         f_old = data_ntw.frequency.f
#         f_new = freq.f
#         if len(f_old) == len(f_new) and np.allclose(f_old, f_new):
#             data_aligned = data_ntw
#         else:
#             S_old = data_ntw.s
#             S_new = np.zeros((len(f_new), 2, 2), dtype=complex)
#             for i in range(2):
#                 for j in range(2):
#                     yr = np.interp(f_new, f_old, np.real(S_old[:, i, j]))
#                     yi = np.interp(f_new, f_old, np.imag(S_old[:, i, j]))
#                     S_new[:, i, j] = yr + 1j * yi
#             z0_old = data_ntw.z0
#             if z0_old.ndim == 1:
#                 z0_new = np.tile(z0_old[0], (len(f_new), 2))
#             else:
#                 z0_new = np.zeros((len(f_new), 2))
#                 for p in range(2):
#                     z0_new[:, p] = np.interp(f_new, f_old, z0_old[:, p])
#             data_aligned = rf.Network(frequency=freq, s=S_new, z0=z0_new)

#         f = freq.f
#         n = cfg.n_widths
#         mask11 = (f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)
#         mask22 = (f > f0_s22 - n * kappa_s22) & (f < f0_s22 + n * kappa_s22)
#         mask21 =(f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)
#         mask12 =(f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)


#         if not np.any(mask11):
#             raise ValueError("S11 mask is empty: check f0_s11/kappa_s11/n_widths.")
#         if not np.any(mask22):
#             raise ValueError("S22 mask is empty: check f0_s22/kappa_s22/n_widths.")
#         if not np.any(mask21):
#             raise ValueError("S21 mask is empty: no frequencies outside S11/S22 windows.")

#         freq11 = rf.Frequency.from_f(f[mask11], unit="Hz")
#         freq22 = rf.Frequency.from_f(f[mask22], unit="Hz")
#         freq21 = rf.Frequency.from_f(f[mask21], unit="Hz")
#         freq12 = rf.Frequency.from_f(f[mask21], unit="Hz")

#         s11_data = data_aligned.s[mask11, 0, 0]
#         s22_data = data_aligned.s[mask22, 1, 1]
#         s21_data = data_aligned.s[mask21, 1, 0]
#         s12_data = data_aligned.s[mask12, 0, 1]

#         def _depth(s_complex):
#             m = np.abs(s_complex)
#             return max(float(np.max(m) - np.min(m)), eps)

#         S11_scale = _depth(s11_data)
#         S22_scale = _depth(s22_data)
#         depth_floor = max(S11_scale, S22_scale)
#         S11_scale = max(S11_scale, depth_floor)
#         S22_scale = max(S22_scale, depth_floor)
#         S21_scale = max(_depth(s21_data), depth_floor)
#         S12_scale = max(_depth(s12_data), depth_floor)

#         logger.debug("S11 depth scale=%.4e, N11=%d", S11_scale, mask11.sum())
#         logger.debug("S22 depth scale=%.4e, N22=%d", S22_scale, mask22.sum())
#         logger.debug("S21 depth scale=%.4e, N21=%d", S21_scale, mask21.sum())

#         def residuals(x):
#             Le, Ce = float(x[0]), float(x[1])
#             lom11 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq11, Z0)
#             lom22 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq22, Z0)
#             lom21 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq21, Z0)
#             lom12 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq12, Z0)
#             if phase is not None:
#                 r11 = lom11.s[:, 0, 0] * phase - s11_data
#                 r22 = lom22.s[:, 1, 1] * phase - s22_data
#                 r21 = lom21.s[:, 1, 0] * phase - s21_data
#                 r12 = lom12.s[:, 0, 1] * phase - s12_data

#                 r11_sc = np.concatenate([r11.real, r11.imag]) / S11_scale
#                 r22_sc = np.concatenate([r22.real, r22.imag]) / S22_scale
#                 r21_sc = np.concatenate([r21.real, r21.imag]) / S21_scale
#                 r12_sc = np.concatenate([r12.real, r12.imag]) / S12_scale
#             else: 
#                 r11 = lom11.s[:, 0, 0] - s11_data
#                 r22 = lom22.s[:, 1, 1] - s22_data
#                 r21 = lom21.s[:, 1, 0] - s21_data
#                 r12 = lom12.s[:, 0, 1] - s12_data

#                 r11_sc = np.concatenate([r11.real, r11.imag]) / S11_scale
#                 r22_sc = np.concatenate([r22.real, r22.imag]) / S22_scale
#                 r21_sc = np.concatenate([r21.real, r21.imag]) / S21_scale
#                 r12_sc = np.concatenate([r12.real, r12.imag]) / S21_scale

#             return np.concatenate([r11_sc, r22_sc, r21_sc, r12_sc])
#             # lom21  = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq21, Z0)
#             # r21_sc = np.concatenate([(lom21.s[:, 1, 0] - s21_data).real,
#             #                         (lom21.s[:, 1, 0] - s21_data).imag]) / S21_scale
#             # r12_sc = np.concatenate([(lom21.s[:, 0, 1] - s12_data).real,
#             #                         (lom21.s[:, 0, 1] - s12_data).imag]) / S21_scale
#             # return np.concatenate([r11_sc, r22_sc, r21_sc, r12_sc])

#         return residuals
    
#     def _make_windowed_residuals_complex(
#         self,
#         freq: rf.Frequency,
#         data_ntw: rf.Network,
#         Cc1: float,
#         Cc2: float,
#         f0_s11: float,
#         kappa_s11: float,
#         f0_s22: float,
#         kappa_s22: float,
#         f0_s12: float,
#         f0_s21: float,
#         Z0: float,
#         eps: float = 1e-18,
#     ):
#         cfg = self.config
#         f_old = data_ntw.frequency.f
#         f_new = freq.f
#         if len(f_old) == len(f_new) and np.allclose(f_old, f_new):
#             data_aligned = data_ntw
#         else:
#             S_old = data_ntw.s
#             S_new = np.zeros((len(f_new), 2, 2), dtype=complex)
#             for i in range(2):
#                 for j in range(2):
#                     yr = np.interp(f_new, f_old, np.real(S_old[:, i, j]))
#                     yi = np.interp(f_new, f_old, np.imag(S_old[:, i, j]))
#                     S_new[:, i, j] = yr + 1j * yi
#             z0_old = data_ntw.z0
#             if z0_old.ndim == 1:
#                 z0_new = np.tile(z0_old[0], (len(f_new), 2))
#             else:
#                 z0_new = np.zeros((len(f_new), 2))
#                 for p in range(2):
#                     z0_new[:, p] = np.interp(f_new, f_old, z0_old[:, p])
#             data_aligned = rf.Network(frequency=freq, s=S_new, z0=z0_new)

#         f = freq.f
#         n = cfg.n_widths
#         mask11 = (f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)
#         mask22 = (f > f0_s22 - n * kappa_s22) & (f < f0_s22 + n * kappa_s22)
#         mask21 = (f > f0_s22 - n * kappa_s22) & (f < f0_s22 + n * kappa_s22)

#         if not np.any(mask11):
#             raise ValueError("S11 mask is empty: check f0_s11/kappa_s11/n_widths.")
#         if not np.any(mask22):
#             raise ValueError("S22 mask is empty: check f0_s22/kappa_s22/n_widths.")
#         if not np.any(mask21):
#             raise ValueError("S21 mask is empty: no frequencies outside S11/S22 windows.")

#         freq11 = rf.Frequency.from_f(f[mask11], unit="Hz")
#         freq22 = rf.Frequency.from_f(f[mask22], unit="Hz")
#         freq21 = rf.Frequency.from_f(f[mask21], unit="Hz")

#         s11_data = data_aligned.s[mask11, 0, 0]
#         s22_data = data_aligned.s[mask22, 1, 1]
#         s21_data = data_aligned.s[mask21, 1, 0]
#         s12_data = data_aligned.s[mask21, 0, 1]

#         def _depth(s_complex):
#             m = np.abs(s_complex)
#             return max(float(np.max(m) - np.min(m)), eps)

#         S11_scale = _depth(s11_data)
#         S22_scale = _depth(s22_data)
#         depth_floor = max(S11_scale, S22_scale)
#         S11_scale = max(S11_scale, depth_floor)
#         S22_scale = max(S22_scale, depth_floor)
#         S21_scale = max(_depth(s21_data), depth_floor)

#         logger.debug("S11 depth scale=%.4e, N11=%d", S11_scale, mask11.sum())
#         logger.debug("S22 depth scale=%.4e, N22=%d", S22_scale, mask22.sum())
#         logger.debug("S21 depth scale=%.4e, N21=%d", S21_scale, mask21.sum())

#         def residuals(x):
#             Le, Ce, phi = float(x[0]), float(x[1]), float(x[2])
#             phase = np.exp(1j * phi)

#             lom11 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq11, Z0)
#             lom22 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq22, Z0)
#             lom21 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq21, Z0)

#             r11 = lom11.s[:, 0, 0] * phase - s11_data
#             r22 = lom22.s[:, 1, 1] * phase - s22_data
#             r21 = lom21.s[:, 1, 0] * phase - s21_data
#             #r12 = lom21.s[:, 0, 1] * phase - s12_data

#             r11_sc = np.concatenate([r11.real, r11.imag]) / S11_scale
#             r22_sc = np.concatenate([r22.real, r22.imag]) / S22_scale
#             r21_sc = np.concatenate([r21.real, r21.imag]) / S21_scale
#             #r12_sc = np.concatenate([r12.real, r12.imag]) / S21_scale

#             return np.concatenate([r11_sc, r22_sc, r21_sc])
#             #return np.concatenate([r11_sc, r22_sc, r21_sc, r12_sc])

#         return residuals
    
#     def _make_windowed_residuals_z(
#         self,
#         freq: rf.Frequency,
#         data_ntw: rf.Network,
#         Cc1: float,
#         Cc2: float,
#         f0_s11: float,
#         kappa_s11: float,
#         f0_s22: float,
#         kappa_s22: float,
#         Z0: float,
#         eps: float = 1e-18,
#         ):
#         cfg = self.config
#         f_old = data_ntw.frequency.f
#         f_new = freq.f
#         if len(f_old) == len(f_new) and np.allclose(f_old, f_new):
#             data_aligned = data_ntw
#         else:
#             S_old = data_ntw.s
#             S_new = np.zeros((len(f_new), 2, 2), dtype=complex)
#             for i in range(2):
#                 for j in range(2):
#                     yr = np.interp(f_new, f_old, np.real(S_old[:, i, j]))
#                     yi = np.interp(f_new, f_old, np.imag(S_old[:, i, j]))
#                     S_new[:, i, j] = yr + 1j * yi
#             z0_old = data_ntw.z0
#             if z0_old.ndim == 1:
#                 z0_new = np.tile(z0_old[0], (len(f_new), 2))
#             else:
#                 z0_new = np.zeros((len(f_new), 2))
#                 for p in range(2):
#                     z0_new[:, p] = np.interp(f_new, f_old, z0_old[:, p])
#             data_aligned = rf.Network(frequency=freq, s=S_new, z0=z0_new)

#         f = freq.f
#         n = cfg.n_widths
#         mask11 = (f > f0_s11 - n * kappa_s11) & (f < f0_s11 + n * kappa_s11)
#         mask22 = (f > f0_s22 - n * kappa_s22) & (f < f0_s22 + n * kappa_s22)

#         if not np.any(mask11):
#             raise ValueError("Z11 mask is empty: check f0_s11/kappa_s11/n_widths.")
#         if not np.any(mask22):
#             raise ValueError("Z22 mask is empty: check f0_s22/kappa_s22/n_widths.")

#         freq11 = rf.Frequency.from_f(f[mask11], unit="Hz")
#         freq22 = rf.Frequency.from_f(f[mask22], unit="Hz")

#         # Use Z-parameters directly
#         z11_data = data_aligned.z[mask11, 0, 0]
#         z22_data = data_aligned.z[mask22, 1, 1]

#         def _depth(z_complex):
#             m = np.abs(z_complex)
#             return max(float(np.max(m) - np.min(m)), eps)

#         Z11_scale = _depth(z11_data)
#         Z22_scale = _depth(z22_data)
#         depth_floor = max(Z11_scale, Z22_scale)
#         Z11_scale = max(Z11_scale, depth_floor)
#         Z22_scale = max(Z22_scale, depth_floor)

#         logger.debug("Z11 depth scale=%.4e, N11=%d", Z11_scale, mask11.sum())
#         logger.debug("Z22 depth scale=%.4e, N22=%d", Z22_scale, mask22.sum())

#         def residuals(x):
#             Le, Ce = float(x[0]), float(x[1])
#             lom11 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq11, Z0)
#             lom22 = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, freq22, Z0)

#             r11 = lom11.z[:, 0, 0] - z11_data
#             r22 = lom22.z[:, 1, 1] - z22_data

#             r11_sc = np.concatenate([r11.real, r11.imag]) / Z11_scale
#             r22_sc = np.concatenate([r22.real, r22.imag]) / Z22_scale

#             return np.concatenate([r11_sc, r22_sc])

#         return residuals
#         # ------------------------------------------------------------------
#     # fit()
#     # ------------------------------------------------------------------

#     def fit(
#         self,
#         freq: rf.Frequency,
#         data_ntw: rf.Network,
#         Cc1: float,
#         Cc2: float,
#         Ctog1: float,
#         Ctog2: float,
#         d: float,
#         cpw_params: CPWParams,
#         Z0: float = 50.0,
#         **kwargs,
#     ) -> None:
#         """
#         Two-stage optimisation to fit Leff and Ceff to data_ntw.

#         Stage 1: coarse Nelder-Mead scan over ω0, constrained Ceff.
#             Uses an explicitly constructed 1-port CPW network so that the
#             Stage 1 data points come from a true single-port reflection
#             measurement rather than a 2-port S11 view.
#         Stage 2: windowed least_squares refinement on S11 + S22.

#         Parameters
#         ----------
#         freq : rf.Frequency
#             Frequency grid for the optimisation (should be centred on the
#             resonance with enough bandwidth to capture the linewidth).
#         data_ntw : rf.Network
#             2-port reference network (CPW model or measured data).
#             Used for Stage 2 S11 + S22 residuals.
#         Cc1, Cc2 : float
#             Coupling capacitances in Farads.
#         Ctog1, Ctog2 : float
#             Shunt-to-ground capacitances in Farads.
#         d : float
#             Resonator length in metres — used to build the 1-port CPW network.
#         cpw_params : CPWParams
#             CPW geometry — used to build the 1-port CPW network.
#         Z0 : float
#             Reference impedance in Ohms.
#         """
#         from simpleLOMs.networks.cpw import cpw_resonator_network

#         cfg = self.config

#         # --- Stage 1: explicit 1-port CPW network ---
#         single_port_ntw = cpw_resonator_network(
#             freq, d, Cc1, Cc2, Ctog1, Ctog2,
#             cpw_params=cpw_params, Z0=Z0,
#         )

#         data_points = self._build_sparse_data_points(
#             single_port_ntw,
#             n_dense=cfg.n_dense,
#             n_kappa=cfg.n_kappa,
#         )
#         w0_guess = resonance_from_res11(single_port_ntw) * 2 * np.pi

#         Ceff_guess, Leff_guess, w0_best, scan_results = self._fit_Ceff_Leff(
#             w0_guess=w0_guess,
#             Cc1=Cc1,
#             Cc2=Cc2,
#             Z0=Z0,
#             data_points=data_points,
#             freq=freq,
#         )
#         self.scan_results = scan_results
#         logger.debug("Stage 1: L_guess=%.4e H, C_guess=%.4e F", Leff_guess, Ceff_guess)

#         # --- Stage 2: 2-port windowed least_squares ---
#         kappa_s11 = fwhm_from_trace_db(data_ntw, m=0, n=0, kind="dip")
#         kappa_s22 = fwhm_from_trace_db(data_ntw, m=1, n=1, kind="dip")
#         f0_s11    = resonance(data_ntw, m=0, n=0)
#         f0_s22    = resonance(data_ntw, m=1, n=1)
#         f0_s21    = resonance_from_s_max(data_ntw, m=1, n=0)
#         f0_s12    = resonance_from_s_max(data_ntw, m=0, n=1)

#         residuals_fn = self._make_windowed_residuals(
#             freq=freq,
#             data_ntw=data_ntw,
#             Cc1=Cc1,
#             Cc2=Cc2,
#             f0_s11=f0_s11,
#             kappa_s11=kappa_s11,
#             f0_s22=f0_s22,
#             kappa_s22=kappa_s22,
#             Z0=Z0,
#             # f0_s12 = f0_s12,
#             # f0_s21 = f0_s21,
#         )

#         res = least_squares(
#             residuals_fn,
#             x0=(Leff_guess, Ceff_guess),
#             bounds=(
#                 [Leff_guess - 2e-10, Ceff_guess - 2e-13],
#                 [Leff_guess + 2e-10, Ceff_guess + 2e-13],
#             ),
#             method="trf",
#             jac="3-point",
#             diff_step=1e-4,
#             x_scale="jac",
#             xtol=1e-15,
#             ftol=1e-15,
#             gtol=1e-15,
#             max_nfev=cfg.max_nfev,
#             verbose=2 if cfg.verbose else 0,
#         )

#         self.L = float(res.x[0])
#         self.C = float(res.x[1])
#         logger.debug("Stage 2 result: L=%.4e H, C=%.4e F", self.L, self.C)

#     # ------------------------------------------------------------------
#     # get_network()
#     # ------------------------------------------------------------------

#     def get_network(
#         self,
#         freq: rf.Frequency,
#         Cc1: float,
#         Cc2: float,
#         Z0: float = 50.0,
#         with_grounds: bool = False,
#         Ctog1: float = None,
#         Ctog2: float = None,
#         **kwargs,
#     ) -> rf.Network:
#         """
#         Build the 2-port LC LOM network from the optimised L and C.

#         Parameters
#         ----------
#         freq : rf.Frequency
#         Cc1, Cc2 : float
#             Coupling capacitances in Farads.
#         Z0 : float
#             Reference impedance in Ohms.
#         with_grounds : bool
#             If True, include Ctog1/Ctog2 shunt caps (requires them to
#             be provided).
#         Ctog1, Ctog2 : float or None
#             Required when with_grounds=True.

#         Returns
#         -------
#         rf.Network
#         """
#         self._require_fitted()

#         if with_grounds:
#             if Ctog1 is None or Ctog2 is None:
#                 raise ValueError("Ctog1 and Ctog2 must be provided when with_grounds=True.")
#             return lc_resonator_network_with_grounds_2port(
#                 Leff=self.L, Ceff=self.C,
#                 Cc1=Cc1, Cc2=Cc2,
#                 Ctog1=Ctog1, Ctog2=Ctog2,
#                 freq=freq, Z0=Z0,
#             )

#         return lc_resonator_network_2port(
#             Leff=self.L, Ceff=self.C,
#             Cc1=Cc1, Cc2=Cc2,
#             freq=freq, Z0=Z0,
#         )

#     # ------------------------------------------------------------------
#     # get_params()
#     # ------------------------------------------------------------------

#     def get_params(self) -> dict:
#         """
#         Return fitted parameters and scan diagnostics.

#         Returns
#         -------
#         dict
#             Keys: "L" (H), "C" (F), "scan_results" (ndarray or None).
#         """
#         self._require_fitted()
#         return {
#             "L":            self.L,
#             "C":            self.C,
#             "scan_results": self.scan_results,
#         }

#     def __repr__(self) -> str:
#         if self.is_fitted:
#             return "OptimizedFit(L={:.4e} H, C={:.4e} F)".format(self.L, self.C)
#         return "OptimizedFit(unfitted)"

from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

"""
models/optimized_fit.py
-----------------------
OptimizedFit: numerical optimization over configurable S-parameter residuals
to find the best-fit effective L, C (and optionally phase) for a CPW resonator.

Algorithm summary
-----------------
Stage 1 — coarse scan (fit_Ceff_Leff):
    Scan ω0 in a window around the measured resonance.  For each ω0,
    use Nelder-Mead to optimise Leff with Ceff forced by the resonance
    constraint ω0² * L * (C + Cc1 + Cc2) = 1.  Keep the best (L, C).

Stage 2 — windowed refinement (least_squares):
    Build separate frequency windows around the S11 / S22 / S21 / S12
    resonances (controlled by OptimizationConfig.params).  Minimise the
    complex residuals of the 2-port LC network against the CPW reference
    network.  Phase is optionally included as a free parameter.

Key new knobs in OptimizationConfig
------------------------------------
fit_phase : bool
    If True, a global complex phase e^{i·φ} is fitted together with L and C.
    The fitted phase is stored in self.phase after fit().
s_params : sequence of str  (subset of {"S11","S22","S21","S12"})
    Which S-parameter channels to include in the Stage 2 residual.
    Default: ("S11", "S22").
weights : dict[str, float] or None
    Per-channel weight (before depth normalisation).  Pass None to weight
    all included channels equally (weight = 1.0).
    Example: {"S11": 2.0, "S21": 1.0} — S11 counts twice as much as S21.
"""

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import skrf as rf
from scipy.optimize import minimize, least_squares

from simpleLOMs.models.base import BaseFit
from simpleLOMs.analysis import (
    resonance_from_res11,
    fwhm_from_res11,
    fwhm_from_trace_db,
    resonance,
    # resonance_from_s_max intentionally not imported: see fit() comments.
)
from simpleLOMs.networks.lc import (
    lc_resonator_network,
    lc_resonator_network_2port,
    lc_resonator_network_2port_shifted,
    lc_resonator_network_with_grounds_2port,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# S-parameter channel registry
# ---------------------------------------------------------------------------

# Maps channel name → (data matrix row, col)  in skrf convention
_CHANNEL_IDX: dict[str, tuple[int, int]] = {
    "S11": (0, 0),
    "S22": (1, 1),
    "S21": (1, 0),
    "S12": (0, 1),
}

# Which "resonance" function to use for each channel
_REFLECTION_CHANNELS = {"S11", "S22"}
_TRANSMISSION_CHANNELS = {"S21", "S12"}


# ---------------------------------------------------------------------------
# Helper dataclass for optimizer hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """
    Tuning knobs for the two-stage OptimizedFit algorithm.

    Core algorithm knobs
    --------------------
    w0_window_frac : float
        Half-width of the ω0 scan window as a fraction of ω0_guess.
    n_w0 : int
        Number of ω0 grid points in the coarse scan.
    n_dense : int
        Number of data points sampled around resonance in Stage 1.
    n_kappa : float
        Half-width of the dense sampling window in units of κ (linewidth).
    n_widths : float
        Half-width of each residual window in Stage 2, in units of κ.
    max_nfev : int
        Maximum function evaluations for the Stage 2 least_squares call.
    verbose : bool
        If True, least_squares prints iteration progress.

    Residual configuration
    ----------------------
    fit_phase : bool
        If True, a global scalar phase φ is added to the free parameters so
        that the model prediction is e^{i·φ} · S_model.  The fitted phase is
        stored in OptimizedFit.phase (radians) after fit().
    s_params : sequence of str
        Which S-parameter channels to include in Stage 2.  Any non-empty
        subset of {"S11", "S22", "S21", "S12"}.  Default: ("S11", "S22").
    weights : dict[str, float] or None
        Manual per-channel weight multipliers applied *after* depth
        normalisation.  None → equal weights (all 1.0).
        Example: {"S11": 2.0, "S22": 2.0, "S21": 1.0}
    """
    # --- algorithm ---
    w0_window_frac: float = 0.00001
    n_w0:           int   = 20
    n_dense:        int   = 100
    n_kappa:        float = 0.75
    n_widths:       float = 1.0
    max_nfev:       int   = 100
    verbose:        bool  = False

    # --- residual options ---
    fit_phase: bool                          = False
    s_params:  Sequence[str]                 = field(default_factory=lambda: ["S11", "S22"])
    weights:   dict[str, float] | None       = None

    def __post_init__(self):
        # Normalise s_params to upper-case and validate
        self.s_params = [p.upper() for p in self.s_params]
        unknown = set(self.s_params) - set(_CHANNEL_IDX)
        if unknown:
            raise ValueError(f"Unknown s_params channels: {unknown}. "
                             f"Valid choices: {set(_CHANNEL_IDX)}")
        if not self.s_params:
            raise ValueError("s_params must contain at least one channel.")

    def channel_weight(self, name: str) -> float:
        """Return the manual weight for a channel (default 1.0)."""
        if self.weights is None:
            return 1.0
        return float(self.weights.get(name, 1.0))


# ---------------------------------------------------------------------------
# OptimizedFit class
# ---------------------------------------------------------------------------

class OptimizedFit(BaseFit):
    """
    Lumped LC model via two-stage numerical optimisation.

    Parameters
    ----------
    config : OptimizationConfig, optional
        Hyperparameters and residual options.  Defaults to
        OptimizationConfig() (S11 + S22, no phase fitting).

    Attributes
    ----------
    L : float or None
        Fitted effective inductance in Henries.
    C : float or None
        Fitted effective capacitance in Farads.
    phase : float or None
        Fitted global phase in radians.  None if config.fit_phase is False
        or before fit() is called.
    scan_results : np.ndarray or None
        Diagnostic array from the Stage 1 ω0 scan, shape (n_w0, 4):
        columns are [ω0, Ceff, Leff, sse].

    Examples
    --------
    ::

        from simpleLOMs.models.optimized_fit import OptimizedFit, OptimizationConfig

        # Fit using only S21, with phase as a free parameter
        cfg   = OptimizationConfig(s_params=["S21"], fit_phase=True)
        model = OptimizedFit(config=cfg)
        model.fit(freq, data_ntw=cpw_net, Cc1=3e-14, Cc2=7e-14,
                  Ctog1=4e-14, Ctog2=6e-14, d=7e-3,
                  cpw_params=cpw_params, Z0=50)
        print(model)           # OptimizedFit(L=…, C=…, phase=…)
        net = model.get_network(freq, Cc1=3e-14, Cc2=7e-14, Z0=50)
    """

    def __init__(self, config: OptimizationConfig = None):
        self.config:       OptimizationConfig   = config if config is not None else OptimizationConfig()
        self.L:            float | None         = None
        self.C:            float | None         = None
        self.phase:        float | None         = None
        self.scan_results: np.ndarray | None    = None
        self._data_ntw:    rf.Network | None    = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _align_network(
        self,
        data_ntw: rf.Network,
        freq: rf.Frequency,
    ) -> rf.Network:
        """Interpolate data_ntw onto freq if the grids differ."""
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

    def _build_sparse_data_points(
        self,
        data_ntw: rf.Network,
        n_dense: int,
        n_kappa: float,
    ) -> list[tuple[float, float, float]]:
        """
        Build a sparse, physically-motivated set of (ω, Re(S11), Im(S11)) tuples
        for Stage 1.
        """
        f_hz   = data_ntw.frequency.f
        w      = 2 * np.pi * f_hz
        S11    = data_ntw.s[:, 0, 0]
        re_s11 = np.real(S11)
        im_s11 = np.imag(S11)

        idx_re_min  = np.argmin(re_s11)
        data_points = [(w[idx_re_min], re_s11[idx_re_min], im_s11[idx_re_min])]

        if n_dense > 0:
            f0    = f_hz[idx_re_min]
            kappa = fwhm_from_res11(data_ntw)
            f_dense = np.linspace(
                f0 - n_kappa * kappa,
                f0 + (n_kappa / 2) * kappa,
                n_dense + 2,
            )[1:-1]
            re_dense = np.interp(f_dense, f_hz, re_s11)
            im_dense = np.interp(f_dense, f_hz, im_s11)
            for f_pt, r, i in zip(f_dense, re_dense, im_dense):
                data_points.append((2 * np.pi * f_pt, r, i))

        return data_points

    def _fit_Ceff_Leff(
        self,
        w0_guess: float,
        Cc1: float,
        Cc2: float,
        Z0: float,
        data_points: list,
        freq: rf.Frequency,
    ) -> tuple[float, float, float, np.ndarray]:
        """
        Stage 1: scan ω0 and fit Leff by Nelder-Mead with Ceff constrained.

        Returns (Ceff_best, Leff_best, w0_best, scan_results).
        """
        cfg = self.config
        dp  = np.asarray(data_points, dtype=float)
        freqs  = dp[:, 0]
        target = dp[:, 1] + 1j * dp[:, 2]

        def ZCc1(w):        return 1.0 / (1j * w * Cc1)
        def ZCc2(w):        return 1.0 / (1j * w * Cc2)
        def ZLC(w, Le, Ce): return 1.0 / (1j * w / (Le * (1j*w)**2 + 1/(Le*Ce)) if False
                                           else (1/(1j*w*Le) + 1j*w*Ce)**-1)
        # Simpler inline version used below
        def S11_fn(w, Le, Ce):
            ZL  = 1j * w * Le
            ZC  = 1.0 / (1j * w * Ce)
            ZLC = 1.0 / (1.0/ZL + 1.0/ZC)
            Zsh = 1.0 / (1.0/ZLC + 1j * w * Cc2)
            Zin = 1.0/(1j * w * Cc1) + Zsh
            return (Zin - Z0) / (Zin + Z0)

        w0_min  = w0_guess * (1.0 - cfg.w0_window_frac)
        w0_max  = w0_guess * (1.0 + cfg.w0_window_frac)
        w0_grid = np.linspace(w0_min, w0_max, int(cfg.n_w0))

        best    = None
        results = []

        for w0 in w0_grid:
            def ceff_of_L(Le):
                return 1.0 / (Le * (w0 ** 2)) - (Cc1 + Cc2)

            Le_max   = 1.0 / ((Cc1 + Cc2) * (w0 ** 2))
            Le_upper = 0.999 * Le_max
            x0_local = 0.5 * Le_upper

            def obj(Le_arr):
                Le = float(Le_arr[0])
                if not (0.0 < Le < Le_upper):
                    dist = (0.0 - Le) if Le <= 0.0 else (Le - Le_upper)
                    return 1e6 * (1.0 + dist ** 2)
                Ce = ceff_of_L(Le)
                if Ce <= 0.0 or not np.isfinite(Ce):
                    return 1e6
                pred  = np.array([S11_fn(w, Le, Ce) for w in freqs])
                resid = pred - target
                return float(np.sum(resid.real ** 2 + resid.imag ** 2))

            fit = minimize(
                obj,
                x0=np.array([x0_local]),
                method="Nelder-Mead",
                options=dict(maxiter=3000, xatol=1e-30, fatol=1e-30, disp=False),
            )

            Le_hat = float(np.clip(fit.x[0], np.nextafter(0.0, 1.0), Le_upper))
            Ce_hat = ceff_of_L(Le_hat)
            sse    = float(fit.fun)
            results.append((w0, Ce_hat, Le_hat, sse))

            if best is None or sse < best[-1]:
                best = (w0, Ce_hat, Le_hat, sse)

        w0_best, Ceff_best, Leff_best, _ = best
        return Ceff_best, Leff_best, w0_best, np.array(results)

    # ------------------------------------------------------------------
    # _make_windowed_residuals  (unified, replaces the four old variants)
    # ------------------------------------------------------------------

    def _make_windowed_residuals(
        self,
        freq: rf.Frequency,
        data_ntw: rf.Network,
        Cc1: float,
        Cc2: float,
        f0_s11: float,
        kappa_s11: float,
        f0_s22: float,
        kappa_s22: float,
        f0_s21: float,
        f0_s12: float,
        Z0: float,
        eps: float = 1e-18,
    ):
        """
        Stage 2: build the windowed residual function for least_squares.

        The returned callable signature depends on config.fit_phase:
          - fit_phase=False  →  residuals(x)  where x = [Leff, Ceff]
          - fit_phase=True   →  residuals(x)  where x = [Leff, Ceff, phi]

        Channels included and their weights are taken from config.s_params
        and config.weights.
        """
        cfg            = self.config
        data_aligned   = self._align_network(data_ntw, freq)
        f              = freq.f
        n              = cfg.n_widths

        # --- build per-channel masks ---
        # Each channel is windowed around its own resonance frequency.
        # For S21/S12 we use f0_s21/f0_s12 and the S11 linewidth as a
        # proxy for kappa (transmission FWHM == reflection FWHM for a
        # symmetric Lorentzian, and we have no independent kappa_tx yet).
        # Using f0_s11 as the S21 anchor was wrong: it can miss the peak
        # entirely when the two ports couple differently.
        # Transmission channels use a 2× wider window than n_widths because
        # the S21/S12 Lorentzian tail is much broader in dB and we need
        # enough off-peak points for the optimizer to see the correct shape.
        # Reflection channels keep the tighter n_widths window.
        n_tx = max(n * 2.0, 3.0)
        masks = {
            "S11": (f > f0_s11 - n    * kappa_s11) & (f < f0_s11 + n    * kappa_s11),
            "S22": (f > f0_s22 - n    * kappa_s22) & (f < f0_s22 + n    * kappa_s22),
            "S21": (f > f0_s21 - n * kappa_s11) & (f < f0_s21 + n * kappa_s11),
            "S12": (f > f0_s12 - n * kappa_s11) & (f < f0_s12 + n * kappa_s11),
        }

        # Validate masks for requested channels
        for ch in cfg.s_params:
            if not np.any(masks[ch]):
                raise ValueError(
                    f"{ch} mask is empty — check the resonance frequency / "
                    f"linewidth / n_widths settings."
                )

        # --- extract windowed data and build depth scales ---
        ch_freqs = {}   # channel → rf.Frequency
        ch_data  = {}   # channel → complex S array
        ch_scale = {}   # channel → depth normalisation scalar

        def _depth(s_complex):
            m = np.abs(s_complex)
            return max(float(np.max(m) - np.min(m)), eps)

        # Compute per-group depth floors separately for reflection and
        # transmission channels.  Mixing them causes a fatal scale mismatch:
        # S21/S12 have ~0.4–0.5 linear depth while S11/S22 have ~0.01–0.1,
        # so a shared floor either drowns the transmission residual (if
        # reflection dominates) or inflates it by 10× (the reverse).
        refl_depths = [_depth(data_aligned.s[masks[ch], _CHANNEL_IDX[ch][0], _CHANNEL_IDX[ch][1]])
                       for ch in cfg.s_params if ch in _REFLECTION_CHANNELS]
        tx_depths   = [_depth(data_aligned.s[masks[ch], _CHANNEL_IDX[ch][0], _CHANNEL_IDX[ch][1]])
                       for ch in cfg.s_params if ch in _TRANSMISSION_CHANNELS]

        refl_floor = max(refl_depths) if refl_depths else eps
        tx_floor   = max(tx_depths)   if tx_depths   else eps

        for ch in cfg.s_params:
            mask = masks[ch]
            row, col = _CHANNEL_IDX[ch]
            ch_freqs[ch] = rf.Frequency.from_f(f[mask], unit="Hz")
            ch_data[ch]  = data_aligned.s[mask, row, col]

        for ch in cfg.s_params:
            floor        = refl_floor if ch in _REFLECTION_CHANNELS else tx_floor
            raw_scale    = max(_depth(ch_data[ch]), floor)
            ch_scale[ch] = raw_scale / cfg.channel_weight(ch)
            logger.debug("%s depth scale=%.4e, N=%d, weight=%.2f",
                         ch, ch_scale[ch], masks[ch].sum(), cfg.channel_weight(ch))

        # --- closure over aligned data ---
        def _lom_s(Le, Ce, ch, phase_factor):
            """Compute model S-parameter for channel ch, applying phase."""
            row, col = _CHANNEL_IDX[ch]
            lom = lc_resonator_network_2port(Le, Ce, Cc1, Cc2, ch_freqs[ch], Z0)
            return lom.s[:, row, col] * phase_factor

        if cfg.fit_phase:
            def residuals(x):
                Le  = float(x[0])
                Ce  = float(x[1])
                phi = float(x[2])
                pf  = np.exp(1j * phi)
                parts = []
                for ch in cfg.s_params:
                    pred = _lom_s(Le, Ce, ch, pf)
                    r    = pred - ch_data[ch]
                    parts.append(np.concatenate([r.real, r.imag]) / ch_scale[ch])
                return np.concatenate(parts)
        else:
            def residuals(x):
                Le = float(x[0])
                Ce = float(x[1])
                parts = []
                for ch in cfg.s_params:
                    pred = _lom_s(Le, Ce, ch, 1.0)
                    r    = pred - ch_data[ch]
                    if ch in _TRANSMISSION_CHANNELS:
                        parts.append((np.abs(pred) - np.abs(ch_data[ch])) / ch_scale[ch])
                    else:
                        parts.append(np.concatenate([r.real, r.imag]) / ch_scale[ch])
                # for ch in cfg.s_params:
                #     pred = _lom_s(Le, Ce, ch, 1.0)
                #     r    = pred - ch_data[ch]
                #     parts.append(np.concatenate([r.real, r.imag]) / ch_scale[ch])
                return np.concatenate(parts)

        return residuals

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(
        self,
        freq: rf.Frequency,
        data_ntw: rf.Network,
        Cc1: float,
        Cc2: float,
        Ctog1: float,
        Ctog2: float,
        d: float,
        cpw_params,
        Z0: float = 50.0,
        **kwargs,
    ) -> None:
        """
        Two-stage optimisation to fit Leff, Ceff (and optionally phase) to
        data_ntw.

        Stage 1: coarse Nelder-Mead scan over ω0, constrained Ceff.
            Uses an explicitly constructed 1-port CPW network so that the
            Stage 1 data points come from a true single-port reflection
            measurement rather than a 2-port S11 view.
        Stage 2: windowed least_squares refinement on the channels specified
            in config.s_params.

        Parameters
        ----------
        freq : rf.Frequency
            Frequency grid for the optimisation (should be centred on the
            resonance with enough bandwidth to capture the linewidth).
        data_ntw : rf.Network
            2-port reference network (CPW model or measured data).
        Cc1, Cc2 : float
            Coupling capacitances in Farads.
        Ctog1, Ctog2 : float
            Shunt-to-ground capacitances in Farads.
        d : float
            Resonator length in metres.
        cpw_params : CPWParams
            CPW geometry parameters.
        Z0 : float
            Reference impedance in Ohms.
        """
        from simpleLOMs.networks.cpw import cpw_resonator_network

        cfg = self.config

        # --- Stage 1: explicit 1-port CPW network ---
        single_port_ntw = cpw_resonator_network(
            freq, d, Cc1, Cc2, Ctog1, Ctog2,
            cpw_params=cpw_params, Z0=Z0,
        )

        data_points = self._build_sparse_data_points(
            single_port_ntw,
            n_dense=cfg.n_dense,
            n_kappa=cfg.n_kappa,
        )
        w0_guess = resonance_from_res11(single_port_ntw) * 2 * np.pi

        Ceff_guess, Leff_guess, w0_best, scan_results = self._fit_Ceff_Leff(
            w0_guess=w0_guess,
            Cc1=Cc1,
            Cc2=Cc2,
            Z0=Z0,
            data_points=data_points,
            freq=freq,
        )
        self.scan_results = scan_results
        self._data_ntw = data_ntw
        logger.debug("Stage 1: L_guess=%.4e H, C_guess=%.4e F", Leff_guess, Ceff_guess)

        # --- Stage 2: windowed least_squares ---
        kappa_s11 = fwhm_from_trace_db(data_ntw, m=0, n=0, kind="dip")
        kappa_s22 = fwhm_from_trace_db(data_ntw, m=1, n=1, kind="dip")
        f0_s11    = resonance(data_ntw, m=0, n=0)
        f0_s22    = resonance(data_ntw, m=1, n=1)
        # For this LC topology the S21/S12 transmission peak is always
        # co-located with the S11 reflection dip.  resonance_from_s_max
        # searches the full band and can return the band edge in weakly
        # coupled resonators (shallow peak), placing the window entirely
        # outside freq.f and raising an empty-mask ValueError.  Using
        # f0_s11 as the anchor is both correct for this topology and robust.
        f0_s21 = f0_s11
        f0_s12 = f0_s11

        residuals_fn = self._make_windowed_residuals(
            freq=freq,
            data_ntw=data_ntw,
            Cc1=Cc1,
            Cc2=Cc2,
            f0_s11=f0_s11,
            kappa_s11=kappa_s11,
            f0_s22=f0_s22,
            kappa_s22=kappa_s22,
            f0_s21=f0_s21,
            f0_s12=f0_s12,
            Z0=Z0,
        )

        # --- initial parameter vector and bounds ---
        # Bounds are set as ±50% of the Stage-1 guess rather than fixed
        # absolute offsets.  Fixed offsets (e.g. ±2e-10 H) can be too
        # tight when transmission-only fitting wants to explore a different
        # region of (L, C) space, causing the optimizer to hit the wall
        # immediately and return the Stage-1 guess unchanged.
        L_lo, L_hi = Leff_guess * 0.5, Leff_guess * 1.5
        C_lo, C_hi = Ceff_guess * 0.5, Ceff_guess * 1.5
        x0     = [Leff_guess, Ceff_guess]
        lo     = [L_lo, C_lo]
        hi     = [L_hi, C_hi]

        if cfg.fit_phase:
            x0.append(0.0)      # phi = 0 rad initial guess
            lo.append(-np.pi)
            hi.append( np.pi)

        res = least_squares(
            residuals_fn,
            x0=x0,
            bounds=(lo, hi),
            method="trf",
            jac="3-point",
            diff_step=1e-4,
            x_scale="jac",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=cfg.max_nfev,
            verbose=2 if cfg.verbose else 0,
        )

        self.L = float(res.x[0])
        self.C = float(res.x[1])
        if cfg.fit_phase:
            self.phase = float(res.x[2])
        else:
            self.phase = None

        logger.debug("Stage 2 result: L=%.4e H, C=%.4e F, phase=%s",
                     self.L, self.C,
                     f"{self.phase:.4f} rad" if self.phase is not None else "N/A")

    # ------------------------------------------------------------------
    # get_network()
    # ------------------------------------------------------------------

    def get_network(
        self,
        freq: rf.Frequency,
        Cc1: float,
        Cc2: float,
        Z0: float = 50.0,
        with_grounds: bool = False,
        Ctog1: float = None,
        Ctog2: float = None,
        shifted: bool = False,
        show: bool = False,
        reference: rf.Network = None,
        m: int = 0,
        n: int = 0,
        lom_label: str = "OptimizedFit",
        data_label: str = "CPW",
        save_path: str = None,
        **kwargs,
    ) -> rf.Network:
        """
        Build the 2-port LC LOM network from the optimised L and C.

        Note: the optional fitted phase is *not* applied here because it is a
        calibration correction for the *data*, not a physical property of the
        LC network.  Apply it separately if you need to compare against
        phase-corrected data::

            net_s = net.s * np.exp(1j * model.phase)

        Parameters
        ----------
        freq : rf.Frequency
        Cc1, Cc2 : float
            Coupling capacitances in Farads.
        Z0 : float
            Reference impedance in Ohms.
        with_grounds : bool
            If True, include Ctog1/Ctog2 shunt caps.
        Ctog1, Ctog2 : float or None
            Required when with_grounds=True.
        show : bool
            If True, automatically plot the built network (default False).
        reference : rf.Network, optional
            Overlay reference.  Defaults to the ``data_ntw`` stored during fit().
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
        elif shifted:
            net = lc_resonator_network_2port_shifted(
                Leff=self.L, Ceff=self.C,
                Cc1=Cc1, Cc2=Cc2,
                freq=freq, Z0=Z0,
            )
        else:
            net = lc_resonator_network_2port(
                Leff=self.L, Ceff=self.C,
                Cc1=Cc1, Cc2=Cc2,
                freq=freq, Z0=Z0,
            )

        ref = reference if reference is not None else self._data_ntw
        self._maybe_plot(
            net,
            show=show,
            reference=ref,
            m=m, n=n,
            lom_label=lom_label,
            data_label=data_label,
            save_path=save_path,
        )
        return net

    def plot_scan(self, show: bool = True, save_path: str = None):
        """
        Plot Stage-1 ω₀ scan results (SSE vs resonant frequency).

        Parameters
        ----------
        show : bool
            If True (default), call ``plt.show()``.
        save_path : str, optional

        Returns
        -------
        fig, ax
        """
        from simpleLOMs import plotting
        return plotting.plot_scan(self, show=show, save_path=save_path)

    # ------------------------------------------------------------------
    # get_params()
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """
        Return fitted parameters and scan diagnostics.

        Returns
        -------
        dict
            Keys: "L" (H), "C" (F), "phase" (rad or None),
                  "scan_results" (ndarray or None).
        """
        self._require_fitted()
        return {
            "L":            self.L,
            "C":            self.C,
            "phase":        self.phase,
            "scan_results": self.scan_results,
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            phase_str = (f", phase={self.phase:.4f} rad"
                         if self.phase is not None else "")
            return (f"OptimizedFit(L={self.L:.4e} H, C={self.C:.4e} F"
                    + phase_str + ")")
        return "OptimizedFit(unfitted)"
