#!/usr/bin/env python3
"""Temporary perf script — not part of package. Delete after investigation.

2026-07-18: the fit_lom fix is applied (TUTORIAL_HANDOFF.md §7). Leg (a) now
measures the fixed function; legs (b)/(c)/(extra) hand-build what the OLD
implementation used to do internally and are kept only for comparison.
"""
from __future__ import annotations

import math
import sys
import time

import skrf as rf

from simpleLOMs import CPWParams, OptimizedFit, OptimizationConfig, extract_f0_kappa, fit_lom
from simpleLOMs.networks.cpw import cpw_resonator_network_2port

D = 7e-3
CC = 6e-15
CTOG = 1e-14
Z0 = 50.0
CPW = CPWParams(ep_r=11.45)


def p(msg: str = "") -> None:
    print(msg, flush=True)


def fr_from_lc(L: float, C: float) -> float:
    return 1.0 / (2.0 * math.pi * (L * C) ** 0.5)


def timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    p(f"{label}: {dt:.3f} s")
    return out, dt


def main() -> None:
    p("=== fit_lom optimized vs direct OptimizedFit ===")
    p(f"d={D}, Cc={CC}, Ctog={CTOG}, CPWParams(ep_r={CPW.ep_r})")
    p()

    (L_a, C_a), t_a = timed(
        "(a) fit_lom(..., model='optimized')",
        lambda: fit_lom(D, model="optimized", Cc=CC, Ctog=CTOG, cpw_params=CPW),
    )

    (f0_b, kappa_b), t_b = timed(
        "(b) extract_f0_kappa(...)",
        lambda: extract_f0_kappa(CPW, d=D, Cc=CC, Ctog=CTOG, Z0=Z0),
    )

    net_c = None

    def build_window_net():
        nonlocal net_c
        fr = rf.Frequency(f0_b - 0.5e9, f0_b + 0.5e9, 200_001, unit="Hz")
        net_c = cpw_resonator_network_2port(
            fr, D, CC, CC, CTOG, CTOG, cpw_params=CPW, Z0=Z0
        )
        return fr, net_c

    (_, _), t_c = timed("(c) 200k-pt fr + cpw_resonator_network_2port", build_window_net)
    assert net_c is not None
    fr_c = net_c.frequency

    m_200k = OptimizedFit(config=OptimizationConfig(verbose=False, n_widths=4))

    def fit_200k():
        m_200k.fit(
            fr_c,
            data_ntw=net_c,
            Cc1=CC,
            Cc2=CC,
            Ctog1=CTOG,
            Ctog2=CTOG,
            d=D,
            cpw_params=CPW,
            Z0=Z0,
        )
        return m_200k.L, m_200k.C

    (L_200k, C_200k), t_fit200k = timed(
        "(extra) OptimizedFit on 200k grid only",
        fit_200k,
    )

    m_d = OptimizedFit(config=OptimizationConfig(verbose=False, n_widths=4))

    def direct_optimized():
        fr_d = rf.Frequency(4e9, 12e9, 8001, unit="Hz")
        net_d = cpw_resonator_network_2port(
            fr_d, D, CC, CC, CTOG, CTOG, cpw_params=CPW, Z0=Z0
        )
        m_d.fit(
            fr_d,
            data_ntw=net_d,
            Cc1=CC,
            Cc2=CC,
            Ctog1=CTOG,
            Ctog2=CTOG,
            d=D,
            cpw_params=CPW,
            Z0=Z0,
        )
        return m_d.L, m_d.C

    (L_d, C_d), t_d = timed(
        "(d) direct OptimizedFit (8001 pts, 4–12 GHz)",
        direct_optimized,
    )

    p()
    p("=== timing summary (seconds) ===")
    rows = [
        ("a", "fit_lom optimized E2E", t_a),
        ("b", "extract_f0_kappa", t_b),
        ("c", "200k fr + CPW net", t_c),
        ("extra", "OptimizedFit @ 200k", t_fit200k),
        ("d", "direct OptimizedFit 8k", t_d),
    ]
    for key, desc, t in rows:
        p(f"  {key:5s}  {t:8.3f}  {desc}")
    p(f"  b+c   {t_b + t_c:8.3f}  (prep only, no OptimizedFit)")
    p(f"  b+c+extra {t_b + t_c + t_fit200k:8.3f}  (should ≈ a)")
    p(f"  sum b+c vs a: {t_b + t_c:.3f} vs {t_a:.3f}  (ratio (b+c)/a={(t_b + t_c) / t_a:.2f})")

    p()
    p("=== L, C, f_r comparison ===")
    fr_a = fr_from_lc(L_a, C_a)
    fr_200k = fr_from_lc(L_200k, C_200k)
    fr_d_val = fr_from_lc(L_d, C_d)
    p(f"  extract_f0_kappa f0     = {f0_b:.6e} Hz, kappa = {kappa_b:.6e} Hz")
    p(f"  fit_lom L,C             = {L_a:.6e}, {C_a:.6e}  -> f_r = {fr_a:.6e} Hz")
    p(f"  OptimizedFit @ 200k     = {L_200k:.6e}, {C_200k:.6e}  -> f_r = {fr_200k:.6e} Hz")
    p(f"  direct OptimizedFit 8k  = {L_d:.6e}, {C_d:.6e}  -> f_r = {fr_d_val:.6e} Hz")
    p(f"  |f_r(8k)-f0|/f0        = {abs(fr_d_val - f0_b) / f0_b:.3e}")
    p(f"  |f_r(fit_lom)-f0|/f0   = {abs(fr_a - f0_b) / f0_b:.3e}")


if __name__ == "__main__":
    main()
