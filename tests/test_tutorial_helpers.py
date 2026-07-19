"""Unit tests for tutorial plotting / design helpers."""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import skrf as rf

import simpleLOMs as sl


def test_invert_monotonic_linear():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 8.0, 6.0, 4.0])  # decreasing
    assert abs(sl.invert_monotonic(x, y, 7.0) - 2.5) < 1e-9


def test_invert_monotonic_log():
    x = np.array([1e-15, 2e-15, 4e-15, 8e-15])
    y = 1.0 / (x / 1e-15) ** 2
    got = sl.invert_monotonic(x, y, y[1], log=True)
    assert abs(got - x[1]) / x[1] < 1e-6


def test_measure_sweep():
    vals = np.array([1.0, 2.0, 3.0])
    out = sl.measure_sweep(vals, lambda v: {"sq": v ** 2})
    assert np.allclose(out["value"], vals)
    assert np.allclose(out["sq"], vals ** 2)


def test_notches_below_and_notch_frequency():
    freq = rf.Frequency(1e9, 5e9, 401, unit="Hz")
    # Synthetic notch around 3 GHz
    f = freq.f
    s21 = 1.0 - 0.9 * np.exp(-((f - 3e9) / 0.15e9) ** 2)
    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 1, 0] = s21
    s[:, 0, 1] = s21
    s[:, 0, 0] = 0.1
    s[:, 1, 1] = 0.1
    net = rf.Network(frequency=freq, s=s)
    f0, depth = sl.notch_frequency(net)
    assert abs(f0 - 3.0) < 0.05
    assert depth < 0.2
    notches = sl.notches_below(net, thresh=0.5)
    assert len(notches) >= 1
    assert abs(notches[0] - 3.0) < 0.05


def test_plot_curves_smoke():
    x = np.linspace(0, 1, 5)
    y = x ** 2
    fig, ax = sl.plot_curves(x, y, xlabel="x", ylabel="y", fmt="o-", show=False)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_design_resonator_smoke():
    cpw = sl.CPWParams(ep_r=11.45)
    freq = rf.Frequency(4e9, 12e9, 2001, unit="Hz")
    d, Cc, f0_GHz, rp = sl.design_resonator(
        7.5, 5000, cpw, freq=freq, model="foster", n=4,
        d_range=(6e-3, 8.5e-3), Cc_range=(3e-15, 12e-15),
    )
    assert 5e-3 < d < 10e-3
    assert 1e-15 < Cc < 20e-15
    assert abs(f0_GHz - 7.5) < 0.5
    assert rp.Q > 0


def test_format_comparison_table():
    text = sl.utils.format_comparison_table(
        [("a", "1.0"), ("bb", "2.5")],
        ["name", "val"],
    )
    assert "name" in text
    assert "bb" in text


def _lorentzian_2port(freq: rf.Frequency, f0: float, kappa: float) -> rf.Network:
    f = freq.f
    s21 = (kappa / 2) / ((kappa / 2) + 1j * (f - f0))
    s = np.zeros((len(f), 2, 2), dtype=complex)
    s[:, 1, 0] = s21
    s[:, 0, 1] = s21
    s[:, 0, 0] = 0.05
    s[:, 1, 1] = 0.05
    return rf.Network(frequency=freq, s=s)


def test_circle_fit_windowed_coarse_grid():
    # 401 points over 8 GHz (~20 MHz/pt): the default 51-point smoothing spans
    # ~1 GHz, wider than the 150 MHz line, so the plain fit inflates kappa.
    f0, kappa = 8.0e9, 150e6
    net = _lorentzian_2port(rf.Frequency(4e9, 12e9, 401, unit="Hz"), f0, kappa)
    _, k_plain = sl.circle_fit_f0_kappa(net, 1, 0)
    f0_w, k_w = sl.circle_fit_f0_kappa_windowed(net, 1, 0)
    assert k_plain > 2 * kappa
    assert abs(k_w - kappa) / kappa < 0.05
    assert abs(f0_w - f0) / f0 < 1e-3


def test_circle_fit_windowed_dense_grid_matches_truth():
    f0, kappa = 8.0e9, 150e6
    net = _lorentzian_2port(rf.Frequency(4e9, 12e9, 8001, unit="Hz"), f0, kappa)
    f0_w, k_w = sl.circle_fit_f0_kappa_windowed(net, 1, 0)
    assert abs(k_w - kappa) / kappa < 0.02
    assert abs(f0_w - f0) / f0 < 1e-4


def test_circle_fit_unbracketed_linewidth_warns_and_returns_nan():
    # 780 MHz line probed through a +/-250 MHz window: the half-power points
    # (f0 +/- 390 MHz) lie outside the grid. The old code silently returned
    # 2 grid spacings (~0.2 MHz, a ~4000x understatement); it must now warn
    # and return NaN so retrying callers can widen the window.
    import warnings

    f0, kappa = 6.4e9, 780e6
    net = _lorentzian_2port(rf.Frequency(6.15e9, 6.65e9, 8001, unit="Hz"), f0, kappa)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f0_p, k_p = sl.circle_fit_f0_kappa(net, 1, 0)
        f0_w, k_w = sl.circle_fit_f0_kappa_windowed(net, 1, 0)
    assert np.isnan(k_p)
    assert np.isnan(k_w)
    assert any(issubclass(x.category, RuntimeWarning)
               and "not bracketed" in str(x.message) for x in w)
    assert abs(f0_p - f0) < 10e6  # f0 itself is still located


def test_circle_fit_subgrid_narrow_line_stays_finite():
    # A line far below grid resolution is *bracketed* (single-point spike):
    # it must return a finite grid-scale width without warning, not NaN.
    import warnings

    freq = rf.Frequency(4e9, 12e9, 4001, unit="Hz")  # 2 MHz/pt >> 0.3 MHz line
    net = _lorentzian_2port(freq, 8.0e9, 0.3e6)
    df = freq.f[1] - freq.f[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _, k = sl.circle_fit_f0_kappa(net, 1, 0, smooth_window=None)
    assert np.isfinite(k) and 0 < k <= 4 * df
    assert not any(issubclass(x.category, RuntimeWarning) for x in w)


def test_extract_f0_kappa_ultra_overcoupled_corner():
    # Tutorial-02 grid corner Cc1 = Cc2 = 150 fF: kappa ~ 0.8 GHz is wider
    # than the default 0.4 GHz refine half-span, which used to truncate the
    # dip and silently return the 2-grid-spacing fallback (0.2 MHz at
    # n_fine=8001). The window is now sized from the coarse linewidth; the
    # result must be at physical scale and independent of the grid density.
    cpw = sl.CPWParams(ep_r=11.45)
    f0_a, k_a = sl.extract_f0_kappa(
        cpw, d=7e-3, Cc1=150e-15, Cc2=150e-15, Ctog=1e-14, Z0=50, n_fine=8001)
    f0_b, k_b = sl.extract_f0_kappa(
        cpw, d=7e-3, Cc1=150e-15, Cc2=150e-15, Ctog=1e-14, Z0=50, n_fine=4001)
    assert 0.7e9 < k_a < 0.9e9          # direct -3 dB width: ~792 MHz
    assert abs(k_a - k_b) / k_a < 0.01  # no grid-spacing artifact
    assert 6.5e9 < f0_a < 7.5e9
    assert abs(f0_a - f0_b) < 20e6


def test_extract_f0_kappa_asymmetric_port_reciprocity():
    # The 2-port network mirrors exactly under (Cc1, Ctog1) <-> (Cc2, Ctog2),
    # so swapped couplings must give the same (f0, kappa). Probing reflection
    # at the weak port used to inflate kappa by ~18 % at this corner.
    cpw = sl.CPWParams(ep_r=11.45)
    f0_s, k_s = sl.extract_f0_kappa(
        cpw, d=7e-3, Cc1=150e-15, Cc2=15e-15, Ctog=1e-14, Z0=50, n_fine=8001)
    f0_w, k_w = sl.extract_f0_kappa(
        cpw, d=7e-3, Cc1=15e-15, Cc2=150e-15, Ctog=1e-14, Z0=50, n_fine=8001)
    assert abs(k_s - k_w) / k_s < 0.005
    assert abs(f0_s - f0_w) / f0_s < 1e-4
    assert 0.4e9 < k_s < 0.6e9          # direct -3 dB width: ~483 MHz
