"""
analysis.py
-----------
Pure numerical analysis functions for extracting resonance frequencies
and linewidths from S-parameter networks.

None of these functions build networks or do optimization — they only
inspect existing rf.Network objects.  This makes them easy to test
independently and reuse across models.
"""
from __future__ import annotations
import logging
import warnings

import numpy as np
import skrf as rf
from scipy.signal import savgol_filter, find_peaks

logger = logging.getLogger(__name__)




def _fit_circle_kasa(z: np.ndarray) -> tuple[complex, float]:
    """
    Algebraic Kasa circle fit to complex points z = x + i y.

    Returns
    -------
    center : complex
        Fitted circle center.
    radius : float
        Fitted circle radius.
    """
    x = np.real(z)
    y = np.imag(z)

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)

    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = coeffs

    xc = -a / 2.0
    yc = -b_ / 2.0
    r = np.sqrt(max(xc**2 + yc**2 - c, 0.0))

    return xc + 1j * yc, r


def circle_fit_f0_kappa(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    smooth_window: int | None = 51,
    smooth_polyorder: int = 3,
) -> tuple[float, float]:
    """
    Extract resonance frequency (f0) and linewidth (kappa) from a circle fit
    to S[m,n] in the complex plane (reflection S11/S22).

    Returns
    -------
    f0 : float
        Resonance frequency in Hz.
    kappa : float
        Linewidth (FWHM of the angular velocity peak) in Hz. NaN (with a
        ``RuntimeWarning``) when the half-power points are not bracketed by
        the network's frequency window — i.e. the line is wider than the
        window, or the resonance sits on its edge.
    """
    f = ntwk.frequency.f
    s = ntwk.s[:, m, n]

    if len(f) < 5:
        raise ValueError("Need ≥5 frequency points for circle-fit.")

    center, radius = _fit_circle_kasa(s)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Circle fit failed: non-physical radius.")

    theta = np.unwrap(np.angle(s - center))

    sw = smooth_window
    if sw is not None:
        if sw >= len(theta):
            sw = len(theta) - 1 if len(theta) % 2 == 0 else len(theta)
        if sw < 5:
            sw = None
        elif sw % 2 == 0:
            sw += 1
    if sw is not None:
        theta = savgol_filter(theta, window_length=sw, polyorder=smooth_polyorder)

    dtheta_df = np.abs(np.gradient(theta, f))

    idx_peak = int(np.argmax(dtheta_df))
    # Sub-grid refinement of the peak location by parabolic interpolation through
    # the three points straddling the maximum, so f0 is not quantised to the
    # frequency grid (kappa below is already interpolated to sub-grid accuracy).
    if 0 < idx_peak < len(f) - 1:
        y0, y1, y2 = dtheta_df[idx_peak - 1], dtheta_df[idx_peak], dtheta_df[idx_peak + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))          # stay within the bracketing bin
        f0 = float(f[idx_peak] + delta * (f[idx_peak + 1] - f[idx_peak]))
    else:
        f0 = float(f[idx_peak])

    half_max = dtheta_df[idx_peak] / 2.0
    above = dtheta_df > half_max

    # A crossing index i marks the interval [f_i, f_{i+1}]. The falling edge of
    # a peak that is only one grid point wide sits at i == idx_peak, so the
    # right-side comparison must be >= — with a strict > that resolution-limited
    # (but bracketed) case would be misread as a failure.
    crossings = np.where(np.diff(above.astype(int)))[0]
    left_cross = crossings[crossings < idx_peak]
    right_cross = crossings[crossings >= idx_peak]

    if len(left_cross) == 0 or len(right_cross) == 0:
        # The peak is above half-max by construction, so a missing crossing
        # means |dtheta/df| stays above half-maximum all the way to that edge
        # of the grid (or the peak itself sits on the edge): the half-power
        # point lies outside the analysed band and the linewidth is wider than
        # the window can measure. Returning a number here would understate
        # kappa by the ratio of true linewidth to window width — a ~780 MHz
        # line probed through a 0.8 GHz window used to come back as 2 grid
        # spacings (0.2 MHz) — so report NaN and let callers widen the window.
        warnings.warn(
            "circle_fit_f0_kappa: half-power points of the resonance near "
            f"{f0 / 1e9:.3f} GHz are not bracketed by the analysed window "
            f"[{f[0] / 1e9:.3f}, {f[-1] / 1e9:.3f}] GHz; kappa is unresolved "
            "(returning NaN). Widen the frequency window.",
            RuntimeWarning,
            stacklevel=2,
        )
        kappa = float("nan")
    else:
        iL = left_cross[-1]
        iR = right_cross[0]

        def _interp(i: int) -> float:
            f0_, f1_ = f[i], f[i + 1]
            y0_, y1_ = dtheta_df[i], dtheta_df[i + 1]
            if y1_ == y0_:
                return 0.5 * (f0_ + f1_)
            return f0_ + (half_max - y0_) * (f1_ - f0_) / (y1_ - y0_)

        kappa = float(_interp(iR) - _interp(iL))

    return f0, kappa


def circle_fit_f0_kappa_windowed(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    span_factor: float = 1.5,
    min_points: int = 25,
    smooth_window: int | None = 51,
    smooth_polyorder: int = 3,
) -> tuple[float, float]:
    """
    Circle fit for networks on a fixed frequency grid (FEM exports, VNA
    traces, or models evaluated on such a grid for comparison).

    ``circle_fit_f0_kappa`` smooths the circle phase over a fixed number of
    grid *points*.  On the dense grids the package builds internally that
    span is far narrower than any linewidth, but on an externally supplied
    sweep it can exceed the physical kappa — the extracted linewidth then
    reflects the smoothing filter rather than the resonance.  This variant
    uses the smoothed full-band fit only to *locate* the resonance, then
    re-extracts f0 and kappa from an unsmoothed fit restricted to
    ``f0 ± span_factor * kappa_estimate``.

    Parameters
    ----------
    ntwk : rf.Network
        Network to analyse.
    m, n : int
        S-parameter indices, as in ``circle_fit_f0_kappa``.
    span_factor : float
        Half-width of the extraction window in units of the first-pass
        kappa estimate (which may be inflated; a generous window is
        harmless because the unsmoothed extraction is local).
    min_points : int
        Minimum number of grid points in the window; widened symmetrically
        around the resonance if the first pass yields fewer.
    smooth_window : int or None
        Smoothing window for the first-pass locating fit.
    smooth_polyorder : int
        Polynomial order for the first-pass locating fit.

    Returns
    -------
    f0 : float
        Resonance frequency in Hz.
    kappa : float
        Linewidth in Hz, free of the smoothing-filter bias. NaN when the
        linewidth cannot be bracketed inside the supplied grid (see
        ``circle_fit_f0_kappa``); callers that can re-simulate should retry
        with a wider window.
    """
    f = ntwk.frequency.f

    f0_loc, kappa_loc = circle_fit_f0_kappa(
        ntwk, m, n,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
    )

    if not (np.isfinite(kappa_loc) and kappa_loc > 0):
        # The locating pass could not bracket the half-power points inside
        # this grid (linewidth wider than the span, or resonance on the edge).
        # No sub-window of the same grid can recover that, and re-fitting a
        # min_points sliver of a broad, locally featureless arc risks
        # returning numerical-ripple width as kappa — so propagate the failure.
        return f0_loc, float("nan")

    sel = np.abs(f - f0_loc) <= span_factor * kappa_loc

    if int(sel.sum()) < min_points:
        idx = int(np.argmin(np.abs(f - f0_loc)))
        half = min_points // 2
        lo = max(idx - half, 0)
        hi = min(lo + min_points, len(f))
        lo = max(hi - min_points, 0)
        sel = np.zeros(len(f), dtype=bool)
        sel[lo:hi] = True

    return circle_fit_f0_kappa(ntwk[sel], m, n, smooth_window=None)


def circle_fit_modes(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    n_modes: int = 3,
    prominence: float = None,
    min_spacing_hz: float = None,
    smooth_window: int | None = 51,
    smooth_polyorder: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find multiple resonance modes in S[m,n] using the reflection circle fit.

    Returns
    -------
    f0s : np.ndarray
        Mode frequencies in Hz, length n_modes (NaN-padded).
    kappas : np.ndarray
        Mode linewidths in Hz, length n_modes (NaN-padded).
    """
    f = ntwk.frequency.f
    s = ntwk.s[:, m, n]
    df = f[1] - f[0]

    if min_spacing_hz is None:
        min_spacing_hz = (f[-1] - f[0]) / (10 * n_modes)
    min_distance_pts = max(1, int(min_spacing_hz / df))

    center, _ = _fit_circle_kasa(s)
    theta = np.unwrap(np.angle(s - center))

    sw = smooth_window
    if sw is not None:
        if sw >= len(theta):
            sw = len(theta) - 1 if len(theta) % 2 == 0 else len(theta)
        if sw < 5:
            sw = None
        elif sw % 2 == 0:
            sw += 1
    if sw is not None:
        theta = savgol_filter(theta, window_length=sw, polyorder=smooth_polyorder)

    trace = np.abs(np.gradient(theta, f))

    if prominence is None:
        prominence = 0.0005 * (trace.max() - trace.min())

    peak_indices, _ = find_peaks(
        trace,
        prominence=prominence,
        distance=min_distance_pts,
    )

    if len(peak_indices) > n_modes:
        heights = trace[peak_indices]
        top = np.argsort(heights)[::-1][:n_modes]
        peak_indices = np.sort(peak_indices[top])

    f0s = np.full(n_modes, np.nan)
    kappas = np.full(n_modes, np.nan)

    for k, pidx in enumerate(peak_indices):
        if k >= n_modes:
            break

        half_w_pts = max(20, int(0.5 * min_distance_pts))
        i_lo = max(0, pidx - 5 * half_w_pts)
        i_hi = min(len(f) - 1, pidx + 5 * half_w_pts)

        local_freq = rf.Frequency.from_f(f[i_lo:i_hi + 1], unit="Hz")
        local_s = s[i_lo:i_hi + 1]

        if len(local_s) < 5:
            f0s[k] = float(f[pidx])
            kappas[k] = float("nan")
            continue

        local_ntwk = rf.Network(
            frequency=local_freq,
            s=local_s[:, np.newaxis, np.newaxis],
        )
        try:
            f0_loc, kappa_loc = circle_fit_f0_kappa(
                local_ntwk,
                m=0,
                n=0,
                smooth_window=smooth_window,
                smooth_polyorder=smooth_polyorder,
            )
            f0s[k] = f0_loc
            kappas[k] = kappa_loc
        except Exception:
            f0s[k] = float(f[pidx])
            kappas[k] = float("nan")

    return f0s, kappas


def stitch_shifted_freqs(
    *arrays: np.ndarray,
    dedup_tol_ghz: float = 0.05,
    return_sources: bool = False,
):
    """Merge S11/S22 mode frequency lists, deduplicating nearby modes."""
    tagged = []
    for src_idx, arr in enumerate(arrays):
        for freq in arr[~np.isnan(arr)]:
            tagged.append((freq, src_idx))

    if not tagged:
        empty = np.array([])
        return (empty, np.array([], dtype=object)) if return_sources else empty

    tagged = sorted(tagged, key=lambda x: x[0])

    merged_freqs = []
    merged_sources = []
    cluster_freqs = [tagged[0][0]]
    cluster_srcs = [tagged[0][1]]

    for freq, src in tagged[1:]:
        if freq - cluster_freqs[0] < dedup_tol_ghz:
            cluster_freqs.append(freq)
            cluster_srcs.append(src)
        else:
            merged_freqs.append(float(np.mean(cluster_freqs)))
            merged_sources.append(max(set(cluster_srcs), key=cluster_srcs.count))
            cluster_freqs = [freq]
            cluster_srcs = [src]

    merged_freqs.append(float(np.mean(cluster_freqs)))
    merged_sources.append(max(set(cluster_srcs), key=cluster_srcs.count))

    freqs = np.array(merged_freqs)
    sources = np.array(["s11" if s == 0 else "s22" for s in merged_sources], dtype=object)

    return (freqs, sources) if return_sources else freqs


def resonance_circle_fit(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    smooth_window: int | None = 51,
    smooth_polyorder: int = 3,
) -> float:
    """
    Estimate resonance frequency from a circle fit to S[m,n] in the complex plane.

    Strategy
    --------
    1. Fit a circle to the complex trace S[m,n](f).
    2. Compute the angle of each point about the fitted center.
    3. Take the frequency where |d(theta)/df| is largest.

    This is usually a good reflection-based estimator for S11/S22.

    Parameters
    ----------
    ntwk : rf.Network
    m, n : int
        S-parameter indices.
    smooth_window : int or None
        Optional Savitzky-Golay smoothing window for theta before differentiation.
        Must be odd if provided.
    smooth_polyorder : int
        Polynomial order for Savitzky-Golay smoothing.

    Returns
    -------
    float
        Resonance frequency in Hz.
    """
    f0, _ = circle_fit_f0_kappa(
        ntwk,
        m=m,
        n=n,
        smooth_window=smooth_window,
        smooth_polyorder=smooth_polyorder,
    )
    return f0


def resonance(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    use_max: bool = False,
    method: str = "min_re",
) -> float:
    """
    Estimate resonance frequency from S[m,n].

    Parameters
    ----------
    ntwk : rf.Network
    m, n : int
        S-parameter indices (0-based).
    use_max : bool
        Only used for method='min_re'. If True, find frequency of maximum
        Re(S) instead of minimum.
    method : str
        Resonance estimator. Supported:
            - 'min_re'     : min/max of Re(S)
            - 'circle_fit' : circle fit in complex plane

    Returns
    -------
    float
        Resonance frequency in Hz.
    """
    if method == "circle_fit":
        return resonance_circle_fit(ntwk, m=m, n=n)

    if method == "min_re":
        f = ntwk.frequency.f
        re_s = np.real(ntwk.s[:, m, n])
        idx = np.argmax(re_s) if use_max else np.argmin(re_s)
        return float(f[idx])

    raise ValueError(f"Unknown resonance method: {method}")

# def resonance(ntwk: rf.Network, m: int = 0, n: int = 0, use_max: bool = False) -> float:
#     """
#     Estimate resonance frequency from the real part of S[m,n].

#     By default returns the frequency of the minimum of Re(S), which
#     corresponds to the resonance dip in S11/S22.  Set use_max=True
#     for transmission peaks.

#     Parameters
#     ----------
#     ntwk : rf.Network
#     m, n : int
#         S-parameter indices (0-based).
#     use_max : bool
#         If True, find frequency of maximum Re(S) instead of minimum.

#     Returns
#     -------
#     float
#         Resonance frequency in Hz.
#     """
#     f    = ntwk.frequency.f
#     re_s = np.real(ntwk.s[:, m, n])
#     idx  = np.argmax(re_s) if use_max else np.argmin(re_s)
#     return float(f[idx])


def resonances(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    n_modes: int = 3,
    method: str = "min_re",
    prominence: float = None,
    min_spacing_hz: float = None,
) -> np.ndarray:
    """
    Estimate resonance frequencies for multiple modes from S[m,n].

    Parameters
    ----------
    ntwk : rf.Network
    m, n : int
        S-parameter indices (0-based).
    n_modes : int
        Number of resonant modes to find.
    method : str
        Resonance estimator. Supported:
            - 'min_re'     : finds n_modes minima of Re(S)
            - 'circle_fit' : finds n_modes peaks of |dθ/df| from circle fit
    prominence : float, optional
        Minimum peak prominence passed to scipy.signal.find_peaks.
        If None, defaults to 0.1 * (max - min) of the trace being searched.
    min_spacing_hz : float, optional
        Minimum frequency separation between modes in Hz.
        If None, defaults to (f_max - f_min) / (10 * n_modes).

    Returns
    -------
    np.ndarray
        Resonance frequencies in Hz, sorted ascending, length n_modes.
        If fewer than n_modes peaks are found, the array is padded with NaN.
    """
    from scipy.signal import find_peaks

    f = ntwk.frequency.f
    df = f[1] - f[0]  # assumes uniform spacing

    if min_spacing_hz is None:
        min_spacing_hz = (f[-1] - f[0]) / (10 * n_modes)
    min_distance_pts = max(1, int(min_spacing_hz / df))

    if method == "min_re":
        # Search for minima of Re(S) by inverting the trace
        trace = -np.real(ntwk.s[:, m, n])

    elif method == "circle_fit":
        # Build the angular velocity |dθ/df| trace from the circle fit
        s = ntwk.s[:, m, n]
        x, y = s.real, s.imag

        # Kasa algebraic circle fit
        A = np.column_stack([x, y, np.ones(len(x))])
        b = -(x**2 + y**2)
        coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -coeffs[0] / 2.0, -coeffs[1] / 2.0

        theta = np.unwrap(np.angle((x - cx) + 1j * (y - cy)))
        trace = np.abs(np.gradient(theta, f))

    else:
        raise ValueError(f"Unknown resonance method: {method!r}")

    if prominence is None:
        prominence = 0.1 * (trace.max() - trace.min())

    peak_indices, _ = find_peaks(
        trace,
        prominence=prominence,
        distance=min_distance_pts,
    )

    # Sort by peak height descending, take top n_modes, then re-sort by frequency
    if len(peak_indices) > n_modes:
        heights = trace[peak_indices]
        top = np.argsort(heights)[::-1][:n_modes]
        peak_indices = np.sort(peak_indices[top])

    freqs = f[peak_indices].astype(float)

    # Pad with NaN if not enough peaks found
    if len(freqs) < n_modes:
        freqs = np.concatenate([freqs, np.full(n_modes - len(freqs), np.nan)])

    return freqs

def resonance_from_s_max(network: rf.Network, m: int = 0, n: int = 0) -> float:
    """
    Resonance frequency from the dominant peak in |S[m,n]|.

    Finds all peaks in the magnitude and returns the frequency of the
    largest one.  Useful for transmission parameters (S12/S21) where the
    resonance appears as a peak rather than a dip.

    Parameters
    ----------
    network : rf.Network
    m, n : int
        S-parameter indices (0-based).

    Returns
    -------
    float
        Dominant resonance frequency in Hz.

    Raises
    ------
    ValueError
        If no peaks are found in |S[m,n]|.
    """
    f     = network.frequency.f
    s_mn  = network.s[:, m, n]
    peaks, _ = find_peaks(np.abs(s_mn))

    if len(peaks) == 0:
        raise ValueError("No peaks found in |S[{},{}]|.".format(m, n))

    peak_idx = peaks[np.argmax(np.abs(s_mn)[peaks])]
    return float(f[peak_idx])

def resonances_from_s_max(network: rf.Network, m: int = 0, n: int = 0) -> np.ndarray:
    """
    Resonance frequencies from the peaks in |S[m,n]|.

    Finds all peaks in the magnitude and returns their frequencies.
    Useful for transmission parameters (S12/S21) where the resonances
    appear as peaks rather than dips.

    Parameters
    ----------
    network : rf.Network
    m, n : int
        S-parameter indices (0-based).

    Returns
    -------
    np.ndarray
        Array of resonance frequencies in Hz.

    Raises
    ------
    ValueError
        If no peaks are found in |S[m,n]|.
    """
    f     = network.frequency.f
    s_mn  = network.s[:, m, n]
    peaks, _ = find_peaks(np.abs(s_mn))
    freqs = f[peaks]
    logger.debug("Resonance frequencies (Hz): %s", freqs)

    if len(peaks) == 0:
        raise ValueError("No peaks found in |S[{},{}]|.".format(m, n))
    return freqs


def resonances_from_s(network: rf.Network, m: int = 0, n: int = 0) -> np.ndarray:
    """
    Find all resonance frequencies from peaks in Re(S[m,n]).

    Parameters
    ----------
    network : rf.Network
    m, n : int
        S-parameter indices (0-based).

    Returns
    -------
    np.ndarray
        Array of resonance frequencies in Hz.
    """
    f    = network.frequency.f
    s_mn = network.s[:, m, n]
    peaks, _ = find_peaks(np.real(s_mn))
    freqs = f[peaks]
    logger.debug("Resonance frequencies (Hz): %s", freqs)
    return freqs


def fwhm_from_trace_db(
    ntwk: rf.Network,
    m: int = 0,
    n: int = 0,
    kind: str = "dip",
    smooth: int = None,
) -> float:
    """
    Full-Width at Half Maximum (FWHM) from S-parameter magnitude in dB.

    Uses crossing-interpolation to find the two frequencies where the
    magnitude crosses the half-depth level, giving a more accurate result
    than a pure index-based approach.

    Parameters
    ----------
    ntwk : rf.Network
    m, n : int
        S-parameter indices (0-based).
    kind : {"dip", "peak"}
        "dip"  — resonance appears as a downward dip (typical for S11/S22).
        "peak" — resonance appears as an upward peak (typical for S12/S21).
    smooth : int or None
        Optional moving-average window length for noisy traces.
        Use an odd integer; None disables smoothing.

    Returns
    -------
    float
        FWHM linewidth in Hz.

    Raises
    ------
    ValueError
        If two crossings at the half-depth level cannot be found.
    """
    f      = np.asarray(ntwk.frequency.f, dtype=float)
    s      = ntwk.s[:, m, n]
    mag_db = 20 * np.log10(np.maximum(np.abs(s), 1e-300))

    if smooth is not None and smooth > 1:
        kernel = np.ones(int(smooth)) / int(smooth)
        mag_db = np.convolve(mag_db, kernel, mode="same")

    if kind == "dip":
        i0       = np.argmin(mag_db)
        y0       = mag_db[i0]
        baseline = np.median(np.r_[
            mag_db[:max(3, i0 // 8)],
            mag_db[min(len(mag_db) - 3, i0 + len(mag_db) // 8):],
        ])
        level = 0.5 * (baseline + y0)
        above = mag_db > level
    elif kind == "peak":
        i0       = np.argmax(mag_db)
        y0       = mag_db[i0]
        baseline = np.median(np.r_[
            mag_db[:max(3, i0 // 8)],
            mag_db[min(len(mag_db) - 3, i0 + len(mag_db) // 8):],
        ])
        level = 0.5 * (baseline + y0)
        above = mag_db < level
    else:
        raise ValueError("kind must be 'dip' or 'peak', got '{}'.".format(kind))

    flips = np.where(above[:-1] != above[1:])[0]
    if flips.size < 2:
        raise ValueError("Could not find two crossings for FWHM.")

    left_flips  = flips[flips < i0]
    right_flips = flips[flips >= i0]
    if left_flips.size == 0 or right_flips.size == 0:
        raise ValueError("Could not bracket resonance with crossings.")

    iL = left_flips[-1]
    iR = right_flips[0]

    def _interp_crossing(i: int) -> float:
        x0, x1 = f[i], f[i + 1]
        y0_, y1_ = mag_db[i], mag_db[i + 1]
        if y1_ == y0_:
            return 0.5 * (x0 + x1)
        return x0 + (level - y0_) * (x1 - x0) / (y1_ - y0_)

    return float(_interp_crossing(iR) - _interp_crossing(iL))


def fwhm_from_res11(ntwk: rf.Network) -> float:
    """
    Linewidth from zero crossings of Re(S11).

    Re(S11) crosses zero at the two half-power frequencies of the resonance.
    Uses linear interpolation between samples for accuracy.

    Parameters
    ----------
    ntwk : rf.Network

    Returns
    -------
    float
        Linewidth (distance between the two zero crossings) in Hz.

    Raises
    ------
    ValueError
        If fewer than two zero crossings are found.
    """
    f     = ntwk.frequency.f
    re_s11 = np.real(ntwk.s[:, 0, 0])

    sign_changes = np.where(np.diff(np.sign(re_s11)))[0]
    if len(sign_changes) < 2:
        raise ValueError("Could not determine linewidth: fewer than 2 zero crossings found.")

    zeros = []
    for idx in sign_changes[:2]:
        f1, f2 = f[idx], f[idx + 1]
        y1, y2 = re_s11[idx], re_s11[idx + 1]
        f_zero = f1 - y1 * (f2 - f1) / (y2 - y1)
        zeros.append(f_zero)

    return float(zeros[1] - zeros[0])


def resonance_from_res11(ntwk: rf.Network) -> float:
    """
    Resonance frequency from the minimum of Re(S11).

    Thin convenience wrapper around `resonance` for the common single-port
    S11 case.

    Parameters
    ----------
    ntwk : rf.Network

    Returns
    -------
    float
        Resonance frequency in Hz.
    """
    return resonance(ntwk, m=0, n=0, use_max=False)


def find_resonant_frequency(network: rf.Network) -> complex:
    freqs = network.f
    S21 = network.s[:, 1, 0]

    mag = np.abs(S21)
    peak_idx = np.argmax(mag)
    omega_0 = 2 * np.pi * freqs[peak_idx]

    half_pow = mag[peak_idx] / np.sqrt(2)
    left  = np.where(mag[:peak_idx] < half_pow)[0]
    right = np.where(mag[peak_idx:] < half_pow)[0]
    alpha = (2 * np.pi * freqs[peak_idx + right[0]] - 2 * np.pi * freqs[left[-1]]) / 2

    pole = complex(-alpha, omega_0)
    f0 = pole.imag / (2 * np.pi)
    return f0


def find_resonant_frequency(network: rf.Network) -> complex:
    freqs = network.f
    S21 = network.s[:, 1, 0]

    mag = np.abs(S21)
    peak_idx = np.argmax(mag)
    omega_0 = 2 * np.pi * freqs[peak_idx]

    half_pow = mag[peak_idx] / np.sqrt(2)
    left  = np.where(mag[:peak_idx] < half_pow)[0]
    right = np.where(mag[peak_idx:] < half_pow)[0]
    alpha = (2 * np.pi * freqs[peak_idx + right[0]] - 2 * np.pi * freqs[left[-1]]) / 2

    pole = complex(-alpha, omega_0)
    f0 = pole.imag / (2 * np.pi)
    return f0


def s_db(network: rf.Network, m: int = 1, n: int = 0) -> np.ndarray:
    """Return ``20 log10 |S[m,n]|`` in dB for ``network``."""
    return 20 * np.log10(np.maximum(np.abs(network.s[:, m, n]), 1e-300))


def notch_frequency(
    network: rf.Network,
    m: int = 1,
    n: int = 0,
) -> tuple[float, float]:
    """
    Locate the deepest transmission notch in ``|S[m,n]|``.

    The returned frequency is refined below the frequency-grid spacing by fitting
    a parabola to ``|S|**2`` through the three points straddling the minimum, so
    the result is not quantised to the grid. On the ~1 MHz-spaced grids used in
    the hanger tutorials this tightens the notch location from ``±½`` grid point
    (~0.5 MHz) to a fraction of a grid point; a deep notch is a sharp cusp rather
    than a true parabola, so a small sub-grid residual remains (it is *not*
    exact). The returned depth is the raw grid-point minimum ``|S[m,n]|``.

    Returns
    -------
    f_GHz : float
        Frequency of the deepest dip in GHz (sub-grid refined).
    depth : float
        ``|S[m,n]|`` at the grid-point minimum (linear magnitude).
    """
    s21 = np.abs(network.s[:, m, n])
    f = network.frequency.f / 1e9
    i = int(np.argmin(s21))
    f_min = float(f[i])
    if 0 < i < len(f) - 1:
        # Parabolic sub-grid refinement on |S|**2 (better conditioned near a deep
        # notch than linear |S|). delta is clamped to the bracketing bin.
        y0, y1, y2 = s21[i - 1] ** 2, s21[i] ** 2, s21[i + 1] ** 2
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        f_min = float(f[i] + delta * (f[i + 1] - f[i]))
    return f_min, float(s21[i])


def notches_below(
    network: rf.Network,
    thresh: float = 0.5,
    m: int = 1,
    n: int = 0,
) -> list[float]:
    """
    Frequencies (GHz) of local ``|S[m,n]|`` minima below ``thresh``.

    A coarse multi-notch finder for multiplexed hanger spectra.
    """
    s21 = np.abs(network.s[:, m, n])
    f = network.frequency.f / 1e9
    idxs = np.where(
        (s21[1:-1] < s21[:-2])
        & (s21[1:-1] < s21[2:])
        & (s21[1:-1] < thresh)
    )[0] + 1
    return sorted(float(f[i]) for i in idxs)
