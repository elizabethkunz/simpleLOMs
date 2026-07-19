"""
utils.py
---------
Ease of use functions for processing and viewing data.

"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import skrf as rf
import pandas as pd

def make_results_table(results: dict, round_decimals: int = 4) -> pd.DataFrame:
    """
    Summarise analyze_system() output as a tidy DataFrame.

    Rows are models (CPW, Optimized, Foster, Analytical).
    Columns are f₀, κ, and percent errors for S11 and S22.

    Parameters
    ----------
    results : dict
        Output from analyze_system().
    round_decimals : int
        Decimal places to round to.  Default 4.

    Returns
    -------
    pd.DataFrame
        Indexed by model name.
    """
    rows = []
    for method in ["CPW", "Optimized", "Foster", "Analytical"]:
        row = {"Method": method}
        for port in ["S11", "S22"]:
            row[f"f0 {port} (GHz)"] = results.get(f"{method} f0 {port} (GHz)",   float("nan"))
            row[f"κ {port} (MHz)"]  = results.get(f"{method} kappa {port} (MHz)", float("nan"))
            if method != "CPW":
                row[f"f0 error {port} (%)"] = results.get(f"{method} f0 error {port} (%)", float("nan"))
                row[f"κ error {port} (%)"]  = results.get(f"{method} kappa error {port} (%)", float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).set_index("Method").round(round_decimals)


def make_params_table(results: dict) -> pd.DataFrame:
    """
    Summarize fitted L and C for all three models as a tidy DataFrame.

    Parameters
    ----------
    results : dict
        Output from analyze_system().

    Returns
    -------
    pd.DataFrame
        Indexed by model name, columns L (H) and C (F).
    """
    rows = []
    for key, label in [
        ("foster_model",     "Foster"),
        ("optimized_model",  "Optimized"),
        ("analytical_model", "Analytical"),
    ]:
        model = results.get(key)
        if model is not None and model.is_fitted:
            p = model.get_params()
            rows.append({"Model": label, "L (H)": p["L"], "C (F)": p["C"]})
        else:
            rows.append({"Model": label, "L (H)": float("nan"), "C (F)": float("nan")})

    return pd.DataFrame(rows).set_index("Model")


def make_shift_table(results: dict, port: str = "S11", round_decimals: int = 5) -> pd.DataFrame:
    """
    Hybridised mode frequencies for all models as a tidy DataFrame.

    Parameters
    ----------
    results : dict
        Output from analyze_system().
    port : {"S11", "S22"}
    round_decimals : int

    Returns
    -------
    pd.DataFrame
        Rows are modes, columns are models.
    """
    raise NotImplementedError
    port = port.upper()
    return pd.DataFrame({
        "CPW (GHz)":        results.get(f"CPW shifted freqs {port} (GHz)",        [float("nan")]*3),
        "Foster (GHz)":     results.get(f"Foster shifted freqs {port} (GHz)",     [float("nan")]*3),
        "Optimized (GHz)":  results.get(f"Optimized shifted freqs {port} (GHz)",  [float("nan")]*3),
        "Analytical (GHz)": results.get(f"Analytical shifted freqs {port} (GHz)", [float("nan")]*3),
    }, index=["Mode 1", "Mode 2", "Mode 3"]).round(round_decimals)


def format_comparison_table(rows, headers) -> str:
    """
    Format an aligned plain-text comparison table.

    Parameters
    ----------
    rows : sequence of sequences
        Data rows; each cell is already a string or will be ``str()``-converted.
    headers : sequence of str
        Column headers (same length as each row).

    Returns
    -------
    str
        Multi-line table suitable for ``print``.
    """
    headers = list(headers)
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in str_rows)) if str_rows else len(headers[i])
        for i in range(len(headers))
    ]
    # Right-align numeric-looking columns (all but the first).
    def fmt_cell(i, text, header=False):
        w = widths[i]
        if i == 0 or header:
            return text.ljust(w) if i == 0 else text.rjust(w)
        return text.rjust(w)

    lines = [
        "  ".join(fmt_cell(i, h, header=True) for i, h in enumerate(headers)),
        "  ".join("-" * w for w in widths),
    ]
    for row in str_rows:
        lines.append("  ".join(fmt_cell(i, c) for i, c in enumerate(row)))
    return "\n".join(lines)


def make_shift_error_table(results: dict, round_decimals: int = 3) -> pd.DataFrame:
    """
    Signed percent shift errors vs CPW reference for all three models.

    Parameters
    ----------
    results : dict
        Output from analyze_system().
    round_decimals : int

    Returns
    -------
    pd.DataFrame
        Rows are modes, columns are models.
    """
    raise NotImplementedError
    return pd.DataFrame({
        "Foster error (%)":     results.get("Foster shift errors (%)",     [float("nan")]*3),
        "Optimized error (%)":  results.get("Optimized shift errors (%)",  [float("nan")]*3),
        "Analytical error (%)": results.get("Analytical shift errors (%)", [float("nan")]*3),
    }, index=["Mode 1", "Mode 2", "Mode 3"]).round(round_decimals)
