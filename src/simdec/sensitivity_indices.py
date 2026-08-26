from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

__all__ = ["sensitivity_indices"]

try:
    from IPython.display import display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False


def magic_binning(x: np.ndarray, n_bins_default: int) -> tuple[np.ndarray, int]:
    """
    Python equivalent of the MATLAB magic_binning function.
    Uses equal-frequency binning, groups identical values, and handles NaNs.
    """

    is_nan = np.isnan(x)
    not_nan_x = x[~is_nan]

    idx_sorted = np.argsort(not_nan_x)
    x_sorted = not_nan_x[idx_sorted]
    n_valid = len(x_sorted)

    bin_idx_valid = np.zeros(n_valid, dtype=int)

    unique_vals = np.unique(x_sorted)
    if len(unique_vals) <= n_bins_default:
        _, bin_idx_valid = np.unique(
            x_sorted, return_inverse=True
        )  # x_sorted, not not_nan_x
        bin_idx_valid += 1
    else:
        min_bin_size = n_valid // n_bins_default
        remaining_size = n_valid
        current_edge_idx = min_bin_size - 1  # 0-based index

        b = 1
        start_idx = 0

        while b <= n_bins_default:
            current_bin_size = min_bin_size

            # While the edge is between identical values, move one element further
            while (current_edge_idx < n_valid - 1) and (
                x_sorted[current_edge_idx + 1] == x_sorted[current_edge_idx]
            ):
                current_edge_idx += 1
                current_bin_size += 1

            # Assign bin indices
            bin_idx_valid[start_idx : current_edge_idx + 1] = b
            remaining_size -= current_bin_size

            # Break if not enough elements left for two distinct bins
            if remaining_size < min_bin_size * 2:
                bin_idx_valid[current_edge_idx + 1 :] = b + 1
                break

            start_idx = current_edge_idx + 1
            current_edge_idx += min_bin_size
            b += 1

    bin_idx_valid_orig_order = np.zeros(n_valid, dtype=int)
    bin_idx_valid_orig_order[idx_sorted] = bin_idx_valid

    # NaNs back in (NaNs get bin 0)
    bin_idx = np.zeros(len(x), dtype=int)
    bin_idx[~is_nan] = bin_idx_valid_orig_order

    n_bins_out = np.max(bin_idx) if len(bin_idx) > 0 else 0

    return bin_idx, n_bins_out


def bin_data_1d(
    x: np.ndarray, y: np.ndarray, n_bins_default: int
) -> tuple[np.ndarray, np.ndarray]:
    bin_idx, n_bins_x = magic_binning(x, n_bins_default)

    bin_avg = np.full(n_bins_x, np.nan)
    bin_count = np.full(n_bins_x, np.nan)

    for b in range(1, n_bins_x + 1):
        mask = bin_idx == b
        if np.any(mask):
            bin_avg[b - 1] = np.mean(y[mask])
            bin_count[b - 1] = np.sum(mask)

    return bin_avg, bin_count


def bin_data_2d(xi: np.ndarray, xj: np.ndarray, y: np.ndarray, n_bins_default: int):
    bin_idx_i, n_bins_i = magic_binning(xi, n_bins_default)
    bin_idx_j, n_bins_j = magic_binning(xj, n_bins_default)

    bin_avg_ij = np.full((n_bins_i, n_bins_j), np.nan)
    bin_count_ij = np.full((n_bins_i, n_bins_j), np.nan)
    bin_avg_i = np.full(n_bins_i, np.nan)
    bin_count_i = np.full(n_bins_i, np.nan)
    bin_avg_j = np.full(n_bins_j, np.nan)
    bin_count_j = np.full(n_bins_j, np.nan)

    for n in range(1, n_bins_i + 1):
        mask_i = bin_idx_i == n
        if np.any(mask_i):
            bin_avg_i[n - 1] = np.mean(y[mask_i])
            bin_count_i[n - 1] = np.sum(mask_i)

    for m in range(1, n_bins_j + 1):
        mask_j = bin_idx_j == m
        if np.any(mask_j):
            bin_avg_j[m - 1] = np.mean(y[mask_j])
            bin_count_j[m - 1] = np.sum(mask_j)

    for n in range(1, n_bins_i + 1):
        for m in range(1, n_bins_j + 1):
            mask_ij = (bin_idx_i == n) & (bin_idx_j == m)
            if np.any(mask_ij):
                bin_avg_ij[n - 1, m - 1] = np.mean(y[mask_ij])
                bin_count_ij[n - 1, m - 1] = np.sum(mask_ij)

    # Flatten the 2D matrices and exclude NaNs (empty bins)
    bin_avg_ij_flat = bin_avg_ij.flatten()
    bin_count_ij_flat = bin_count_ij.flatten()

    valid = ~np.isnan(bin_avg_ij_flat)

    return (
        bin_avg_ij_flat[valid],
        bin_count_ij_flat[valid],
        bin_avg_i,
        bin_count_i,
        bin_avg_j,
        bin_count_j,
    )


def number_of_bins(n_runs: int, n_factors: int) -> tuple[int, int]:
    """Optimal number of bins for first & second-order sensitivity_indices indices.

    Linear approximation of experimental results from (Marzban & Lahmer, 2016).
    """
    n_bins_foe = 36 - 2.7 * n_factors + (0.0017 - 0.00008 * n_factors) * n_runs
    n_bins_foe = np.ceil(n_bins_foe)
    if n_bins_foe <= 30:
        n_bins_foe = 10  # setting a limit to fit the experimental results

    n_bins_soe = max(4, np.round(np.sqrt(n_bins_foe)))

    return n_bins_foe, n_bins_soe


def _weighted_var(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    avg = np.average(x, weights=weights)
    variance = np.average((x - avg) ** 2, weights=weights)
    return variance


@dataclass
class SensitivityAnalysisResult:
    si: np.ndarray
    first_order: np.ndarray
    second_order: np.ndarray


def sensitivity_indices(
    inputs: pd.DataFrame | np.ndarray,
    output: pd.DataFrame | np.ndarray,
    print_indices: bool = False,
) -> SensitivityAnalysisResult:
    """Sensitivity indices.

    The sensitivity_indices express how much variability of the output is
    explained by the inputs.

    Parameters
    ----------
    inputs : ndarray or DataFrame of shape (n_runs, n_factors)
        Input variables.
    output : ndarray or DataFrame of shape (n_runs, 1)
        Target variable.
    print_indices : bool, default False
        If True, displays computed indices.

    Returns
    -------
    res : SensitivityAnalysisResult
        An object with attributes:

        si : ndarray of shape (n_factors, 1)
            Sensitivity indices, combined effect of each input.
        foe : ndarray of shape (n_factors, 1)
            First-order effects (also called 'main' or 'individual').
        soe : ndarray of shape (n_factors, n_factors)
            Second-order effects (also called 'interaction').

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> import simdec as sd

    We define first the function that we want to analyse. We use the
    well studied Ishigami function:

    >>> def f_ishigami(x):
    ...     return (np.sin(x[0]) + 7 * np.sin(x[1]) ** 2
    ...             + 0.1 * (x[2] ** 4) * np.sin(x[0]))

    Then we generate inputs using the Quasi-Monte Carlo method of Sobol' in
    order to cover uniformly our space. And we compute outputs of the function.

    >>> rng = np.random.default_rng()
    >>> inputs = qmc.Sobol(d=3, seed=rng).random(2**18)
    >>> inputs = qmc.scale(
    ...     sample=inputs,
    ...     l_bounds=[-np.pi, -np.pi, -np.pi],
    ...     u_bounds=[np.pi, np.pi, np.pi]
    ... )
    >>> output = f_ishigami(inputs.T)

    We can now pass our inputs and outputs to the `sensitivity_indices` function:

    >>> res = sd.sensitivity_indices(inputs=inputs, output=output)
    >>> res.si
    array([0.43157591, 0.44241433, 0.11767249])

    """
    # Handle inputs conversion
    if isinstance(inputs, pd.DataFrame):
        var_names = inputs.columns.tolist()
        cat_cols = inputs.select_dtypes(include=["category", "O", "string"]).columns
        if not cat_cols.empty:
            inputs = inputs.copy()  # Avoid SettingWithCopyWarning
            inputs[cat_cols] = inputs[cat_cols].apply(
                lambda x: x.astype("category").cat.codes
            )
        inputs = inputs.to_numpy()
    else:
        inputs = np.asarray(inputs)
        # Fallback names if it's just a numpy array
        var_names = [f"x{i}" for i in range(inputs.shape[1])]

    # Handle output conversion first, then flatten
    if isinstance(output, (pd.DataFrame, pd.Series)):
        output = output.to_numpy()

    # Flatten output if it's (N, 1)
    output = output.flatten()

    n_runs, n_factors = inputs.shape
    n_bins_foe, n_bins_soe = number_of_bins(n_runs, n_factors)

    # Overall variance of the output
    var_y = np.var(output)

    si = np.empty(n_factors)
    foe = np.empty(n_factors)
    soe = np.zeros((n_factors, n_factors))

    for i in range(n_factors):
        # 1. First-order effects (FOE)
        xi = inputs[:, i]

        bin_avg, bin_count = bin_data_1d(xi, output, int(n_bins_foe))

        valid_foe = ~np.isnan(bin_avg)
        foe[i] = _weighted_var(bin_avg[valid_foe], weights=bin_count[valid_foe]) / var_y

        # 2. Second-order effects (SOE)
        for j in range(n_factors):
            if j <= i:
                continue

            xj = inputs[:, j]

            # Second-order effects (SOE) with magic_binning
            (
                bin_avg_ij,
                bin_count_ij,
                bin_avg_i,
                bin_count_i,
                bin_avg_j,
                bin_count_j,
            ) = bin_data_2d(xi, xj, output, int(n_bins_soe))

            var_ij = _weighted_var(bin_avg_ij, weights=bin_count_ij)

            valid_i = ~np.isnan(bin_avg_i)
            var_i = _weighted_var(bin_avg_i[valid_i], weights=bin_count_i[valid_i])

            valid_j = ~np.isnan(bin_avg_j)
            var_j = _weighted_var(bin_avg_j[valid_j], weights=bin_count_j[valid_j])

            soe[i, j] = (var_ij - var_i - var_j) / var_y

    # Mirror SOE and calculate Combined Effect (SI)
    # SI is FOE + half of all interactions associated with that variable
    soe = soe + soe.T
    for k in range(n_factors):
        si[k] = foe[k] + (soe[:, k].sum() / 2)

    if print_indices:
        if not HAS_IPYTHON:
            warnings.warn(
                "print_indices=True requires ipython to be installed. "
                "Install it with: pip install simdec[display]. Table skipped.",
                stacklevel=2,
            )
        else:
            df_foe = pd.DataFrame(foe, index=var_names, columns=["First-order effect"])
            df_soe = pd.DataFrame(soe, index=var_names, columns=var_names)
            df_si = pd.DataFrame(si, index=var_names, columns=["Combined effect"])

            df_indices = pd.concat([df_foe, df_soe, df_si], axis=1)
            display(df_indices)

    return SensitivityAnalysisResult(si, foe, soe)
