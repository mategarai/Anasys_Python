#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:40:09 2026

@author: mategarai
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import PDM_main as PDM

# =============================================================================
# Pure Helper Functions
# =============================================================================

PROFILE_PARAM_COUNT = {"gaussian": 3, "lorentzian": 3, "voigt": 4}

def stack_ri(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return np.concatenate([z.real, z.imag]).astype(float)

def baseline_mask(x, regions):
    x = np.asarray(x, float)
    m = np.zeros_like(x, dtype=bool)
    for lo, hi in regions:
        m |= (x >= lo) & (x <= hi)
    return m

def build_fit_parameters_old(num_peaks, profile, overrides_p0=None, overrides_bounds=None, **defaults):
    overrides_p0 = overrides_p0 or {}
    overrides_bounds = overrides_bounds or {}
    
    p0 = [
        overrides_p0.get("eps", defaults.get("eps_guess", 1.0)),
        overrides_p0.get("slope", defaults.get("slope_guess", -0.0003))
    ]
    
    lb = [
        overrides_bounds.get("eps", defaults.get("eps_bounds", (0.5, 6.0)))[0],
        overrides_bounds.get("slope", defaults.get("slope_bounds", (-1.0, 1.0)))[0]
    ]
    
    ub = [
        overrides_bounds.get("eps", defaults.get("eps_bounds", (0.5, 6.0)))[1],
        overrides_bounds.get("slope", defaults.get("slope_bounds", (-1.0, 1.0)))[1]
    ]

    pcount = PROFILE_PARAM_COUNT[profile.lower()]
    param_names = ["A", "x0", "sigma", "gamma"][:pcount]

    for pk in range(1, int(num_peaks) + 1):
        for nm in param_names:
            key = f"{nm}{pk}"
            guess_key = f"{nm}_guess"
            bounds_key = f"{nm}_bounds"
            
            p0.append(overrides_p0.get(key, defaults.get(guess_key, 1.0)))
            bnd = overrides_bounds.get(key, defaults.get(bounds_key, (0.0, 1.0)))
            lb.append(bnd[0])
            ub.append(bnd[1])

    return tuple(p0), (tuple(lb), tuple(ub))



def build_fit_parameters(
    profile, peak_centers, center_tolerance, 
    eps_guess, slope_guess, A_guess, sigma_guess, gamma_guess,
    eps_bounds, slope_bounds, A_bounds, sigma_bounds, gamma_bounds
):
    num_peaks = len(peak_centers)
    
    # --- Normalization Helpers ---
    def norm_guess(g):
        """Checks if guess is a list for multiple peaks, or a scalar to be duplicated."""
        if isinstance(g, (list, tuple, np.ndarray)):
            if len(g) != num_peaks:
                raise ValueError(f"Guess list length ({len(g)}) does not match num_peaks ({num_peaks})")
            return g
        return [g] * num_peaks

    def norm_bound(b):
        """Checks if bounds are a list of pairs [[lo,hi], [lo,hi]], or a single pair [lo,hi] to be duplicated."""
        # If the first element is a collection, it's a list of bounds
        if isinstance(b[0], (list, tuple, np.ndarray)):
            if len(b) != num_peaks:
                raise ValueError(f"Bounds list length ({len(b)}) does not match num_peaks ({num_peaks})")
            return b
        # Otherwise, it's a single bound pair like (14.0, 15.0)
        return [b] * num_peaks

    # Normalize per-peak parameters
    A_g = norm_guess(A_guess)
    sig_g = norm_guess(sigma_guess)
    gam_g = norm_guess(gamma_guess)

    A_b = norm_bound(A_bounds)
    sig_b = norm_bound(sigma_bounds)
    gam_b = norm_bound(gamma_bounds)
    
    p0 = [eps_guess, slope_guess]
    lb = [eps_bounds[0], slope_bounds[0]]
    ub = [eps_bounds[1], slope_bounds[1]]

    profile = profile.lower()
    
    for i in range(num_peaks):
        # Center bounds calculated dynamically from list and tolerance
        x0_val = float(peak_centers[i])
        x0_bnd = (x0_val - center_tolerance, x0_val + center_tolerance)
        
        # Use the index 'i' to pull the specific guess/bound for this peak
        p0.extend([A_g[i], x0_val])
        lb.extend([A_b[i][0], x0_bnd[0]])
        ub.extend([A_b[i][1], x0_bnd[1]])
        
        # Apply the dynamically extracted width guesses/bounds to each peak
        if profile == "gaussian":
            p0.append(sig_g[i])
            lb.append(sig_b[i][0])
            ub.append(sig_b[i][1])
        elif profile == "lorentzian":
            p0.append(gam_g[i])
            lb.append(gam_b[i][0])
            ub.append(gam_b[i][1])
        elif profile == "voigt":
            p0.extend([sig_g[i], gam_g[i]])
            lb.extend([sig_b[i][0], gam_b[i][0]])
            ub.extend([sig_b[i][1], gam_b[i][1]])

    return tuple(p0), (tuple(lb), tuple(ub))


def detrend_real_keep_im_fast(z, x, detrend_context):
    """Handles 1D and 2D arrays using explicit context instead of globals."""
    x_base, n_base, sum_x, ols_denom, mask = detrend_context
    z_real_base = z.real[mask]
    
    if z.ndim == 1:
        sum_y = np.sum(z_real_base)
        sum_xy = np.sum(x_base * z_real_base)
        
        m = (n_base * sum_xy - sum_x * sum_y) / ols_denom
        b = (sum_y - m * sum_x) / n_base
        re_line = m * x + b
        im0 = float(np.mean(z.imag[mask]))
        
        z_corr = (z.real - re_line) + 1j * (z.imag - im0)
        return z_corr, (m, b), im0
        
    else:
        sum_y = np.sum(z_real_base, axis=0)
        sum_xy = x_base @ z_real_base
        
        m = (n_base * sum_xy - sum_x * sum_y) / ols_denom
        b = (sum_y - m * sum_x) / n_base
        re_line = x[:, None] * m + b
        im0 = np.mean(z.imag[mask], axis=0)
        
        z_corr = (z.real - re_line) + 1j * (z.imag - im0)
        return z_corr, np.column_stack((m, b)), im0

# =============================================================================
# Multiprocessing Worker
# =============================================================================

def fit_single_pixel(args):
    """Worker function for parallel processing. Must be at root module level."""
    i, y_target, p0_i, bounds_i, x_data, detrend_context, profile, max_nfev, bulk_sample = args

    def model_ri_detrended_wrapper(x, *p):
        z = PDM.PDM_fitting(x, *p, profile=profile, bulk_sample=bulk_sample)
        zcorr, _, _ = detrend_real_keep_im_fast(z, x, detrend_context)
        return stack_ri(zcorr)

    try:
        popt, pcov = curve_fit(
            model_ri_detrended_wrapper,
            x_data,
            y_target,
            p0=p0_i,
            bounds=bounds_i,
            method="trf",
            jac="2-point",
            max_nfev=max_nfev,
        )
        
        # --- R-squared Calculation ---
        y_pred = model_ri_detrended_wrapper(x_data, *popt)
        ss_res = np.sum((y_target - y_pred)**2)
        ss_tot = np.sum((y_target - np.mean(y_target))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return i, True, popt, r_squared
    except Exception as e:
        return i, False, str(e), np.nan

# =============================================================================
# Main Pipeline Function
# =============================================================================

def process_spectra_array_old(
    data_input,
    profile="voigt",
    num_peaks=2,
    fit_window=None,
    baseline_regions=None,
    p0_overrides=None,
    bounds_overrides=None,
    max_nfev=30000,
    **defaults
):
    if profile not in PROFILE_PARAM_COUNT:
        raise ValueError(f"Unknown PROFILE={profile!r}.")

    tic = time.perf_counter()

    # Load Data dynamically based on input type
    if isinstance(data_input, (str, Path)):
        raw = pd.read_csv(data_input, header=None)
        xcm_full = raw.iloc[:, 0].to_numpy(dtype=float)
        spectra_raw = raw.iloc[:, 1:]
        spectra = spectra_raw.map(lambda s: complex(str(s).replace(" ", "")) if pd.notnull(s) else 0j).to_numpy(np.complex128)
        
    elif isinstance(data_input, pd.DataFrame):
        # Handle DataFrames loaded exactly like the CSV vs direct xarray exports
        if type(data_input.index) is pd.RangeIndex:
            xcm_full = data_input.iloc[:, 0].to_numpy(dtype=float)
            spectra_raw = data_input.iloc[:, 1:]
        else:
            xcm_full = data_input.index.to_numpy(dtype=float)
            spectra_raw = data_input

        # Check if values are strings (need parsing) or native complex numbers
        if not spectra_raw.empty and isinstance(spectra_raw.iloc[0, 0], str):
            spectra = spectra_raw.map(lambda s: complex(str(s).replace(" ", "")) if pd.notnull(s) else 0j).to_numpy(np.complex128)
        else:
            spectra = spectra_raw.fillna(0j).to_numpy(np.complex128)
            
    else:
        raise TypeError("data_input must be a file path (str/Path) or a pandas DataFrame.")
    
    
    
    N, M = spectra.shape

    spectra = spectra.real - 1j * spectra.imag

    # Apply Window
    if fit_window is None:
        x_fit = xcm_full
        spectra_fit = spectra
    else:
        win = (xcm_full >= fit_window[0]) & (xcm_full <= fit_window[1])
        x_fit = xcm_full[win]
        spectra_fit = spectra[win, :]

    # Setup Baseline Context
    baseline_regions = baseline_regions or [(1575.0, 1593.0), (1680.0, 1730.0)]
    base_m = baseline_mask(x_fit, baseline_regions)
    x_base = x_fit[base_m]
    n_base = len(x_base)
    sum_x = np.sum(x_base)
    sum_x2 = np.sum(x_base**2)
    ols_denom = n_base * sum_x2 - sum_x**2
    
    detrend_context = (x_base, n_base, sum_x, ols_denom, base_m)

    # Detrend Array
    spectra_fit_norm, re_drift_mb, im_base = detrend_real_keep_im_fast(spectra_fit, x_fit, detrend_context)

    # Build Parameters
    p0, bounds = build_fit_parameters(
        num_peaks, profile, p0_overrides, bounds_overrides, **defaults
    )

    # Parallel Execution Setup
    tasks = [
        (i, stack_ri(spectra_fit_norm[:, i]), p0, bounds, x_fit, detrend_context, profile, max_nfev) 
        for i in range(M)
    ]
    
    fit_params = np.full((M, len(p0)), np.nan, dtype=float)
    fit_success = np.zeros(M, dtype=bool)
    fit_r2 = np.full(M, np.nan, dtype=float)
    
    num_cores = max(1, os.cpu_count() - 1)
    optimal_chunksize = max(1, M // (num_cores * 4))

    print(f"Loaded x: {xcm_full.shape}, spectra: {spectra.shape} (N={N}, M={M})")
    print(f"Starting parallel fit across {num_cores} cores...")

    completed = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in executor.map(fit_single_pixel, tasks, chunksize=optimal_chunksize):
            i, success, res_data, r2_val = result # Unpack R2
            fit_success[i] = success
            
            if success:
                fit_params[i, :] = res_data
                fit_r2[i] = r2_val
            else:
                print(f"\nPixel {i:3d} FIT FAILED: {res_data}")
                
            completed += 1
            if completed % max(1, M // 100) == 0 or completed == M:
                percent_float = completed / M
                filled_length = int(20 * percent_float)
                bar = "=" * filled_length + "-" * (20 - filled_length)
                print(f"\rPDM Fitting Spectra: [{bar}] {int(percent_float * 100)}% ({completed}/{M})", end="", flush=True)

    toc = time.perf_counter()
    print("\nFitting complete!")
    print(f"\n\nSuccessful fits: {np.sum(fit_success)} / {M}")
    print(f"Elapsed time: {toc - tic:.4f} seconds")

    # Return a dictionary containing everything needed for further analysis/plotting
    return {
        "x_fit": x_fit,
        "spectra_fit": spectra_fit,
        "spectra_fit_norm": spectra_fit_norm,
        "fit_params": fit_params,
        "fit_success": fit_success,
        "fit_r2": fit_r2,
        "re_drift_mb": re_drift_mb,
        "im_base": im_base,
        "detrend_context": detrend_context,
        "p0": p0,
        "M": M
    }

def process_spectra_array(
    data_input,
    profile="voigt",
    peak_centers=None,
    center_tolerance=2.0,
    fit_window=None,
    baseline_regions=None,
    max_nfev=4000,
    # Strictly flat kwargs
    eps_guess=1.0, slope_guess=-0.0003, A_guess=50000.0, sigma_guess=6.0, gamma_guess=6.0,
    eps_bounds=(0.5, 2.0), slope_bounds=(-1.0, 1.0), A_bounds=(1000.0, 60000.0), 
    sigma_bounds=(4.0, 8.0), gamma_bounds=(4.0, 8.0),
    bulk_sample=False
):
    if not peak_centers:
        raise ValueError("peak_centers list must be provided.")
        
    if profile not in PROFILE_PARAM_COUNT:
        raise ValueError(f"Unknown PROFILE={profile!r}.")

    tic = time.perf_counter()

    # --- KEEP YOUR EXISTING DATA LOADING AND DETRENDING BLOCKS HERE ---
    if isinstance(data_input, (str, Path)):
        raw = pd.read_csv(data_input, header=None)
        xcm_full = raw.iloc[:, 0].to_numpy(dtype=float)
        spectra_raw = raw.iloc[:, 1:]
        spectra = spectra_raw.map(lambda s: complex(str(s).replace(" ", "")) if pd.notnull(s) else 0j).to_numpy(np.complex128)
        
    elif isinstance(data_input, pd.DataFrame):
        if type(data_input.index) is pd.RangeIndex:
            xcm_full = data_input.iloc[:, 0].to_numpy(dtype=float)
            spectra_raw = data_input.iloc[:, 1:]
        else:
            xcm_full = data_input.index.to_numpy(dtype=float)
            spectra_raw = data_input

        if not spectra_raw.empty and isinstance(spectra_raw.iloc[0, 0], str):
            spectra = spectra_raw.map(lambda s: complex(str(s).replace(" ", "")) if pd.notnull(s) else 0j).to_numpy(np.complex128)
        else:
            spectra = spectra_raw.fillna(0j).to_numpy(np.complex128)
            
    else:
        raise TypeError("data_input must be a file path (str/Path) or a pandas DataFrame.")
    
    N, M = spectra.shape
    spectra = spectra.real - 1j * spectra.imag

    if fit_window is None:
        x_fit = xcm_full
        spectra_fit = spectra
    else:
        win = (xcm_full >= fit_window[0]) & (xcm_full <= fit_window[1])
        x_fit = xcm_full[win]
        spectra_fit = spectra[win, :]
        
    base_m = baseline_mask(x_fit, baseline_regions)
    x_base = x_fit[base_m]
    n_base = len(x_base)
    sum_x = np.sum(x_base)
    sum_x2 = np.sum(x_base**2)
    ols_denom = n_base * sum_x2 - sum_x**2
    
    detrend_context = (x_base, n_base, sum_x, ols_denom, base_m)

    spectra_fit_norm, re_drift_mb, im_base = detrend_real_keep_im_fast(spectra_fit, x_fit, detrend_context)

    # Build Parameters using only the flat inputs
    p0, bounds = build_fit_parameters(
        profile, peak_centers, center_tolerance, 
        eps_guess, slope_guess, A_guess, sigma_guess, gamma_guess,
        eps_bounds, slope_bounds, A_bounds, sigma_bounds, gamma_bounds
    )
    
    # Parallel Execution Setup
    tasks = [
        (i, stack_ri(spectra_fit_norm[:, i]), p0, bounds, x_fit, detrend_context, profile, max_nfev,bulk_sample) 
        for i in range(M)
    ]
    fit_params = np.full((M, len(p0)), np.nan, dtype=float)
    fit_success = np.zeros(M, dtype=bool)
    fit_r2 = np.full(M, np.nan, dtype=float)
    
    num_cores = max(1, os.cpu_count() - 1)
    optimal_chunksize = max(1, M // (num_cores * 4))

    print(f"Loaded x: {xcm_full.shape}, spectra: {spectra.shape} (N={N}, M={M})")
    print(f"Starting parallel fit across {num_cores} cores...")

    completed = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in executor.map(fit_single_pixel, tasks, chunksize=optimal_chunksize):
            i, success, res_data, r2_val = result # Unpack R2
            fit_success[i] = success
            
            if success:
                fit_params[i, :] = res_data
                fit_r2[i] = r2_val
            else:
                print(f"\nPixel {i:3d} FIT FAILED: {res_data}")
                
            completed += 1
            if completed % max(1, M // 100) == 0 or completed == M:
                percent_float = completed / M
                filled_length = int(20 * percent_float)
                bar = "=" * filled_length + "-" * (20 - filled_length)
                print(f"\rFitting Spectra (PDM): [{bar}] {int(percent_float * 100)}% ({completed}/{M})", end="", flush=True)

    toc = time.perf_counter()
    print(f"\n\nSuccessful fits: {np.sum(fit_success)} / {M}")
    print(f"Elapsed time: {toc - tic:.4f} seconds")

    # Return a dictionary containing everything needed for further analysis/plotting
    return {
        "x_fit": x_fit,
        "spectra_fit": spectra_fit,
        "spectra_fit_norm": spectra_fit_norm,
        "fit_params": fit_params,
        "fit_success": fit_success,
        "fit_r2": fit_r2,
        "re_drift_mb": re_drift_mb,
        "im_base": im_base,
        "detrend_context": detrend_context,
        "p0": p0,
        "M": M
    }

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_2d_maps_old(results, nx, ny):
    """Generates the 2D maps for parameters."""
    M = results["M"]
    if nx * ny != M:
        raise ValueError(f"nx*ny must equal M. Got {nx*ny} but M={M}.")

    fit_params = results["fit_params"]
    fit_success = results["fit_success"]
    p0_ref = np.asarray(results["p0"], dtype=float)
    
    P = fit_params.shape[1]
    param_labels = ["eps", "slope", "amp", "center", "sigma", "gamma"]

    base = 2
    A_idx, x0_idx = base + 0, base + 1

    A_all = fit_params[:, A_idx]
    x0_all = fit_params[:, x0_idx]

    A_close = np.isclose(A_all, p0_ref[A_idx], rtol=0.01, atol=1e-12)
    x0_close = np.isclose(x0_all, p0_ref[x0_idx], rtol=0.01, atol=1e-12)

    invalid_global = (~fit_success) | (A_close & x0_close)
    invalid_global_2d = invalid_global.reshape(ny, nx)
    maps = fit_params.reshape(ny, nx, P).copy()

    rows, cols = 2, min(3, P)
    if P > 6: rows = int(np.ceil(P / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for k in range(min(P, len(axes))):
        param_map = maps[:, :, k].copy()
        invalid_param = invalid_global_2d | (param_map == p0_ref[k])
        param_map[invalid_param] = np.nan

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="black")

        im = axes[k].imshow(param_map, origin="lower", aspect="equal", cmap=cmap)
        axes[k].set_title(f"Fit map: {param_labels[k] if k < len(param_labels) else f'param[{k}]'}")
        axes[k].set_xlabel("x pixel")
        axes[k].set_ylabel("y pixel")
        fig.colorbar(im, ax=axes[k])
    
    for k in range(P, len(axes)):
        axes[k].axis('off')
        
    plt.tight_layout()
    plt.show()

def plot_2d_maps(results, nx, ny, profile="voigt"):
    """Generates the 2D maps for parameters with dynamic peak labeling."""
    M = results["M"]
    if nx * ny != M:
        raise ValueError(f"nx*ny must equal M. Got {nx*ny} but M={M}.")

    fit_params = results["fit_params"]
    fit_success = results["fit_success"]
    p0_ref = np.asarray(results["p0"], dtype=float)
    
    P = fit_params.shape[1]
    
    # --- DYNAMIC LABEL GENERATOR ---
    param_labels = ["eps", "slope"]
    pcount = PROFILE_PARAM_COUNT[profile.lower()]
    num_peaks = (P - 2) // pcount  # Calculate number of peaks based on total parameters

    for i in range(1, num_peaks + 1):
        param_labels.extend([f"Peak {i} Amp", f"Peak {i} Center"])
        if profile.lower() == "gaussian":
            param_labels.append(f"Peak {i} Sigma")
        elif profile.lower() == "lorentzian":
            param_labels.append(f"Peak {i} Gamma")
        elif profile.lower() == "voigt":
            param_labels.extend([f"Peak {i} Sigma", f"Peak {i} Gamma"])
    # -------------------------------

    base = 2
    A_idx, x0_idx = base + 0, base + 1

    A_all = fit_params[:, A_idx]
    x0_all = fit_params[:, x0_idx]

    A_close = np.isclose(A_all, p0_ref[A_idx], rtol=0.01, atol=1e-12)
    x0_close = np.isclose(x0_all, p0_ref[x0_idx], rtol=0.01, atol=1e-12)

    invalid_global = (~fit_success) | (A_close & x0_close)
    invalid_global_2d = invalid_global.reshape(ny, nx)
    maps = fit_params.reshape(ny, nx, P).copy()

    # Create grid based on actual number of parameters
    cols = 3
    rows = int(np.ceil(P / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for k in range(min(P, len(axes))):
        param_map = maps[:, :, k].copy()
        invalid_param = invalid_global_2d | (param_map == p0_ref[k])
        param_map[invalid_param] = np.nan

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="black")

        im = axes[k].imshow(param_map, origin="lower", aspect="equal", cmap=cmap)
        
        # Use the dynamically generated label, fallback to param[k] just in case
        title_label = param_labels[k] if k < len(param_labels) else f'param[{k}]'
        axes[k].set_title(f"Fit map: {title_label}")
        axes[k].set_xlabel("x pixel")
        axes[k].set_ylabel("y pixel")
        fig.colorbar(im, ax=axes[k])
    
    for k in range(P, len(axes)):
        axes[k].axis('off')
        
    plt.tight_layout()
    plt.show()


def get_individual_peaks(x, popt, num_peaks, profile, detrend_context,bulk_sample):
    """Generates detrended line shapes for individual peaks."""
    pcount = PROFILE_PARAM_COUNT[profile]
    base_idx = 2  # indices 0 and 1 are eps and slope
    
    peaks_detrended = []
    
    for i in range(num_peaks):
        popt_single = np.copy(popt)
        
        # Zero out amplitudes of all other peaks
        for j in range(num_peaks):
            if i != j:
                A_idx = base_idx + j * pcount
                popt_single[A_idx] = 0.0
                
        # Evaluate single peak model and detrend
        z_single = PDM.PDM_fitting(x, *popt_single, profile=profile,bulk_sample=bulk_sample)
        z_single_det, _, _ = detrend_real_keep_im_fast(z_single, x, detrend_context)
        peaks_detrended.append(z_single_det)
        
    return peaks_detrended




def plot_individual_fits(results, profile="voigt", num_peaks=2, plot_every=10, show_individual_peaks=True, grid_n=5,bulk_sample=False):
    """Plots grouped manual previews for individual pixels on n x n grids."""
    if not plot_every:
        return

    x_fit = results["x_fit"]
    # spectra_fit = results["spectra_fit"]
    spectra_fit_norm = results["spectra_fit_norm"]
    fit_params = results["fit_params"]
    fit_success = results["fit_success"]
    re_drift_mb = results["re_drift_mb"]
    # im_base = results["im_base"]
    detrend_context = results["detrend_context"]
    M = results["M"]

    # Gather all valid indices to plot
    valid_indices = [i for i in range(M) if fit_success[i] and (i % plot_every == 0)]
    
    if not valid_indices:
        print("No successful fits to plot.")
        return

    plots_per_fig = grid_n * grid_n

    # Iterate through the valid pixels in chunks of n*n
    for chunk_start in range(0, len(valid_indices), plots_per_fig):
        chunk = valid_indices[chunk_start : chunk_start + plots_per_fig]
        
        # Create 3 new figures for this specific batch
        # fig_raw, axs_raw = plt.subplots(grid_n, grid_n, figsize=(4 * grid_n, 3.5 * grid_n))
        fig_det, axs_det = plt.subplots(grid_n, grid_n, figsize=(4 * grid_n, 3.5 * grid_n))
        fig_cmp, axs_cmp = plt.subplots(grid_n, grid_n, figsize=(4 * grid_n, 4 * grid_n))
        
        # fig_raw.suptitle("Raw Overlay", fontsize=16)
        fig_det.suptitle("Detrended (with peaks)", fontsize=16)
        fig_cmp.suptitle("Complex Plane", fontsize=16)

        # Flatten the axis arrays to iterate over them easily
        # axs_raw_flat = np.array(axs_raw).flatten()
        axs_det_flat = np.array(axs_det).flatten()
        axs_cmp_flat = np.array(axs_cmp).flatten()

        for idx, i in enumerate(chunk):
            zhat_raw = PDM.PDM_fitting(x_fit, *fit_params[i, :], profile=profile,bulk_sample=bulk_sample)
            zhat_det, _, _ = detrend_real_keep_im_fast(zhat_raw, x_fit, detrend_context)

            m, b = re_drift_mb[i]
            # re_line = m * x_fit + b
            # zhat_overlay = (zhat_det.real + re_line) + 1j * (zhat_det.imag + im_base[i])
            # zraw = spectra_fit[:, i]
            zdet = spectra_fit_norm[:, i]

            # # --- Panel 1: Raw Overlay ---
            # ax = axs_raw_flat[idx]
            # ax.plot(x_fit, zraw.real, label="data Re")
            # ax.plot(x_fit, zraw.imag, label="data Im")
            # ax.plot(x_fit, zhat_overlay.real, "--k", label="total fit Re")
            # ax.plot(x_fit, zhat_overlay.imag, "--", color="gray", label="total fit Im")
            # ax.set_title(f"Spectrum {i}")
            # # ax.set_xlabel("Freq. (cm$^{-1}$)")
            # ax.set_box_aspect(1)
            
            # --- Panel 2: Detrended ---
            ax = axs_det_flat[idx]
            ax.plot(x_fit, zdet.real, label="data Re")
            ax.plot(x_fit, zdet.imag, label="data Im")
            ax.plot(x_fit, zhat_det.real, "--k", label="total fit Re")
            ax.plot(x_fit, zhat_det.imag, "--", color="gray", label="total fit Im")
            # ax.set_box_aspect(1)
            
            if show_individual_peaks:
                peaks_det = get_individual_peaks(x_fit, fit_params[i, :], num_peaks, profile, detrend_context,bulk_sample=bulk_sample)
                for p_idx, peak_z in enumerate(peaks_det):
                    ax.plot(x_fit, peak_z.imag, ':', label=f"Peak {p_idx+1} Im")

            ax.set_title(f"Spectrum {i}")
            # ax.set_xlabel("Freq. (cm$^{-1}$)")
            
            if idx == 0:  # Only put the legend on the first subplot to save space
                ax.legend(loc='best', fontsize=5)
            
            # --- Panel 3: Complex Plane ---
            ax = axs_cmp_flat[idx]
            ax.plot(zdet.real, zdet.imag, label="data", lw=1)
            ax.plot(zhat_det.real, zhat_det.imag, "--k", label="fit", lw=1.5)
            ax.set_title(f"Spectrum {i}")
            ax.set_xlabel("Re")
            ax.set_ylabel("Im")
            ax.set_aspect('equal')

        # Turn off any unused subplots in the grid
        for idx in range(len(chunk), plots_per_fig):
            # axs_raw_flat[idx].axis('off')
            axs_det_flat[idx].axis('off')
            axs_cmp_flat[idx].axis('off')
            
        # fig_raw.tight_layout()
        fig_det.tight_layout()
        fig_cmp.tight_layout()

    plt.show()



