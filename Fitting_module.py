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
from phase_utils import baseline_phase_poly

# =============================================================================
# Pure Helper Functions
# =============================================================================

PROFILE_PARAM_COUNT = {"lorentzian": 3, "voigt": 4}

def stack_ri(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return np.concatenate([z.real, z.imag]).astype(float)

def baseline_mask(x, regions):
    x = np.asarray(x, float)
    m = np.zeros_like(x, dtype=bool)
    for lo, hi in regions:
        m |= (x >= lo) & (x <= hi)
    return m



def _norm_per_peak(value, num_peaks, name, pair=False):
    """
    One entry per peak, whether the user gave a scalar/pair or a list.

    For single values (pair=False) a one-item list counts as a scalar, so
    5.0, [5.0] and [5.0, 5.0, 5.0] all mean the same thing for three peaks.
    """
    if pair:
        if value is None:
            return None
        v = list(value)
        if isinstance(v[0], (list, tuple, np.ndarray)):
            if len(v) != num_peaks:
                raise ValueError(f"{name} has {len(v)} entries but there are "
                                 f"{num_peaks} peak_centers.")
            return [tuple(float(a) for a in b) for b in v]
        return [tuple(float(a) for a in v)] * num_peaks
    if np.ndim(value) > 0:
        v = [float(a) for a in np.ravel(value)]
        if len(v) == 1:
            return v * num_peaks
        if len(v) != num_peaks:
            raise ValueError(f"{name} has {len(v)} entries but there are "
                             f"{num_peaks} peak_centers. Give one value for "
                             "all peaks, or one per peak.")
        return v
    return [float(value)] * num_peaks


def build_fit_parameters(
    profile, peak_centers, center_tolerance,
    A_guess, sigma_guess, gamma_guess,
    A_bounds, sigma_bounds, gamma_bounds,
    fwhm_bounds=None, fit_window=None, edge_frac=0.02,
):
    """
    p0 and bounds for the PDM fit.

    Returns (p0, (lower, upper), param_labels) - THREE values now.

    fwhm_bounds : the same per-peak [min, max] FWHM list the standard fitter
        uses. When given it REPLACES gamma_bounds, because for the Lorentz
        oscillator used here gamma is the FWHM. For voigt it bounds the
        Lorentzian part only; sigma keeps sigma_bounds, so the total Voigt
        FWHM can still exceed fwhm_max - a box constraint cannot express the
        combination.
    fit_window : (wmin, wmax). Centre bounds are clipped into it, so a peak is
        never chased outside the data being fitted. The standard fitter already
        does this; the PDM did not.
    edge_frac : how far inside a bound a guess that sits ON that bound is moved.
        A parameter starting exactly on a bound gives 'trf' a one-sided
        derivative and routinely stalls there. A guess that is already interior
        is left alone.

    center_tolerance may be a scalar, a one-item list (both apply to every
    peak) or one value per peak in peak_centers order, e.g. [5, 3, 5, 10].

    NOTE ON GUESSING CENTRES: peak_centers here are oscillator frequencies w0
    in eps(w), NOT the positions where peaks appear in the measured phase. For
    a strong near-field mode the phase maximum sits roughly 10 cm^-1 ABOVE w0,
    so centres read off a phase plot need to be shifted down before use.
    """
    num_peaks = len(peak_centers)
    profile = profile.lower()

    A_g   = _norm_per_peak(A_guess,     num_peaks, "A_guess")
    sig_g = _norm_per_peak(sigma_guess, num_peaks, "sigma_guess")
    gam_g = _norm_per_peak(gamma_guess, num_peaks, "gamma_guess")
    A_b   = _norm_per_peak(A_bounds,     num_peaks, "A_bounds",     pair=True)
    sig_b = _norm_per_peak(sigma_bounds, num_peaks, "sigma_bounds", pair=True)
    gam_b = _norm_per_peak(gamma_bounds, num_peaks, "gamma_bounds", pair=True)

    if fwhm_bounds is not None and len(fwhm_bounds):
        gam_b = _norm_per_peak(fwhm_bounds, num_peaks, "fwhm_bounds", pair=True)

    tol = _norm_per_peak(center_tolerance, num_peaks, "center_tolerance")

    w_lo, w_hi = (float(min(fit_window)), float(max(fit_window))) \
        if fit_window is not None else (-np.inf, np.inf)

    def inside(v, lo, hi):
        """
        Move a guess off a bound, and ONLY off a bound.

        A fraction of the bound span is the wrong step when the span is huge:
        with A_bounds [0, 1e9] a 2% pad would drag A_guess=5e4 up to 2e7. So a
        strictly interior guess is returned untouched, and only a guess sitting
        on (or outside) a bound is moved just inside it.
        """
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return v
        span = hi - lo
        pad = min(edge_frac * span, 0.5 * span)
        eps = 1e-9 * span
        if v < lo or v > hi:
            return float(np.clip(v, lo + pad, hi - pad))
        if v - lo <= eps:
            return float(lo + pad)
        if hi - v <= eps:
            return float(hi - pad)
        return float(v)

    p0, lb, ub, labels, problems = [], [], [], [], []

    for i in range(num_peaks):
        c = float(peak_centers[i])
        c_lo, c_hi = max(w_lo, c - tol[i]), min(w_hi, c + tol[i])
        if not tol[i] > 0:
            problems.append(f"  peak {i+1}: center_tolerance {tol[i]:g} must be > 0")
            c_lo, c_hi = w_lo, w_hi
        elif c_lo >= c_hi:
            problems.append(
                f"  peak {i+1}: centre {c} +/- {tol[i]} does not overlap the fit "
                f"window [{w_lo:.1f}, {w_hi:.1f}]")
            c_lo, c_hi = w_lo, w_hi

        p0 += [inside(A_g[i], *A_b[i]), inside(c, c_lo, c_hi)]
        lb += [A_b[i][0], c_lo]
        ub += [A_b[i][1], c_hi]
        labels += [f"peak_{i+1}_amplitude", f"peak_{i+1}_center"]

        if profile == "lorentzian":
            p0 += [inside(gam_g[i], *gam_b[i])]
            lb += [gam_b[i][0]]; ub += [gam_b[i][1]]
            labels += [f"peak_{i+1}_gamma"]
        elif profile == "voigt":
            p0 += [inside(gam_g[i], *gam_b[i]), inside(sig_g[i], *sig_b[i])]
            lb += [gam_b[i][0], sig_b[i][0]]
            ub += [gam_b[i][1], sig_b[i][1]]
            labels += [f"peak_{i+1}_gamma", f"peak_{i+1}_sigma"]
        else:
            raise ValueError(f"Unknown profile {profile!r}. Use 'lorentzian' or 'voigt'.")

    for k, (v, lo, hi) in enumerate(zip(p0, lb, ub)):
        if not lo < hi:
            problems.append(f"  {labels[k]}: bounds [{lo}, {hi}] need min < max")
    if problems:
        raise ValueError("Cannot build valid PDM fit bounds:\n" + "\n".join(problems))

    return tuple(p0), (tuple(lb), tuple(ub)), tuple(labels)


def at_bound_matrix(fit_params, bounds, fit_success, p0, rtol=1e-6):
    """
    (M, P) boolean: which fitted parameters finished sitting on a bound.

    A clamped parameter was not fitted, it was stopped. R^2 stays high while it
    happens - a railed centre still gives R^2 ~ 0.86 - so this has to be
    reported separately or the failure is invisible.
    """

    lo = np.asarray(bounds[0], float)
    hi = np.asarray(bounds[1], float)
    # tolerance on the parameter's own scale, not the bound span: with
    # A_bounds [0, 1e9] a span-based tolerance is 1000 and flags every A < 1000
    p0 = np.abs(np.asarray(p0, float))
    tol_lo = np.maximum(rtol * np.maximum(p0, np.abs(lo)), 1e-12)
    tol_hi = np.maximum(rtol * np.maximum(p0, np.abs(hi)), 1e-12)
    hit = (np.abs(fit_params - lo) <= tol_lo) | (np.abs(fit_params - hi) <= tol_hi)
    
    
    return hit & np.asarray(fit_success, bool)[:, None]

def _with_instr(p, profile, gamma_instr):
    """Copy of p with the apodization's Lorentzian width added to every gamma.
    gamma is slot 2 of each peak for both 'lorentzian' [A, x0, gamma] and
    'voigt' [A, x0, gamma, sigma], so it is every pcount-th entry from 2."""
    if not gamma_instr:
        return p
    p = np.array(p, dtype=float)
    p[2::PROFILE_PARAM_COUNT[profile.lower()]] += gamma_instr
    return p


def _flatten_phase(z, x, regions, order=2):
    """Remove a polynomial phase fitted on the baseline regions - the same step
    batch_phase_correct applies to every measured S and R, through the same
    function (phase_utils.baseline_phase_poly: per-region unwrap, 2*pi
    alignment, one polynomial). Before 2026-11-17 the model was unwrapped over
    the joined regions in one go while the data were unwrapped per region."""
    p = baseline_phase_poly(x, z, regions, order)
    return z * np.exp(-1j * np.polyval(p, x))


def _model_vec(p, x_full, win, detrend_context, profile, bulk_sample, d_film_nm,
               eps_inf, gamma_instr, free_scale, y_target=None, harmonic=2):
    """The model exactly as the data were processed. Returns (vector, scale)."""
    z = PDM.PDM_fitting(x_full, *_with_instr(p, profile, gamma_instr), eps_inf=eps_inf,
                        profile=profile, bulk_sample=bulk_sample, d_film_nm=d_film_nm,
                        harmonic=harmonic)
    z = _flatten_phase(z, x_full, detrend_context[6], order=detrend_context[5])
    zcorr, _, _ = detrend_real_keep_im_fast(z, x_full, detrend_context)
    m = stack_ri(zcorr[win])
    c = 1.0
    if free_scale and y_target is not None:
        c = float(m @ y_target) / float(m @ m)     # closed-form real scale
    return c * m, c

def detrend_real_keep_im_fast(z, x, detrend_context):
    """Handles 1D and 2D arrays using explicit context instead of globals."""
    x_base, n_base, sum_x, ols_denom, mask = detrend_context[:5]
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
    (i, y_target, p0_i, bounds_i, x_full, win, detrend_context,
     profile, max_nfev, bulk_sample, d_film_nm, eps_inf,
     gamma_instr, free_scale, harmonic) = args

    if not np.all(np.isfinite(y_target)):
        return i, False, "non-finite values in spectrum", np.nan, None, np.nan

    def model_ri_detrended_wrapper(x, *p):
        return _model_vec(p, x_full, win, detrend_context, profile, bulk_sample,
                          d_film_nm, eps_inf, gamma_instr, free_scale, y_target, harmonic)[0]

    try:
        popt, pcov = curve_fit(
            model_ri_detrended_wrapper,
            x_full[win],
            y_target,
            p0=p0_i,
            bounds=bounds_i,
            method="trf",
            jac="2-point",
            max_nfev=max_nfev,
        )

        # --- R-squared Calculation ---
        y_pred, c = _model_vec(popt, x_full, win, detrend_context, profile, bulk_sample,
                               d_film_nm, eps_inf, gamma_instr, free_scale, y_target, harmonic)
        ss_res = np.sum((y_target - y_pred)**2)
        ss_tot = np.sum((y_target - np.mean(y_target))**2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)

        return i, True, popt, r_squared, perr, c
    except Exception as e:
        return i, False, str(e), np.nan, None, np.nan

# =============================================================================
# Shared-envelope background (pdm_background = "envelope", added 2026-11-17)
# =============================================================================
#
#   S/R(w) = c * B(w) * PDM(w; oscillators)
#   B(w)   = P_amp(t) * exp(i * P_phase(t)),   t = (w - w_mid) / w_half
#
# w_mid and w_half are the centre and half-width of the envelope range (the span
# of the baseline regions), so t runs from -1 to +1 for any dataset and the
# polynomial coefficients stay O(1). The scaling does not change the fit: a
# polynomial in t describes the same curves as one in w.
#
# B is a smooth background that MULTIPLIES the whole spectrum (reference /
# laser-spectrum mismatch, far-field factor). It is fitted once on the array
# average, together with the oscillators, over the span of the baseline regions,
# and then held fixed: per pixel only the complex scale c (about 1) and the
# oscillator parameters are free. No Re/Im detrend and no phase flattening is
# applied to data or model - B's polynomial phase takes over what the per-
# spectrum phase correction did. With envelope_shared=False every pixel fits
# all of B itself (the cross-check for gamma maps).

def envelope_eval(t, amp_coef, phase_coef):
    """B(t) = P_amp(t) * exp(i P_phase(t)) (np.polyval coefficient order)."""
    return np.polyval(amp_coef, t) * np.exp(1j * np.polyval(phase_coef, t))


def _pdm_eval(p, x, pdm_kw):
    """PDM S/R for oscillator parameters p (intrinsic gammas; the apodization
    width is added here, as in _model_vec)."""
    return PDM.PDM_fitting(x, *_with_instr(p, pdm_kw["profile"], pdm_kw["gamma_instr"]),
                           eps_inf=pdm_kw["eps_inf"], profile=pdm_kw["profile"],
                           bulk_sample=pdm_kw["bulk_sample"], d_film_nm=pdm_kw["d_film_nm"],
                           harmonic=pdm_kw["harmonic"])


def _zero_amplitudes(p, profile, keep=None):
    """Copy of p with every oscillator amplitude set to 0, except peak `keep`."""
    q = np.array(p, dtype=float)
    pc = PROFILE_PARAM_COUNT[profile.lower()]
    for j in range(len(q) // pc):
        if keep is None or j != keep:
            q[j * pc] = 0.0
    return q


def _resonant_r2(z, model, base, win):
    """R^2 of the resonant part inside the fit window: data and model divided
    by the off-resonance background (c*B*PDM with every amplitude 0), minus 1,
    Re and Im stacked. Measures the band, not the baseline, so it is on the
    same footing as the detrend-mode R^2."""
    d = stack_ri(z[win] / base[win] - 1.0)
    m = stack_ri(model[win] / base[win] - 1.0)
    ss_tot = np.sum((d - d.mean()) ** 2)
    return 1.0 - np.sum((d - m) ** 2) / ss_tot if ss_tot > 0 else 0.0


def _fit_envelope_full(x, t, z, p0_osc, bounds_osc, ka, kp, pdm_kw, max_nfev,
                       start=None):
    """
    Fit B(t; all coefficients) * PDM(oscillators) to ONE spectrum.
    start : optional (amp_coef, phase_coef) initial values; otherwise they are
            taken from z / PDM(p0_osc).
    Returns (popt_osc, amp_coef, phase_coef, perr_osc).
    """
    n_osc = len(p0_osc)
    if start is None:
        ratio = z / _pdm_eval(p0_osc, x, pdm_kw)
        amp0 = np.polyfit(t, np.abs(ratio), ka)
        ph0 = np.polyfit(t, np.unwrap(np.angle(ratio)), kp)
    else:
        amp0, ph0 = start

    def f(_x, *q):
        q = np.asarray(q)
        m = envelope_eval(t, q[n_osc:n_osc + ka + 1], q[n_osc + ka + 1:]) \
            * _pdm_eval(q[:n_osc], x, pdm_kw)
        return stack_ri(m)

    lb = list(bounds_osc[0]) + [-np.inf] * (ka + kp + 2)
    ub = list(bounds_osc[1]) + [np.inf] * (ka + kp + 2)
    popt, pcov = curve_fit(f, x, stack_ri(z), p0=list(p0_osc) + list(amp0) + list(ph0),
                           bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=max_nfev)
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(len(popt), np.nan)
    return popt[:n_osc], popt[n_osc:n_osc + ka + 1], popt[n_osc + ka + 1:], perr[:n_osc]


def fit_single_pixel_envelope(args):
    """
    Worker for one spectrum in envelope mode. Must stay at module level.

    shared=True : params = oscillators + (Re c, Im c), B fixed.
    shared=False: params = oscillators + all envelope coefficients.
    Returns (i, success, popt_osc | message, r2, perr_osc, c, coefs | None).
    c is the pixel's background relative to the shared/average B at t = 0.
    """
    (i, z, p0_osc, bounds_osc, x, t, win, B, amp_avg, ph_avg, ka, kp,
     shared, pdm_kw, max_nfev) = args
    if not np.all(np.isfinite(z)):
        return i, False, "non-finite values in spectrum", np.nan, None, np.nan, None
    n_osc = len(p0_osc)
    try:
        if shared:
            m0 = B * _pdm_eval(p0_osc, x, pdm_kw)
            c0 = np.vdot(m0, z) / np.vdot(m0, m0)

            def f(_x, *q):
                q = np.asarray(q)
                return stack_ri((q[n_osc] + 1j * q[n_osc + 1]) * B * _pdm_eval(q[:n_osc], x, pdm_kw))

            lb = list(bounds_osc[0]) + [-np.inf, -np.inf]
            ub = list(bounds_osc[1]) + [np.inf, np.inf]
            popt, pcov = curve_fit(f, x, stack_ri(z), p0=list(p0_osc) + [c0.real, c0.imag],
                                   bounds=(lb, ub), method="trf", x_scale="jac",
                                   max_nfev=max_nfev)
            p_osc = popt[:n_osc]
            c = popt[n_osc] + 1j * popt[n_osc + 1]
            env = c * B
            coefs = None
            perr = np.sqrt(np.diag(pcov))[:n_osc] if np.all(np.isfinite(pcov)) \
                else np.full(n_osc, np.nan)
        else:
            p_osc, amp, ph, perr = _fit_envelope_full(
                x, t, z, p0_osc, bounds_osc, ka, kp, pdm_kw, max_nfev, start=(amp_avg, ph_avg))
            env = envelope_eval(t, amp, ph)
            c = env[np.argmin(np.abs(t))] / B[np.argmin(np.abs(t))]
            coefs = np.concatenate([amp, ph])
        model = env * _pdm_eval(p_osc, x, pdm_kw)
        base = env * _pdm_eval(_zero_amplitudes(p_osc, pdm_kw["profile"]), x, pdm_kw)
        r2 = _resonant_r2(z, model, base, win)
        return i, True, p_osc, r2, perr, c, coefs
    except Exception as e:
        return i, False, str(e), np.nan, None, np.nan, None


def envelope_curves(results, i, profile, bulk_sample, d_film_nm, eps_inf):
    """
    For plotting one pixel of an envelope-mode result, inside the fit window:
    (x, data, model, [peak_1, peak_2, ...]), each as S/R / background - 1.
    """
    env = results["envelope"]
    x, t, win = env["x"], env["t"], results["win"]
    pdm_kw = dict(profile=profile, gamma_instr=results.get("gamma_instr", 0.0), eps_inf=eps_inf,
                  bulk_sample=bulk_sample, d_film_nm=d_film_nm, harmonic=results.get("harmonic", 2))
    if env["shared"]:
        E = env["c"][i] * env["B"]
    else:
        k = env["order"][0] + 1
        E = envelope_eval(t, env["coefs"][i, :k], env["coefs"][i, k:])
    p = results["fit_params"][i, :]
    base = E * _pdm_eval(_zero_amplitudes(p, profile), x, pdm_kw)
    model = E * _pdm_eval(p, x, pdm_kw)
    n_pk = len(p) // PROFILE_PARAM_COUNT[profile.lower()]
    peaks = [(E * _pdm_eval(_zero_amplitudes(p, profile, keep=j), x, pdm_kw) / base - 1.0)[win]
             for j in range(n_pk)]
    data = results["spectra_env"][:, i]
    return x[win], (data / base - 1.0)[win], (model / base - 1.0)[win], peaks


def _process_envelope(xcm_full, spectra, p0, bounds, param_labels, fit_window,
                      baseline_regions, envelope_order, envelope_shared, pdm_kw,
                      max_nfev, tic):
    """Envelope-mode body of process_spectra_array. spectra are already
    conjugated (absorption = +Im)."""
    N, M = spectra.shape
    regions = [r for r in baseline_regions if len(r) == 2]
    lo, hi = min(min(r) for r in regions), max(max(r) for r in regions)
    rng = (xcm_full >= lo) & (xcm_full <= hi)
    x = xcm_full[rng]
    z_all = spectra[rng, :]
    w_mid = 0.5 * (x.min() + x.max())
    w_half = 0.5 * (x.max() - x.min()) or 1.0      # t = -1 ... +1 over the range
    t = (x - w_mid) / w_half
    win = np.ones_like(x, dtype=bool) if fit_window is None else \
        (x >= min(fit_window)) & (x <= max(fit_window))
    ka, kp = (int(v) for v in envelope_order)

    finite = np.all(np.isfinite(z_all), axis=0)
    if not finite.any():
        raise ValueError("No spectrum is finite over the envelope range.")
    z_avg = z_all[:, finite].mean(axis=1)

    print(f"Envelope background: amplitude order {ka}, phase order {kp}, "
          f"{'shared by all spectra' if envelope_shared else 'fitted per spectrum'}, "
          f"range {x.min():.1f}-{x.max():.1f} cm^-1 ({x.size} points)")

    # 1. Background shape (and starting oscillators) from the average spectrum
    p_avg, amp_avg, ph_avg, _ = _fit_envelope_full(x, t, z_avg, p0, bounds, ka, kp,
                                                   pdm_kw, max_nfev)
    B = envelope_eval(t, amp_avg, ph_avg)

    # 2. Every spectrum, starting from the average fit
    tasks = [(i, z_all[:, i], tuple(p_avg), bounds, x, t, win, B, amp_avg, ph_avg,
              ka, kp, bool(envelope_shared), pdm_kw, max_nfev) for i in range(M)]

    fit_params = np.full((M, len(p0)), np.nan, dtype=float)
    fit_errors = np.full((M, len(p0)), np.nan, dtype=float)
    fit_r2 = np.full(M, np.nan, dtype=float)
    fit_success = np.zeros(M, dtype=bool)
    c_all = np.full(M, np.nan, dtype=complex)
    coefs = None if envelope_shared else np.full((M, ka + kp + 2), np.nan)
    failures = []
    for i, ok, res, r2, perr, c, cf in _run_pool(fit_single_pixel_envelope, tasks, M):
        fit_success[i] = ok
        if ok:
            fit_params[i, :] = res
            fit_errors[i, :] = perr
            fit_r2[i] = r2
            c_all[i] = c
            if cf is not None:
                coefs[i, :] = cf
        else:
            failures.append(res)

    _report_fit_summary(fit_success, failures, tic, M)
    at_bound = _report_at_bound(fit_params, bounds, fit_success, p0, param_labels, M)

    # Resonant part (S/R / background - 1) of every spectrum, for the plots
    profile = pdm_kw["profile"]
    res_norm = np.full((int(win.sum()), M), np.nan, dtype=complex)
    for i in np.flatnonzero(fit_success):
        E = c_all[i] * B if envelope_shared else \
            envelope_eval(t, coefs[i, :ka + 1], coefs[i, ka + 1:])
        base = E * _pdm_eval(_zero_amplitudes(fit_params[i], profile), x, pdm_kw)
        res_norm[:, i] = (z_all[:, i] / base - 1.0)[win]

    return {
        "background": "envelope",
        "x_fit": x[win],
        "spectra_fit": z_all[win, :],
        "spectra_fit_norm": res_norm,          # S/R / background - 1, fit window
        "spectra_env": z_all,                  # conjugated S/R over the envelope range
        "fit_params": fit_params,
        "fit_errors": fit_errors,
        "fit_scale": np.abs(c_all),            # |c|: pixel background / array average
        "fit_bg_phase_deg": np.degrees(np.angle(c_all)),
        "gamma_instr": pdm_kw["gamma_instr"],
        "free_scale": False,
        "harmonic": pdm_kw["harmonic"],
        "fit_success": fit_success,
        "fit_r2": fit_r2,
        "re_drift_mb": None,
        "im_base": None,
        "detrend_context": None,
        "p0": p0,
        "param_labels": param_labels,
        "at_bound": at_bound,
        "M": M,
        "x_full": x,
        "win": win,
        "envelope": {
            "x": x, "t": t, "w_mid": w_mid, "t_scale": w_half,
            "order": (ka, kp), "shared": bool(envelope_shared),
            "amp_coef": amp_avg, "phase_coef": ph_avg, "B": B,
            "c": c_all, "coefs": coefs,
            "average_spectrum": z_avg, "average_params": p_avg,
        },
    }


# =============================================================================
# Main Pipeline Function
# =============================================================================


def _run_pool(worker, tasks, M):
    """Run worker over tasks in parallel, with the progress bar. Yields results
    in completion order (each carries its own index)."""
    num_cores = min(60, max(1, (os.cpu_count() or 2) - 1))
    optimal_chunksize = max(1, M // (num_cores * 4))
    print(f"Starting parallel fit across {num_cores} cores...")
    completed = 0
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in executor.map(worker, tasks, chunksize=optimal_chunksize):
            yield result
            completed += 1
            if completed % max(1, M // 100) == 0 or completed == M:
                percent_float = completed / M
                filled_length = int(20 * percent_float)
                bar = "=" * filled_length + "-" * (20 - filled_length)
                print(f"\rFitting Spectra (PDM): [{bar}] {int(percent_float * 100)}% ({completed}/{M})", end="", flush=True)


def _report_fit_summary(fit_success, failures, tic, M):
    toc = time.perf_counter()
    print(f"\n\nSuccessful fits: {np.sum(fit_success)} / {M}")
    if failures:
        from collections import Counter
        for msg, k in Counter(failures).most_common():
            print(f"\033[93m    {k} x {msg}\033[0m")
    print(f"Elapsed time: {toc - tic:.4f} seconds")


def _report_at_bound(fit_params, bounds, fit_success, p0, param_labels, M):
    """Parameters that finished ON a bound were clamped, not fitted. R^2 does not
    reveal this, so report it explicitly. Returns the (M, P) matrix."""
    at_bound = at_bound_matrix(fit_params, bounds, fit_success, p0=p0)
    if at_bound.any():
        print("\033[93mParameters clamped to a bound (the fit could not move them):\033[0m")
        for k, nm in enumerate(param_labels):
            n_hit = int(at_bound[:, k].sum())
            if n_hit:
                print(f"\033[93m    {nm:22s} {n_hit:4d}/{M} pixels   "
                      f"bounds [{bounds[0][k]:g}, {bounds[1][k]:g}]\033[0m")
        if any(at_bound[:, k].sum() for k, nm in enumerate(param_labels)
               if nm.endswith("_center")):
            print("\033[93m  Widen center_tolerance or shift the guesses down.\033[0m")
    return at_bound


def process_spectra_array(
    data_input,
    profile="voigt",
    peak_centers=None,
    center_tolerance=2.0,
    fit_window=None,
    baseline_regions=None,
    max_nfev=20000,
    # Strictly flat kwargs
    eps_inf=2.5, A_guess=50000.0, sigma_guess=6.0, gamma_guess=6.0,
    A_bounds=(1000.0, 60000.0), 
    sigma_bounds=(4.0, 8.0), gamma_bounds=(4.0, 8.0),
    fwhm_bounds=None,
    bulk_sample=False,
    d_film_nm=300.0,
    gamma_instr=0.0,
    free_scale=False,
    harmonic=2,
    phase_order=2,
    background="detrend",
    envelope_order=(3, 2),
    envelope_shared=True,
):
    """
    PDM fit of every spectrum in data_input (complex S/R, the fitter conjugates).

    background : how the off-resonance background is handled
        "detrend"  - (original) phase flattening + straight line subtracted from
                     Re + constant subtracted from Im, applied to data and model;
                     residual over fit_window.
        "envelope" - S/R = c * B(w) * PDM: a smooth multiplicative background B
                     (polynomial amplitude x exp(i polynomial phase), orders
                     envelope_order), fitted with the oscillators over the span
                     of baseline_regions. envelope_shared=True fits B once on the
                     array average and keeps its shape for every spectrum (only
                     a complex scale c per spectrum); False fits B per spectrum.
                     fit_window then only sets the R^2 window and the plots.
                     free_scale and phase_order are not used.
    """
    if not peak_centers:
        raise ValueError("peak_centers list must be provided.")
        
    if profile not in PROFILE_PARAM_COUNT:
        raise ValueError(f"Unknown PROFILE={profile!r}.")
    background = str(background).lower()
    if background not in ("detrend", "envelope"):
        raise ValueError(f"background {background!r}: use 'detrend' or 'envelope'.")

    tic = time.perf_counter()

    # --- Load ---
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
            # spectra = spectra_raw.fillna(0j).to_numpy(np.complex128)
            spectra = spectra_raw.to_numpy(np.complex128)
            
    else:
        raise TypeError("data_input must be a file path (str/Path) or a pandas DataFrame.")
    
    N, M = spectra.shape
    spectra = spectra.real - 1j * spectra.imag
    print(f"Loaded x: {xcm_full.shape}, spectra: {spectra.shape} (N={N}, M={M})")

    # Build Parameters using only the flat inputs
    p0, bounds, param_labels = build_fit_parameters(
        profile=profile,
        peak_centers=peak_centers,
        center_tolerance=center_tolerance,
        A_guess=A_guess, sigma_guess=sigma_guess, gamma_guess=gamma_guess,
        A_bounds=A_bounds, sigma_bounds=sigma_bounds, gamma_bounds=gamma_bounds,
        fwhm_bounds=fwhm_bounds,
        fit_window=fit_window,
    )

    if background == "envelope":
        if free_scale:
            print("Note: pdm_free_scale is ignored with the envelope background "
                  "(its complex scale c replaces it).")
        pdm_kw = dict(profile=profile, gamma_instr=gamma_instr, eps_inf=eps_inf,
                      bulk_sample=bulk_sample, d_film_nm=d_film_nm, harmonic=harmonic)
        return _process_envelope(xcm_full, spectra, p0, bounds, param_labels, fit_window,
                                 baseline_regions, envelope_order, envelope_shared,
                                 pdm_kw, max_nfev, tic)

    # --- background = "detrend" (unchanged method) ---
    # Detrend over the whole carried range so baseline_regions are independent
    # of the fit window.
    base_m = baseline_mask(xcm_full, baseline_regions)
    x_base = xcm_full[base_m]
    n_base = len(x_base)
    if n_base < 2:
        raise ValueError(
            f"baseline_regions give {n_base} points inside the loaded range "
            f"[{xcm_full.min():.1f}, {xcm_full.max():.1f}] - need at least 2."
        )
    sum_x = np.sum(x_base)
    sum_x2 = np.sum(x_base**2)
    ols_denom = n_base * sum_x2 - sum_x**2
    # the phase-correction order and the regions travel with the baseline mask,
    # so every model evaluation (fit and plots) flattens exactly like the data:
    # (x_base, n_base, sum_x, ols_denom, mask, phase_order, regions)
    detrend_context = (x_base, n_base, sum_x, ols_denom, base_m, int(phase_order),
                       tuple(tuple(float(v) for v in r) for r in baseline_regions if len(r) == 2))

    spectra_norm, re_drift_mb, im_base = detrend_real_keep_im_fast(
        spectra, xcm_full, detrend_context)

    # The residual is evaluated only inside the fit window.
    if fit_window is None:
        win = np.ones_like(xcm_full, dtype=bool)
    else:
        win = (xcm_full >= min(fit_window)) & (xcm_full <= max(fit_window))

    x_fit = xcm_full[win]
    spectra_fit = spectra[win, :]
    spectra_fit_norm = spectra_norm[win, :]

    # Parallel Execution Setup
    tasks = [
        (i, stack_ri(spectra_fit_norm[:, i]), p0, bounds, xcm_full, win, 
         detrend_context, profile, max_nfev,bulk_sample, d_film_nm, eps_inf,
         gamma_instr, free_scale, harmonic)
        for i in range(M)
    ]
    fit_params = np.full((M, len(p0)), np.nan, dtype=float)
    fit_errors = np.full((M, len(p0)), np.nan, dtype=float)
    fit_scale = np.full(M, np.nan, dtype=float)
    fit_success = np.zeros(M, dtype=bool)
    fit_r2 = np.full(M, np.nan, dtype=float)
    failures = []

    for i, success, res_data, r2_val, perr, c in _run_pool(fit_single_pixel, tasks, M):
        fit_success[i] = success
        if success:
            fit_params[i, :] = res_data
            fit_errors[i, :] = perr
            fit_scale[i] = c
            fit_r2[i] = r2_val
        else:
            failures.append(res_data)

    _report_fit_summary(fit_success, failures, tic, M)
    at_bound = _report_at_bound(fit_params, bounds, fit_success, p0, param_labels, M)

    # Return a dictionary containing everything needed for further analysis/plotting
    return {
        "background": "detrend",
        "x_fit": x_fit,
        "spectra_fit": spectra_fit,
        "spectra_fit_norm": spectra_fit_norm,
        "fit_params": fit_params,
        "fit_errors": fit_errors,
        "fit_scale": fit_scale,
        "gamma_instr": gamma_instr,
        "free_scale": free_scale,
        "harmonic": harmonic,
        "fit_success": fit_success,
        "fit_r2": fit_r2,
        "re_drift_mb": re_drift_mb,
        "im_base": im_base,
        "detrend_context": detrend_context,
        "p0": p0,
        "param_labels": param_labels,
        "at_bound": at_bound,
        "M": M,
        "x_full": xcm_full,
        "win": win,
    }

# =============================================================================
# Plotting Functions
# =============================================================================



def get_individual_peaks(x_full, win, popt, num_peaks, profile, detrend_context,bulk_sample,d_film_nm, eps_inf,
                         gamma_instr=0.0, scale=1.0, harmonic=2):
    """Generates detrended line shapes for individual peaks."""
    pcount = PROFILE_PARAM_COUNT[profile.lower()]
    base_idx = 0  # indices 0 and 1 are eps and slope
    
    peaks_detrended = []
    
    for i in range(num_peaks):
        popt_single = np.copy(popt)
        
        # Zero out amplitudes of all other peaks
        for j in range(num_peaks):
            if i != j:
                A_idx = base_idx + j * pcount
                popt_single[A_idx] = 0.0
                
        # Evaluate single peak model and detrend
        z_single = PDM.PDM_fitting(x_full, *_with_instr(popt_single, profile, gamma_instr),
                                   eps_inf=eps_inf, profile=profile,
                                   bulk_sample=bulk_sample, d_film_nm=d_film_nm,
                                   harmonic=harmonic)
        z_single = _flatten_phase(z_single, x_full, detrend_context[6], order=detrend_context[5])
        z_single_det, _, _ = detrend_real_keep_im_fast(z_single, x_full, detrend_context)
        peaks_detrended.append(scale * z_single_det[win])
        
    return peaks_detrended




def plot_individual_fits(results, profile="voigt", num_peaks=2, plot_every=10, show_individual_peaks=True, grid_n=5, bulk_sample=False, d_film_nm=300.0, eps_inf=2.5):
    """Plots grouped manual previews for individual pixels on n x n grids."""
    if not plot_every:
        return

    envelope_mode = results.get("background") == "envelope"
    x_fit = results["x_fit"]
    # spectra_fit = results["spectra_fit"]
    spectra_fit_norm = results["spectra_fit_norm"]
    fit_params = results["fit_params"]
    fit_success = results["fit_success"]
    re_drift_mb = results["re_drift_mb"]
    # im_base = results["im_base"]
    detrend_context = results["detrend_context"]
    M = results["M"]
    x_full = results["x_full"]
    win    = results["win"]

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
        fig_det, axs_det = plt.subplots(grid_n, grid_n, figsize=(4 * grid_n, 3.5 * grid_n))
        fig_cmp, axs_cmp = plt.subplots(grid_n, grid_n, figsize=(4 * grid_n, 4 * grid_n))
        
        # fig_raw.suptitle("Raw Overlay", fontsize=16)
        fig_det.suptitle("S/R \u00f7 background \u2212 1 (with peaks)" if envelope_mode
                         else "Detrended (with peaks)", fontsize=16)
        fig_cmp.suptitle("Complex Plane", fontsize=16)

        # Flatten the axis arrays to iterate over them easily
        axs_det_flat = np.array(axs_det).flatten()
        axs_cmp_flat = np.array(axs_cmp).flatten()

        for idx, i in enumerate(chunk):
            g_instr = results.get("gamma_instr", 0.0)
            if envelope_mode:
                # resonant part: data and model divided by c*B*PDM(no oscillator)
                _, zdet, zhat_det, peaks_env = envelope_curves(
                    results, i, profile, bulk_sample, d_film_nm, eps_inf)
            else:
                c = results["fit_scale"][i] if results.get("free_scale") else 1.0
                zhat_raw = PDM.PDM_fitting(x_full, *_with_instr(fit_params[i, :], profile, g_instr),
                                           eps_inf=eps_inf, profile=profile,
                                           bulk_sample=bulk_sample, d_film_nm=d_film_nm,
                                           harmonic=results.get("harmonic", 2))
                zhat_raw = _flatten_phase(zhat_raw, x_full, detrend_context[6], order=detrend_context[5])
                zhat_det = c * detrend_real_keep_im_fast(zhat_raw, x_full, detrend_context)[0][win]
                zdet = spectra_fit_norm[:, i]
            
            # --- Panel 2: Detrended ---
            ax = axs_det_flat[idx]
            ax.plot(x_fit, zdet.real, label="data Re")
            ax.plot(x_fit, zdet.imag, label="data Im")
            ax.plot(x_fit, zhat_det.real, "--k", label="total fit Re")
            ax.plot(x_fit, zhat_det.imag, "--", color="gray", label="total fit Im")
            # ax.set_box_aspect(1)
            
            if show_individual_peaks:
                if envelope_mode:
                    peaks_det = peaks_env
                else:
                    peaks_det = get_individual_peaks(x_full, win, fit_params[i, :], num_peaks,
                                                     profile, detrend_context,
                                                     bulk_sample=bulk_sample, d_film_nm=d_film_nm, eps_inf=eps_inf,
                                                     gamma_instr=g_instr, scale=c,
                                                     harmonic=results.get("harmonic", 2))
                
                for p_idx, peak_z in enumerate(peaks_det):
                    ax.plot(x_fit, peak_z.imag, ':', label=f"Peak {p_idx+1} Im")

            ax.set_title(f"Spectrum {i}")
            
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



