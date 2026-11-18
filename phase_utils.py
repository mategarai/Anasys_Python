#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_utils.py - the one place where the baseline phase of a complex spectrum
is fitted.

Used by
  snom_utils.phase_correct_spectrum   every measured sample and reference spectrum
  Fitting_module._flatten_phase       the PDM model in pdm_background "detrend"

Both used to unwrap the phase differently (per region with 2*pi alignment for
the data, the joined regions in one go for the model). Sharing this function
makes the data and the model go through exactly the same step.

Pure numpy, so Fitting_module can import it without the GUI and xarray
dependencies of snom_utils.
"""

import numpy as np


def baseline_phase_poly(wavenumbers, complex_spectrum, fit_regions, order=2):
    """
    Polynomial coefficients (np.polyfit order, highest power first) of the phase
    of `complex_spectrum` over `fit_regions`.

    Each region is unwrapped on its own. Every later region is then given the
    whole multiple of 2*pi that lets ONE polynomial of `order` fit all regions
    best. Unwrapping the concatenated regions would treat the gap between them
    as a single step, and extrapolating a fit to the first region across the
    gap can pick the wrong branch when that region is narrow.

    Raises ValueError when no region contains any point.
    """
    wavenumbers = np.asarray(wavenumbers, dtype=float)
    complex_spectrum = np.asarray(complex_spectrum)

    seg_w, seg_p = [], []
    for wmin, wmax in fit_regions:
        m = (wavenumbers >= wmin) & (wavenumbers <= wmax)
        if m.sum() == 0:
            continue
        seg_w.append(wavenumbers[m])
        seg_p.append(np.unwrap(np.angle(complex_spectrum[m])))
    if not seg_w:
        raise ValueError(
            f"None of the phase_fit_regions {list(fit_regions)} contains a data point "
            f"(data span {wavenumbers.min():.1f}-{wavenumbers.max():.1f} cm^-1)."
        )

    for k in range(1, len(seg_w)):
        x = np.concatenate(seg_w[:k + 1])
        deg = min(order, len(x) - 1)
        best = None
        for n2pi in range(-4, 5):
            y = np.concatenate(seg_p[:k] + [seg_p[k] + 2 * np.pi * n2pi])
            rss = np.sum((y - np.polyval(np.polyfit(x, y, deg), x)) ** 2)
            if best is None or rss < best[0]:
                best = (rss, n2pi)
        seg_p[k] = seg_p[k] + 2 * np.pi * best[1]

    return np.polyfit(np.concatenate(seg_w), np.concatenate(seg_p), order)
