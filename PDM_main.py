"""
PDM_main.py — Physically correct dielectric model for nano-FTIR point dipole model

Main change vs your previous version:
  - The sample dielectric function is built from COMPLEX oscillators (KK-consistent),
    so Re[ε(ω)] is dispersive and Im[ε(ω)] is absorptive.

Supported oscillator broadening models (choose via profile=...):
  1) profile="lorentzian"  : Complex Lorentz oscillator
        ε(ω) = ε_inf + Σ  S / (ω0^2 - ω^2 - i γ ω)
     Parameters per oscillator: (S, w0, gamma)

  2) profile="voigt"    : Inhomogeneous (Gaussian) broadening of the Lorentz oscillator
     implemented as a discrete convolution over ω0:
        ε(ω) = ε_inf + Σ ∫ dω0' G(ω0'; ω0, sigma) * S/(ω0'^2 - ω^2 - i γ ω)
     Parameters per oscillator: (S, w0, gamma, sigma)
     (sigma is the std dev of the Gaussian distribution of resonance centers)

Notes:
  - x is in cm^-1 (wavenumber). We keep your original form with ω0^2 - ω^2 - iγω.
  - You can easily add multiple oscillators by stacking parameters.
  - The rest of the PDM (beta, alpha, lock-in demod) remains as in your optimized code.

Performance:
  - LockIn is vectorized and fast
  - Pt interpolators cached
  - For voigt: we keep the number of quadrature points small (N_INH) and vectorize it.
"""

import numpy as np
import numpy.fft as fft
from scipy.interpolate import interp1d

# =============================================================================
# Tip / tapping parameters (unchanged)
# =============================================================================
Tip_rad  = 20        # nm
Tip_amp  = 50 / 2    # nm
Tip_freq = 250e3     # Hz
sampling = 20        # samples per cycle

Time  = np.arange(0, 30 / Tip_freq, (1 / Tip_freq) / sampling)
Zdist = Tip_amp * np.cos(np.pi * Tip_freq * Time) + Tip_amp  # keep your original
Tip_h = Tip_rad + Zdist

inv_h3 = 1.0 / (Tip_h.astype(float) ** 3)
dt = float(Time[1] - Time[0])

FFT_freq = fft.fftshift(fft.fftfreq(Time.size, dt))
idx_2H = int(np.argmin(np.abs(FFT_freq - 2.0 * Tip_freq)))


# Pre-scale the tip distance array
inv_h3_scaled = inv_h3 / (16.0 * np.pi)

# Cache for Voigt distribution weights
_voigt_cache = {}
def _get_voigt_uw(N_INH, span):
    key = (N_INH, span)
    if key not in _voigt_cache:
        u = np.linspace(-span, span, int(N_INH))[None, :, None]
        w = np.exp(-0.5 * u**2)
        w /= np.sum(w)
        _voigt_cache[key] = (u, w)
    return _voigt_cache[key]
# =============================================================================
# Gold dielectric function (as in your script)
# =============================================================================
gomegaf = 8.06e2
gomegat = 2.16e2
gomegap = 6.20e4

def GoldE(ROIcm):
    ROIcm = np.asarray(ROIcm, dtype=float)
    gEps1 = -(gomegap**2) / (ROIcm**2 + gomegat**2)
    gEps2 = (gomegap**2) * gomegaf / (1e6 * ((ROIcm / 100.0) ** 3) + ROIcm * (gomegat**2))
    return gEps1, gEps2

# =============================================================================
# Platinum dielectric function (cached interpolators)
# =============================================================================
wavenumbers = np.array(
    [600, 8.07E+02, 1.05E+03, 1.21E+03, 1.37E+03, 1.61E+03,
     2.42E+03, 3.23E+03, 4.03E+03, 4.84E+03, 5.65E+03, 6.45E+03,
     8.07E+03, 1.21E+04, 1.61E+04],
    dtype=float
)
E1 = np.array(
    [-2400, -1.83E+03, -1.25E+03, -9.04E+02, -6.92E+02, -5.39E+02,
     -2.46E+02, -1.22E+02, -4.42E+01, -1.92E+01, -1.40E+01, -2.14E+01,
     -2.58E+01, -1.72E+01, -1.13E+01],
    dtype=float
)
E2 = np.array(
    [950, 1.18E+03, 7.28E+02, 5.10E+02, 3.68E+02, 2.83E+02,
     1.27E+02, 6.40E+01, 6.03E+01, 6.93E+01, 7.80E+01, 7.48E+01,
     5.63E+01, 2.96E+01, 1.87E+01],
    dtype=float
)

_pt_f1 = interp1d(wavenumbers, E1, kind="linear", bounds_error=False, fill_value="extrapolate")
_pt_f2 = interp1d(wavenumbers, E2, kind="linear", bounds_error=False, fill_value="extrapolate")

def PlatinumE(ROIcm):
    ROIcm = np.asarray(ROIcm, dtype=float)
    return _pt_f1(ROIcm), _pt_f2(ROIcm)

# =============================================================================
# Utilities
# =============================================================================
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y

# =========================
# PDM_main.py (drop-in add-on)
# Adds a thin-film-on-gold reflection model (air/film/gold) and uses COMPLEX r_p
# =========================

# --- Fresnel helpers (complex, p-polarized) ---

def _kz_in_medium(eps_j, eps0, sin_theta, k0):
    """
    k_z,j = k0 * sqrt(eps_j - eps0 * sin^2(theta))
    where k0 = 2*pi/lambda in the SAME length units as the thickness d.
    """
    return k0 * np.sqrt(eps_j - eps0 * (sin_theta**2))

def fresnel_rp_halfspace(eps2, theta_deg=30.0, eps0=1.0):
    """
    Complex p-polarized Fresnel field reflection coefficient r_p for:
      medium 0 (eps0) -> medium 2 (eps2)
    Using wavelength-independent form via k_z ratios (k0 cancels).
    """
    theta = np.deg2rad(theta_deg)
    sin_t = np.sin(theta)

    eps0_c = np.complex128(eps0)
    eps2 = np.asarray(eps2, dtype=np.complex128)

    # Use "kz/k0" to avoid caring about absolute wavelength
    kz0 = np.sqrt(eps0_c - eps0_c * sin_t**2)   # = sqrt(eps0)*cos(theta) for eps0 real
    kz2 = np.sqrt(eps2   - eps0_c * sin_t**2)

    # r_p = (eps2*kz0 - eps0*kz2)/(eps2*kz0 + eps0*kz2)
    rp = (eps2 * kz0 - eps0_c * kz2) / (eps2 * kz0 + eps0_c * kz2)
    return rp

def fresnel_rp_layered_fast(eps_film, d_nm, eps2, kz0, kz2, k0, theta_deg=30.0, eps0=1.0):
    """
    Optimized complex p-polarized r_p for 3-layer stack.
    kz0 (air) and kz2 (gold) are precalculated.
    """
    eps1 = np.asarray(eps_film, dtype=np.complex128)
    eps0_c = np.complex128(eps0)

    sin_t = np.sin(np.deg2rad(theta_deg))
    d_cm = float(d_nm) * 1e-7

    # Only the film's kz needs to be calculated dynamically during the fit
    kz1 = _kz_in_medium(eps1, eps0_c, sin_t, k0)

    # p-polarized interface reflection coefficients
    r01 = (eps1 * kz0 - eps0_c * kz1) / (eps1 * kz0 + eps0_c * kz1)
    r12 = (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2)

    phase = np.exp(2j * kz1 * d_cm)
    r012 = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    return r012


def fresnel_rp_layered(eps_film, d_nm, eps_sub, x_cm, theta_deg=30.0, eps0=1.0):
    """
    Complex p-polarized r_p for 3-layer stack:
        medium 0: air (eps0)
        medium 1: film (eps_film), thickness d_nm
        medium 2: substrate (eps_sub) (gold)

    Uses standard thin-film formula:
      r012 = (r01 + r12*exp(2 i kz1 d)) / (1 + r01*r12*exp(2 i kz1 d))

    IMPORTANT:
      Here kz1 depends on wavelength, so we DO need k0 = 2*pi/lambda.
      x_cm is wavenumber in cm^-1, so lambda_cm = 1/x_cm, and k0 (in cm^-1) = 2*pi*x_cm.
      Convert d_nm -> d_cm using 1 nm = 1e-7 cm.
    """
    theta = np.deg2rad(theta_deg)
    sin_t = np.sin(theta)

    eps0_c = np.complex128(eps0)
    eps1 = np.asarray(eps_film, dtype=np.complex128)
    eps2 = np.asarray(eps_sub,  dtype=np.complex128)
    x_cm = np.asarray(x_cm, dtype=float)

    d_cm = float(d_nm) * 1e-7
    k0 = 2.0 * np.pi * x_cm  # since lambda_cm = 1/x_cm

    kz0 = _kz_in_medium(eps0_c, eps0_c, sin_t, k0)
    kz1 = _kz_in_medium(eps1,   eps0_c, sin_t, k0)
    kz2 = _kz_in_medium(eps2,   eps0_c, sin_t, k0)

    # p-polarized interface reflection coefficients (field)
    r01 = (eps1 * kz0 - eps0_c * kz1) / (eps1 * kz0 + eps0_c * kz1)
    r12 = (eps2 * kz1 - eps1   * kz2) / (eps2 * kz1 + eps1   * kz2)

    phase = np.exp(2j * kz1 * d_cm)
    r012 = (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)
    return r012

# =============================================================================
# Vectorized lock-in (2H)
# =============================================================================

# Precompute globally using your existing FFT_freq and Time arrays
phase_2H = np.exp(-2j * np.pi * FFT_freq[idx_2H] * Time)

def LockIn_complex(Beta, rp, alpha_0, E_inc=1.0):
    pref = (alpha_0 * (1.0 + rp)**2 * E_inc)[:, None]
    denom = 1.0 - (alpha_0 * Beta)[:, None] * inv_h3_scaled[None, :]
    return np.dot(pref / denom, phase_2H)

# =============================================================================
# Physically correct complex oscillators for ε(ω)
# =============================================================================

def eps_lorentz_osc(x, S, w0, gamma):
    """
    Complex Lorentz oscillator contribution in wavenumber units.

    ε(ω) += S / (w0^2 - w^2 - i*gamma*w)

    Parameters
    ----------
    x : array (cm^-1)
    S : oscillator strength (sets amplitude; units depend on convention)
    w0 : center (cm^-1)
    gamma : damping (cm^-1)   (often ~ FWHM-ish; this form uses gamma*w)
    """
    x = np.asarray(x, dtype=float)
    return S / (w0**2 - x**2 - 1j * gamma * x)

def eps_voigt_inhom(x, S, w0, gamma, sigma, N_INH=11, span=4.0):
    """
    Inhomogeneous (Gaussian) broadening of the Lorentz oscillator by distributing w0.

    Approximate:
      ∫ dω0'  G(ω0'; w0, sigma) * S/(ω0'^2 - ω^2 - iγ ω)

    We use a fixed grid ω0' = w0 + u*sigma, u in [-span, span]
    with Gaussian weights, normalized.

    Parameters
    ----------
    sigma : std dev of center distribution (cm^-1)
    N_INH : number of quadrature points (odd is nice)
    span  : +/- span*sigma integration window (typically 3-5)
    """
    x = np.asarray(x, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        return eps_lorentz_osc(x, S, w0, gamma)

    # quadrature points in units of sigma
    u = np.linspace(-span, span, int(N_INH))
    w0p = w0 + u * sigma  # (N_INH,)

    # Gaussian weights, normalized
    w = np.exp(-0.5 * u**2)
    w /= np.sum(w)

    # Vectorize: compute oscillator for each w0p and weight-sum
    # shape (N_INH, Nx)
    denom = (w0p[:, None] ** 2) - (x[None, :] ** 2) - 1j * gamma * x[None, :]
    contrib = (S * w[:, None]) / denom
    return np.sum(contrib, axis=0)

# =============================================================================
# Build ε(ω) from oscillators
# =============================================================================
_PROFILE_PARAM_COUNT = {
    "lorentzian": 3,  # (S, w0, gamma)
    "voigt":   4,  # (S, w0, gamma, sigma)
}

def build_eps_sample(x, eps_inf, num_osc, osc_args, profile="lorentz", N_INH=21, span=4.0):
    x = np.asarray(x, dtype=float)
    eps = np.full_like(x, eps_inf, dtype=np.complex128)
    n = int(round(num_osc))
    
    if n == 0:
        return eps

    profile = str(profile).lower()
    P = np.asarray(osc_args, dtype=float).reshape((n, -1))

    if profile == "lorentzian":
        S, w0, gam = P[:, 0, None], P[:, 1, None], P[:, 2, None]
        eps += np.sum(S / (w0**2 - x**2 - 1j * gam * x), axis=0)
        
    elif profile == "voigt":
        S, w0, gam, sig = P[:, 0, None, None], P[:, 1, None, None], P[:, 2, None, None], P[:, 3, None, None]
        u, w = _get_voigt_uw(N_INH, span)
        w0p = w0 + u * sig
        denom = w0p**2 - x**2 - 1j * gam * x
        eps += np.sum((S * w) / denom, axis=(0, 1))
        
    else:
        raise ValueError(f"Unknown profile={profile!r}. Use 'lorentzian' or 'voigt'.")

    return eps


# =============================================================================
# PDM: now uses physical ε(ω)
# =============================================================================
# Initialize cache
_pdm_cache = {'x_arr': np.array([])}

def PDM(x, numOsc, eps_inf, *osc_args, profile="lorentzian",
        d_film_nm=300.0, theta_deg=30.0, N_INH=21, span=4.0):

    x = np.asarray(x, dtype=float)

    # Check if x is the same as the previous call to avoid recalculating statics
    if len(x) != len(_pdm_cache.get('x_arr', [])) or x[0] != _pdm_cache['x_arr'][0] or x[-1] != _pdm_cache['x_arr'][-1]:
        _pdm_cache['x_arr'] = x
        
        # Calculate and cache static reference data for this x-axis
        AuE1, AuE2 = GoldE(x)
        _pdm_cache['E_gold'] = AuE1 + 1j * AuE2
        Beta_ref = (_pdm_cache['E_gold'] - 1.0) / (_pdm_cache['E_gold'] + 1.0)
        rp_ref = fresnel_rp_halfspace(_pdm_cache['E_gold'], theta_deg=theta_deg, eps0=1.0)
        
        PtE1, PtE2 = PlatinumE(x)
        E_tip = PtE1 + 1j * PtE2
        _pdm_cache['alpha_0'] = ((E_tip - 1.0) * 4.0 * np.pi * Tip_rad ** 3) / (E_tip + 2.0)
        
        _pdm_cache['s2_ref'] = LockIn_complex(Beta_ref, rp_ref, _pdm_cache['alpha_0'], 1.0)

        # --- NEW: Fresnel Cache ---
        eps0_c = np.complex128(1.0)
        k0 = 2.0 * np.pi * x
        sin_t = np.sin(np.deg2rad(theta_deg))
        
        _pdm_cache['k0'] = k0
        _pdm_cache['kz0'] = _kz_in_medium(eps0_c, eps0_c, sin_t, k0)
        _pdm_cache['kz2'] = _kz_in_medium(_pdm_cache['E_gold'], eps0_c, sin_t, k0)

    # Retrieve static data
    E_gold = _pdm_cache['E_gold']
    alpha_0 = _pdm_cache['alpha_0']
    s2_ref = _pdm_cache['s2_ref']
    k0 = _pdm_cache['k0']
    kz0 = _pdm_cache['kz0']
    kz2 = _pdm_cache['kz2']

    # Evaluate dynamic sample calculations
    E_samp = build_eps_sample(x, eps_inf, numOsc, osc_args,
                              profile=profile, N_INH=N_INH, span=span)

    Beta_samp = (E_samp - 1.0) / (E_samp + 1.0)
    
    # Route to the optimized Fresnel function
    rp_samp = fresnel_rp_layered_fast(
        E_samp, d_film_nm, E_gold, kz0, kz2, k0, theta_deg=theta_deg, eps0=1.0
    )

    s2_samp = LockIn_complex(Beta_samp, rp_samp, alpha_0, 1.0)
    return s2_samp / s2_ref


# =============================================================================
# Fitting wrapper
# =============================================================================
def PDM_fitting(x, eps_inf, slope, *osc_args,
                profile="lorentzian", d_film_nm=1.0, theta_deg=30.0, N_INH=21, span=4.0):
    # infer numOsc from osc_args and profile
    pcount = _PROFILE_PARAM_COUNT[str(profile).lower()]
    numOsc = len(osc_args) // pcount

    z = PDM(x, numOsc, eps_inf, *osc_args,
            profile=profile, d_film_nm=d_film_nm, theta_deg=theta_deg, N_INH=N_INH, span=span)

    # Keep your minimal real-only linear baseline tweak on the real part
    # (still OK because you detrend both model+data inside the objective)
    if len(osc_args) > 1:
        x0_ref = osc_args[1]  # x0 of the first oscillator
    else:
        x0_ref = np.mean(x)

    return (z.real + abs((x - x0_ref) * slope)) + 1j * z.imag