#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar	9 21:44:31 2026
@author: mategarai
"""

import argparse
import json
import re
import warnings
import dataclasses
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

import Fitting_module as pfm

try:
    get_ipython()
    IN_IPYTHON = True
except NameError:
    IN_IPYTHON = False
    for _bk in ("QtAgg", "TkAgg", "MacOSX"):
        try:
            plt.switch_backend(_bk)
            break
        except Exception:
            continue

import snom_utils
from axz_parser import load_axz_as_dict

mpl.rcParams["savefig.dpi"] = 300
plt.ion()


@dataclass
class ProcessSettings:

    target_folder: Path

    array_num: Optional[int] = None
    array_format: str = "Array_{num}.axz"
    array_scan_dim: List[int] = field(default_factory=lambda: [])
    samp_nums: List[str] = field(default_factory=lambda: [])
    ref_nums: List[str] = field(default_factory=lambda: [])
    target_signal: str = "//ZI/DEV533/DEMODS/{harmonic}/X"
    ref_format: str = "{num}_AuRef_intfgm2D_{harmonic}.txt"
    samp_format: str = "{num}_*_intfgm2D_{harmonic}.txt"
    plotlims: List[float] = field(default_factory=lambda: [])
    phase_fit_regions: List[tuple] = field(default_factory=lambda: [[],[]])
    # Exponential apodization decay constant as a fraction of the mirror travel.
    # 0.25 reproduces the original behaviour. It adds 5/(pi*decay_frac*L_mm)
    # cm^-1 of Lorentzian FWHM to every band; that width is removed from the
    # fitted widths automatically. 0 or null = no exponential.
    decay_frac: Optional[float] = 0.4
    # Polynomial order of the phase correction fitted over phase_fit_regions.
    # Used for every reference and sample spectrum AND for the PDM model, so the
    # three can never disagree.
    phase_correction_order: int = 2
    afmcolormap: str = "afmhot"
    hyperspectracmap: str = "jet"

    # --- HARMONIC SELECTION ---
    # The physical harmonic order (2 = second harmonic). Where it is stored is
    # read from the data: the 'ZI X2'-style channel labels in the .axz, and the
    # 'Channel selection' flags in each point spectrum's _params.txt (files are
    # numbered sequentially over the selected demods). {harmonic} in
    # target_signal / ref_format / samp_format is filled with the demod index /
    # file number found that way. A harmonic that was not recorded is reported
    # with the ones that were.
    harmonic: int = 2

    # --- RATIO MODE ---
    # When on, the processed quantity becomes the ratio of two harmonics and
    # REPLACES S/R everywhere: plots, the standard fit, the drift maps and
    # every export.
    #   ratio_harmonics = [numerator, denominator], physical harmonic orders,
    #       e.g. [3, 2] for S3/S2.
    #   ratio_use_reference -> each harmonic is divided by its own AuRef first,
    #       i.e. (S_n/R_n)/(S_m/R_m). False gives the raw S_n/S_m, which needs
    #       no reference at all.
    ratio_mode: bool = False
    ratio_harmonics: List[int] = field(default_factory=lambda: [3,2])
    ratio_use_reference: bool = True

    # --- PLOTTING TOGGLES ---
    plot_intfgm: bool = False
    plot_allspectra: bool = False
    plot_referenced_spectra: bool = False
    plot_afm: bool = False
    drift_correct: bool = False
    fit_spectra: bool = False
    plot_fitresults: bool = False
    plot_fitstatistics: bool = False
    # Repair the start-of-scan spike in the AFM frames (first scan lines),
    # filling from the retrace when the file has one
    afm_fix_scan_start: bool = True
    # No longer used: drift is interpolated between AFM frames by timestamp.
    # Kept so existing configs still load without a warning.
    drift_poly_degree: Optional[int] = 2

    # --- FIGURE EXPORT ---
    # Every figure the run opens is written into the export folder, not just
    # the referenced-spectra one. Needs EXPORT.
    #   figure_dpi    -> resolution for the saved PNGs. The fit grids are
    #                    15x15 inches, so 300 dpi gives ~4500 px files; drop
    #                    to 150 if the folder gets unwieldy.
    #   close_figures -> close each figure once it is saved. Off by default so
    #                    interactive sessions keep their windows; worth turning
    #                    on for a large array, where the fit grids alone can
    #                    leave dozens of figures open.
    figure_dpi: int = 300
    close_figures: bool = False

    # --- Fitting ---
    signal_type: str = "phase"
    fit_wmin: Optional[float] = None
    fit_wmax: Optional[float] = None
    peak_shape: str = "lorentzian"
    peak_centers: List[float] = field(default_factory=lambda: [])
    # Max shift (cm^-1) of each centre from its guess. One number applies to
    # every peak; a list gives one per peak, in peak_centers order:
    #   "center_tolerance": 5.0            same as [5.0]
    #   "center_tolerance": [5, 3, 5, 10]
    center_tolerance: Union[float, List[float]] = field(default_factory=float)
    fwhm_bounds: List[tuple] = field(default_factory=lambda: [])
    map_parameter: List[str] = field(default_factory=lambda: [])
    shared_colormap: bool = False
    interpolation_res: int = field(default_factory=int)
    relative_array_coords: List[float] = field(default_factory=lambda: [])
    plotgrid_n: int = 4

    correlations_to_check: List[tuple] = field(default_factory=lambda: [])

    # --- Fit-quality filter ---
    # Points whose fit R^2 is below this are left out of the spatial maps
    # (x in the scatter map, blank in the interpolated map) and out of the
    # referenced-spectra plot. null/None turns the filter off.
    #   A number  -> the same threshold for every fit, e.g. 0.9
    #   A dict    -> one per fit, e.g. {"PDM": 0.95, "Standard": 0.8}
    # The two R^2 values are not on the same scale: PDM is scored on the
    # stacked Re/Im spectrum, Standard on the 1D signal_type spectrum.
    r2_threshold: Optional[Union[float, dict]] = None

    EXPORT: bool = True
    export_foldername: Optional[Path] = None

    # --- PDM Fitting Toggles ---
    pdm_fit: bool = False
    # Fit a real scale factor c on the model (closed form, no extra optimiser
    # parameter). Absorbs an absolute |S/R| normalisation error; exported as
    # 'scale'. Costs precision on A, so leave off unless c is needed.
    pdm_free_scale: bool = False
    pdm_profile: str = "lorentzian"
    pdm_plot_every: int = 1
    # How the off-resonance background is handled in the PDM fit (2026-11-17):
    #   "detrend"  -> original: the phase correction, then a straight line
    #                 subtracted from Re and a constant from Im (data and model),
    #                 residual over fit_wmin-fit_wmax.
    #   "envelope" -> S/R = c * B(w) * PDM: B is a smooth background that
    #                 multiplies the spectrum (amplitude polynomial x exp(i phase
    #                 polynomial)), fitted together with the oscillators over the
    #                 span of phase_fit_regions. fit_wmin/fit_wmax then only set
    #                 the R^2 window and the plots; pdm_free_scale is ignored.
    # envelope_order  -> [amplitude order, phase order] of B. Use the lowest
    #                    orders that follow the off-resonance background (check
    #                    the envelope plot); high orders can bend under a band and
    #                    absorb part of it. Choose them before looking at results
    #                    and keep them fixed when comparing datasets.
    # envelope_shared -> true: B is fitted once on the array average and only a
    #                    complex scale c varies per spectrum (exported as 'scale'
    #                    and 'bg_phase_deg'); false: B is fitted per spectrum
    #                    (use for point spectra taken at different times, and as
    #                    a cross-check of gamma maps).
    pdm_background: str = "detrend"
    envelope_order: List[int] = field(default_factory=lambda: [3, 2])
    envelope_shared: bool = True

    # --- Flat PDM Parameters ---
    bulk_sample: bool = False
    d_film_nm: float = 300.0
    eps_inf: float = 2.5
    A_guess: float = 5.0e4
    sigma_guess: float = 10.0
    gamma_guess: float = 10.0
    A_bounds: List[float] = field(default_factory=lambda: [0,1e9])
    sigma_bounds: List[float] = field(default_factory=lambda: [1.0, 20.0])
    gamma_bounds: List[float] = field(default_factory=lambda: [1.0, 20.0])

    def __post_init__(self):
        # Cast paths from loaded JSON strings
        if isinstance(self.target_folder, str):
            self.target_folder = Path(self.target_folder)
        if self.export_foldername and isinstance(self.export_foldername, str):
            self.export_foldername = Path(self.export_foldername)

        if self.plotlims:
            if self.fit_wmin is None:
                self.fit_wmin = float(self.plotlims[0])
            if self.fit_wmax is None:
                self.fit_wmax = float(self.plotlims[1])

        if self.fit_spectra or self.pdm_fit:
            tols = np.atleast_1d(np.asarray(self.center_tolerance, dtype=float))
            if tols.size == 0 or not np.all(tols > 0):
                raise ValueError(
                    "center_tolerance must be > 0 when fitting is enabled (every "
                    f"value, if it is a list; got {self.center_tolerance!r}). A zero "
                    "tolerance makes the lower and upper centre bounds equal, which "
                    "curve_fit rejects."
                )

        if isinstance(self.r2_threshold, dict):
            unknown = set(self.r2_threshold) - {"PDM", "Standard"}
            if unknown:
                raise ValueError(
                    f"r2_threshold keys must be 'PDM' and/or 'Standard'; got {sorted(unknown)}."
                )

        self._validate_harmonics()

    def _validate_harmonics(self):
        """
        Every way ratio mode can silently produce nonsense, caught up front.

        The dangerous one is a format string with no {harmonic} placeholder:
        both harmonics would then resolve to the same files and the ratio would
        come out as exactly 1 everywhere, with no error anywhere.
        """
        if not self.ratio_mode:
            return

        if len(self.ratio_harmonics) != 2:
            raise ValueError(
                "ratio_mode is on, so ratio_harmonics must be "
                "[numerator, denominator] harmonic orders, e.g. [3, 2] for S3/S2; "
                f"got {self.ratio_harmonics!r}."
            )

        num_h, den_h = self.ratio_harmonics
        if num_h == den_h:
            raise ValueError(
                f"ratio_harmonics numerator and denominator are both {num_h}; "
                "the ratio would be 1 everywhere."
            )

        problems = []
        if self.array_num is not None and "{harmonic}" not in self.target_signal:
            problems.append(
                f"  target_signal = {self.target_signal!r} has no {{harmonic}}, so both "
                "harmonics would read the same demod channel."
            )
        if self.samp_nums and "{harmonic}" not in self.samp_format:
            problems.append(
                f"  samp_format = {self.samp_format!r} has no {{harmonic}}, so both "
                "harmonics would match the same files."
            )
        if self.ratio_use_reference and "{harmonic}" not in self.ref_format:
            problems.append(
                f"  ref_format = {self.ref_format!r} has no {{harmonic}}, so both "
                "harmonics would be divided by the same reference."
            )
        if problems:
            raise ValueError(
                "ratio_mode needs a {harmonic} placeholder in the format strings "
                "it varies:\n" + "\n".join(problems)
            )

        if self.pdm_fit:
            raise ValueError(
                "pdm_fit cannot be combined with ratio_mode yet.\n"
                "  PDM_main models S/R for a single demodulation order (2H is "
                "hardcoded), so fitting it to a harmonic ratio would fit the "
                "wrong model and report a healthy R^2 while doing it.\n"
                "  A ratio-mode PDM is a separate change: the lock-in has to "
                "become harmonic-aware, and alpha_0 and (1+r_p)^2 then cancel "
                "exactly in the ratio.\n"
                "  Set pdm_fit to false, or turn ratio_mode off."
            )

    def validate(self):
        """Checks that need a complete config. Called from load_config, not
        __post_init__, so --generate can still build the empty template."""
        errs = []
        if len(self.plotlims) != 2:
            errs.append("plotlims must be [min, max]")
        if not [r for r in self.phase_fit_regions if len(r) == 2]:
            errs.append("phase_fit_regions needs at least one [min, max] pair")
        if self.peak_shape.lower() not in ("lorentzian", "gaussian"):
            errs.append(f"peak_shape {self.peak_shape!r}: use 'lorentzian' or 'gaussian'")
        if self.signal_type.lower() not in ("phase", "amplitude", "real", "imaginary", "imag", "complex"):
            errs.append(f"signal_type {self.signal_type!r} not recognised")
        if int(self.phase_correction_order) != self.phase_correction_order or not 0 <= self.phase_correction_order <= 3:
            errs.append("phase_correction_order must be an integer 0-3")
        if self.decay_frac is not None and self.decay_frac < 0:
            errs.append("decay_frac must be >= 0 (0 or null = no exponential)")
        if self.fit_spectra or self.pdm_fit:
            n_tol, n_pk = np.atleast_1d(self.center_tolerance).size, len(self.peak_centers)
            if n_tol not in (1, n_pk):
                errs.append(f"center_tolerance has {n_tol} values but there are {n_pk} "
                            "peak_centers; give one value for all peaks, or one per peak")
        if self.pdm_fit:
            if self.pdm_profile.lower() not in ("lorentzian", "voigt"):
                errs.append(f"pdm_profile {self.pdm_profile!r}: use 'lorentzian' or 'voigt'")
            if not 2 <= int(self.harmonic) <= 15:
                errs.append(f"PDM needs a harmonic between 2 and 15; got {self.harmonic}")
            if self.eps_inf <= 1.0:
                errs.append("eps_inf <= 1 means no sample background; use n^2 of the material")
            if str(self.pdm_background).lower() not in ("detrend", "envelope"):
                errs.append(f"pdm_background {self.pdm_background!r}: use 'detrend' or 'envelope'")
            if str(self.pdm_background).lower() == "envelope":
                eo = list(self.envelope_order) if np.ndim(self.envelope_order) else []
                if len(eo) != 2 or any(int(v) != v or not 0 <= v <= 6 for v in eo):
                    errs.append("envelope_order must be [amplitude order, phase order], "
                                f"integers 0-6; got {self.envelope_order!r}")
        for h in [self.harmonic] + list(self.ratio_harmonics):
            if int(h) != h or h < 1:
                errs.append(f"harmonic orders must be integers >= 1; got {h!r}")
        if errs:
            raise ValueError("Config problems:\n  " + "\n  ".join(errs))

    def save_config(self, filepath: Path):
        """Exports the current settings to a JSON file."""
        data = dataclasses.asdict(self)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4, default=str)

    def export_dir(self, generated_name: str) -> Path:
        """
        Where outputs go.

        export_foldername unset         -> <target_folder>/<generated_name>
        export_foldername a name        -> <target_folder>/<export_foldername>
          (or a relative path, e.g. "fits/run1")
        export_foldername absolute path -> that folder, used as-is
          ("~" is expanded, so "~/Desktop/run1" counts as absolute)
        """
        if self.export_foldername:
            out = Path(self.export_foldername).expanduser()
            return out if out.is_absolute() else Path(self.target_folder) / out
        return Path(self.target_folder) / generated_name

    @classmethod
    def load_config(cls, filepath: Path, **kwargs):
        """Loads settings from JSON and overrides with any provided kwargs."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Configs from before 2026-11-05 gave the file / demod number plus a
        # harmonic_offset. Convert once, so they keep selecting the same data.
        if "harmonic_offset" in data:
            offset = int(data.pop("harmonic_offset"))
            old_h = data.get("harmonic", 1)
            data["harmonic"] = int(old_h) + offset
            if data.get("ratio_harmonics"):
                data["ratio_harmonics"] = [int(h) + offset for h in data["ratio_harmonics"]]
            print_warning(
                f"{Path(filepath).name} uses the old harmonic numbering (harmonic_offset = "
                f"{offset}): reading harmonic {old_h} as harmonic order {data['harmonic']}"
                + (f" and ratio_harmonics as {data['ratio_harmonics']}" if data.get("ratio_harmonics") else "")
                + ". Update the file: give the true harmonic order(s) and delete harmonic_offset."
            )

        if "fwhm_bounds" in data:
            data["fwhm_bounds"] = [tuple(x) for x in data["fwhm_bounds"]]
        if "correlations_to_check" in data:
            data["correlations_to_check"] = [
                tuple(x) for x in data["correlations_to_check"]
            ]
        
        
        data.update(kwargs)
        valid_fields = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            print_warning(f"Ignoring unknown config keys: {unknown}")

        config = cls(**filtered_data)
        config.validate()
        return config



def print_warning(msg: str):
    """Prints a yellow warning message to the console."""
    print(f"\033[93m{msg}\033[0m")


# ----------------------- Figure export -----------------------

def _slug(text: str, limit: int = 48) -> str:
    """A filename-safe fragment: alphanumerics, spaces -> underscores."""
    if not text:
        return ""
    first = str(text).splitlines()[0]
    keep = "".join(c if (c.isalnum() or c in " -_") else " " for c in first)
    return "_".join(keep.split())[:limit].strip("_")


def _figure_label(fig) -> str:
    """
    A readable discriminator for one figure out of several in the same block.

    Only taken from a single-panel figure (one plot plus at most a colorbar),
    where the axes title describes the whole figure - that is the map case,
    where the useful name is 'Drift-Corrected Locations: Peak 1 Amplitude'. On
    a grid of subplots the first panel's title says nothing about the rest, so
    those stay as plain indices.
    """
    if len(fig.axes) > 2:
        return ""
    text = fig.axes[0].get_title() if fig.axes else ""
    return _slug(text)


def _label_figure(fig, label: str):
    """
    Puts `label` (batch and harmonic, e.g. 'Array 3 2H') in a figure's title,
    unless a title already contains it:
      suptitle present      -> 'Array 3 2H: <suptitle>'
      one panel (+colorbar) -> 'Array 3 2H: <axes title>'
      grid, no suptitle     -> suptitle 'Array 3 2H'
    """
    sup = getattr(fig, "_suptitle", None)
    texts = [sup.get_text() if sup is not None else ""]
    texts += [ax.get_title(loc=loc) for ax in fig.axes for loc in ("center", "left", "right")]
    if any(label in t for t in texts):
        return
    if sup is not None and sup.get_text():
        sup.set_text(f"{label}: {sup.get_text()}")      # keeps its font settings
    elif len(fig.axes) <= 2 and fig.axes:
        ax = fig.axes[0]
        loc = next((l for l in ("center", "left", "right") if ax.get_title(loc=l)), "center")
        old = ax.get_title(loc=loc)
        ax.set_title(f"{label}: {old}" if old else label, loc=loc)
    else:
        fig.suptitle(label)
        # constrained layout makes room by itself; anything else (tight
        # layout leaves a placeholder engine behind) needs a tight-layout rerun
        if "Constrained" not in type(fig.get_layout_engine()).__name__:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.tight_layout()


@contextmanager
def _save_new_figures(config, export_dir: Path, stem: str, label: Optional[str] = None):
    """
    Saves every figure opened inside the block into export_dir.

    The plotting helpers in snom_utils and Fitting_module build their own
    figures and do not return them, so rather than changing all of their
    signatures this records which figure numbers existed before the call and
    writes out whatever is new afterwards. Runs in a finally block, so figures
    produced before an exception are still written.

    label: when given (batch and harmonic, e.g. 'Array 3 2H'), it is put in
    the title of every new figure, on screen and in the saved file.
    """
    before = set(plt.get_fignums())
    try:
        yield
    finally:
        new = [n for n in plt.get_fignums() if n not in before]

        # File names come from the titles as the helper drew them, so they
        # don't change when the label is added below
        names = {}
        if new and config.EXPORT:
            multi = len(new) > 1
            for k, num in enumerate(new, 1):
                name = stem
                if multi:
                    name = f"{stem}_{k:02d}"
                    discriminator = _figure_label(plt.figure(num))
                    if discriminator:
                        name = f"{name}_{discriminator}"
                names[num] = name

        if label:
            for num in new:
                _label_figure(plt.figure(num), label)

        if names:
            export_dir.mkdir(parents=True, exist_ok=True)
            written = []
            for num, name in names.items():
                fig = plt.figure(num)
                path = export_dir / f"{name}.png"
                try:
                    fig.savefig(path, dpi=config.figure_dpi, bbox_inches="tight")
                    written.append(path.name)
                except Exception as exc:
                    print_warning(f"Could not save figure {path.name}: {exc}")
            if written:
                head = written[0] if len(written) == 1 else f"{written[0]} (+{len(written)-1} more)"
                print(f"Figure(s) saved to: {head}")

        if config.close_figures:
            for num in new:
                plt.close(num)


# ----------------------- Harmonic bookkeeping -----------------------

def _sub_harmonic(template: str, harmonic) -> str:
    """
    Replace {harmonic} in a format string, leaving {num} for later.

    str.format would raise KeyError on the {num} field, so this is a plain
    replace. A template with no {harmonic} comes back untouched.
    """
    return str(template).replace("{harmonic}", str(harmonic))


def _harmonics_in_use(config) -> List[int]:
    """The harmonic orders this run needs, numerator first."""
    if config.ratio_mode:
        return [int(h) for h in config.ratio_harmonics]
    return [int(config.harmonic)]


def _harmonic_label(config) -> str:
    """For figure titles: '2H', or '3H/2H' in ratio mode."""
    if config.ratio_mode:
        num_h, den_h = config.ratio_harmonics
        return f"{int(num_h)}H/{int(den_h)}H"
    return f"{int(config.harmonic)}H"


def _batch_dir(config, batch: str) -> Path:
    """
    Output folder of one batch ('Array_3' or 'Point_Spectra'), named with the
    harmonic: Extracted_Data_Array_3_2H, or Extracted_Data_Array_3_3H-2H_ratio
    in ratio mode. A user-given export_foldername is used as-is.
    """
    if config.ratio_mode:
        num_h, den_h = config.ratio_harmonics
        harmonic = f"{int(num_h)}H-{int(den_h)}H_ratio"
    else:
        harmonic = f"{int(config.harmonic)}H"
    return config.export_dir(f"Extracted_Data_{batch}_{harmonic}")


_AXZ_LABEL = re.compile(r"^ZI\s*([A-Za-z]+)\s*(\d+)$")


def _axz_signal_for_harmonic(data: dict, template: str, harmonic: int):
    """
    The .axz signal path that holds `harmonic`, from the spectrum channel
    labels ('ZI X2' = X of the 2nd harmonic on //ZI/.../DEMODS/1/X).
    {harmonic} in `template` stands for the demod index, found here.

    Returns (path or None, {harmonic: path} of what is recorded, note).
    """
    spectra = data["Document"]["SNOMSpectra"]["AXDSNOMSpectrum"]
    first = spectra[0] if isinstance(spectra, list) else spectra
    channels = (first.get("DataSources") or {}).get("AXDDataChannel", [])
    if isinstance(channels, dict):
        channels = [channels]
    pattern = re.compile(
        "^" + re.escape(template).replace(re.escape("{harmonic}"), r"(\d+)") + "$"
    )
    recorded = {}
    for ch in channels:
        path = snom_utils._axz_text(ch.get("DataSignal"))
        label = snom_utils._axz_text(ch.get("Label"))
        if path and label and pattern.match(path):
            m = _AXZ_LABEL.match(str(label).strip())
            if m:
                recorded[int(m.group(2))] = path
    if recorded:
        return recorded.get(int(harmonic)), recorded, ""
    # No labelled channels: fall back on the usual wiring, but say so
    path = _sub_harmonic(template, int(harmonic) - 1)
    return path, {}, ("no harmonic labels found on the .axz channels; assuming "
                      f"demod {int(harmonic) - 1} holds harmonic {harmonic}")


def _params_file(data_file: Path) -> Path:
    """'201_AuRef_intfgm2D_1.txt' -> '201_AuRef_params.txt' (LabVIEW naming)."""
    return data_file.with_name(re.sub(r"_intfgm2D_\d+\.txt$", "_params.txt", data_file.name))


def _channel_selection(params_file: Path) -> Optional[str]:
    """The 'Channel selection' flags of a point-spectrum params file, e.g.
    '011100': one flag per demod 0, 1, 2, ...; demod d records harmonic d + 1."""
    try:
        for line in Path(params_file).read_text(errors="ignore").splitlines():
            if line.strip().lower().startswith("channel selection"):
                flags = line.split()[-1]
                if flags and set(flags) <= {"0", "1"}:
                    return flags
    except OSError:
        pass
    return None


def _txt_files_for_harmonic(target_folder: Path, fmt: str, num, harmonic: int, exclude=()):
    """
    The point-spectrum files of measurement(s) `num` that hold `harmonic`.

    The LabVIEW program stores only the selected demods and numbers the files
    sequentially, so file 1 is the first selected demod. Each measurement's
    _params.txt says which demods were selected. Returns (files, problems),
    one problem string per measurement that doesn't have the harmonic.
    """
    harmonic = int(harmonic)
    if "{harmonic}" not in fmt:
        # Fixed file name: nothing to choose, but check it holds the harmonic
        files, problems = [], []
        for f in sorted(target_folder.glob(str(fmt).format(num=num))):
            if f.resolve() in exclude:
                continue
            m = re.search(r"_intfgm2D_(\d+)\.txt$", f.name)
            flags = _channel_selection(_params_file(f)) if m else None
            if flags is not None:
                selected = [d for d, flag in enumerate(flags) if flag == "1"]
                index = int(m.group(1))
                held = selected[index - 1] + 1 if 1 <= index <= len(selected) else None
                if held != harmonic:
                    problems.append(
                        f"{f.name} holds harmonic {held if held else '?'} "
                        f"(Channel selection {flags} = harmonics "
                        f"{[d + 1 for d in selected]}), not harmonic {harmonic}"
                    )
                    continue
            files.append(f)
        return files, problems
    probe = re.compile("^" + re.escape(_sub_harmonic(fmt, "\x00").format(num=num))
                       .replace(r"\*", ".*").replace("\x00", r"(\d+)") + "$")
    groups = {}
    for f in sorted(target_folder.glob(_sub_harmonic(fmt, "*").format(num=num))):
        m = probe.match(f.name)
        if m and f.resolve() not in exclude:
            groups.setdefault(_params_file(f), {})[int(m.group(1))] = f
    files, problems = [], []
    for params, by_index in groups.items():
        flags = _channel_selection(params)
        if flags is None:
            problems.append(f"{params.name} is missing or has no 'Channel selection', "
                            "so the harmonic of its files is unknown")
            continue
        selected = [d for d, flag in enumerate(flags) if flag == "1"]
        recorded = [d + 1 for d in selected]
        if harmonic - 1 not in selected:
            problems.append(f"{params.name}: harmonic {harmonic} was not recorded "
                            f"(Channel selection {flags} = harmonics {recorded})")
            continue
        index = selected.index(harmonic - 1) + 1
        if index not in by_index:
            problems.append(f"{params.name}: harmonic {harmonic} should be file "
                            f"{index}, which is missing")
            continue
        files.append(by_index[index])
    return files, problems


def _quantity_label(config) -> Optional[str]:
    """
    The symbol for the processed quantity, used in axis labels.

    None means 'use the existing S/R labels unchanged', so normal-mode plots
    come out byte-identical to before.
    """
    if not config.ratio_mode:
        return None
    num_h, den_h = config.ratio_harmonics
    return f"S_{{{num_h}}}/S_{{{den_h}}}"


def _output_tag(config) -> str:
    """Filename fragment that keeps ratio outputs from overwriting S/R ones."""
    if not config.ratio_mode:
        return ""
    num_h, den_h = config.ratio_harmonics
    return f"_ratio_S{num_h}_S{den_h}"


def _align_complex(da: xr.DataArray, target_wavenumber) -> xr.DataArray:
    """
    Put a complex spectrum on another wavenumber axis.

    Interpolating real and imaginary parts separately, as the reference
    alignment has always done - interpolating amplitude and phase would need
    unwrapping first.
    """
    if da.wavenumber.equals(target_wavenumber):
        return da
    return (
        da.real.interp(wavenumber=target_wavenumber)
        + 1j * da.imag.interp(wavenumber=target_wavenumber)
    )


def _data_range(config):
    """
    Widest wavenumber span the processed data must carry.

    Includes the fit window and the phase-fit regions because processing needs
    them, and plotlims so plots are not clipped. plotlims does not affect any
    result - only how much data is available to display and export.
    """
    regions = [r for r in config.phase_fit_regions if len(r) == 2]
    lo = [config.fit_wmin, config.plotlims[0]] + [min(r) for r in regions]
    hi = [config.fit_wmax, config.plotlims[1]] + [max(r) for r in regions]
    return min(lo), max(hi)

def correct_bounds(value, bounds, name):
    """
    Clips value(s) to bounds. Handles both single peak scalars
    and multi-peak lists/arrays.
    """

    def _check_scalar(v, b, label):
        low, high = b
        if v < low:
            print_warning(f"{label}: ({v}) is below limit. Clipping to {low}.")
            return low
        if v > high:
            print_warning(f"{label}: ({v}) is above limit. Clipping to {high}.")
            return high # Fixed bug: previously set to low
        return v

    # --- Multi-Peak Case ---
    if isinstance(value, (list, tuple, np.ndarray)):
        num_peaks = len(value)

        # Normalize bounds: if it's one pair [lo, hi], duplicate it for all peaks
        if not isinstance(bounds[0], (list, tuple, np.ndarray)):
            normalized_bounds = [bounds] * num_peaks
        else:
            if len(bounds) != num_peaks:
                raise ValueError(f"Length of {name} bounds must match length of guesses.")
            normalized_bounds = bounds

        return [
            _check_scalar(v, b, f"{name} Peak {i+1}")
            for i, (v, b) in enumerate(zip(value, normalized_bounds))
        ]

    # --- Single Peak Case ---
    return _check_scalar(value, bounds, name)


def _r2_threshold_for(config, fit_type):
    """The R^2 threshold that applies to one fit type, or None if off."""
    thr = config.r2_threshold
    if isinstance(thr, dict):
        thr = thr.get(fit_type)
    return None if thr is None else float(thr)


def _r2_pass_mask(fit_df: pd.DataFrame, config, fit_type: str) -> np.ndarray:
    """
    True for each point whose fit R^2 meets the threshold for this fit type.

    With no threshold set every point passes, so plots are unchanged. With a
    threshold set, a failed fit (R^2 = NaN) does not pass.
    """
    n = len(fit_df)
    thr = _r2_threshold_for(config, fit_type)
    if thr is None:
        return np.ones(n, dtype=bool)

    r2 = fit_df["r_squared"].to_numpy(dtype=float)
    keep = np.isfinite(r2) & (r2 >= thr)

    n_low = int((np.isfinite(r2) & (r2 < thr)).sum())
    n_failed = int((~np.isfinite(r2)).sum())
    if n_low or n_failed:
        print_warning(
            f"{fit_type}: excluding {n_low + n_failed}/{n} points from plots "
            f"({n_low} with R^2 < {thr:g}, {n_failed} failed fits)."
        )
    return keep


def _per_point(values, n_rows: int, fit_type: str, column: str = "intfgm_pp") -> np.ndarray:
    """A per-point array as a fit-table column; NaN (with a warning) if the
    table does not have one row per point, rather than misaligning rows."""
    values = np.asarray(values, dtype=float)
    if len(values) == n_rows:
        return values
    print_warning(f"{fit_type} fit has {n_rows} rows but there are {len(values)} "
                  f"interferograms; {column} left empty.")
    return np.full(n_rows, np.nan)


def _plot_referenced_spectra(target_da, keep, title_label, y_label, config):
    """
    Plots every kept spectrum in light grey and their mean in red.
    Points where keep is False are left out of both.
    """
    wn = np.asarray(target_da.wavenumber.values, dtype=float)
    vals = np.asarray(target_da.values)          # (point, wavenumber)
    shown = vals[keep]
    n_excluded = int((~keep).sum())

    fig, ax2 = plt.subplots(figsize=(7, 5))

    if shown.shape[0] > 0:
        ax2.plot(wn, shown.T, color="black", alpha=0.1)
        with warnings.catch_warnings():
            # all-NaN wavenumber columns at the edges are expected
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ax2.plot(wn, np.nanmean(shown, axis=0), color="red")
    else:
        print_warning(f"{title_label}: every spectrum was excluded by r2_threshold; nothing to plot.")

    title = f"{title_label} ({config.signal_type.capitalize()})"
    if config.ratio_mode:
        kind = "gold-normalised" if config.ratio_use_reference else "self-referenced"
        title += f"\n{kind} harmonic ratio"
    if n_excluded:
        title += f"\n{n_excluded}/{len(keep)} excluded (fit R$^2$ below threshold)"
    ax2.set_title(title)
    ax2.set_xlabel(r"$\tilde{\nu}$ [cm$^{-1}$]")
    ax2.set_ylabel(y_label)
    if config.plotlims:
        ax2.set_xlim(config.plotlims)
    return fig

# ----------------------- Core Processing Functions -----------------------


def _extract_target_signal(
    normalized_sample: xr.DataArray, signal_type: str, quantity: Optional[str] = None
) -> tuple[xr.DataArray, str]:
    """
    Helper to extract the correct signal type and its axis label.

    quantity is the symbol to show in the axis label. None keeps the original
    S/R wording, so single-harmonic output is unchanged.
    """
    if quantity is None:
        sig_map = {
            "amplitude": (np.abs(normalized_sample), r"Amplitude ($|S|/|R|$)"),
            "real": (np.real(normalized_sample), r"Re($S/R$)"),
            "imaginary": (-np.imag(normalized_sample), r"Im($S/R$)"),
            "imag": (-np.imag(normalized_sample), r"Im($S/R$)"),
            "complex": (normalized_sample, r"$S/R$")
        }
        default_label = r"$\Phi_{NF}$ [deg.]"
    else:
        sig_map = {
            "amplitude": (np.abs(normalized_sample), rf"Amplitude ($|{quantity}|$)"),
            "real": (np.real(normalized_sample), rf"Re(${quantity}$)"),
            "imaginary": (-np.imag(normalized_sample), rf"Im(${quantity}$)"),
            "imag": (-np.imag(normalized_sample), rf"Im(${quantity}$)"),
            "complex": (normalized_sample, rf"${quantity}$")
        }
        default_label = rf"$\Delta\Phi$(${quantity}$) [deg.]"

    data, label = sig_map.get(
        signal_type.lower(),
        (
            -np.rad2deg(np.angle(normalized_sample)),
            default_label,
        ),  # Default to phase
    )
    return normalized_sample.copy(data=data), label


def _reference_for_harmonic(target_folder: Path, config: ProcessSettings, harmonic):
    """
    Loads, baseline-corrects, FFTs and phase-corrects the AuRef files for one
    harmonic.

    Returns (flat_ref_spectra, master_reference, reference_corrected, ref_files,
    reference_raw). reference_corrected carries the same single baseline
    make_single_spec removes and is for interferogram_pp only; the spectra and
    the interferogram plot use reference_raw.
    """
    ref_files, problems = [], []
    for num in config.ref_nums:
        files, why = _txt_files_for_harmonic(target_folder, config.ref_format, num, harmonic)
        if not files and not why:
            why = [f"no file matches {_sub_harmonic(config.ref_format, '*').format(num=num)!r}"]
        ref_files += files
        problems += why
    for why in problems:
        print_warning(f"Warning: reference {why}; left out.")
    if not ref_files:
        raise FileNotFoundError(
            f"No reference in {target_folder} holds harmonic {harmonic} "
            f"(ref_nums {config.ref_nums}):\n  " + "\n  ".join(problems)
        )

    reference_interferograms = snom_utils.package_point_interferograms(ref_files)

    # One baseline subtraction per sweep, done inside make_single_spec on the
    # RAW sweeps. This baseline-corrected copy (same fit) is only for
    # interferogram_pp.
    reference_corrected = snom_utils.baseline_correct_sweeps(reference_interferograms)

    ref_spectra = snom_utils.process_all_spectra(
        reference_interferograms, pad_pow=2, auto_center=True,
        decay_frac=config.decay_frac,
    )
    data_lo, data_hi = _data_range(config)
    flat_ref_spectra = snom_utils.batch_phase_correct(
        ref_spectra,
        fit_regions=config.phase_fit_regions,
        out_wmin=data_lo,
        out_wmax=data_hi,
        correction_order=config.phase_correction_order,
    )

    return (
        flat_ref_spectra,
        flat_ref_spectra.mean(dim="point"),
        reference_corrected,
        ref_files,
        reference_interferograms,
    )


def _flat_sample_spectra(sample_interfs, config: ProcessSettings, name: str):
    """
    Baseline-correct, FFT and phase-correct one harmonic's sample interferograms.

    The phase correction MUST happen here, per harmonic, before any ratio is
    taken. make_single_spec rolls each interferogram by its own Hilbert-envelope
    maximum, and a one-sample difference between harmonics puts a linear phase
    ramp into the ratio (about 24 degrees at 1340 cm^-1 for a typical padded
    length). Correcting each harmonic separately absorbs that ramp into its own
    polynomial fit; dividing first and correcting once would not.

    Returns (baseline_corrected_interferograms, flat_complex_spectra).
    The baseline-corrected interferograms carry the same single baseline
    make_single_spec removes and are for interferogram_pp only; the spectra are
    computed from the RAW sweeps, so no baseline is subtracted twice.
    """
    sample_corrected = snom_utils.baseline_correct_sweeps(sample_interfs)
    sample_corrected.name = f"{name}_Baseline_Corrected_Intensity"

    sample_spectra = snom_utils.process_all_spectra(
        sample_interfs, pad_pow=2, auto_center=True,
        decay_frac=config.decay_frac,
    )

    data_lo, data_hi = _data_range(config)
    flat_sample_spectra = snom_utils.batch_phase_correct(
        sample_spectra,
        fit_regions=config.phase_fit_regions,
        out_wmin=data_lo,
        out_wmax=data_hi,
        correction_order=config.phase_correction_order,
    )

    return sample_corrected, flat_sample_spectra


def _normalized_quantity(interfs_by_h, refs_by_h, config: ProcessSettings,
                         title_label: str):
    """
    Build the complex quantity the rest of the pipeline works on.

    Single-harmonic mode : S/R, exactly as before.
    Ratio mode           : (S_n/R_n) / (S_m/R_m), or S_n/S_m when
                           ratio_use_reference is False.

    Returns (normalized, corrected_interfs_numerator, flat_spectra_numerator).
    The numerator's interferograms and spectra are handed back so plot_intfgm
    and plot_allspectra keep showing a real measured harmonic.
    """
    harmonics = _harmonics_in_use(config)

    per_h = {}
    for h in harmonics:
        corrected, flat = _flat_sample_spectra(
            interfs_by_h[h], config, f"{title_label}_{h}H"
        )

        if config.ratio_mode and not config.ratio_use_reference:
            per_h[h] = (corrected, flat, flat)
            continue

        master_reference = refs_by_h[h][1]
        aligned_reference = _align_complex(
            master_reference, flat.wavenumber
        )
        per_h[h] = (corrected, flat, flat / aligned_reference)

    if not config.ratio_mode:
        h = harmonics[0]
        corrected, flat, normalized = per_h[h]
        return normalized, corrected, flat

    num_h, den_h = harmonics
    num_corrected, num_flat, num_norm = per_h[num_h]
    _, _, den_norm = per_h[den_h]

    if num_norm.sizes["point"] != den_norm.sizes["point"]:
        raise ValueError(
            f"Harmonic {num_h} has {num_norm.sizes['point']} spectra but harmonic "
            f"{den_h} has {den_norm.sizes['point']}. The ratio needs one "
            "denominator spectrum per numerator spectrum, in the same order."
        )

    # The two harmonics can land on slightly different wavenumber axes: for
    # point spectra each harmonic's files are truncated to their own shortest
    # sweep, which changes the padded FFT length.
    den_aligned = _align_complex(den_norm, num_norm.wavenumber)
    den_aligned = den_aligned.assign_coords(point=num_norm.point)

    kind = "(S/R ratio)" if config.ratio_use_reference else "(raw harmonic ratio)"
    print(f"  Harmonic ratio {num_h}/{den_h} {kind}")

    return num_norm / den_aligned, num_corrected, num_flat


def _export_envelope(results, config, export_dir: Path, stem: str, fig_title: str):
    """
    Envelope-mode exports:
      <stem>_PDM_envelope.csv   the shared background B(w), and the average
                                spectrum with its fit
      <stem>_PDM_envelope.png   what was divided out, amplitude and phase
    """
    env = results["envelope"]
    x = env["x"]
    p_avg = env["average_params"]
    pdm_kw = dict(profile=config.pdm_profile, gamma_instr=results["gamma_instr"],
                  eps_inf=config.eps_inf, bulk_sample=config.bulk_sample,
                  d_film_nm=config.d_film_nm, harmonic=results["harmonic"])
    background = env["B"] * pfm._pdm_eval(
        pfm._zero_amplitudes(p_avg, config.pdm_profile), x, pdm_kw)
    model = env["B"] * pfm._pdm_eval(p_avg, x, pdm_kw)
    avg = env["average_spectrum"]

    # The fitter works in conj(S/R) (absorption = +Im), so the phase of these
    # arrays is Phi_NF = -arg(S/R), the convention of the phase spectra output.
    df = pd.DataFrame({
        "wavenumber": x,
        "B_real": env["B"].real, "B_imag": env["B"].imag,
        "B_abs": np.abs(env["B"]), "B_phase_deg": np.degrees(np.angle(env["B"])),
        "average_abs": np.abs(avg), "average_phase_deg": np.degrees(np.angle(avg)),
        "background_abs": np.abs(background),
        "background_phase_deg": np.degrees(np.angle(background)),
        "fit_abs": np.abs(model), "fit_phase_deg": np.degrees(np.angle(model)),
    })
    df.to_csv(export_dir / f"{stem}_PDM_envelope.csv", index=False)

    with _save_new_figures(config, export_dir, f"{stem}_PDM_envelope", label=fig_title):
        fig, (ax_a, ax_p) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        for ax in (ax_a, ax_p):
            for r in config.phase_fit_regions:
                if len(r) == 2:
                    ax.axvspan(min(r), max(r), color="0.93", zorder=0)
        ax_a.plot(x, np.abs(avg), color="#2a78d6", lw=2, label="array average |S/R|")
        ax_a.plot(x, np.abs(model), "--", color="#262624", lw=1.5, label="envelope x PDM (fit)")
        ax_a.plot(x, np.abs(background), ":", color="#8a8a85", lw=2,
                  label="background (envelope x PDM without oscillators)")
        ax_a.set_ylabel("|S/R|")
        ax_a.legend(fontsize=7, frameon=False)
        ax_p.plot(x, np.degrees(np.angle(avg)), color="#2a78d6", lw=2, label="array average")
        ax_p.plot(x, np.degrees(np.angle(model)), "--", color="#262624", lw=1.5, label="fit")
        ax_p.plot(x, np.degrees(np.angle(background)), ":", color="#8a8a85", lw=2,
                  label="background")
        ax_p.set_ylabel(r"$\Phi_{NF}$ [deg.]")
        ax_p.set_xlabel(r"Wavenumber [cm$^{-1}$]")
        ka, kp = env["order"]
        ax_a.set_title(f"Envelope background (amplitude order {ka}, phase order {kp}, "
                       f"{'shared' if env['shared'] else 'per spectrum'}); "
                       "grey bands: phase_fit_regions", fontsize=9)
        fig.tight_layout()
    print(f"Envelope background saved to: {stem}_PDM_envelope.csv / .png")


def _process_and_plot_samples(
    interfs_by_h: dict,
    title_label: str,
    target_folder: Path,
    config: ProcessSettings,
    refs_by_h: dict,
    export_dir: Path,
):
    """
    Processes one batch of sample interferograms.

    interfs_by_h : {file-name harmonic number -> interferogram DataArray}
    refs_by_h    : {file-name harmonic number -> tuple from
                    _reference_for_harmonic}, empty when no reference is needed
    """
    # Figure titles carry the harmonic: 'Array 3 2H' ('Array 3 3H/2H' for a
    # ratio). Plots of the raw numerator harmonic use its own label.
    fig_title = f"{title_label} {_harmonic_label(config)}"
    print(f"Processing {fig_title}...")

    if config.EXPORT:
        # --- Make sure the export folder exists! ---
        export_dir.mkdir(parents=True, exist_ok=True)

    normalized_sample, sample_corrected, flat_sample_spectra = _normalized_quantity(
        interfs_by_h, refs_by_h, config, title_label
    )

    num_h = _harmonics_in_use(config)[0]
    tag = _output_tag(config)
    quantity = _quantity_label(config)

    # Peak-to-peak amplitude (uV) of each point's baseline-corrected
    # interferogram, for the harmonic being analysed (the numerator in ratio
    # mode). A direct handle on laser-power drift: it goes into every fit table
    # as 'intfgm_pp', so it can be mapped and correlated like r_squared.
    intfgm_pp = snom_utils.interferogram_pp(sample_corrected)
    ref_pp = snom_utils.interferogram_pp(refs_by_h[num_h][2]) if num_h in refs_by_h else None
    msg = f"  Interferogram p-p: median {np.nanmedian(intfgm_pp):.0f} uV " \
          f"(range {np.nanmin(intfgm_pp):.0f}-{np.nanmax(intfgm_pp):.0f})"
    if ref_pp is not None:
        msg += f"; reference {np.nanmean(ref_pp):.0f} uV"
    print(msg)

    # Mean level of each RAW interferogram (uV), 'intfgm_offset' in the fit
    # tables: the non-interferometric, spectrally integrated signal at this
    # harmonic - a handle on broadband scattering and laser power that does
    # not depend on the fringes. Baseline correction removes it, so it is
    # taken before.
    intfgm_offset = snom_utils.interferogram_offset(interfs_by_h[num_h])
    ref_offset = None
    if num_h in refs_by_h:
        ref_offset = snom_utils.interferogram_offset(
            snom_utils.package_point_interferograms(refs_by_h[num_h][3])
        )
    msg = f"  Interferogram offset: median {np.nanmedian(intfgm_offset):.1f} uV " \
          f"(range {np.nanmin(intfgm_offset):.1f} to {np.nanmax(intfgm_offset):.1f})"
    if ref_offset is not None:
        msg += f"; reference {np.nanmean(ref_offset):.0f} uV"
    print(msg)

    # Background level and tilt of each normalised spectrum (S/R, or the
    # harmonic ratio), measured on the phase_fit_regions points only - no fit
    # model - so they exist for every fit mode, for point spectra and arrays.
    # 'bg_tilt' (% per 100 cm^-1) follows the spectral shape of the background,
    # e.g. the laser spectrum drifting away from the one the reference was
    # recorded with, which p-p and offset cannot show. 'bg_level' is |S/R| in
    # the middle of the regions. Both go into the fit tables and the drift plot.
    bg_level, bg_tilt = snom_utils.background_tilt(
        normalized_sample.wavenumber.values,
        normalized_sample.transpose("point", "wavenumber").values,
        config.phase_fit_regions,
    )
    q_print = (f"{config.ratio_harmonics[0]}H/{config.ratio_harmonics[1]}H ratio"
               if config.ratio_mode else "S/R")
    if np.isfinite(bg_tilt).any():
        print(f"  Background tilt of |{q_print}| in phase_fit_regions: median "
              f"{np.nanmedian(bg_tilt):+.1f} %/100 cm^-1 (range {np.nanmin(bg_tilt):+.1f} to "
              f"{np.nanmax(bg_tilt):+.1f}); level median {np.nanmedian(bg_level):.3g}")
    else:
        print_warning("Background tilt not computed: fewer than 3 data points inside "
                      "phase_fit_regions.")

    # Lorentzian FWHM the exponential apodization adds to every band. S and R
    # get the same window, so it carries straight into S/R.
    gamma_instr = snom_utils.instrumental_lorentz_fwhm(
        interfs_by_h[num_h].stage_position_mm.values[0], config.decay_frac
    )
    print(f"  Apodization: decay_frac={config.decay_frac} adds "
          f"{gamma_instr:.2f} cm^-1 Lorentzian FWHM to every band")

    safe_title = title_label.replace(" ", "_")
    num_title = f"{title_label} {num_h}H"

    if config.plot_intfgm:
        with _save_new_figures(
            config, export_dir, f"{safe_title}_{num_h}H_samples_intfgm_plot", label=num_title
        ):
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            snom_utils.plot_intfgm(interfs_by_h[num_h], num_title, ax,
                                   decay_frac=config.decay_frac)
            ax.set_ylabel(rf"X{num_h} signal ($\mu$V)")

    if config.plot_allspectra:
        # Shows the numerator harmonic against its own reference - a ratio has
        # no meaningful "reference vs sample" pair to compare.
        if refs_by_h:
            with _save_new_figures(
                config, export_dir, f"{safe_title}_{num_h}H_all_spectra", label=num_title
            ):
                snom_utils.plot_all_spectra(
                    refs_by_h[num_h][0], flat_sample_spectra, config.plotlims
                )
        else:
            print_warning(
                "plot_allspectra needs a reference; skipped because "
                "ratio_use_reference is false."
            )

    target_da, y_label = _extract_target_signal(
        normalized_sample, config.signal_type, quantity=quantity
    )
    complex_da, ycomp_label = _extract_target_signal(
        normalized_sample, "complex", quantity=quantity
    )

    complex_df_t = None
    if config.EXPORT or config.pdm_fit:
        # Evaluate to Pandas once to serve both export and fitting blocks
        complex_df_t = complex_da.T.to_pandas().dropna(how='all')

    # --- Fitting ---
    # Runs before the referenced-spectra plot so that plot can leave out
    # points whose fit R^2 is below config.r2_threshold.
    results_bundle = {"PDM": None, "Standard": None}

    if config.pdm_fit:

        # Handle user input errors
        gamma_guess = correct_bounds(value=config.gamma_guess,bounds=config.gamma_bounds,name="Gamma")
        sigma_guess = correct_bounds(value=config.sigma_guess,bounds=config.sigma_bounds,name="Sigma")
        A_guess = correct_bounds(value=config.A_guess,bounds=config.A_bounds,name="Amplitude")

        results = pfm.process_spectra_array(

            data_input=complex_df_t, # Native complex DataFrame reused
            profile=config.pdm_profile,
            peak_centers=config.peak_centers,
            center_tolerance=config.center_tolerance,
            fit_window=[config.fit_wmin, config.fit_wmax],
            baseline_regions=config.phase_fit_regions,

            # Explicit flat parameters
            eps_inf=        config.eps_inf,
            A_guess=        A_guess,
            sigma_guess=    sigma_guess,
            gamma_guess=    gamma_guess,

            A_bounds=       config.A_bounds,
            sigma_bounds=   config.sigma_bounds,
            gamma_bounds=   config.gamma_bounds,
            fwhm_bounds=    config.fwhm_bounds,

            bulk_sample=    config.bulk_sample,
            d_film_nm=      config.d_film_nm,
            gamma_instr=    gamma_instr,
            free_scale=     config.pdm_free_scale,
            harmonic=       int(config.harmonic),
            phase_order=    config.phase_correction_order,
            background=     config.pdm_background,
            envelope_order= config.envelope_order,
            envelope_shared=config.envelope_shared,
        )
        envelope_mode = results.get("background") == "envelope"
        if envelope_mode and config.EXPORT:
            _export_envelope(results, config, export_dir, f"{safe_title}{tag}", fig_title)

        if config.plot_fitresults:
            # 5. Plot the individual fits
            with _save_new_figures(config, export_dir, f"{safe_title}{tag}_PDM_fits", label=fig_title):
                pfm.plot_individual_fits(
                    results,
                    profile=config.pdm_profile,
                    num_peaks=len(config.peak_centers),
                    plot_every=config.pdm_plot_every,
                    show_individual_peaks=True,
                    grid_n=config.plotgrid_n,
                    bulk_sample=config.bulk_sample,
                    d_film_nm=config.d_film_nm,
                    eps_inf=config.eps_inf
                )


        fit_params = results["fit_params"]
        fit_success = results["fit_success"]
        P = fit_params.shape[1]

        # 1. Column names come from the fitter, so they always match its packing order
        param_labels = list(results["param_labels"])
        pcount = pfm.PROFILE_PARAM_COUNT[config.pdm_profile.lower()]
        num_peaks = P // pcount

        pdm_fit_df = pd.DataFrame(fit_params, columns=param_labels)
        err_df = pd.DataFrame(results["fit_errors"], columns=[f"{c}_err" for c in param_labels])

        # 2. Calculate FWHM and Area to match the requested format
        ordered_cols = ["point", "fit_success", "r_squared", "r2_pass", "intfgm_pp",
                        "intfgm_offset", "bg_level", "bg_tilt", "eps_inf", "gamma_instr", "scale"]
        if envelope_mode:
            ordered_cols.append("bg_phase_deg")

        for i in range(1, num_peaks + 1):
            amp = pdm_fit_df[f"peak_{i}_amplitude"]

            # Convert raw widths to FWHM
            if config.pdm_profile.lower() == "lorentzian":
                # For a physical oscillator, Gamma is effectively the FWHM
                fwhm = pdm_fit_df[f"peak_{i}_gamma"]
            elif config.pdm_profile.lower() == "gaussian":
                fwhm = pdm_fit_df[f"peak_{i}_sigma"] * 2.355
            elif config.pdm_profile.lower() == "voigt":
                f_g = pdm_fit_df[f"peak_{i}_sigma"] * 2.355
                f_l = pdm_fit_df[f"peak_{i}_gamma"]
                # Voigt FWHM approximation
                fwhm = 0.5346 * f_l + np.sqrt(0.2166 * f_l**2 + f_g**2)

            # Assign calculated columns
            pdm_fit_df[f"peak_{i}_fwhm"] = fwhm
            pdm_fit_df[f"peak_{i}_delta_eps"] = amp / pdm_fit_df[f"peak_{i}_center"]**2

            # Append to the ordered list
            ordered_cols.extend([
                f"peak_{i}_center",
                f"peak_{i}_amplitude",
                f"peak_{i}_fwhm",
                f"peak_{i}_delta_eps"
            ])

            # Keep Voigt-specific raw parameters if needed
            if config.pdm_profile.lower() == "voigt":
                ordered_cols.extend([f"peak_{i}_sigma", f"peak_{i}_gamma"])

        # 3. Add Metadata
        pdm_fit_df["point"] = range(results["M"])
        pdm_fit_df["r_squared"] = results["fit_r2"]
        pdm_fit_df["fit_success"] = fit_success
        pdm_fit_df["eps_inf"] = config.eps_inf
        pdm_fit_df["gamma_instr"] = gamma_instr          # already removed from every gamma
        # detrend: 1.0 unless pdm_free_scale. envelope: |c|, the spectrum's
        # background level relative to the shared background (about 1).
        pdm_fit_df["scale"] = results["fit_scale"]
        if envelope_mode:
            pdm_fit_df["bg_phase_deg"] = results["fit_bg_phase_deg"]
        pdm_fit_df["intfgm_pp"] = _per_point(intfgm_pp, len(pdm_fit_df), "PDM")
        pdm_fit_df["intfgm_offset"] = _per_point(intfgm_offset, len(pdm_fit_df), "PDM", "intfgm_offset")
        pdm_fit_df["bg_level"] = _per_point(bg_level, len(pdm_fit_df), "PDM", "bg_level")
        pdm_fit_df["bg_tilt"] = _per_point(bg_tilt, len(pdm_fit_df), "PDM", "bg_tilt")
        pdm_fit_df = pd.concat([pdm_fit_df, err_df], axis=1)
        pdm_fit_df["r2_pass"] = _r2_pass_mask(pdm_fit_df, config, "PDM")
        
        if "at_bound" in results:
            for k, nm in enumerate(results["param_labels"]):
                pdm_fit_df[f"{nm}_at_bound"] = results["at_bound"][:, k]
            ordered_cols += [f"{nm}_at_bound" for nm in results["param_labels"]]

        # Reorder DataFrame (1-sigma errors last)
        pdm_fit_df = pdm_fit_df[ordered_cols + list(err_df.columns)]
        results_bundle["PDM"] = pdm_fit_df

        if config.plot_fitstatistics and config.correlations_to_check:
            with _save_new_figures(
                config, export_dir, f"{safe_title}{tag}_PDM_correlations", label=fig_title
            ):
                snom_utils.plot_correlations(results_bundle["PDM"], config.correlations_to_check,fitting_type = "PDM")

        # --- Export PDM Fit Results ---
        if config.EXPORT:
            save_filename_pdm = export_dir / f"{safe_title}{tag}_PDM_fit_results.csv"

            pdm_fit_df.to_csv(save_filename_pdm, index=False)
            print(f"PDM Fit results saved to: {save_filename_pdm.name}")

    if config.fit_spectra:

        with _save_new_figures(config, export_dir, f"{safe_title}{tag}_standard_fits", label=fig_title):
            std_fit_df = snom_utils.fit_all_spectra(
                da=target_da,
                wmin=config.fit_wmin,
                wmax=config.fit_wmax,
                peak_centers=config.peak_centers,
                shape=config.peak_shape,
                center_tolerance=config.center_tolerance,
                fwhm_bounds=config.fwhm_bounds,
                grid_n=config.plotgrid_n if config.plot_fitresults else 0,
            )
        std_fit_df["r2_pass"] = _r2_pass_mask(std_fit_df, config, "Standard")
        std_fit_df.insert(
            std_fit_df.columns.get_loc("r_squared") + 1 if "r_squared" in std_fit_df else len(std_fit_df.columns),
            "intfgm_pp", _per_point(intfgm_pp, len(std_fit_df), "Standard"),
        )
        std_fit_df.insert(
            std_fit_df.columns.get_loc("intfgm_pp") + 1,
            "intfgm_offset", _per_point(intfgm_offset, len(std_fit_df), "Standard", "intfgm_offset"),
        )
        for k, (col, vals) in enumerate((("bg_level", bg_level), ("bg_tilt", bg_tilt))):
            std_fit_df.insert(std_fit_df.columns.get_loc("intfgm_offset") + 1 + k,
                              col, _per_point(vals, len(std_fit_df), "Standard", col))
        if config.peak_shape.lower() == "lorentzian" and gamma_instr:
            # exact for Lorentzians: the apodization width simply adds
            for j in range(1, len(config.peak_centers) + 1):
                std_fit_df[f"peak_{j}_fwhm_intrinsic"] = std_fit_df[f"peak_{j}_fwhm"] - gamma_instr

        results_bundle["Standard"] = std_fit_df

        if config.plot_fitstatistics and config.correlations_to_check:
            with _save_new_figures(
                config, export_dir, f"{safe_title}{tag}_standard_correlations", label=fig_title
            ):
                snom_utils.plot_correlations(results_bundle["Standard"], config.correlations_to_check,fitting_type = "Standard")


        if config.EXPORT:
            save_filename_fits = export_dir / f"{safe_title}{tag}_fit_results.csv"

            std_fit_df.to_csv(save_filename_fits, index=False)
            print(f"Fit results saved to: {save_filename_fits.name}")

    # A spectrum is shown only if it passes the threshold of every fit that ran.
    n_points = target_da.sizes["point"]
    keep = np.ones(n_points, dtype=bool)
    for fit_type, fit_df in results_bundle.items():
        if fit_df is None:
            continue
        if len(fit_df) != n_points:
            print_warning(
                f"{fit_type} fit has {len(fit_df)} rows but there are {n_points} "
                "spectra; its R^2 filter is not applied to the spectra plot."
            )
            continue
        keep &= fit_df["r2_pass"].to_numpy(dtype=bool)

    # Drift check: interferogram p-p, offset and background tilt against point
    # number. Needs no fit.
    if config.plot_fitstatistics:
        row_length = (config.array_scan_dim[1]
                      if config.array_num is not None and len(config.array_scan_dim) == 2 else None)
        with _save_new_figures(config, export_dir, f"{safe_title}{tag}_intfgm_pp", label=num_title):
            snom_utils.plot_intfgm_pp(intfgm_pp, num_title, ref_pp=ref_pp,
                                      offset=intfgm_offset, ref_offset=ref_offset,
                                      tilt=bg_tilt,
                                      tilt_label=rf"tilt of $|{quantity or 'S/R'}|$ (% / 100 cm$^{{-1}}$)",
                                      row_length=row_length,
                                      keep=keep if len(keep) == len(intfgm_pp) else None)

    # --- Output (independent of plotting) ---
    if config.EXPORT:
        save_filename = export_dir / f"{safe_title}{tag}_{config.signal_type}_spectra_output.csv"
        save_complex_filename = export_dir / f"{safe_title}{tag}_complex_spectra_output.csv"
        save_complextranspose_filename = export_dir / f"{safe_title}{tag}_complex_transpose_spectra_output.csv"

        # The CSVs keep every spectrum; use r2_pass in the fit-results CSV
        # to filter them later.
        target_da.to_pandas().dropna(axis=1, how='all').to_csv(save_filename)
        # target_da.to_pandas().to_csv(save_filename)
        complex_da.to_pandas().to_csv(save_complex_filename)
        print(f"Processed spectra saved to: {save_filename.name}")
        print(f"Processed spectra saved to: {save_complex_filename.name}")

        # Format and save using the pre-calculated dataframe
        df_formatted = complex_df_t.astype(str).replace({r'\(': '', r'\)': ''}, regex=True)
        df_formatted.to_csv(save_complextranspose_filename, header=False)

    # --- Plotting ---
    if config.plot_referenced_spectra:
        fig = _plot_referenced_spectra(target_da, keep, fig_title, y_label, config)

        if config.EXPORT:
            save_fig_filename = export_dir / f"{safe_title}{tag}_{config.signal_type}_plot.png"
            fig.savefig(save_fig_filename, dpi=config.figure_dpi, bbox_inches="tight")
            print(f"Figure saved to: {save_fig_filename.name}")

        if config.close_figures:
            plt.close(fig)
        else:
            plt.show()

    return results_bundle


def process_spectra(target_folder: Path, config: ProcessSettings):
    """Main workflow to process references and samples based on config."""

    harmonics = _harmonics_in_use(config)
    need_reference = (not config.ratio_mode) or config.ratio_use_reference

    if need_reference and not config.ref_nums:
        raise ValueError("Execution halted: 'ref_nums' cannot be empty. A reference is required.")

    # 1. Process the Master Reference, once per harmonic in use
    refs_by_h = {}
    if need_reference:
        for h in harmonics:
            refs_by_h[h] = _reference_for_harmonic(target_folder, config, h)
            print(f"Reference loaded for harmonic {h}: "
                  f"{[f.name for f in refs_by_h[h][3]]}")

        if config.plot_intfgm:
            # The reference figures belong with whichever batch this run is
            # producing; the array folder wins when both are present.
            ref_fig_dir = _batch_dir(
                config,
                f"Array_{config.array_num}"
                if config.array_num is not None
                else "Point_Spectra"
            )
            for h in harmonics:
                with _save_new_figures(
                    config, ref_fig_dir, f"AuRef_{h}H_intfgm_plot"
                ):
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    snom_utils.plot_intfgm(refs_by_h[h][4], f"AuRef {h}H", ax,
                                           decay_frac=config.decay_frac)
                    ax.set_ylabel(rf"X{h} signal ($\mu$V)")

    array_fit_df = None
    data = None

    # 2. Process Array Data
    if config.array_num is not None:
        axz_file = target_folder / config.array_format.format(num=config.array_num)

        if not axz_file.exists():
            raise ValueError(f"Execution halted: AXZ file {axz_file.name} not found.")

        data = load_axz_as_dict(axz_file)

        # Centralized Export Path Logic
        array_export_dir = _batch_dir(config, f"Array_{config.array_num}")

        if config.EXPORT:
            snom_utils.export_axz_contents(
                data, array_export_dir, afm_format="gwy"
            )

        # One demod channel per harmonic, all from the same .axz, found from
        # the channel labels
        signals, missing, recorded = {}, [], {}
        for h in harmonics:
            path, recorded, note = _axz_signal_for_harmonic(data, config.target_signal, h)
            if note:
                print_warning(f"Warning: {axz_file.name}: {note}.")
            if path is None:
                missing.append(h)
            else:
                signals[h] = path
                print(f"Array {config.array_num}: harmonic {h} -> {path}")

        if missing:
            print_warning(
                f"Warning: {axz_file.name} has no {config.target_signal!r} channel for "
                f"harmonic(s) {missing}; it records harmonics {sorted(recorded)}. "
                "Skipping the array spectra (AFM steps still run)."
            )
        else:
            interfs_by_h = {
                h: snom_utils.package_axz_interferograms(data, target_signal=path)
                for h, path in signals.items()
            }

            array_fit_df = _process_and_plot_samples(
                interfs_by_h,
                f"Array {config.array_num}",
                target_folder,
                config,
                refs_by_h,
                array_export_dir,
            )

        if config.EXPORT:
            df = pd.DataFrame(
                snom_utils.extract_raw_spectra_coords(data),
                columns=["X_Absolute", "Y_Absolute"],
            )
            # The folder may not exist yet when the spectra were skipped
            array_export_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(
                array_export_dir / "raw_coordinates.csv",
                index=False,
            )

    # 3. Process Point Spectra Data
    if config.samp_nums:
        # Reference files for every harmonic in use are excluded, so a
        # reference sharing a number is never picked up as a sample.
        ref_paths = {
            f.resolve() for entry in refs_by_h.values() for f in entry[3]
        }

        interfs_by_h = {}
        for h in harmonics:
            samp_fmt = config.samp_format
            samp_files = []

            for num in config.samp_nums:
                matches, problems = _txt_files_for_harmonic(
                    target_folder, config.samp_format, num, h, exclude=ref_paths
                )
                for why in problems:
                    print_warning(f"Warning: sample {why}; skipped.")

                if not matches:
                    if not problems:
                        print_warning(
                            f"No sample file matched "
                            f"'{_sub_harmonic(samp_fmt, '*').format(num=num)}' "
                            f"for samp_num '{num}'."
                        )
                    continue
                if len(matches) > 1:
                    print_warning(
                        f"samp_num '{num}' matched {len(matches)} files; using all: "
                        f"{[m.name for m in matches]}"
                    )
                samp_files.extend(matches)

            if not samp_files:
                print_warning(
                    f"No point-spectra files found in {target_folder} for "
                    f"samp_nums={config.samp_nums} using samp_format='{samp_fmt}'."
                )
                continue

            interfs_by_h[h] = snom_utils.package_point_interferograms(samp_files)

        missing_h = [h for h in harmonics if h not in interfs_by_h]
        if missing_h:
            print_warning(
                f"Skipping point spectra: no files found for harmonic(s) {missing_h}."
            )
        else:
            point_export_dir = _batch_dir(config, "Point_Spectra")

            _process_and_plot_samples(
                interfs_by_h,
                "Point Spectra",
                target_folder,
                config,
                refs_by_h,
                point_export_dir,
            )

    return data, array_fit_df


def process_afm_drift(
    axzdata: dict, target_folder: Path, config: ProcessSettings, fit_results_bundle=None
):
    """Processes AFM drift correction and maps spectral coordinates for all fit types."""
    if config.array_num is None:
        return

    # Needed up front now, because the AFM figures below are saved into it.
    base_export = _batch_dir(config, f"Array_{config.array_num}")
    tag = _output_tag(config)
    map_title = f"Array {config.array_num} {_harmonic_label(config)}"

    # 1. Base AFM Processing
    processor = snom_utils.AFMArray(axzdata)
    processor.extract_scans(fix_scan_start=config.afm_fix_scan_start)
    processor.align_rows(method="median", mask_percentile=80)
    processor.flatten_scans(degree=1, method="2D", mask_percentile=80)

    if config.plot_afm:
        with _save_new_figures(config, base_export, "AFM_scans_raw"):
            processor.plot_scans(show_corrected=False, cmap=config.afmcolormap)

    # 2. Drift Correction Processing
    if config.drift_correct:
        if config.drift_poly_degree is not None:
            print("Note: drift_poly_degree is no longer used; drift is interpolated "
                  "between AFM frames by timestamp.")
        processor.calculate_and_apply_drift()
        with _save_new_figures(config, base_export, "AFM_drift"):
            processor.plot_drift()

        if config.plot_afm:
            with _save_new_figures(config, base_export, "AFM_scans_drift_corrected"):
                processor.plot_scans(show_corrected=True, cmap=config.afmcolormap)

    # Guard clause to abort before mapping if prerequisites are not met
    if not config.drift_correct or fit_results_bundle is None:
        return

    # 3. Spectral Alignment Calibration
    if len(config.relative_array_coords) != 2:
        calibration_result = snom_utils.calibrate_start_point(target_folder)
    else:
        calibration_result = config.relative_array_coords

    if calibration_result is None:
        print("Skipping spectra alignment (calibration aborted).")
        return

    print(f"Relative array coordinates: {calibration_result}")
    rel_x, rel_y = calibration_result
    processor.align_and_plot_spectra(rel_x, rel_y)

    # 4. Map Coordinates for ALL DataFrames in the Bundle
    # Type normalization ensures backward compatibility with older single-DataFrame scripts
    if isinstance(fit_results_bundle, pd.DataFrame):
        fit_results_bundle = {"fit": fit_results_bundle}

    base_export.mkdir(parents=True, exist_ok=True)

    for fit_type, df_to_map in fit_results_bundle.items():
        if df_to_map is None:
            continue

        print(f"Applying drift correction to '{fit_type}' coordinates...")

        # Points that failed the R^2 threshold are drawn as x / left blank
        keep_mask = (
            df_to_map["r2_pass"].to_numpy(dtype=bool)
            if "r2_pass" in df_to_map.columns
            else None
        )

        # Apply mapping using the specific dataframe from the current loop iteration
        with _save_new_figures(config, base_export, f"maps_{fit_type}{tag}", label=map_title):
            df_mapped = processor.apply_drift_to_spectra(
                fit_results_df=df_to_map,
                map_parameter=config.map_parameter,
                shared_colormap=config.shared_colormap,
                interp_resolution=config.interpolation_res,
                grid_shape=config.array_scan_dim,
                fitting_type = fit_type,      # the title label says harmonic / ratio
                afmcolormap = config.afmcolormap,
                hyperspectracmap = config.hyperspectracmap,
                keep_mask = keep_mask,

            )

        if config.EXPORT:
            # Append the fit_type suffix to prevent file overwriting
            save_path = base_export / f"corrected_coordinates_{fit_type}{tag}.csv"
            df_mapped.to_csv(save_path, index=False)
            print(f"Corrected '{fit_type}' map coordinates saved to: {save_path.name}")



def main():


    parser = argparse.ArgumentParser(
        description="Process s-SNOM data from a JSON config."
    )
    parser.add_argument(
        "config_file",
        type=Path,
        nargs="?",
        help="Path to the JSON configuration file to run the processing.",
    )
    parser.add_argument(
        "--generate",
        type=Path,
        nargs="?",
        const=Path("config.json"),
        metavar="FILEPATH",
        help="Generate a default JSON configuration file. Defaults to config.json in the current directory.",
    )

    args = parser.parse_args()

    # Handle the config generation routine
    if args.generate:
        # "." defaults the data search path to the current working directory
        default_config = ProcessSettings(target_folder=".")
        default_config.save_config(args.generate)

        # .resolve() prints the absolute path so the user knows exactly where it went
        print(f"Default configuration file generated at '{args.generate.resolve()}'.")
        return

    # Enforce standard usage if not generating
    if args.config_file is None:
        parser.error("The config_file argument is required unless --generate is used.")

    if not args.config_file.exists():
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return

    print(f"Loading configuration from {args.config_file.name}...")
    config = ProcessSettings.load_config(args.config_file)

    target_folder = Path(config.target_folder)

    if not target_folder.exists():
        print(f"Error: Target folder '{target_folder}' does not exist.")
        return

    # Execute Pipeline
    axzdata, fit_bundle = process_spectra(target_folder, config)

    if config.plot_afm or config.drift_correct:
        process_afm_drift(axzdata, target_folder, config, fit_results_bundle=fit_bundle)



if __name__ == "__main__":
    try:
        main()
    finally:
        if not IN_IPYTHON:
            plt.show(block=True)        # blocks; runs even if main() raised
