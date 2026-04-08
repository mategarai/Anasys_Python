#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar	9 21:44:31 2026
@author: mategarai
"""

import argparse
import json
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl

import Fitting_module as pfm

try:
    # Check if we are in Spyder / IPython
    get_ipython()
except NameError:
    mpl.use("Qt5Agg")

import snom_utils
from axz_parser import load_axz_as_dict

mpl.rcParams["savefig.dpi"] = 300
plt.ion()


@dataclass
class ProcessSettings:

    target_folder: str

    array_num: Optional[int] = None
    array_format: str = "Array_{num}.axz"
    array_scan_dim: List[int] = field(default_factory=list)
    samp_nums: List[str] = field(default_factory=list)
    ref_nums: List[str] = field(default_factory=list)
    target_signal: str = "//ZI/DEV533/DEMODS/1/X"
    ref_format: str = "{num}_AuRef_intfgm2D_1.txt"
    plotlims: List[float] = field(default_factory=list)
    phase_fit_regions: List[tuple] = field(default_factory=list)

    # --- PLOTTING TOGGLES ---
    plot_intfgm: bool = False
    plot_allspectra: bool = False
    plot_referenced_spectra: bool = False
    plot_afm: bool = False
    drift_correct: bool = False
    fit_spectra: bool = False
    plot_fitresults: bool = False
    plot_fitstatistics: bool = False
    drift_poly_degree: int = 4

    # --- Fitting ---
    signal_type: str = "phase"
    fit_wmin: Optional[float] = None
    fit_wmax: Optional[float] = None
    peak_shape: str = "lorentzian"
    peak_centers: List[float] = field(default_factory=list)
    center_tolerance: float = field(default_factory=float)
    fwhm_bounds: List[tuple] = field(default_factory=list)
    map_parameter: List[str] = field(default_factory=list)
    shared_colormap: bool = False
    interpolation_res: int = field(default_factory=int)
    relative_array_coords: List[float] = field(default_factory=list)
    plotgrid_n: int = 4

    correlations_to_check: List[tuple] = field(default_factory=list)

    EXPORT: bool = True
    export_foldername: Optional[str] = None
    
    # --- PDM Fitting Toggles ---
    pdm_fit: bool = False
    pdm_profile: str = "voigt"
    pdm_plot_every: int = 10
    
    # --- Flat PDM Parameters ---
    eps_guess: float = 1.0
    slope_guess: float = -0.0003
    A_guess: float = 50000.0
    x0_guess: float = 1622.0
    sigma_guess: float = 6.0
    gamma_guess: float = 6.0
    eps_bounds: List[float] = field(default_factory=lambda: [0.5, 2.0])
    slope_bounds: List[float] = field(default_factory=lambda: [-1.0, 1.0])
    A_bounds: List[float] = field(default_factory=lambda: [1000.0, 60000.0])
    x0_bounds: List[float] = field(default_factory=lambda: [1595.0, 1680.0])
    sigma_bounds: List[float] = field(default_factory=lambda: [4.0, 8.0])
    gamma_bounds: List[float] = field(default_factory=lambda: [4.0, 8.0])


    def __post_init__(self):
        # Only attempt to extract values if plotlims is populated
        if self.plotlims:
            if self.fit_wmin is None:
                self.fit_wmin = float(self.plotlims[0])
            if self.fit_wmax is None:
                self.fit_wmax = float(self.plotlims[1])

    def save_config(self, filepath: Path):
        """Exports the current settings to a JSON file."""
        data = dataclasses.asdict(self)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_config(cls, filepath: Path, **kwargs):
        """Loads settings from JSON and overrides with any provided kwargs."""
        with open(filepath, "r") as f:
            data = json.load(f)

        if "fwhm_bounds" in data:
            data["fwhm_bounds"] = [tuple(x) for x in data["fwhm_bounds"]]
        if "correlations_to_check" in data:
            data["correlations_to_check"] = [
                tuple(x) for x in data["correlations_to_check"]
            ]

        data.update(kwargs)
        valid_fields = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


# ----------------------- Core Processing Functions -----------------------


def _extract_target_signal(
    normalized_sample: xr.DataArray, signal_type: str
) -> tuple[xr.DataArray, str]:
    """Helper to extract the correct signal type and its axis label."""
    sig_map = {
        "amplitude": (np.abs(normalized_sample), r"Amplitude ($|S|/|R|$)"),
        "real": (np.real(normalized_sample), r"Re($S/R$)"),
        "imaginary": (-np.imag(normalized_sample), r"Im($S/R$)"),
        "imag": (-np.imag(normalized_sample), r"Im($S/R$)"),
        "complex": (normalized_sample, r"$S/R$")
    }

    data, label = sig_map.get(
        signal_type.lower(),
        (
            -np.rad2deg(np.angle(normalized_sample)),
            r"$\Phi_{NF}$ [deg.]",
        ),  # Default to phase
    )
    return normalized_sample.copy(data=data), label


def _process_and_plot_samples(
    sample_interfs,
    title_label: str,
    target_folder: Path,
    config: ProcessSettings,
    flat_ref_spectra: xr.DataArray,
    master_reference: xr.DataArray,
    reference_corrected: xr.DataArray,
    export_dir: Path,
):
    """Processes a specific batch of sample interferograms against the master reference."""
    print(f"Processing {title_label}...")

    if config.EXPORT:
        # --- Make sure the export folder exists! ---
        export_dir.mkdir(parents=True, exist_ok=True)

    sample_corrected, _ = xr.apply_ufunc(
        snom_utils.baseline_correct,
        sample_interfs,
        input_core_dims=[["step"]],
        output_core_dims=[["step"], ["step"]],
        vectorize=True,
        kwargs={"deg": 3},
    )
    sample_corrected.name = f"{title_label}_Baseline_Corrected_Intensity"

    sample_spectra = snom_utils.process_all_spectra(
        sample_corrected, pad_pow=2, auto_center=True
    )
    flat_sample_spectra = snom_utils.batch_phase_correct(
        sample_spectra,
        fit_regions=config.phase_fit_regions,
        out_wmin=config.plotlims[0],
        out_wmax=config.plotlims[1],
        correction_order=2,
    )

    if config.plot_allspectra:
        snom_utils.plot_all_spectra(
            flat_ref_spectra, flat_sample_spectra, config.plotlims
        )


    if not master_reference.wavenumber.equals(flat_sample_spectra.wavenumber):
        aligned_ref_real = master_reference.real.interp(
            wavenumber=flat_sample_spectra.wavenumber
        )
        aligned_ref_imag = master_reference.imag.interp(
            wavenumber=flat_sample_spectra.wavenumber
        )
        aligned_reference = aligned_ref_real + 1j * aligned_ref_imag
    else:
        aligned_reference = master_reference
            
    normalized_sample = flat_sample_spectra / aligned_reference
    target_da, y_label = _extract_target_signal(normalized_sample, config.signal_type)
    
    complex_da, ycomp_label = _extract_target_signal(normalized_sample, "complex")

    # --- Plotting & Output ---
    if config.plot_intfgm:
        snom_utils.plot_intfgm(reference_corrected, "Reference")
        snom_utils.plot_intfgm(sample_corrected, f"{title_label} Samples")

    if config.plot_referenced_spectra:
        fig, ax2 = plt.subplots(figsize=(7, 5))
        target_da.plot.line(
            x="wavenumber",
            hue="point",
            ax=ax2,
            add_legend=False,
            color="black",
            alpha=0.1,
        )

        if config.EXPORT:
            # --- Dynamically name the file so Arrays and Points don't overwrite each other ---
            safe_title = title_label.replace(" ", "_")
            save_filename = export_dir / f"{safe_title}_{config.signal_type}_spectra_output.csv"
            save_complex_filename = export_dir / f"{safe_title}_complex_spectra_output.csv"
            save_complextranspose_filename = export_dir / f"{safe_title}_complex_transpose_spectra_output.csv"

            target_da.to_pandas().to_csv(save_filename)
            complex_da.to_pandas().to_csv(save_complex_filename)
            print(f"Processed spectra saved to: {save_filename.name}")
            print(f"Processed spectra saved to: {save_complex_filename.name}")
            
            df = complex_da.T.to_pandas()
            df = df.dropna(how='all')
            df = df.astype(str).replace({r'\(': '', r'\)': ''}, regex=True)
            df.to_csv(save_complextranspose_filename, header=False)

        target_da.mean(dim="point").plot.line(
            x="wavenumber", hue="point", ax=ax2, add_legend=False, color="red"
        )
        ax2.set_title(f"{title_label} ({config.signal_type.capitalize()})")
        ax2.set_xlabel(r"Wavenumber [cm$^{-1}$]")
        ax2.set_ylabel(y_label)
        plt.show()



    if config.pdm_fit:
        
        # 1. Keep native complex numbers. Do not convert to strings.
        df = complex_da.T.to_pandas().dropna(how='all')
        
        results = pfm.process_spectra_array(
            data_input=df,
            profile=config.pdm_profile,
            peak_centers=config.peak_centers,
            center_tolerance=config.center_tolerance,
            fit_window=config.plotlims if config.plotlims else None,
            baseline_regions=config.phase_fit_regions,
            
            # Explicit flat parameters
            eps_guess=config.eps_guess,
            slope_guess=config.slope_guess,
            A_guess=config.A_guess,
            sigma_guess=config.sigma_guess,
            gamma_guess=config.gamma_guess,
            eps_bounds=config.eps_bounds,
            slope_bounds=config.slope_bounds,
            A_bounds=config.A_bounds,
            sigma_bounds=config.sigma_bounds,
            gamma_bounds=config.gamma_bounds
        )
        
        # 4. Plot 2D maps only if the dataset is a 2D array
        if "Array" in title_label and len(config.array_scan_dim) == 2:
            pfm.plot_2d_maps(
                results, 
                nx=config.array_scan_dim[0], 
                ny=config.array_scan_dim[1],
                profile=config.pdm_profile
            )
        
        if config.plot_fitresults:
            # 5. Plot the individual fits
            pfm.plot_individual_fits(
                results, 
                profile=config.pdm_profile, 
                num_peaks=len(config.peak_centers), 
                plot_every=config.pdm_plot_every, 
                show_individual_peaks=True,
                grid_n=config.plotgrid_n
            )
    
    
    # --- Fitting ---
    fit_results_df = None
    
    if config.fit_spectra:
        
        fit_results_df = snom_utils.fit_all_spectra(
            da=target_da,
            wmin=config.fit_wmin,
            wmax=config.fit_wmax,
            peak_centers=config.peak_centers,
            shape=config.peak_shape,
            center_tolerance=config.center_tolerance,
            fwhm_bounds=config.fwhm_bounds,
            grid_n=config.plotgrid_n if config.plot_fitresults else 0,
        )
        
        if config.plot_fitstatistics and config.correlations_to_check:
            snom_utils.plot_correlations(fit_results_df, config.correlations_to_check)

        if config.EXPORT:
            safe_title = title_label.replace(" ", "_")
            save_filename_fits = export_dir / f"{safe_title}_fit_results.csv"

            fit_results_df.to_csv(save_filename_fits, index=False)
            print(f"Fit results saved to: {save_filename_fits.name}")

    return fit_results_df


def process_spectra(target_folder: Path, config: ProcessSettings):
    """Main workflow to process references and samples based on config."""
    
    if not config.ref_nums:
        raise ValueError("Execution halted: 'ref_nums' cannot be empty. A reference is required.")
    # 1. Process Master Reference
    ref_files = [
        target_folder / config.ref_format.format(num=num) for num in config.ref_nums
    ]
    reference_interferograms = snom_utils.package_point_interferograms(ref_files)

    reference_corrected, _ = xr.apply_ufunc(
        snom_utils.baseline_correct,
        reference_interferograms,
        input_core_dims=[["step"]],
        output_core_dims=[["step"], ["step"]],
        vectorize=True,
        kwargs={"deg": 3},
    )

    ref_spectra = snom_utils.process_all_spectra(
        reference_corrected, pad_pow=2, auto_center=True
    )
    flat_ref_spectra = snom_utils.batch_phase_correct(
        ref_spectra,
        fit_regions=config.phase_fit_regions,
        out_wmin=config.plotlims[0],
        out_wmax=config.plotlims[1],
        correction_order=2,
    )

    array_fit_df = None
    data = None
    master_reference = flat_ref_spectra.mean(dim="point")

    # 2. Process Array Data
    if config.array_num is not None:
        axz_file = target_folder / config.array_format.format(num=config.array_num)

        if not axz_file.exists():
            raise FileNotFoundError(
                f"Execution halted: AXZ file {axz_file.name} not found."
            )
        data = load_axz_as_dict(axz_file)
        
        # --- DEFINE ARRAY EXPORT FOLDER ---
        array_export_name = config.export_foldername if config.export_foldername else f"Extracted_Data_Array_{config.array_num}"
        array_export_dir = target_folder / array_export_name
        
        if config.EXPORT:
            snom_utils.export_axz_contents(
                data, array_export_dir, afm_format="gwy"
            )

        sample_interfs = snom_utils.package_axz_interferograms(
            data, target_signal=config.target_signal
        )

        array_fit_df = _process_and_plot_samples(
            sample_interfs,
            f"Array {config.array_num}",
            target_folder,
            config,
            flat_ref_spectra,
            master_reference,
            reference_corrected,
            array_export_dir,
        )
        
        if config.EXPORT:
            df = pd.DataFrame(
                snom_utils.extract_raw_spectra_coords(data),
                columns=["X_Absolute", "Y_Absolute"],
            )
            df.to_csv(
                array_export_dir / "raw_coordinates.csv",
                index=False,
            )

    # 3. Process Point Spectra Data
    if config.samp_nums:
        
        # Scan the directory once
        all_intfgm_files = list(target_folder.glob(f"*{config.ref_format[-15:]}"))
        lsamp = len(all_intfgm_files)
        
        print(f"{lsamp}")
        print(f"*{config.ref_format[-15:]}")
        
        samp_files = [
            f for f in all_intfgm_files 
            if any(f.name.startswith(str(num)) for num in config.samp_nums)
        ]
        
        if samp_files:
            # --- DEFINE POINT SPECTRA EXPORT FOLDER ---
            point_export_name = config.export_foldername if config.export_foldername else "Extracted_Data_Point_Spectra"
            point_export_dir = target_folder / point_export_name
            sample_interfs = snom_utils.package_point_interferograms(samp_files)
            _process_and_plot_samples(
                sample_interfs,
                "Point Spectra",
                target_folder,
                config,
                flat_ref_spectra,
                master_reference,
                reference_corrected,
                point_export_dir,
            )

    return data, array_fit_df


def process_afm_drift(
    axzdata: dict, target_folder: Path, config: ProcessSettings, fit_results_df=None
):
    """Processes AFM drift correction. Uses guard clauses to prevent deep nesting."""
    if config.array_num is None:
        return

    data = axzdata
    processor = snom_utils.AFMArray(data)
    processor.extract_scans()
    processor.align_rows(method="median", mask_percentile=80)
    processor.flatten_scans(degree=1, method="2D", mask_percentile=80)

    if config.plot_afm:
        processor.plot_scans(show_corrected=False, num_stdev=5)

    if config.drift_correct:
        processor.calculate_and_apply_drift()
        processor.plot_and_fit_drift(poly_degree=config.drift_poly_degree)

        if config.plot_afm:
            processor.plot_scans(show_corrected=config.drift_correct, num_stdev=5)

    if not config.drift_correct:
        return

    if not config.relative_array_coords:
        calibration_result = snom_utils.calibrate_start_point(target_folder)
    else:
        calibration_result = config.relative_array_coords
    

    print(f"Relative array coordinates: {calibration_result}")
    
    
    if calibration_result is None:
        print("Skipping spectra alignment (calibration aborted).")
        return

    rel_x, rel_y = calibration_result
    processor.align_and_plot_spectra(rel_x, rel_y)

    df = processor.apply_poly_drift_to_spectra(
        fit_results_df=fit_results_df,
        map_parameter=config.map_parameter,
        shared_colormap=config.shared_colormap,
        interp_resolution=config.interpolation_res,
        grid_shape=config.array_scan_dim,
    )
    if config.EXPORT:
        array_export_name = config.export_foldername if config.export_foldername else f"Extracted_Data_Array_{config.array_num}"
        save_folder = (
            target_folder / array_export_name / "corrected_coordinates.csv"
        )
        df.to_csv(save_folder, index=False)
        print(f"Corrected map coordinates saved to: {save_folder}")



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
    axzdata, fit_df = process_spectra(target_folder, config)

    if config.plot_afm or config.drift_correct:
        process_afm_drift(axzdata, target_folder, config, fit_results_df=fit_df)
    
    try:
        get_ipython()
    except NameError:
        plt.show(block=True)

if __name__ == "__main__":
    main()
