#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 19:25:12 2026

@author: mategarai
"""

"""
Helper functions for data processing and analysis.

This module contains utility functions for loading data, signal processing, 
and plotting to keep main analysis scripts clean.
"""

# Standard Library Imports
import base64
import math
import sys
import traceback
from pathlib import Path
from PIL import Image
from collections import Counter

# Third-Party GUI & Plotting Imports
import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PyQt5.QtWidgets import QApplication, QFileDialog

# Third-Party Data & Math Imports
import numpy as np
import pandas as pd
import xarray as xr
from skimage.registration import phase_cross_correlation

# from skimage.restoration import inpaint_biharmonic
import gwyfile

# SciPy Imports (Grouped together)
from scipy.ndimage import shift, uniform_filter1d, zoom
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.special import erf

mpl.rcParams["savefig.dpi"] = 300
# ==========================================
# Data Loading & I/O
# ==========================================


class AFMArray:

    def __init__(self, axz_dict):
        """Initialize with the dictionary loaded from your AXZ file."""
        self.raw_dict = axz_dict
        self.scans = None  # Will hold the 3D numpy array of images
        self.drift_path = None  # Will hold the (Y, X) cumulative shifts
        self.corrected_scans = None  # Will hold the aligned images

    def extract_scans(self, channel_index=0, flatten=True):
        """
        Extracts and decodes the 2D AFM scans. Automatically detects how many
        channels are saved per scan based on the XML TimeStamp.
        Filters out scans that have pixel resolution OR physical dimension mismatches.
        """
        scan_list = []
        try:
            height_maps = self.raw_dict["Document"]["HeightMaps"]["HeightMap"]
            if not isinstance(height_maps, list):
                height_maps = [height_maps]

            first_timestamp = height_maps[0].get("TimeStamp", None)

            channel_stride = 1
            for h_map in height_maps[1:]:
                if h_map.get("TimeStamp", None) == first_timestamp:
                    channel_stride += 1
                else:
                    break

            print(f"Auto-detected {channel_stride} channels per scan.")
            filtered_maps = height_maps[channel_index::channel_stride]

            for h_map in filtered_maps:
                # --- 1. Extract Pixel Resolution (Shape) ---
                x_val = h_map["Resolution"]["X"]
                y_val = h_map["Resolution"]["Y"]
                if isinstance(x_val, dict):
                    x_val = list(x_val.values())[0]
                if isinstance(y_val, dict):
                    y_val = list(y_val.values())[0]
                res_x, res_y = int(x_val), int(y_val)

                # --- 2. Extract Physical Dimensions (Size) ---
                size_x = h_map["Size"]["X"]
                size_y = h_map["Size"]["Y"]
                if isinstance(size_x, dict):
                    size_x = list(size_x.values())[0]
                if isinstance(size_y, dict):
                    size_y = list(size_y.values())[0]
                phys_size = (float(size_x), float(size_y))

                # --- 3. Decode Matrix ---
                b64_string = h_map["SampleBase64"]
                if isinstance(b64_string, dict):
                    b64_string = list(b64_string.values())[0]

                decoded_bytes = base64.b64decode(b64_string)
                scan_1d = np.frombuffer(decoded_bytes, dtype=np.float32)
                scan_matrix = scan_1d.reshape((res_y, res_x))
                scan_matrix = np.flipud(scan_matrix)

                if flatten:
                    Y, X = np.indices(scan_matrix.shape)
                    x_flat = X.flatten()
                    y_flat = Y.flatten()
                    z_flat = scan_matrix.flatten()

                    A = np.c_[x_flat, y_flat, np.ones_like(x_flat)]
                    C, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

                    plane = C[0] * X + C[1] * Y + C[2]
                    scan_matrix = scan_matrix - plane

                scan_matrix = scan_matrix - np.min(scan_matrix)

                # Store a tuple of the matrix AND its physical size
                scan_list.append((scan_matrix, phys_size))

            # --- 4. Enforce Uniform Shape AND Dimensions ---
            if not scan_list:
                print("Error: No scans were extracted.")
                return None

            # Create a "signature" for each scan: ((res_y, res_x), (phys_x, phys_y))
            signatures = [(scan.shape, phys) for scan, phys in scan_list]
            most_common_signature = Counter(signatures).most_common(1)[0][0]
            target_shape, target_phys = most_common_signature

            uniform_scans = []
            for i, (scan, phys) in enumerate(scan_list):
                if scan.shape == target_shape and phys == target_phys:
                    uniform_scans.append(scan)
                else:
                    print(
                        f"Warning: Scan {i} skipped due to mismatch. "
                        f"Expected shape/size {target_shape} / {target_phys}, "
                        f"got {scan.shape} / {phys}."
                    )

            self.scans = np.array(uniform_scans)
            self.physical_size = target_phys

            print(
                f"Physical scan size: {self.physical_size[0]} x {self.physical_size[1]}"
            )
            print(
                f"Successfully extracted {len(self.scans)} scans of shape {target_shape}."
            )

        except Exception as e:
            print(f"Extraction failed: {e}")
            traceback.print_exc()

        return self.scans

    def calculate_and_apply_drift(self):
        """Calculates the frame-by-frame drift and aligns the images."""

        upsample_factor = 10

        if self.scans is None or len(self.scans) < 2:
            print("Not enough scans extracted to perform drift correction.")
            return

        num_frames = len(self.scans)
        self.corrected_scans = np.zeros_like(self.scans)
        self.corrected_scans[0] = self.scans[0]

        # Store the [Y, X] drift for each frame relative to frame 0
        self.drift_path = np.zeros((num_frames, 2))
        cumulative_shift = np.array([0.0, 0.0])

        for i in range(1, num_frames):
            shift_vector, _, _ = phase_cross_correlation(
                reference_image=self.scans[i - 1],
                moving_image=self.scans[i],
                upsample_factor=upsample_factor,
            )

            cumulative_shift += shift_vector
            self.drift_path[i] = cumulative_shift

            self.corrected_scans[i] = shift(
                input=self.scans[i], shift=cumulative_shift, mode="nearest"
            )

        return self.corrected_scans

    def plot_and_fit_drift(self, poly_degree=2):
        """Fits a polynomial to the X and Y drift paths, saves the fit, and plots them."""

        if not hasattr(self, "drift_path") or self.drift_path is None:
            print("Drift path is empty. Run calculate_and_apply_drift() first.")
            return None, None

        frames = np.arange(len(self.drift_path))
        y_drift = self.drift_path[:, 0]
        x_drift = self.drift_path[:, 1]  # Note: skimage returns (Y, X)

        # Bypass fitting if degree is 0, setting coefficients to zero
        if poly_degree == 0:
            p_y_coeffs = [0.0]
            p_x_coeffs = [0.0]
        else:
            p_y_coeffs = np.polyfit(frames, y_drift, poly_degree)
            p_x_coeffs = np.polyfit(frames, x_drift, poly_degree)

        # --- NEW: Save the poly1d objects to the class instance ---
        self.drift_poly_y = np.poly1d(p_y_coeffs)
        self.drift_poly_x = np.poly1d(p_x_coeffs)

        # Create a high-resolution array of frames for smooth plotting
        frames_high_res = np.linspace(0, len(self.drift_path) - 1, 500)

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # X Drift Plot
        ax1.scatter(frames, x_drift, label="Measured Drift", color="blue")
        ax1.plot(
            frames_high_res,
            self.drift_poly_x(frames_high_res),
            label=f"Fit (Degree {poly_degree})",
            color="red",
            linestyle="--",
        )
        ax1.set_title("X-Axis Drift over Time")
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Shift (Pixels)")
        ax1.legend()
        ax1.grid(True)

        # Y Drift Plot
        ax2.scatter(frames, y_drift, label="Measured Drift", color="green")
        ax2.plot(
            frames_high_res,
            self.drift_poly_y(frames_high_res),
            label=f"Fit (Degree {poly_degree})",
            color="red",
            linestyle="--",
        )
        ax2.set_title("Y-Axis Drift over Time")
        ax2.set_xlabel("Frame Number")
        ax2.set_ylabel("Shift (Pixels)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

        return self.drift_path, (self.drift_poly_y, self.drift_poly_x)

    def align_rows(self, method="median", mask_percentile=80):
        """
        Removes horizontal striping by aligning the baseline of each row.

        Parameters:
        -----------
        method : str
            'median' (recommended for AFM to ignore outliers) or 'mean'.
        mask_percentile : float or None
            If provided, calculates the row's mean/median using only pixels
            below this percentile, preventing tall features from shifting the row down.
        """

        if self.scans is None:
            print("Error: No scans found. Run extract_scans() first.")
            return

        # Using .copy() so we don't accidentally modify the original raw data reference
        scans = np.array(self.scans).copy()

        if mask_percentile is not None:

            thresholds = np.percentile(scans, mask_percentile, axis=2, keepdims=True)

            masked_data = np.where(scans < thresholds, scans, np.nan)
        else:
            masked_data = scans

        if method == "median":
            offsets = np.nanmedian(masked_data, axis=2, keepdims=True)
        else:
            offsets = np.nanmean(masked_data, axis=2, keepdims=True)

        if np.any(np.isnan(offsets)):
            if method == "median":
                fallback = np.median(scans, axis=2, keepdims=True)
            else:
                fallback = np.mean(scans, axis=2, keepdims=True)
            offsets = np.where(np.isnan(offsets), fallback, offsets)

        scans -= offsets

        scan_mins = np.min(scans, axis=(1, 2), keepdims=True)
        scans -= scan_mins

        self.scans = scans
        print(f"Successfully aligned rows using the vectorized {method} method.")

    def flatten_scans(self, degree=1, method="2D", mask_percentile=80):
        """
        Applies polynomial background correction to all extracted scans using
        optimized linear algebra and matrix operations.
        """
        if self.scans is None:
            print("Error: No scans found.")
            return

        # Convert to 3D array: (num_scans, rows, cols)
        scans = np.array(self.scans).copy()
        num_scans, rows, cols = scans.shape

        # 1. VECTORIZED MASKING
        if mask_percentile is not None:
            # Calculate threshold for every scan in one step
            thresholds = np.percentile(
                scans, mask_percentile, axis=(1, 2), keepdims=True
            )
            mask = scans < thresholds
        else:
            mask = np.ones_like(scans, dtype=bool)

        # 2. METHOD 2D: Matrix Multiplication Approach
        if method == "2D":
            # Build the coordinate grids ONCE for all scans
            Y_indices, X_indices = np.indices((rows, cols))
            X_flat = X_indices.ravel()
            Y_flat = Y_indices.ravel()

            # Build the Vandermonde (Design) Matrix ONCE
            A_cols = []
            for i in range(degree + 1):
                for j in range(degree + 1 - i):
                    A_cols.append((X_flat**i) * (Y_flat**j))

            # A_full shape is (rows*cols, number_of_coefficients)
            A_full = np.column_stack(A_cols)

            for s_idx in range(num_scans):
                scan_flat = scans[s_idx].ravel()
                mask_flat = mask[s_idx].ravel()

                # Filter matrix and data using the mask
                # Fallback to all pixels if the mask excluded everything
                if not np.any(mask_flat):
                    mask_flat = np.ones_like(mask_flat, dtype=bool)

                A_fit = A_full[mask_flat]
                z_fit = scan_flat[mask_flat]

                # Solve least squares
                C, _, _, _ = np.linalg.lstsq(A_fit, z_fit, rcond=None)

                # VECTORIZED SURFACE RECONSTRUCTION
                # Instead of looping over pixels and degrees, matrix multiply the
                # full design matrix by the coefficients (A @ C)
                bg_surface = (A_full @ C).reshape(rows, cols)
                scans[s_idx] -= bg_surface

        # 3. METHOD X/Y: Optimized 1D Evaluation
        elif method in ["X", "Y"]:
            is_x = method == "X"
            primary_len = rows if is_x else cols
            axis_len = cols if is_x else rows
            axis = np.arange(axis_len)

            for s_idx in range(num_scans):
                scan = scans[s_idx]
                scan_mask = mask[s_idx]

                for i in range(primary_len):
                    line_data = scan[i, :] if is_x else scan[:, i]
                    line_mask = scan_mask[i, :] if is_x else scan_mask[:, i]

                    if not np.any(line_mask):
                        line_mask = np.ones_like(line_mask, dtype=bool)

                    x_fit = axis[line_mask]
                    z_fit = line_data[line_mask]

                    coeffs = np.polyfit(x_fit, z_fit, degree)

                    # Use np.polyval instead of instantiating np.poly1d objects
                    background_line = np.polyval(coeffs, axis)

                    if is_x:
                        scan[i, :] -= background_line
                    else:
                        scan[:, i] -= background_line
        else:
            print(f"Unknown method '{method}'.")
            return

        # 4. GLOBAL VECTORIZED MIN-SHIFT
        scan_mins = np.min(scans, axis=(1, 2), keepdims=True)
        scans -= scan_mins

        self.scans = scans
        print(f"Successfully applied degree {degree} '{method}' polynomial flattening.")

    def plot_scans(self, show_corrected=False, num_stdev=2):
        """
        Plots all AFM scans from the array in a square grid with a shared color scale.

        Parameters:
        -----------
        show_corrected : bool
            If True, plots the drift-corrected scans instead of the raw ones.
        num_stdev : float
            The number of standard deviations from the mean to use for the color scale limits.
            Values outside this range will be overexposed/clipped to the end colors.
        """

        data_to_plot = self.corrected_scans if show_corrected else self.scans

        if data_to_plot is None:
            print("No data to plot. Run extraction first.")
            return

        num_plots = len(data_to_plot)

        # --- Calculate limits using Mean and Standard Deviation ---
        mean_val = np.mean(data_to_plot)
        std_val = np.std(data_to_plot)

        global_vmin = mean_val - (num_stdev * std_val)
        global_vmax = mean_val + (num_stdev * std_val)

        # Calculate grid dimensions to be as close to a square as possible
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        # Create the figure with dynamically scaled sizing
        # Add constrained_layout=True to the subplots call
        fig, axes = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
        )

        # Flatten the axes array to iterate through it easily 1D
        if num_plots == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()

        title_prefix = "Corrected" if show_corrected else "Raw"

        # Set up physical dimensions if they exist
        extent_kwargs = {}
        xaxis_label = "Pixels"
        yaxis_label = "Pixels"
        if hasattr(self, "physical_size") and self.physical_size is not None:
            extent_kwargs["extent"] = [
                0,
                self.physical_size[0],
                0,
                self.physical_size[1],
            ]
            xaxis_label = "X (µm)"
            yaxis_label = "Y (µm)"

        im = None

        for idx, ax in enumerate(axes_flat):
            if idx < num_plots:
                # Plot the actual data using the stdev-based vmin and vmax
                im = ax.imshow(
                    data_to_plot[idx],
                    cmap="afmhot",
                    origin="lower",
                    vmin=global_vmin,
                    vmax=global_vmax,
                    **extent_kwargs,
                )
                ax.set_title(f"{title_prefix} Frame {idx}")
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel(yaxis_label)
            else:
                # Turn off the axes for any empty spots in the grid
                ax.axis("off")

        # Add a single global colorbar for the entire figure
        if im is not None:
            cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.02, pad=0.04)
            cbar.set_label("Height (nm)")

        plt.show()

    def align_and_plot_spectra(self, rel_x, rel_y):
        """
        Uses the calibrated relative start point to calculate the shift
        between the instrument's absolute motor coordinates and the local plot.
        """

        # 1. Convert the relative click to physical micrometers
        local_start_x = rel_x * self.physical_size[0]
        local_start_y = rel_y * self.physical_size[1]

        # # 2. Extract the absolute coordinates of all points from the XML
        # absolute_coords = []
        # spectra = self.raw_dict["Document"]["SNOMSpectra"]["AXDSNOMSpectrum"]
        # if not isinstance(spectra, list):
        #     spectra = [spectra]

        # for spec in spectra:
        #     loc = spec["Location"]

        #     # X extraction
        #     x_val = loc["X"]
        #     if isinstance(x_val, dict):
        #         abs_x = float(x_val.get("#text", list(x_val.values())[0]))
        #     else:
        #         abs_x = float(x_val)

        #     # Y extraction
        #     y_val = loc["Y"]
        #     if isinstance(y_val, dict):
        #         abs_y = float(y_val.get("#text", list(y_val.values())[0]))
        #     else:
        #         abs_y = float(y_val)

        #     absolute_coords.append([abs_x, abs_y])

        # absolute_coords = np.array(absolute_coords)
        
        # 2. Extract the absolute coordinates of all points
        absolute_coords = extract_raw_spectra_coords(self.raw_dict)
        
        if len(absolute_coords) == 0:
            print("Error: Could not extract coordinates from XML.")
            return

        # 3. Calculate the global offset using the FIRST point
        offset_x = local_start_x - absolute_coords[0, 0]
        offset_y = local_start_y - absolute_coords[0, 1]

        # 4. Apply this offset to ALL points
        self.spectra_local_coords = absolute_coords + [offset_x, offset_y]

        print(f"Aligned {len(self.spectra_local_coords)} points to local coordinates.")

    def apply_poly_drift_to_spectra(
        self,
        fit_results_df=None,
        map_parameter=None,
        bg_scan_idx=0,
        shared_colormap=False,
        interp_resolution=None,
        grid_shape=None,
    ):
        """
        Evaluates the previously saved polynomial fit for each spectrum point
        assuming they were taken sequentially, and plots the actual deformed
        measurement locations onto the specified AFM scan.
        Supports map_parameter as a string or a list of strings to plot multiple maps.
        If shared_colormap=True, locks the color scale across all generated plots.
        """

        if not hasattr(self, "drift_path") or self.drift_path is None:
            print("Error: No drift path found. Run calculate_and_apply_drift() first.")
            return

        if not hasattr(self, "drift_poly_y") or getattr(self, "drift_poly_y") is None:
            print("Error: No polynomial fit found. Run plot_and_fit_drift() first.")
            return

        if (
            not hasattr(self, "spectra_local_coords")
            or self.spectra_local_coords is None
        ):
            print(
                "Error: No spectra coordinates found. Run align_and_plot_spectra() first."
            )
            return

        # 1. Map the spectra to the "frame timeline"
        num_spectra = len(self.spectra_local_coords)
        num_frames = len(self.drift_path)
        spectra_timeline = np.linspace(0, num_frames - 1, num_spectra)

        # 2. Evaluate the SAVED continuous polynomial drift for each point
        dy_pix = self.drift_poly_y(spectra_timeline)
        dx_pix = self.drift_poly_x(spectra_timeline)

        # 3. Convert pixel drift to physical micrometers
        res_y, res_x = self.scans[bg_scan_idx].shape
        pixel_size_x = self.physical_size[0] / res_x
        pixel_size_y = self.physical_size[1] / res_y

        dx_um = dx_pix * pixel_size_x
        dy_um = dy_pix * pixel_size_y

        # 4. Apply the shift to the points
        self.corrected_spectra_coords = np.zeros_like(self.spectra_local_coords)
        self.corrected_spectra_coords[:, 0] = self.spectra_local_coords[:, 0] + dx_um
        self.corrected_spectra_coords[:, 1] = self.spectra_local_coords[:, 1] + dy_um

        print(f"Applied stored polynomial drift to {num_spectra} points.")

        # Ensure we are working with 1D numpy arrays for plotting
        x_coords = self.corrected_spectra_coords[:, 0]
        y_coords = self.corrected_spectra_coords[:, 1]

        # Define the physical extent for all plots to force true aspect ratio
        extent = [0, self.physical_size[0], 0, self.physical_size[1]]

        # Standardize map_parameter into a list
        if isinstance(map_parameter, str):
            parameters_to_plot = [map_parameter]
        elif isinstance(map_parameter, list):
            parameters_to_plot = map_parameter
        else:
            parameters_to_plot = [None]

        # =======================================================
        # GLOBAL COLORMAP CALCULATOR
        # =======================================================
        global_vmin, global_vmax = None, None
        if shared_colormap and fit_results_df is not None:
            all_valid_vals = []
            for param in parameters_to_plot:
                if param is not None and param in fit_results_df.columns:
                    all_valid_vals.append(fit_results_df[param].values)

            if all_valid_vals:
                combined_vals = np.concatenate(all_valid_vals)
                # Find the absolute min and max, ignoring any NaNs (failed fits)
                global_vmin = np.nanmin(combined_vals)
                global_vmax = np.nanmax(combined_vals)

        # Loop through each requested parameter
        for param in parameters_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.imshow(
                self.scans[bg_scan_idx],
                extent=extent,
                aspect="equal",
                cmap="afmhot",
                origin="lower",
                alpha=1,
            )

            if fit_results_df is not None and param is not None:
                if param in fit_results_df.columns:
                    colors = fit_results_df[param].values

                    valid_mask = ~np.isnan(colors)
                    invalid_mask = np.isnan(colors)

                    # Determine limits for THIS specific plot
                    if shared_colormap and global_vmin is not None:
                        c_min, c_max = global_vmin, global_vmax
                    else:
                        c_min, c_max = np.nanmin(colors), np.nanmax(colors)

                    # Add vmin and vmax to lock the scale
                    scatter = ax.scatter(
                        x_coords[valid_mask],
                        y_coords[valid_mask],
                        c=colors[valid_mask],
                        cmap="jet",
                        vmin=c_min,
                        vmax=c_max,
                        s=150,
                        edgecolor="k",
                        linewidth=0.5,
                        zorder=5,
                    )

                    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label(param.replace("_", " ").title(), fontsize=12)

                    if invalid_mask.any():
                        ax.scatter(
                            x_coords[invalid_mask],
                            y_coords[invalid_mask],
                            color="grey",
                            marker="x",
                            s=40,
                            zorder=6,
                            label="Fit Failed",
                        )
                        ax.legend()

                    ax.set_title(
                        f"Drift-Corrected Locations: {param.replace('_', ' ').title()}"
                    )
                    ax.set_xlabel(r"X ($\mu$m)")
                    ax.set_ylabel(r"Y ($\mu$m)")

                    # =======================================================
                    # INTERPOLATED HEATMAP FIGURE
                    # =======================================================
                    
                    if valid_mask.any():
                        # Determine grid dimensions
                        if grid_shape is not None:
                            rows, cols = grid_shape
                            if rows * cols != len(x_coords):
                                print(f"Warning: grid_shape {grid_shape} does not match {len(x_coords)} points. Using scatter plot only.")
                                continue
                        else:
                            # INFER GRID SHAPE FROM RAW COORDINATES
                            # Round to 2 decimal places (10 nanometers) to handle floating-point precision noise
                            unique_x = len(np.unique(np.round(self.spectra_local_coords[:, 0], decimals=2)))
                            unique_y = len(np.unique(np.round(self.spectra_local_coords[:, 1], decimals=2)))
                            
                            rows = unique_y
                            cols = unique_x
                            
                            # Validation fallback: if X * Y doesn't match total points (e.g., sparse or arbitrary points)
                            if rows * cols != len(x_coords):
                                print(f"Warning: Inferred grid {rows}x{cols} does not match {len(x_coords)} total points. Skipping heatmap interpolation.")
                                continue

                        # Linear arrays (1D) cannot generate a 2D mesh surface.
                        # The scatter plot above already handles their visualization.
                        if rows < 2 or cols < 2:
                            continue

                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        ax2.imshow(
                            self.scans[bg_scan_idx],
                            extent=extent,
                            aspect="equal",
                            cmap="afmhot",
                            origin="lower",
                            alpha=1,
                        )

                        # 1. Reshape the 1D arrays into 2D matrices using explicit dimensions
                        X_2D = x_coords.reshape((rows, cols))
                        Y_2D = y_coords.reshape((rows, cols))
                        C_2D = colors.reshape((rows, cols))

                        # 2. Set upsampling factor 
                        # Use a tuple to allow independent scaling if necessary in the future
                        zoom_factors = (interp_resolution, interp_resolution) if interp_resolution else (1, 1)

                        # 3. Upsample the deformed coordinate grid and the values
                        # order=1 uses bilinear interpolation, order=3 uses cubic
                        X_highres = zoom(X_2D, zoom_factors, order=1)
                        Y_highres = zoom(Y_2D, zoom_factors, order=1)
                        C_highres = zoom(C_2D, zoom_factors, order=3)

                        # 4. Plot the high-resolution deformed map
                        heatmap = ax2.pcolormesh(
                            X_highres,
                            Y_highres,
                            C_highres,
                            cmap="jet",
                            shading="nearest",
                            alpha=1,
                            zorder=2,
                            vmin=c_min,
                            vmax=c_max,
                        )

                        cbar2 = fig2.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
                        cbar2.set_label(param.replace("_", " ").title(), fontsize=12)

                        # Optional: overlay the scatter points to verify alignment
                        ax2.scatter(
                            x_coords,
                            y_coords,
                            color="black",
                            marker=".",
                            s=10,
                            alpha=0.3,
                            zorder=5,
                        )

                        ax2.set_title(
                            f"Deformed Pixel Map: {param.replace('_', ' ').title()}"
                        )
                        ax2.set_xlabel(r"X ($\mu$m)")
                        ax2.set_ylabel(r"Y ($\mu$m)")
                        ax2.set_xlim(extent[0], extent[1])
                        ax2.set_ylim(extent[2], extent[3])

                        fig2.tight_layout()
                    # =======================================================

                else:
                    print(
                        f"Warning: '{param}' not found in fit results. Falling back to cyan"
                    )
                    ax.scatter(x_coords, y_coords, color="cyan", s=40, zorder=5)
            else:
                ax.scatter(x_coords, y_coords, color="cyan", s=40, zorder=5)

            fig.tight_layout()

        plt.show()

        df = pd.DataFrame({"map_x": x_coords, "map_y": y_coords})
        return df


def calibrate_start_point(target_folder):
    """
    Opens a native Qt file dialog to select a PNG image, then allows
    the user to click to define the AFM scan area and the location of
    the first spectrum point.

    Returns:
        tuple: (relative_x, relative_y) as a fraction of the scan size (0.0 to 1.0).
    """

    # This grabs it without trying to start a conflicting second one.
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    initial_dir = str(Path(target_folder).resolve())

    print("Opening file browser...")
    print("Select AFM Image for Calibration")
    # Open the native Qt file dialog
    img_path_str, _ = QFileDialog.getOpenFileName(
        None,
        "Select AFM Image for Calibration",
        initial_dir,
        "PNG Images (*.png);;All Files (*)",
    )

    if not img_path_str:
        print("Calibration cancelled: No file selected.")
        return None

    img_path = Path(img_path_str)
    print(f"Selected image: {img_path.name}")

    # Load and display the selected image
    img = mpimg.imread(str(img_path))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(
        "Interactive Calibration\n1. Top-Left of Scan\n2. Bottom-Right of Scan\n3. First Spectrum Point"
    )
    ax.axis("off")

    ax.cursor = Cursor(ax, useblit=True, color="red", linewidth=1)

    print("Waiting for 3 clicks on the image...")
    print(" 1. Click the TOP-LEFT corner of the AFM scan area.")
    print(" 2. Click the BOTTOM-RIGHT corner of the AFM scan area.")
    print(" 3. Click the exact location of the FIRST spectrum point.")

    # timeout=0 means it waits indefinitely
    clicks = plt.ginput(3, timeout=0)
    plt.close(fig)

    if len(clicks) < 3:
        print("Calibration cancelled: Did not receive 3 clicks.")
        return None

    # Process the clicks
    x_tl, y_tl = clicks[0]
    x_br, y_br = clicks[1]
    x_p1, y_p1 = clicks[2]

    scan_width = x_br - x_tl
    scan_height = y_br - y_tl

    rel_x = (x_p1 - x_tl) / scan_width

    # Invert Y to match the AFM origin='lower' standard
    rel_y_from_top = (y_p1 - y_tl) / scan_height
    rel_y = 1.0 - rel_y_from_top

    print("\nCalibration Successful!")
    print(
        f"First point is at {rel_x*100:.1f}% width, {rel_y*100:.1f}% height (from bottom)."
    )
    
    return [float(rel_x), float(rel_y)]


def extract_axz_interferogram(spectrum_dict, target_signal):
    """Finds and decodes the target interferogram from a spectrum dictionary."""

    ig_list = spectrum_dict.get("Interferograms", {}).get("AXDSNOMInterferogram", [])

    for ig in ig_list:
        if ig.get("DataSignal", {}).get("Text") == target_signal:
            try:
                b64_data = ig["Data"]["Text"]
                data_array = np.frombuffer(base64.b64decode(b64_data), dtype=np.float32)

                sweep_start = float(ig["SweepStart"]["Text"])
                sweep_end = float(ig["SweepEnd"]["Text"])
                points = int(ig["Points"]["Text"])

                # Check for invalid parameters that would break linspace
                if np.isnan(sweep_start) or np.isnan(sweep_end) or points <= 1:
                    return None, None

                z_axis = (
                    np.linspace(sweep_start, sweep_end, points) * 1e2
                )  # Units in cm

                return z_axis, data_array
            except (KeyError, ValueError, TypeError):
                # Catch any missing or corrupted text fields
                return None, None

    return None, None


def package_axz_interferograms(data, target_signal):

    spectra_list = data["Document"]["SNOMSpectra"]["AXDSNOMSpectrum"]
    all_intensities = []
    all_stage_positions = []
    expected_length = None

    # 2. Extract Data
    for i, spec in enumerate(spectra_list):
        stage_pos, intensity = extract_axz_interferogram(
            spec, target_signal=target_signal
        )

        # Error check: ensure all arrays have the same number of points
        if expected_length is None:
            expected_length = len(intensity)
        elif len(intensity) != expected_length:
            raise ValueError(
                f"Shape mismatch at point {i}: "
                f"Found {len(intensity)} points, expected {expected_length}."
            )

        all_intensities.append(intensity)
        all_stage_positions.append(stage_pos)

    # Convert lists to 2D NumPy arrays (Shape: #points rows, N columns)
    intensity_matrix = np.array(all_intensities)/1e6  # This is in uV divide by 1e6 for Volts
    stage_position_matrix = np.array(all_stage_positions)
    num_points = intensity_matrix.shape[0]  # number of points
    num_steps = intensity_matrix.shape[1]  # The length of the interferogram

    # 3. Build the simplified xarray
    sample_interferorgams = xr.DataArray(
        data=intensity_matrix,  # Piece 3: Your intensity array
        dims=("point", "step"),
        coords={
            "point": np.arange(num_points),  # Piece 1: Your point # (0 to 63)
            "step": np.arange(num_steps),  # The raw index of the array
            # Piece 2: Your stage position array, linked to each specific point and step
            "stage_position_mm": (("point", "step"), stage_position_matrix * 10.0),
        },
        name="Sample_Interferograms",
    )
    return sample_interferorgams


def package_point_interferograms(file_paths):
    """
    Reads multiple interferograms from a list of CSV/txt files and packages them into an xarray.DataArray.
    Assumes Column 0 is stage position. Extracts valid intensities from every other column
    (Col 1, Col 3, Col 5...) while ignoring the intermediate columns.
    Automatically truncates all sweeps to match the length of the shortest sweep.
    """
    raw_intensities = []
    raw_stage_positions = []

    for i, file_path in enumerate(file_paths):
        # Read the file. delimiter=',' handles the CSV format.
        try:
            data = np.loadtxt(file_path, delimiter=",")
        except ValueError:
            # Fallback: If it fails, try without a delimiter in case
            # they are actually tab- or space-separated .txt files
            data = np.loadtxt(file_path)

        # The first column is always the position
        stage_pos = data[:, 0] / 1.0e3

        # Get the total number of columns in this file
        num_columns = data.shape[1]
        # num_columns = 3

        # Loop through intensity columns: start at index 1, step by 2
        for col_idx in range(1, num_columns, 2):
            intensity = data[:, col_idx]

            raw_intensities.append(intensity)
            raw_stage_positions.append(stage_pos)

    # 1. Find the shortest array across ALL loaded interferograms
    min_length = min(len(arr) for arr in raw_intensities)

    # 2. Truncate all arrays to the minimum length and build matrices
    intensity_matrix = np.array([arr[:min_length] for arr in raw_intensities])
    stage_position_matrix = np.array([arr[:min_length] for arr in raw_stage_positions])

    num_points = intensity_matrix.shape[0]
    num_steps = intensity_matrix.shape[1]  # The length of the interferogram

    # 3. Build the simplified xarray
    sample_interferograms = xr.DataArray(
        data=intensity_matrix,
        dims=("point", "step"),
        coords={
            "point": np.arange(num_points),
            "step": np.arange(num_steps),
            # Multiply by 10.0 to convert to your specified units (cm)
            "stage_position_mm": (("point", "step"), stage_position_matrix),
        },
        name="Sample_Interferograms",
    )

    return sample_interferograms


# ==========================================
# Signal Processing & Math
# ==========================================


def phase_correct_spectrum(
    wavenumbers,
    complex_spectrum,
    fit_regions,
    out_wmin=None,
    out_wmax=None,
    correction_order=1,
):
    """
    Corrects instrumental phase twist by fitting a polynomial to specified background regions,
    and applies it over an independent output range.
    """
    # 1. Create a mask for just the background fitting regions
    fit_mask = np.zeros_like(wavenumbers, dtype=bool)
    for wmin, wmax in fit_regions:
        fit_mask |= (wavenumbers >= wmin) & (wavenumbers <= wmax)

    wN_fit = wavenumbers[fit_mask]
    spec_fit = complex_spectrum[fit_mask]

    # 2. Extract and unwrap the phase for fitting
    raw_phase = np.unwrap(np.angle(spec_fit))

    # 3. Fit a polynomial to the phase twist ONLY on the background
    p = np.polyfit(wN_fit, raw_phase, correction_order)

    # 4. Define the output range (decoupled from fit limits)
    if out_wmin is None:
        out_wmin = wavenumbers.min()
    if out_wmax is None:
        out_wmax = wavenumbers.max()

    out_mask = (wavenumbers >= out_wmin) & (wavenumbers <= out_wmax)
    wN_out = wavenumbers[out_mask]
    spec_out = complex_spectrum[out_mask]

    # 5. Generate and apply the correction curve for the ENTIRE output region
    phase_fit = np.polyval(p, wN_out)
    untwistedspec = spec_out * np.exp(-1j * phase_fit)

    return wN_out, untwistedspec


def baseline_correct(y, deg=3, exclude_arry=None):
    """
    Baseline subtraction for interferograms
     -intfgm: 1D array of intenisties
     -deg: degree of polynomial to fit to the interferogram
     -exclude_arry: boolean array with same dimension as y 1=exclude, 0=include
    """
    # Default to no exclusions if none provided
    if exclude_arry is None:
        exclude_arry = np.zeros_like(y, dtype=bool)

    # Create the x-axis (positions)
    pos = np.arange(len(y))

    # Fit polynomial only to the points NOT excluded
    p = np.polyfit(pos[~exclude_arry], y[~exclude_arry], deg)

    # Evaluate the polynomial across all positions
    fit = np.polyval(p, pos)

    # Subtract the baseline from the original signal
    y = y - fit

    return y, fit


def apodization(pos, intfgm):
    # 1. Baseline correct the interferogram
    # (Assuming our baseline_correct function from earlier is defined)
    int_corr, _ = baseline_correct(intfgm)

    # 2. Smooth the absolute value
    window_size = max(1, len(int_corr) // 50)
    envelope = uniform_filter1d(np.abs(int_corr), size=window_size)

    # 3. Find indices for max and half-max
    max_i = np.argmax(envelope)

    # np.argmax on a boolean array
    # by returning the index of the first True value
    half_i = np.argmax(envelope > (np.max(envelope) / 2))

    # 4. Calculate standard deviation of the Gaussian pulse
    st_dev = np.abs(pos[max_i] - pos[half_i])

    # Prevent division by zero just in case
    if st_dev == 0:
        st_dev = 1e-9

    # 5. Calculate CDF using the error function
    cdf = 0.5 * (1 + erf((pos - pos[half_i] + st_dev) / (np.sqrt(2) * st_dev)))

    # 6. Calculate exponential decay
    exp_tc = (np.max(pos) - np.min(pos)) / 4.0
    exp_decay = np.exp(-(pos - pos[max_i]) / exp_tc)

    # 7. Final Apodization function
    apod_function = exp_decay * cdf

    return int_corr * apod_function, apod_function


def make_single_spec(pos_z, int_z, pad_pow=2, auto_center_intfgm=True):
    """
    Calculates the spectrum from an interferogram.
    Returns wavenumbers (w_ns) and complex values (x_complex).
    """
    # 1. Remove zeros (acts as a filter for unrecorded/padded data points)
    valid_mask = pos_z != 0
    pos = pos_z[valid_mask]
    int_val = int_z[valid_mask]

    # 2. Baseline subtraction and position normalization
    int_val = int_val - np.mean(int_val)
    pos = pos - np.min(pos)

    # 3. Calculate and apply filtering (apodization) function
    int_filtered, _ = apodization(pos, int_val)

    # 4. Zero-padding to the next power of 2 (optimized for FFT speed)
    target_l = int(2 ** (np.ceil(np.log2(len(int_filtered))) + pad_pow))
    pad_amount = target_l - len(int_filtered)
    # np.pad adds zeros to the end of the array
    int_pad = np.pad(int_filtered, (0, pad_amount), mode="constant")

    # 5. Shift so maximum is centered (Phase correction prep)
    if auto_center_intfgm:
        envelope = np.abs(hilbert(int_filtered))
        max_i = np.argmax(envelope)
        # Simply roll the array so the peak is at the very beginning
        int_pad = np.roll(int_pad, -max_i)

    # 6. Calculate spectrum using Fast Fourier Transform
    # Run the FFT immediately while the peak is at index 0
    x_complex = np.fft.fft(int_pad)

    # 7. Calculate wavenumbers
    # Assuming pos is in millimeters, dividing by 10 converts to cm for cm^-1
    d = (2 * np.max(pos) / 10) * (len(int_pad) / len(int_val))
    w_n = 1 / d

    # Create the wavenumber axis (1-based index equivalent to MATLAB's 1:length)
    w_ns = np.arange(1, len(int_pad) + 1) * w_n

    return w_ns, x_complex


def process_all_spectra(spectra_da, pad_pow=2, auto_center=True):
    """
    Loops through an xarray of interferograms, applies make_single_spec to each,
    and returns a new xarray containing the complex frequency spectra.
    """
    # 1. Strip the xarray overhead ONCE before the loop
    # We extract the pure numpy matrices to iterate over them at C-speed
    pos_matrix = spectra_da.stage_position_mm.values
    int_matrix = spectra_da.values
    points_coords = spectra_da.point.values
    num_points = int_matrix.shape[0]

    # 2. Process the FIRST point to get the wavenumber axis and output shape
    wNs, first_complex = make_single_spec(
        pos_matrix[0], int_matrix[0], pad_pow=pad_pow, auto_center_intfgm=auto_center
    )

    # 3. PRE-ALLOCATE the output matrix
    complex_matrix = np.empty((num_points, len(first_complex)), dtype=np.complex128)

    # Insert the first point we already calculated
    complex_matrix[0] = first_complex

    # 4. Loop through the remaining points
    for i in range(1, num_points):
        _, X_complex = make_single_spec(
            pos_matrix[i],
            int_matrix[i],
            pad_pow=pad_pow,
            auto_center_intfgm=auto_center,
        )
        complex_matrix[i] = X_complex

    # 5. Build and return the brand new xarray
    frequency_spectra = xr.DataArray(
        data=complex_matrix,
        dims=("point", "wavenumber"),
        coords={"point": points_coords, "wavenumber": wNs},
        name="Complex_Spectrum",
    )

    return frequency_spectra


def batch_phase_correct(
    frequency_spectra, fit_regions, out_wmin=None, out_wmax=None, correction_order=2
):
    """
    Loops through an xarray of complex spectra, applies phase correction using background regions,
    and returns a new xarray over the specified output limits.
    """
    wavenumbers = frequency_spectra.wavenumber.values
    num_points = frequency_spectra.sizes["point"]

    all_untwisted_spectra = []
    new_wN_axis = None

    for i in range(num_points):
        spec_1d = frequency_spectra.values[i, :]

        wN_slice, untwisted_spec = phase_correct_spectrum(
            wavenumbers, spec_1d, fit_regions, out_wmin, out_wmax, correction_order
        )

        if new_wN_axis is None:
            new_wN_axis = wN_slice

        all_untwisted_spectra.append(untwisted_spec)

    untwisted_matrix = np.array(all_untwisted_spectra)

    phase_corrected_spectra = xr.DataArray(
        data=untwisted_matrix,
        dims=("point", "wavenumber"),
        coords={"point": frequency_spectra.point.values, "wavenumber": new_wN_axis},
        name="Phase_Corrected",
    )

    return phase_corrected_spectra


# ==========================================
# Plotting & Visualization
# ==========================================


def plot_intfgm(intfgm, title, pad_pow=2, auto_center=True):
    """
    Loops through an xarray of interferograms, applies pre-FFT processing
    (baseline, apodization, padding, centering), builds a new xarray, and plots it.
    """
    num_points = intfgm.sizes["point"]

    processed_intfgm = []
    stage_position = []
    target_l = None  # Track padded length

    for i in range(num_points):
        # 1. Extract the raw 1D arrays for this specific point
        pos_1d = intfgm.stage_position_mm.values[i, :]
        int_1d = intfgm.values[i, :]

        # Remove zeros (acts as a filter for unrecorded/padded data points)
        valid_mask = pos_1d != 0
        pos = pos_1d[valid_mask]
        int_val = int_1d[valid_mask]

        # 2. Baseline subtraction and position normalization
        int_val = int_val - np.mean(int_val)
        pos = pos - np.min(pos)

        # Extract the average step size so we can extrapolate the x-axis later
        dx = np.mean(np.diff(pos))

        # 3. Calculate and apply filtering (apodization) function
        # (Assuming apodization is defined elsewhere in your script)
        int_filtered, _ = apodization(pos, int_val)

        # 4. Zero-padding to the next power of 2
        target_l = int(2 ** (np.ceil(np.log2(len(int_filtered))) + pad_pow))
        pad_amount = target_l - len(int_filtered)

        # Pad the intensity with zeros
        int_pad = np.pad(int_filtered, (0, pad_amount), mode="constant")

        # Extrapolate the position array to match the new padded length perfectly
        pos_pad = np.arange(target_l) * dx

        if auto_center:
            max_i = np.argmax(int_pad)
            # Find the physical position of the peak, and subtract it from the whole array
            center_pos = pos_pad[max_i]
            pos_pad = pos_pad - center_pos

        processed_intfgm.append(int_pad)
        stage_position.append(pos_pad)

    # 6. Convert the lists into 2D NumPy matrices
    intfgm_matrix = np.array(processed_intfgm)
    stage_position_matrix = np.array(stage_position)
    num_steps = target_l  # Now num_steps is defined!

    # 7. Build the xarray
    processed_interferograms = xr.DataArray(
        data=intfgm_matrix,
        dims=("point", "step"),
        coords={
            "point": np.arange(num_points),
            "step": np.arange(num_steps),
            # Note: Check your original units here. * 10.0 converts mm to cm?
            "stage_position_mm": (("point", "step"), stage_position_matrix),
        },
        name=r"X2 Signal ($\mu$V)",
    )

    # 8. Plot the interferograms
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # 1. Generate a smooth list of colors from the 'viridis' colormap
    num_lines = processed_interferograms.sizes["point"]
    gradient_colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    # 2. Force the axis to use this color gradient for the next plot
    ax.set_prop_cycle(color=gradient_colors)
    processed_interferograms.plot.line(
        x="stage_position_mm", hue="point", ax=ax, add_legend=False, xlim=(-0.1, 0.2)
    )
    ax.set_title(title + " Interferograms")
    plt.tight_layout()
    plt.show


def plot_all_spectra(reference_spectra, sample_spectra, plotlims):

    sample_phase_data = np.rad2deg(np.angle(sample_spectra))
    reference_phase_data = np.rad2deg(np.angle(reference_spectra))
    sample_phase_spectra = sample_spectra.copy(data=sample_phase_data)
    reference_phase_spectra = reference_spectra.copy(data=reference_phase_data)

    sample_abs_data = np.abs(sample_spectra)
    reference_abs_data = np.abs(reference_spectra)
    sample_abs_spectra = sample_spectra.copy(data=sample_abs_data)
    reference_abs_spectra = reference_spectra.copy(data=reference_abs_data)

    sample_real_data = np.real(sample_spectra)
    reference_real_data = np.real(reference_spectra)
    sample_real_spectra = sample_spectra.copy(data=sample_real_data)
    reference_real_spectra = reference_spectra.copy(data=reference_real_data)

    sample_im_data = np.imag(sample_spectra)
    reference_im_data = np.imag(reference_spectra)
    sample_im_spectra = sample_spectra.copy(data=sample_im_data)
    reference_im_spectra = reference_spectra.copy(data=reference_im_data)

    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    reference_real_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[0, 0], add_legend=False
    )
    ax[0, 0].set_title("Real Reference")

    reference_im_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[0, 1], add_legend=False
    )
    ax[0, 1].set_title("Imaginary Reference")

    reference_abs_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[1, 0], add_legend=False
    )
    ax[1, 0].set_title("Amplitude Reference")

    reference_phase_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[1, 1], add_legend=False
    )
    ax[1, 1].set_title("Phase Reference")

    sample_real_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[0, 2], add_legend=False
    )
    ax[0, 2].set_title("Real Sample")

    sample_im_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[0, 3], add_legend=False
    )
    ax[0, 3].set_title("Imaginary Sample")

    sample_abs_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[1, 2], add_legend=False
    )
    ax[1, 2].set_title("Amplitude Sample")

    sample_phase_spectra.plot.line(
        x="wavenumber", hue="point", ax=ax[1, 3], add_legend=False
    )
    ax[1, 3].set_title("Phase Sample")

    for axi in ax.flatten():
        axi.set_xlim(plotlims)
        axi.set_ylabel("Amplitude")
        axi.ticklabel_format(style="sci", axis="y", scilimits=(-2, 1))
        axi.set_xlim(plotlims)

    plt.tight_layout()
    plt.show()


def _generate_peak_model(n_peaks, shape, x_center):
    """Returns a compiled objective function so Python doesn't rebuild it every loop."""
    shape = shape.lower()

    def multi_peak_model(x_vals, *params):
        y_calc = np.zeros_like(x_vals)
        for i in range(n_peaks):
            a, c, w = params[i * 3 : i * 3 + 3]
            
            if shape == "gaussian":
                y_calc += a * np.exp(-((x_vals - c) ** 2) / (2 * w**2))
                
            elif shape == "lorentzian":
                y_calc += a * (w**2 / ((x_vals - c) ** 2 + w**2))

        slope = params[-2]
        intercept = params[-1]
        return y_calc + slope * (x_vals - x_center) + intercept

    return multi_peak_model


def fit_spectral_region(
    da,
    point_index,
    wmin,
    wmax,
    peak_centers,
    shape="lorentzian",
    center_tolerance=10.0,
    fwhm_bounds=(2.0, 50.0),
    plot=True,
    ax=None,
):
    """Fits Gaussian or Lorentzian peaks to an xarray DataArray with a linear sloping background."""
    spectrum = da.isel(point=point_index)

    w_sorted = np.sort([wmin, wmax])
    region = spectrum.where(
        (spectrum.wavenumber >= w_sorted[0]) & (spectrum.wavenumber <= w_sorted[1]),
        drop=True,
    )

    x = region.wavenumber.values
    y = region.values

    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) < 5:
        print(
            f"Error: Not enough valid data points in this region to fit point {point_index}."
        )
        return None

    n_peaks = len(peak_centers)
    x_center = (w_sorted[0] + w_sorted[1]) / 2.0

    # Build bounds list
    if (
        isinstance(fwhm_bounds, tuple)
        and len(fwhm_bounds) == 2
        and isinstance(fwhm_bounds[0], (int, float))
    ):
        fwhm_bounds_list = [fwhm_bounds] * n_peaks
    elif isinstance(fwhm_bounds, list) and len(fwhm_bounds) == n_peaks:
        fwhm_bounds_list = fwhm_bounds
    else:
        print(
            f"Error: fwhm_bounds must be a single (min, max) tuple or a list of {n_peaks} tuples."
        )
        return None

    # Get the pre-compiled model
    multi_peak_model = _generate_peak_model(n_peaks, shape, x_center)
    w_factor = 2 * np.sqrt(2 * np.log(2)) if shape.lower() == "gaussian" else 2.0

    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min

    amp_guess = y_range
    baseline_guess = np.percentile(y, 5)
    slope_guess = (y[-1] - y[0]) / (x[-1] - x[0]) if len(x) > 1 else 0.0

    p0, bounds_lower, bounds_upper = [], [], []

    for i, c in enumerate(peak_centers):
        peak_fwhm_min, peak_fwhm_max = fwhm_bounds_list[i]
        w_min, w_max = peak_fwhm_min / w_factor, peak_fwhm_max / w_factor
        width_guess = np.clip(
            (w_sorted[1] - w_sorted[0]) / (n_peaks * 5) / w_factor, w_min, w_max
        )

        p0.extend([amp_guess, c, width_guess])
        bounds_lower.extend([0, max(w_sorted[0], c - center_tolerance), w_min])
        bounds_upper.extend([np.inf, min(w_sorted[1], c + center_tolerance), w_max])

    p0.extend([slope_guess, baseline_guess])
    bounds_lower.extend([-np.abs(slope_guess) * 10 - 0.1, y_min - y_range * 0.5])
    bounds_upper.extend(
        [
            np.abs(slope_guess) * 10 + 0.1,
            max(y_min + (y_range * 0.15), baseline_guess + 1e-5),
        ]
    )

    try:
        popt, pcov = curve_fit(
            multi_peak_model, x, y, p0=p0, bounds=(bounds_lower, bounds_upper)
        )
    except RuntimeError:
        print(f"Fit failed to converge for point {point_index}.")
        return None

    if plot or ax is not None:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(x, y, "k.", alpha=0.6, label="Raw Data")
        x_smooth = np.linspace(w_sorted[0], w_sorted[1], 500)
        ax.plot(
            x_smooth,
            multi_peak_model(x_smooth, *popt),
            "r-",
            linewidth=2,
            label="Total Fit",
        )

        baseline_fit = popt[-2] * (x_smooth - x_center) + popt[-1]
        ax.plot(x_smooth, baseline_fit, ":", color="grey", alpha=0.8, label="Baseline")

        for i in range(n_peaks):
            a, c, w = popt[i * 3 : i * 3 + 3]
            actual_fwhm = w * w_factor

            if shape.lower() == "gaussian":
                single_y = (
                    a * np.exp(-((x_smooth - c) ** 2) / (2 * w**2)) + baseline_fit
                )
            else:
                single_y = a * (w**2 / ((x_smooth - c) ** 2 + w**2)) + baseline_fit

            ax.plot(
                x_smooth,
                single_y,
                "--",
                label=f"Peak {i+1}: {c:.1f} (w={actual_fwhm:.1f})",
            )

        ax.set_title(f"Point {point_index}")

    return popt


def fit_all_spectra(
    da,
    wmin,
    wmax,
    peak_centers,
    shape="lorentzian",
    center_tolerance=10.0,
    fwhm_bounds=(2.0, 50.0),
    grid_n=5,
):

    num_points = da.sizes["point"]
    results = []
    do_plot = grid_n > 0

    # Constants
    if shape.lower() == "gaussian":
        w_factor = 2 * np.sqrt(2 * np.log(2))
        area_factor = np.sqrt(2 * np.pi)
    else:
        w_factor = 2.0
        area_factor = np.pi

    print(f"Starting fit for {num_points} spectra...")

    # --- OPTIMIZATION 2: Pre-slice the Xarray ONCE ---
    w_sorted = np.sort([wmin, wmax])
    region = da.where(
        (da.wavenumber >= w_sorted[0]) & (da.wavenumber <= w_sorted[1]), drop=True
    )
    x_full = region.wavenumber.values
    y_matrix = region.values  # NumpPy Matrix of shape (num_points, num_wavenumbers)

    # Keep only finite x columns
    valid_x_mask = np.isfinite(x_full)
    x = x_full[valid_x_mask]
    y_matrix = y_matrix[:, valid_x_mask]

    n_peaks = len(peak_centers)
    x_center = (w_sorted[0] + w_sorted[1]) / 2.0
    multi_peak_model = _generate_peak_model(n_peaks, shape, x_center)

    # Standardize bounds format
    if isinstance(fwhm_bounds, tuple) and len(fwhm_bounds) == 2:
        fwhm_bounds_list = [fwhm_bounds] * n_peaks
    else:
        fwhm_bounds_list = fwhm_bounds

    if do_plot:
        plots_per_fig = grid_n * grid_n
        axes_flat = None

    last_popt = None  # Tracker for the Warm Start

    update_interval = max(1, num_points // 20)  # Updates every 10%
    bar_length = 20  # How wide the progress bar is in characters

    for i in range(num_points):
        y_raw = y_matrix[i]

        # --- Mask out any isolated NaNs/Infs in this specific row ---
        valid_mask = np.isfinite(y_raw)

        # Handle rows that are entirely NaN or have too few points to fit safely
        if np.sum(valid_mask) < 5:
            popt = None
            current_ax = None
            last_popt = None  # Reset warm start if we hit a bad pixel
        else:
            # Apply the mask so curve_fit only sees clean, valid numbers
            y_fit = y_raw[valid_mask]
            x_fit = x[valid_mask]

            # Calculate bounds dynamically for this specific pixel using clean data
            y_min, y_max = np.min(y_fit), np.max(y_fit)
            y_range = y_max - y_min

            p0, bounds_lower, bounds_upper = [], [], []
            amp_guess = y_range
            baseline_guess = np.percentile(y_fit, 5)
            slope_guess = (
                (y_fit[-1] - y_fit[0]) / (x_fit[-1] - x_fit[0])
                if len(x_fit) > 1
                else 0.0
            )

            for j, c in enumerate(peak_centers):
                w_min, w_max = (
                    fwhm_bounds_list[j][0] / w_factor,
                    fwhm_bounds_list[j][1] / w_factor,
                )
                width_guess = np.clip(
                    (w_sorted[1] - w_sorted[0]) / (n_peaks * 5) / w_factor, w_min, w_max
                )

                p0.extend([amp_guess, c, width_guess])
                bounds_lower.extend([0, max(w_sorted[0], c - center_tolerance), w_min])
                bounds_upper.extend(
                    [np.inf, min(w_sorted[1], c + center_tolerance), w_max]
                )

            p0.extend([slope_guess, baseline_guess])
            bounds_lower.extend(
                [-np.abs(slope_guess) * 10 - 0.1, y_min - y_range * 0.5]
            )
            bounds_upper.extend(
                [
                    np.abs(slope_guess) * 10 + 0.1,
                    max(y_min + (y_range * 0.15), baseline_guess + 1e-5),
                ]
            )

            # WARM START
            if last_popt is not None:
                if np.all(last_popt > bounds_lower) and np.all(
                    last_popt < bounds_upper
                ):
                    p0 = last_popt
            try:
                # --- THE FIX: Pass the cleaned x_fit and y_fit arrays ---
                popt, pcov = curve_fit(
                    multi_peak_model,
                    x_fit,
                    y_fit,
                    p0=p0,
                    bounds=(bounds_lower, bounds_upper),
                )
                last_popt = popt
            except RuntimeError:
                popt = None
                last_popt = None  # Reset if failed

        # 1. Figure management
        if do_plot:
            plot_idx = i % plots_per_fig
            if plot_idx == 0:
                fig, axes = plt.subplots(
                    grid_n, grid_n, figsize=(15, 15), constrained_layout=True
                )
                axes_flat = axes.flatten()
            current_ax = axes_flat[plot_idx]

            # Replicate plotting logic for speed without calling fit_spectral_region
            if popt is not None:
                current_ax.plot(x_fit, y_fit, "k.", alpha=0.6, label="Raw")
                x_smooth = np.linspace(w_sorted[0], w_sorted[1], 500)
                current_ax.plot(
                    x_smooth, multi_peak_model(x_smooth, *popt), "r-", linewidth=2
                )

                baseline_fit = popt[-2] * (x_smooth - x_center) + popt[-1]
                current_ax.plot(x_smooth, baseline_fit, ":", color="grey", alpha=0.8)

                for j in range(n_peaks):
                    a, c, w = popt[j * 3 : j * 3 + 3]
                    if shape.lower() == "gaussian":
                        single_y = (
                            a * np.exp(-((x_smooth - c) ** 2) / (2 * w**2))
                            + baseline_fit
                        )
                    else:
                        single_y = (
                            a * (w**2 / ((x_smooth - c) ** 2 + w**2)) + baseline_fit
                        )
                    current_ax.plot(x_smooth, single_y, "--")
            current_ax.set_title(f"Point {i}")

        # 3. Parameter extraction
        row_data = {"point": i}
        if popt is not None:
            row_data["baseline"] = popt[-1]
            for j in range(n_peaks):
                a, c, w = popt[j * 3 : j * 3 + 3]
                row_data[f"peak_{j+1}_center"] = c
                row_data[f"peak_{j+1}_amplitude"] = a
                row_data[f"peak_{j+1}_fwhm"] = w * w_factor
                row_data[f"peak_{j+1}_area"] = a * w * area_factor
        else:
            row_data["baseline"] = np.nan
            for j in range(n_peaks):
                for metric in ["center", "amplitude", "fwhm", "area"]:
                    row_data[f"peak_{j+1}_{metric}"] = np.nan

        results.append(row_data)

        # 4. Clean up the final figure
        if do_plot and i == num_points - 1:
            for j in range(plot_idx + 1, plots_per_fig):
                axes_flat[j].axis("off")

        if (i + 1) % update_interval == 0 or (i + 1) == num_points:
            percent_float = (i + 1) / num_points
            filled_length = int(bar_length * percent_float)

            # Create the visual bar (e.g., '========----------')
            bar = "=" * filled_length + "-" * (bar_length - filled_length)

            print(
                f"\rFitting Spectra: [{bar}] {int(percent_float * 100)}% ({i + 1}/{num_points})",
                end="",
                flush=True,
            )

    print()

    if do_plot:
        plt.show()

    print("Fitting complete!")
    return pd.DataFrame(results)


def plot_correlations(df, pairs):
    """
    Plots scatter correlations between user-specified pairs of fitted parameters.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the fit results.
    pairs : list of tuples
        A list where each tuple contains the column names for the X and Y axes.
        Example: [('peak_1_center', 'peak_1_amplitude'), ('peak_1_fwhm', 'peak_2_center')]
    """
    num_plots = len(pairs)
    if num_plots == 0:
        print("No pairs provided to plot.")
        return

    # Dynamically calculate grid size (max 3 columns wide)
    cols = min(3, num_plots)
    rows = int(np.ceil(num_plots / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
    )

    # Force axes into a flat list even if there's only 1 plot
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (x_col, y_col) in enumerate(pairs):
        ax = axes[i]

        # 1. Error handling if you type the wrong column name
        if x_col not in df.columns or y_col not in df.columns:
            ax.text(
                0.5,
                0.5,
                f"Missing Column(s):\n{x_col} or {y_col}",
                ha="center",
                va="center",
                color="red",
            )
            ax.axis("off")
            continue

        # 2. Drop rows where the fit failed (NaNs) so the math doesn't crash
        valid_data = df.dropna(subset=[x_col, y_col])

        if len(valid_data) == 0:
            ax.text(0.5, 0.5, "No valid data to plot", ha="center", va="center")
            continue

        x_data = valid_data[x_col]
        y_data = valid_data[y_col]
        points = valid_data["point"]

        # 3. Scatter plot colored by the point index
        scatter = ax.scatter(
            x_data, y_data, c=points, cmap="viridis", alpha=0.8, edgecolor="k", s=40
        )

        # 4. Calculate correlation coefficient (R)
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]

        # Format the titles and labels to look clean
        ax.set_title(f"Pearson R = {corr_coef:.2f}", fontsize=11, fontweight="bold")
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_box_aspect(1)

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.55)
        cbar.set_label("Spectrum Point Index")

    # Hide any unused subplots if your list length doesn't perfectly fill the grid
    for j in range(num_plots, len(axes)):
        axes[j].axis("off")

    plt.show()


# ==========================================
# AFM drift correction
# ==========================================


def extract_all_afm_images(data_dict):
    """
    Extracts all AFM and optical images from a parsed AXZ dictionary.

    Returns:
    - A dictionary mapping channel names (e.g., 'Height 1', 'Height 1_Retrace')
      to their corresponding 2D NumPy arrays.
    """
    extracted_images = {}

    # Navigate safely to the HeightMaps list
    afm_scans = data_dict.get("Document", {}).get("HeightMaps", {}).get("HeightMap", [])

    # If there is only one image, the parser might return a dict instead of a list
    if isinstance(afm_scans, dict):
        afm_scans = [afm_scans]

    for scan in afm_scans:
        # 1. Dig out the Label/Channel Name
        label = scan.get("Label", "")
        if not label:
            tags = scan.get("Tags", {}).get("Tag", [])
            if isinstance(tags, dict):
                tags = [tags]
            for tag in tags:
                attrs = tag.get("Attributes", {})
                if attrs.get("Name") in ["Channel", "DataType", "Label"]:
                    label = attrs.get("Value", label)

        if not label:
            label = f"Unknown_Channel_{len(extracted_images)}"

        # 2. Handle Trace vs. Retrace (Preventing dictionary overwrites)
        original_label = label
        counter = 1
        while label in extracted_images:
            if counter == 1:
                label = f"{original_label}_Retrace"
            else:
                label = f"{original_label}_Retrace_{counter}"
            counter += 1

        # 3. Decode the Base64 data into a NumPy matrix
        base64_string = scan.get("SampleBase64", {}).get("Text", "")
        if base64_string:
            decoded = base64.b64decode(base64_string)

            # Extract resolution (default to 100x100 if tags are missing)
            res_x = int(scan.get("Resolution", {}).get("X", {}).get("Text", 100))
            res_y = int(scan.get("Resolution", {}).get("Y", {}).get("Text", 100))

            try:
                # Convert binary to 32-bit float array and reshape to 2D
                img_matrix = np.frombuffer(decoded, dtype=np.float32).reshape(
                    res_y, res_x
                )
                extracted_images[label] = img_matrix
            except ValueError:
                print(
                    f"Warning: Could not reshape '{original_label}' into {res_y}x{res_x}"
                )

    return extracted_images


# ==========================================
# Misc functions
# ==========================================


def load_folder_to_xarray(
    folder_path,
    pattern="*.csv",
    x_col=0,
    y_col=1,
    x_name="wavelength_or_freq",
    y_name="intensity",
    delimiter=",",
    **kwargs,
):
    """
    Reads multiple files from a folder and compiles them into a 2D xarray.DataArray.

    Parameters:
    - folder_path (str/Path): Directory containing the files.
    - pattern (str): File matching criterion (e.g., "*.csv", "*.txt").
    - x_col (int or str): Index or name of the column to use as the X-axis (default: 0).
    - y_col (int or str): Index or name of the column to use as the Y-axis (default: 1).
    - x_name (str): The name to assign to the X-axis in the final xarray.
    - y_name (str): The name of the data values in the final xarray.
    - delimiter (str): Delimiter used in the files (default: ",").
    - **kwargs: Any extra arguments are passed directly to pandas.read_csv
                (e.g., skiprows=10, header=None, sep='\t').

    Returns:
    - xr.DataArray: A 2D array with dimensions (x_name, sample).
    """
    folder = Path(folder_path)
    files = list(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found matching '{pattern}' in {folder_path}")

    data_dict = {}

    for file in files:
        try:
            # Read only the two specific columns we need using pandas
            df = pd.read_csv(file, sep=delimiter, usecols=[x_col, y_col], **kwargs)

            # Identify which column is X and which is Y based on their positions
            x_col_name = df.columns[0]
            y_col_name = df.columns[1]

            # Set X as the index, and extract Y as a Series
            series = df.set_index(x_col_name)[y_col_name]

            # Name the series after the file name (without the extension)
            series.name = file.stem

            data_dict[file.stem] = series

        except Exception as e:
            print(f"Warning: Could not process '{file.name}'. Error: {e}")

    # Concat aligns all files by their X-axis index automatically.
    # If one file has 400.1nm and another lacks it, Pandas fills it with NaN.
    combined_df = pd.concat(data_dict.values(), axis=1)

    # Clean up the axis names for xarray
    combined_df.index.name = x_name
    combined_df.columns.name = "sample"

    # Convert to xarray DataArray
    da = xr.DataArray(combined_df)
    da.name = y_name

    return da


def extract_raw_spectra_coords(axz_dict):
    """
    Extracts raw absolute X and Y coordinates of spectra locations from an AXZ dictionary.

    Parameters:
    -----------
    axz_dict : dict
        The parsed dictionary from the AXZ file.

    Returns:
    --------
    np.ndarray
        A 2D array of shape (N, 2) containing the raw [X, Y] coordinates.
    """
    absolute_coords = []

    # Navigate through the expected XML-to-dict hierarchy
    try:
        spectra = axz_dict["Document"]["SNOMSpectra"]["AXDSNOMSpectrum"]
    except KeyError:
        return np.array([])

    # Standardize to list for iteration (handles single-spectrum files)
    if not isinstance(spectra, list):
        spectra = [spectra]

    for spec in spectra:
        loc = spec.get("Location", {})
        x_val = loc.get("X")
        y_val = loc.get("Y")
        
        if x_val is None or y_val is None:
            continue # Skip invalid points
            
        if isinstance(x_val, dict):
            # Handles XMLattributes parsed as dictionaries (e.g., {'#text': 'value'})
            abs_x = float(x_val.get("#text", list(x_val.values())[0]))
        else:
            abs_x = float(x_val)

        if isinstance(y_val, dict):
            abs_y = float(y_val.get("#text", list(y_val.values())[0]))
        else:
            abs_y = float(y_val)

        absolute_coords.append([abs_x, abs_y])

    return np.array(absolute_coords)


def save_afm_images(data_dict, output_folder, file_format="csv"):
    """Saves extracted AFM and optical height maps with dynamically constructed units and labels."""
    output_dir = Path(output_folder) / "Heightmaps"
    output_dir.mkdir(parents=True, exist_ok=True)

    afm_scans = data_dict.get("Document", {}).get("HeightMaps", {}).get("HeightMap", [])
    if isinstance(afm_scans, dict):
        afm_scans = [afm_scans]

    for i, scan in enumerate(afm_scans):
        label = scan.get("Attributes", {}).get("Label", f"Scan_{i}")
        clean_label = "".join([c for c in label if c.isalnum() or c in " _-"]).rstrip()

        # Combine UnitPrefix (e.g., 'n') and Units (e.g., 'm')
        prefix = scan.get("UnitPrefix", {}).get("Text", "")
        base_unit = scan.get("Units", {}).get("Text", "m")
        z_unit = f"{prefix}{base_unit}"

        b64_string = scan.get("SampleBase64", {}).get("Text", "")
        if not b64_string:
            continue

        decoded = base64.b64decode(b64_string)
        res_x = int(scan.get("Resolution", {}).get("X", {}).get("Text", 100))
        res_y = int(scan.get("Resolution", {}).get("Y", {}).get("Text", 100))

        try:
            img_matrix = np.frombuffer(decoded, dtype=np.float32).reshape(res_y, res_x)

            if file_format.lower() == "csv":
                save_path = output_dir / f"{clean_label}_({z_unit}).csv"
                np.savetxt(save_path, img_matrix, delimiter=",")
            elif file_format.lower() == "gwy":
                try:
                    phys_x_raw = float(scan.get("Size", {}).get("X", {}).get("Text", 1.0))
                    phys_y_raw = float(scan.get("Size", {}).get("Y", {}).get("Text", 1.0))

                    # 1. Scale physical dimensions to base meters
                    phys_x_m = phys_x_raw * 1e-6
                    phys_y_m = phys_y_raw * 1e-6

                    # 2. Map the extracted prefix to its mathematical multiplier
                    prefix_multipliers = {
                        "m": 1e-3,  # milli
                        "u": 1e-6,  # micro
                        "n": 1e-9,  # nano
                        "p": 1e-12, # pico
                        "": 1.0     # no prefix
                    }
                    
                    # Convert the raw matrix values to the base unit (e.g., nm -> m)
                    z_multiplier = prefix_multipliers.get(prefix.lower(), 1.0)
                    base_img_matrix = img_matrix.astype(np.float64) * z_multiplier

                    # 3. Explicitly define Gwyddion unit objects using BASE units
                    unit_xy = gwyfile.objects.GwySIUnit(unitstr="m")
                    unit_z = gwyfile.objects.GwySIUnit(unitstr=base_unit) # Pass "m" or "V", not "nm" or "mV"

                    # 4. Package the data field
                    data_field = gwyfile.objects.GwyDataField(
                        base_img_matrix,
                        xreal=phys_x_m,
                        yreal=phys_y_m,
                        si_unit_xy=unit_xy,
                        si_unit_z=unit_z,
                    )

                    # 5. Save the container to disk
                    container = gwyfile.objects.GwyContainer()
                    container["/0/data/title"] = clean_label
                    container["/0/data"] = data_field

                    save_path = output_dir / f"{clean_label}.gwy"
                    container.tofile(str(save_path))

                except ImportError:
                    print("The 'gwyfile' library is required to save native .gwy files. Saving as .csv instead.")
                    save_path = output_dir / f"{clean_label}_({z_unit}).csv"
                    np.savetxt(save_path, img_matrix, delimiter=",")
        except ValueError:
            print(f"Warning: Could not reshape '{label}' into {res_y}x{res_x}")


def save_video_pictures(data_dict, output_folder):
    """
    Decodes optical images using the Document/Images/Image hierarchy.
    Directly decodes raw pixel arrays based on the XML resolution tags.
    """
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = data_dict.get("Document", {}).get("Images", {}).get("Image", [])
    if isinstance(videos, dict):
        videos = [videos]

    for i, vid in enumerate(videos):
        b64_string = vid.get("SampleBase64", {}).get("Text", "")
        if not b64_string:
            continue

        img_data = base64.b64decode(b64_string)

        label = vid.get("Attributes", {}).get("Label", f"Optical_Image_{i}")
        clean_label = "".join([c for c in label if c.isalnum() or c in " _-"]).rstrip()
        save_path = output_dir / f"{clean_label}.png"

        try:
            res_x = int(vid.get("Resolution", {}).get("X", {}).get("Text", 0))
            res_y = int(vid.get("Resolution", {}).get("Y", {}).get("Text", 0))

            if res_x > 0 and res_y > 0:
                bytes_per_pixel = len(img_data) // (res_x * res_y)
                mode_map = {1: "L", 3: "RGB", 4: "RGBA"}

                if bytes_per_pixel in mode_map:
                    shape = (
                        (res_y, res_x)
                        if bytes_per_pixel == 1
                        else (res_y, res_x, bytes_per_pixel)
                    )
                    img_array = np.frombuffer(img_data, dtype=np.uint8).reshape(shape)

                    img_mode = mode_map[bytes_per_pixel]
                    Image.fromarray(img_array, mode=img_mode).save(save_path)
                    print(
                        f"Saved '{clean_label}.png' ({res_x}x{res_y}, Mode: {img_mode})"
                    )
                else:
                    print(
                        f"Warning: Raw image '{clean_label}' has unexpected byte ratio ({bytes_per_pixel} bytes/pixel)."
                    )
            else:
                print(f"Warning: Could not determine resolution for '{clean_label}'.")

        except Exception as e:
            print(f"Warning: Failed to decode '{clean_label}'. Error: {e}")


def save_all_interferograms(data_dict, output_folder):
    """Saves spectral data, mapping Y-axis units from DataSources and X-axis from SweepUnits."""
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    spectra_list = (
        data_dict.get("Document", {}).get("SNOMSpectra", {}).get("AXDSNOMSpectrum", [])
    )
    if isinstance(spectra_list, dict):
        spectra_list = [spectra_list]

    grouped_signals = {}
    z_axis_reference = {}
    z_unit_reference = {}
    y_unit_reference = {}

    for point_idx, spec in enumerate(spectra_list):
        # 1. Build a map of DataSignal -> Units from the DataSources block
        y_units_map = {}
        data_channels = spec.get("DataSources", {}).get("AXDDataChannel", [])
        if isinstance(data_channels, dict):
            data_channels = [data_channels]

        for channel in data_channels:
            sig = channel.get("DataSignal", {}).get("Text")
            unit = channel.get("Units", {}).get("Text", "a.u.")
            if sig:
                y_units_map[sig] = unit

        # 2. Extract the actual interferograms
        ig_list = spec.get("Interferograms", {}).get("AXDSNOMInterferogram", [])
        if isinstance(ig_list, dict):
            ig_list = [ig_list]

        for ig in ig_list:
            sig_name = ig.get("DataSignal", {}).get("Text", "Unknown_Signal")
            clean_name = sig_name.replace("/", "_").replace("\\", "_").strip("_")

            b64_data = ig.get("Data", {}).get("Text", "")
            if not b64_data:
                continue

            data_array = np.frombuffer(base64.b64decode(b64_data), dtype=np.float32)

            if clean_name not in grouped_signals:
                grouped_signals[clean_name] = []

                # Assign units
                z_unit = ig.get("SweepUnits", {}).get("Text", "mm")
                z_unit_reference[clean_name] = z_unit
                y_unit_reference[clean_name] = y_units_map.get(sig_name, "a.u.")

                # Construct Z-axis
                sweep_start = float(ig.get("SweepStart", {}).get("Text", 0))
                sweep_end = float(ig.get("SweepEnd", {}).get("Text", 1))
                points = int(ig.get("Points", {}).get("Text", len(data_array)))

                z_axis_reference[clean_name] = np.linspace(
                    sweep_start, sweep_end, points
                )

            grouped_signals[clean_name].append(data_array)

    # 3. Write arrays to disk
    for sig_name, data_lists in grouped_signals.items():
        z_array = z_axis_reference[sig_name]
        z_unit = z_unit_reference[sig_name]
        y_unit = y_unit_reference[sig_name]

        intensity_matrix = np.vstack(data_lists).T
        out_matrix = np.column_stack((z_array, intensity_matrix))

        header_cols = [f"Sweep_Position_({z_unit})"] + [
            f"Point_{i}_({y_unit})" for i in range(len(data_lists))
        ]
        header_str = ",".join(header_cols)

        save_path = output_dir / f"Interferograms_{sig_name}.csv"
        np.savetxt(save_path, out_matrix, delimiter=",", header=header_str, comments="")


def export_axz_contents(data_dict, output_folder, afm_format="csv"):
    """
    Master function to unpack an AXZ dictionary and save its visual
    and spectral components to disk.
    """
    print(f"Exporting AXZ contents to {output_folder}...")
    save_afm_images(data_dict, output_folder, file_format=afm_format)
    save_video_pictures(data_dict, output_folder)
    save_all_interferograms(data_dict, output_folder)
    print("Export complete.")


if __name__ == "__main__":
    pass
