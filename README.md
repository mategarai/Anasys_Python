# Anasys NanoIR _s_-SNOM Data Processing
## Overview
The Anasys_Processing.py pipeline automates the extraction, processing, and visualization of Anasys NanoIR s-SNOM data. It processes raw interferograms from both hyperspectral maps (.axz files) and point spectra (.txt files), performs phase correction using a master reference, fits analytical models to the spectral data, and projects the fit results onto drift-corrected AFM topography scans. The workflow is controlled via a single JSON configuration file.

## 0. Required modules
`axz_parser.py` This module handles the extraction of Anasys .axz container files and the parsing of their internal XML structures into accessible Python dictionaries.

`snom_utils.py` list of all helper functions for data processing, handling and plotting

`installpackages.py` installs all required packages

`Anasys_Processing.py` Main code used to process .axz data and .txt spectra

Make sure `Anasys_Processing.py` is in the same folder as `snom_utils.py`.

## 1. Execution Instructions
- **Generate a Template Configuration File** \
  Each measurement file or collection of files should have its own config.json file. This ensures that the data processing is reproducible and saved, and prevents having to reconfigure ref. files, fitting parameters etc. every time one wishes to process an array spectrum or related point spectra. To create a blank configuration file with default settings in your current directory:

```sh
  python3 Anasys_Processing.py --generate
```
(You can specify a custom name: python _Anasys_Processing.py --generate 'path/my_settings.json'_)
- **Run the Pipeline** \
  Execute the script by passing your configured JSON file as an argument:
```sh
python3 Anasys_Processing.py 'path/config.json'
```


## 2. Configuration Parameters (config.json)
The following tables define all allowable parameters in the configuration file.

### Core Data & File Selection
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `target_folder` | String| Valid directory path | The path containing your raw .axz and .txt data files.
|  `array_num`| Integer, null | Any integer, or null | The numerical identifier of the AXZ array to process (e.g., 1 for Array_1). Set to null to skip array processing. |
| `array_format` | String | Format string | Naming convention for arrays. Default: "Array_{num}.axz". |
| `samp_nums` | List [String] | e.g., ["002", "003"] | Identifiers for individual sample point spectra files to process. Leave empty [] to skip point spectra.  |
| `ref_nums` | List [String] | e.g., ["001", "002"] | Required. Identifiers for the reference files. |
| `ref_format` | String | Format string | Naming convention for references. Default: "{num}_AuRef_intfgm2D_2.txt". |
| `target_signal` | String | Valid XML path | The demodulator channel extracted from AXZ files. Default: "//ZI/DEV533/DEMODS/1/X". |
| `EXPORT` | Boolean | true, false | Toggles the exporting of processed spectra, coordinates, and fit results to CSV files. |
| `export_foldername` | String, null | Valid directory path | Optional custom output directory. If null, data exports to a generated folder inside `target_folder`. |

### Plotting & Visualizations
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `plot_intfgm` | Boolean | true, false | Displays the raw interferograms before baseline correction. |
| `plot_allspectra` | Boolean | true, false | Displays an overlay of all phase-corrected spectra. |
| `plot_referenced_spectra` | Boolean | true, false | Displays the final normalized target signal ($S/R$). |
| `plot_afm` | Boolean | true, false | Displays the extracted AFM height and phase maps. |
| `plot_fitresults` | Boolean | true, false | Generates n×n grid previews of individual pixel fits to verify optimizer performance. |
| `plot_fitstatistics` | Boolean | true, false | Generates correlation plots based on the `correlations_to_check` parameter. |
| `plotgrid_n` | Integer | e.g., 4 | Determines the dimensions (n × n) of the grid figures used for plotting individual pixel fits. |
| `correlations_to_check` | List [List [String]] | e.g., [["peak_1_amplitude", "peak_2_amplitude"]] | Pairs of output parameters to plot against each other to check for statistical correlations. |

### Spectral Processing & Limits
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `plotlims` | List [Float] | [min, max] | The global wavenumber range (cm⁻¹) for data output and plotting. |
| `phase_fit_regions` | List [List [Float]] | e.g., [[1000, 1100], [1800, 1900]] | Background regions free of sample absorption. A 2nd-order polynomial is fit here to correct instrumental phase twist. |
| `signal_type` | String | "phase", "amplitude", "real", "imaginary", "complex" | The component of the normalized complex signal ($S/R$) used for downstream plotting and standard fitting. |

### Standard Peak Fitting Settings
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `fit_spectra` | Boolean | true, false | Toggles the standard analytical peak fitting routine. |
| `fit_wmin / fit_wmax` | Float, null | Any Float | Subsets plotlims specifically for peak fitting. If null, defaults to the plotlims boundaries.|
| `peak_shape` | String | "lorentzian", "gaussian" | The mathematical model used for multi-peak fitting.|
| `peak_centers` | List [Float] | e.g., [1600.0, 1650.0] | Initial center guesses for peaks (cm⁻¹). The length of this list dictates the number of peaks fitted.|
| `center_tolerance` | Float | e.g., 10.0 | The maximum allowable shift (cm⁻¹) from the initial center guess during optimization.|
| `fwhm_bounds` | List [List [Float]] | e.g., [[5, 20], [10, 30]] | The [min, max] Full-Width Half-Maximum limits (cm⁻¹) for each defined peak. Must match the length of peak_centers.|

### Point Dipole Model (PDM) Physical Fitting
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `pdm_fit` | Boolean | true, false | Toggles the physically rigorous, complex Point Dipole Model fitting routine. |
| `pdm_profile` | String | "lorentzian", "voigt", "gaussian" | The oscillator model used to build the physical sample dielectric function. |
| `pdm_plot_every` | Integer | e.g., 10 | Interval at which to plot individual PDM fit previews (e.g., every 10th pixel). |
| `bulk_sample` | Boolean | true, false | Sets the Fresnel reflection model. `true` uses an infinitely thick half-space. `false` uses a thin-film on Gold model. |
| `eps_guess` | Float | e.g., 2.45 | Initial guess for the high-frequency background dielectric constant ($\epsilon_\infty$). |
| `eps_bounds` | List [Float] | [min, max] | Boundary limits for the background dielectric constant. |
| `A_guess` / `A_bounds` | Float / List | e.g., 50000.0 / [0, 1e9] | Initial guess and bounds for the physical oscillator strength ($S$). |
| `slope_guess` / `slope_bounds` | Float / List | e.g., -0.0003 / [-0.1, 0.1] | Initial guess and bounds for the linear real-drift detrending. |
| `gamma_guess` / `gamma_bounds` | Float / List | e.g., 8.0 / [2.0, 25.0] | Initial guess and bounds for the Lorentzian damping/width parameter. |
| `sigma_guess` / `sigma_bounds` | Float / List | e.g., 6.0 / [2.0, 15.0] | Initial guess and bounds for the Gaussian inhomogeneous width parameter (used only if `pdm_profile` is "voigt"). |

### AFM Mapping & Drift Correction
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `drift_correct` | Boolean | true, false | Toggles frame-by-frame cross-correlation drift tracking. Requires interactive user clicks to calibrate the first point. |
| `drift_poly_degree` | Integer | Integer | Degree of the polynomial fit applied to the tracked X/Y drift. Set to 0 to bypass polynomial fitting. |
| `map_parameter` | List [String] | "peak_1_area", "peak_1_center", "peak_1_fwhm", "peak_1_amplitude", "eps", "slope" | Which fitted parameter(s) to overlay onto the AFM maps. Must exactly match output CSV column names. |
| `shared_colormap` | Boolean | true, false | Locks the colormap limits across all generated spatial maps based on global minimum and maximum values. |
| `array_scan_dim` | List [Integer] | e.g., [10, 10] | The [X, Y] dimensions of the scan grid. Used for reconstructing regular meshes during high-resolution interpolation. If empty, calculates automatically based on coordinates. |
| `interpolation_res` | Integer | e.g., 3 | The upsampling factor for the interpolated spatial heatmaps. |
| `relative_array_coords` | List [Float] | e.g., [0.1234, 0.2892] | Calibrated relative array start point on the first AFM scan. If left empty, the user selects it manually and coordinates print in the terminal. |


## 3. Point Dipole Model (PDM) Fitting

### Overview
The Point Dipole Model (PDM) uses a more physically rigorous fitting tecnhique by fitting the physical dielectric function ($\epsilon$) of the material to the complex IR spectrum. It simulates the physical interaction between the AFM tip and the sample, extracting true physical oscillator strengths rather than just near-field amplitudes.


### Basic Instructions
To enable physical modeling, update the PDM block in your `config.json`:
1. Set `"pdm_fit": true`.
2. Choose your oscillator `"pdm_profile"` (`"lorentzian"` for standard physical damping, or `"voigt"` to account for inhomogeneous spatial/Gaussian broadening).
3. Ensure `"eps_guess"` is set to the physically known background dielectric constant of your material.

### Tuning Considerations
* **Lock the Background Dielectric ($\epsilon_\infty$):** The `eps` parameter sets the non-resonant background reflectivity. It should approximate the square of the material's refractive index ($n^2$). **Do not let the optimizer use it as a free parameter.** If `eps_bounds` are too loose, the optimizer will artificially inflate the background to scale the peak amplitudes.
* **Set the Geometry:** Use the `"bulk_sample"` toggle. Set to `true` for infinitely thick bulk crystals, or `false` if you are measuring a thin film on a highly reflective substrate (like gold), which requires the 3-layer Fresnel reflection model.
* **Computational Cost:** PDM math requires complex integration and vectorized lock-in demodulation. It is significantly slower than simple curve fitting even when parallelized. Test your parameters on a single point spectrum or use a high `"pdm_plot_every"` value before committing to a high-resolution 2D array.
