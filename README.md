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

  _python3 Anasys_Processing.py --generate_   
(You can specify a custom name: python _Anasys_Processing.py --generate path/my_settings.json_)
- **Run the Pipeline** \
  Execute the script by passing your configured JSON file as an argument:

  _python3 Anasys_Processing.py config.json_

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

### Spectral Processing & Limits
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `plotlims` | List [Float] | [min, max] | The global wavenumber range (cm⁻¹) for data output and plotting. |
| `phase_fit_regions` | List [List [Float]] | e.g., [[1000, 1100], [1800, 1900]] | Background regions free of sample absorption. A 2nd-order polynomial is fit here to correct instrumental phase twist. |
| `signal_type` | String | "phase", "amplitude", "real", "imaginary" | The component of the normalized complex signal (S/R) used for downstream plotting and fitting. |

### Peak Fitting Settings
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `fit_spectra` | Boolean | true, false | Toggles the analytical peak fitting routine.
| `fit_wmin / fit_wmax` | Float, null | Any Float | Subsets plotlims specifically for peak fitting. If null, defaults to the plotlims boundaries.|
| `peak_shape` | String | "lorentzian", "gaussian" | The mathematical model used for multi-peak fitting.|
| `peak_centers` | List [Float] | e.g., [1600.0, 1650.0] | Initial center guesses for peaks (cm⁻¹). The length of this list dictates the number of peaks fitted.|
| `center_tolerance` | Float | e.g., 10.0 | The maximum allowable shift (cm⁻¹) from the initial center guess during optimization.|
| `fwhm_bounds` | List [List [Float]] | e.g., [[5, 20], [10, 30]] | The [min, max] Full-Width Half-Maximum limits (cm⁻¹) for each defined peak. Must match the length of peak_centers.|

### AFM Mapping & Drift Correction
| Parameter | Type | Allowed Values | Description |
| :--: | :--: | :--: | :--: |
| `drift_correct` | Boolean | true, false | Toggles frame-by-frame cross-correlation drift tracking. Requires interactive user clicks to calibrate the first point. |
| `drift_poly_degree` | Integer | Integer | Degree of the polynomial fit applied to the tracked X/Y drift. Set to 0 to bypass polynomial fitting. |
| `map_parameter` | List [String] | "peak_1_area", "peak_1_center", "peak_1_amplitude", "peak_1_fwhm" | Which fitted parameter(s) to overlay onto the AFM maps. Must exactly match output column names. |
| `shared_colormap` | Boolean | true, false | Locks the colormap limits across all generated spatial maps based on global minimum and maximum values. |
| `array_scan_dim` | List [Integer] | e.g., [10, 10] | The [X, Y] dimensions of the scan grid. Used for reconstructing regular meshes during high-resolution interpolation. If empty, it calculates it automatically based on coordinate values |
| `interpolation_res` | Integer | e.g., 3 | The upsampling factor for the interpolated spatial heatmaps.
| `relative_array_coords` | List [Float] | e.g., [0.12342,0.289218] | calibrated relative array start point on first AFM scan. If left empty the user can select it manually. The coordinates will be printed as "Relative array coordinates:" in the terminal. |


