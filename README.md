# Anasys NanoIR _s_-SNOM Data Processing
## Overview
The Anasys_Processing.py pipeline automates the extraction, processing, and visualization of Anasys NanoIR s-SNOM data. It processes raw interferograms from both hyperspectral maps (.axz files) and point spectra (.txt files), performs phase correction using a master reference, fits analytical models to the spectral data, and projects the fit results onto drift-corrected AFM topography scans. The workflow is controlled via a single JSON configuration file.

The full documentation, `Documentation/Pipeline_Master_Document.pdf`, writes out every processing step as mathematics with the function that performs it, and lists every config key, every output file and the known caveats. This README is the quick reference.

## 0. Required modules
`axz_parser.py` This module handles the extraction of Anasys .axz container files and the parsing of their internal XML structures into accessible Python dictionaries.

`snom_utils.py` list of all helper functions for data processing, handling and plotting

`installpackages.py` installs all required packages

`Anasys_Processing.py` Main code used to process .axz data and .txt spectra

`Fitting_module.py` the PDM fitter (both background modes) and its plots

`PDM_main.py` the point dipole model; the tip, tapping and illumination constants are at the top (section 3)

`phase_utils.py` the baseline-phase fit shared by the phase correction (`snom_utils`) and the PDM model (`Fitting_module`)

Make sure all of these files are in the same folder as `Anasys_Processing.py`.

## 1. Execution Instructions
- **Generate a Template Configuration File** \
  Each measurement file or collection of files should have its own config.json file. This ensures that the data processing is reproducible and saved, and prevents having to reconfigure ref. files, fitting parameters etc. every time one wishes to process an array spectrum or related point spectra. To create a blank configuration file with default settings in your current directory:

```sh
  python3 Anasys_Processing.py --generate
```
(You can specify a custom name: `python3 Anasys_Processing.py --generate path/my_settings.json`)
- **Run the Pipeline** \
  Execute the script by passing your configured JSON file as an argument:
```sh
python3 Anasys_Processing.py 'path/config.json'
```


## 2. Configuration Parameters (config.json)
Every key is a field of `ProcessSettings` in `Anasys_Processing.py`. The config is checked when it is loaded and all problems are listed at once; keys the pipeline does not know are ignored with a warning. "Default" is what `--generate` writes; where the shipped `config.json` template differs, both are given. The full reference, with the mathematics behind each key, is Section 14 of `Documentation/Pipeline_Master_Document.pdf`.

### Data & File Selection
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `target_folder` | String | required | Folder with the raw .axz and .txt files; outputs are written inside it. A relative path is resolved from the directory the script is run from, so an absolute path is safest. |
| `array_num` | Integer, null | null | Array to process: the file `array_format` with `{num}` filled in (e.g. 3 → `Array_3.axz`). null skips the array and all AFM steps. |
| `array_format` | String | `"Array_{num}.axz"` | File-name template for arrays, e.g. "Array{num}.axz". |
| `samp_nums` | List [String] | [] | Point-spectrum measurement numbers, e.g. ["002", "003"]. Every sweep of every matching file becomes one point of a single "Point Spectra" batch. Empty skips point spectra. |
| `ref_nums` | List [String] | [] | Gold reference measurement numbers. All their sweeps are phase-corrected and averaged into one master reference per harmonic. Required, unless `ratio_mode` runs with `ratio_use_reference: false`. |
| `ref_format` | String | `"{num}_AuRef_intfgm2D_{harmonic}.txt"` | Reference file template. `{harmonic}` becomes the file number that holds `harmonic`, read from each measurement's `_params.txt` "Channel selection" flags (e.g. `011100` = harmonics 2, 3, 4, saved as files `_1`, `_2`, `_3`). A fixed name (no `{harmonic}`) is checked against `_params.txt` and left out with a warning if it holds another harmonic. |
| `samp_format` | String | `"{num}_*_intfgm2D_{harmonic}.txt"` | Point-spectrum template: `*` matches the sample name and `{harmonic}` is resolved as for `ref_format`. Reference files are never taken as samples. |
| `target_signal` | String | `"//ZI/DEV533/DEMODS/{harmonic}/X"` | Demodulator channel read from the .axz. `{harmonic}` becomes the demod that holds `harmonic`, found from the channel labels (`ZI X2` = 2nd harmonic = `DEMODS/1`). |
| `harmonic` | Integer | 2 | True harmonic order to process (2 to 15 with `pdm_fit`). You never need the demod or file number; each run prints what it matched. A harmonic that was not recorded is reported together with the ones that were: a sample without it is skipped, an .axz without it skips the array spectra (AFM steps still run), and the run stops if no reference holds it. |

### Ratio Mode
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `ratio_mode` | Boolean | false | Process the ratio of two harmonics instead of S/R. The ratio replaces S/R everywhere downstream: plots, standard fit, maps and exports (file names get `_ratio_Sn_Sm`, the folder `_nH-mH_ratio`). Cannot be combined with `pdm_fit`. |
| `ratio_harmonics` | List [Integer] | [3, 2] | [numerator, denominator] true harmonic orders, e.g. [3, 2] for S3/S2. The format strings must contain `{harmonic}`. |
| `ratio_use_reference` | Boolean | true | true: each harmonic is divided by its own AuRef first, (S_n/R_n)/(S_m/R_m). false: the raw S_n/S_m, which needs no reference. |

### Spectral Processing
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `plotlims` | List [Float] | required | [min, max] (cm⁻¹) for the plots and the exported spectra, and the default for `fit_wmin`/`fit_wmax`. It changes how much of the spectrum is kept, not the results. |
| `phase_fit_regions` | List [List [Float]] | required (at least one pair) | Absorption-free regions, e.g. [[1500, 1560], [1790, 1840]]. Used for: the phase correction of every sample and reference spectrum; the Re/Im detrend and the model's phase flattening (PDM `"detrend"`); the fitted range, from the first to the last region (PDM `"envelope"`); and `bg_level`/`bg_tilt`. Place regions on both sides of the bands, clear of band wings and inside the laser spectrum. Results can depend strongly on this choice, so keep it fixed when comparing datasets. |
| `phase_correction_order` | Integer 0–3 | 2 | Order of the polynomial fitted to the phase over `phase_fit_regions` (each region unwrapped on its own and the regions aligned by 2π; `phase_utils.baseline_phase_poly`). The same order is used for the reference, the sample and the PDM model. |
| `decay_frac` | Float, null | 0.4 (template: 0.25) | Exponential apodization: decay length = `decay_frac` × mirror travel L. It broadens every band by a Lorentzian of FWHM Γ_instr = 5 / (π · decay_frac · L[mm]) cm⁻¹, printed every run. The PDM γ and the standard fit's `peak_j_fwhm_intrinsic` have it removed. Smaller values give less noise but broader bands; 0.25–0.5 is typical. 0 or null switches the exponential off. |
| `signal_type` | String | "phase" | Signal for the standard fit, the referenced-spectra plot and `<batch>_<signal_type>_spectra_output.csv`: "phase" (Φ_NF = −arg(S/R), degrees), "amplitude", "real", "imaginary", or "complex" (export only; it cannot be used with the standard fit). Absorption is positive in "phase" and "imaginary". The PDM always uses the complex S/R. |

### Standard Peak Fitting
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `fit_spectra` | Boolean | false | Fits peaks plus a straight baseline to `signal_type`, one spectrum at a time (each fit starts from the previous spectrum's result when that lies inside the bounds). |
| `fit_wmin` / `fit_wmax` | Float, null | null (= `plotlims`) | Fit window. For the PDM it is the residual window in `"detrend"` mode, and only the R² and plot window in `"envelope"` mode. Centre bounds are clipped into it. |
| `peak_shape` | String | "lorentzian" | "lorentzian" or "gaussian". Amplitudes are ≥ 0, so peaks must point up in `signal_type`. |
| `peak_centers` | List [Float] | [] | Initial centres (cm⁻¹); the length sets the number of peaks, for both fits. For the PDM these are the oscillator frequencies ω₀ of ε, which lie a few cm⁻¹ below the phase maximum for strong bands. |
| `center_tolerance` | Float or List | 0.0 | Max shift (cm⁻¹) of each centre from its guess: one number (or [5.0]) for every peak, or one per peak in `peak_centers` order, e.g. [5, 3, 5, 10]. Must be > 0 when fitting. Used by both fits. |
| `fwhm_bounds` | List [List [Float]] | [] | One [min, max] per peak, e.g. [[5, 20], [10, 30]]. Required by the standard fit, where it bounds the measured FWHM (including Γ_instr). For the PDM, a non-empty list replaces `gamma_bounds` and bounds the intrinsic γ. |

### Point Dipole Model (PDM) Fitting
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `pdm_fit` | Boolean | false | Fits the point dipole model to the complex S/R (section 3). |
| `pdm_profile` | String | "lorentzian" | "lorentzian" (A, ω₀, γ per oscillator) or "voigt" (adds σ, the Gaussian spread of ω₀). |
| `pdm_background` | String | "detrend" | How the off-resonance background is handled: "detrend" or "envelope" (section 3). |
| `envelope_order` | List [Integer] | [3, 2] | [amplitude order, phase order] of the envelope B, `"envelope"` mode only. Use the lowest orders that follow the off-resonance background in `<batch>_PDM_envelope.png`; high orders can bend under a band and absorb part of it. Choose them before looking at results and keep them fixed when comparing datasets. |
| `envelope_shared` | Boolean | true | `"envelope"` mode only. true: B is fitted once on the batch average and only a complex scale c varies per spectrum, so the background cannot trade off with the oscillator from pixel to pixel. false: B is fitted per spectrum; use it for point spectra taken at different times and as a cross-check of γ maps. |
| `pdm_free_scale` | Boolean | false | `"detrend"` mode only: a free real scale on the model, exported as `scale`. It is degenerate with A, so use it only as a diagnostic. Ignored in `"envelope"` mode. |
| `bulk_sample` | Boolean | false | true: semi-infinite sample. false: a film of thickness `d_film_nm` on gold. |
| `d_film_nm` | Float | 300.0 | Film thickness in nm (film model only). |
| `eps_inf` | Float > 1 | 2.5 | Background permittivity ε∞ of the sample, held fixed (not fitted). Use ≈ n² of the material in the measured range, not its static permittivity. |
| `A_guess` / `A_bounds` | Float or List / List | 5.0e4 / [0, 1e9] | Oscillator strength A (cm⁻²). One value for every oscillator, or one per oscillator (bounds as a list of [min, max] pairs). If the bounds are given per oscillator, give the guess as a list too. |
| `gamma_guess` / `gamma_bounds` | Float or List / List | 10.0 / [1, 20] | Intrinsic damping γ (cm⁻¹; the FWHM of a Lorentz oscillator). Γ_instr is added inside the model, so the fitted γ excludes the apodization broadening. |
| `sigma_guess` / `sigma_bounds` | Float or List / List | 10.0 / [1, 20] | Gaussian spread σ of ω₀ (cm⁻¹), `"voigt"` only. |
| `pdm_plot_every` | Integer | 1 | Interval for the PDM fit previews (with `plot_fitresults`), e.g. 10 = every 10th spectrum; 0 = no previews. |

### Fit Quality, Plotting & Statistics
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `r2_threshold` | Float, dict, null | null (template: 0.8) | Points whose fit R² is below it are left out of the referenced-spectra plot, drawn as × or left blank in the maps, and drawn hollow and left out of the Pearson R in the correlation plots; failed fits never pass. The CSVs keep every point (filter with `r2_pass`). One number for every fit, or one per fit, e.g. {"PDM": 0.95, "Standard": 0.8}: the two R² are computed on different quantities and are not comparable. |
| `plot_intfgm` | Boolean | false | Interferograms after the single baseline subtraction and the apodization window, i.e. exactly what is Fourier-transformed (sample and reference). |
| `plot_allspectra` | Boolean | false | Phase-corrected reference and sample spectra (Re, Im, amplitude, phase) before normalisation. |
| `plot_referenced_spectra` | Boolean | false | Every normalised spectrum (`signal_type`) in grey with the mean in red; spectra failing `r2_threshold` are left out. |
| `plot_fitresults` | Boolean | false | n×n grids of individual fits (standard and PDM) to check the optimizer. |
| `plot_fitstatistics` | Boolean | false | The drift plot `<batch>_intfgm_pp.png` (`intfgm_pp`, `intfgm_offset` and `bg_tilt` against point number, one panel each with running medians; made even when no fit runs) and the correlation plots of `correlations_to_check`. In the drift plot, points failing `r2_threshold` are drawn hollow but stay in the running median (the monitors do not depend on the fit); in the correlation plots they are drawn hollow grey and left out of the Pearson R. |
| `correlations_to_check` | List [List [String]] | [] | Pairs of fit-table columns to plot against each other, e.g. [["peak_1_center", "peak_1_fwhm"], ["peak_1_amplitude", "intfgm_pp"]]. |
| `plotgrid_n` | Integer | 4 | Size n of the n×n preview grids. |

### AFM Mapping & Drift Correction
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `plot_afm` | Boolean | false | Plots the levelled AFM frames, and the drift-corrected frames when `drift_correct` is on. |
| `drift_correct` | Boolean | false | Drift tracking from the AFM frames recorded during the map. Each frame is registered against its previous 2-3 frames; every match gets a quality score (overlap correlation) and the good matches are solved together, so one bad frame cannot shift the rest. The drift is interpolated to each spectrum's own timestamp. If drift carries points outside the first AFM frame, the maps use a mosaic of all aligned frames as background. The drift plot (`AFM_drift`) shows the correction per frame and per spectrum, the match quality and, for comparison, the instrument's own DriftCorrection (recorded, not applied to the tip). A non-zero scan angle is handled automatically. Requires `relative_array_coords` or interactive clicks to place the first point. Maps of the fit results are only made when this is on. |
| `afm_fix_scan_start` | Boolean | true | Repairs the start-of-scan artefact of the AFM frames: a short spike or ramp at the end of the first scan line(s), recorded before the z feedback settles. Pixels far outside the normal line-to-line variation of their column are filled from the retrace of the same channel when the file has one, otherwise from the neighbouring lines. |
| `relative_array_coords` | List [Float] | [] | Position of the first spectrum as fractions of the AFM frame width and height, measured from the bottom-left corner, e.g. [0.03, 0.96]. If empty, you select an image and click the scan corners and the first point; the values are printed, so copy them here for reproducible runs. |
| `map_parameter` | List [String] | [] | Fit-table columns to overlay on the AFM image, e.g. ["peak_1_center", "peak_1_fwhm", "peak_1_delta_eps", "r_squared", "intfgm_pp", "bg_tilt"]. Names must match the CSV column names exactly; the same list is used for the PDM and the standard table. |
| `array_scan_dim` | List [Integer] | [] | [rows, cols] of the array, for the interpolated ("deformed pixel") map: rows × cols must equal the number of spectra, taken in raster (not serpentine) order. Also sets the row length in the drift plot. If empty, it is inferred from the coordinates. |
| `interpolation_res` | Integer | 0 | Upsampling factor of the interpolated map (0 or 1 = none). |
| `shared_colormap` | Boolean | false | One colour scale for all parameters in `map_parameter`. Only useful when they share units: mixing e.g. a centre and a width makes both maps uniform. |

### Export & Display
| Parameter | Type | Default | Description |
| :--: | :--: | :--: | :-- |
| `EXPORT` | Boolean | true | Writes the spectra and fit-result CSVs, all figures and, for arrays, the .axz contents (AFM channels as .gwy files, every interferogram channel as CSV). |
| `export_foldername` | String, null | null | Optional custom output folder. A plain name (e.g. `"my_fits"`) or relative path is created inside `target_folder`; an absolute path (or one starting with `~`) is used as-is. If null, data exports to a generated folder inside `target_folder`, named with the batch and harmonic: `Extracted_Data_Array_3_2H`, `Extracted_Data_Point_Spectra_3H`, or `Extracted_Data_Array_3_3H-2H_ratio` in `ratio_mode`. A custom name is used as-is, so give 2H and 3H runs different names to keep them apart. Figure titles carry the same label (e.g. "Array 3 2H: Standard"). |
| `figure_dpi` | Integer | 300 | Resolution of the saved PNGs. The fit grids are about 15 inches wide, so 150 keeps large arrays manageable. |
| `close_figures` | Boolean | false | Closes each figure once it is saved. Worth turning on for large arrays, where the fit grids alone can leave dozens of windows open. |
| `afmcolormap` / `hyperspectracmap` | String | "afmhot" / "jet" | Matplotlib colour maps for the AFM background and for the mapped values. |

### Legacy Keys
| Parameter | Status | Description |
| :--: | :--: | :-- |
| `harmonic_offset` | Removed | Configs that still have it are converted when loaded (harmonic and ratio_harmonics + offset), with a warning asking you to update the file. |
| `drift_poly_degree` | Unused | Drift is interpolated between AFM frames by timestamp; the key is accepted so older configs still load. |

### Output Columns of the Fit Tables
Not config keys, but columns of `<batch>_fit_results.csv` (standard fit) and `<batch>_PDM_fit_results.csv`. Use them in `map_parameter` and `correlations_to_check` like any fitted parameter.

| Column | Unit | Description |
| :--: | :--: | :-- |
| `r_squared`, `r2_pass` | — | Fit R², and whether it passes `r2_threshold`. |
| `peak_j_center`, `peak_j_amplitude`, `peak_j_fwhm` | cm⁻¹, see right, cm⁻¹ | Standard fit: centre, amplitude in the unit of `signal_type`, and the measured FWHM. PDM: ω₀, oscillator strength A (cm⁻²), and the intrinsic FWHM (γ for Lorentzian). |
| `peak_j_area`, `peak_j_fwhm_intrinsic` | — , cm⁻¹ | Standard fit only: peak area, and FWHM − Γ_instr (Lorentzian peaks). |
| `peak_j_delta_eps` | — | PDM only: A/ω₀², the dimensionless oscillator strength. |
| `*_err`, `*_at_bound` | — | PDM only: 1σ from the fit covariance (not a reliable error bar; a split-half test is better), and whether the parameter ended on a bound (it was then stopped, not fitted, even if R² looks good). |
| `gamma_instr`, `eps_inf` | cm⁻¹, — | PDM only: the Γ_instr already removed from every γ, and the ε∞ used. |
| `scale` / `bg_phase_deg` | — / deg | Detrend mode: `scale` is 1.0 unless `pdm_free_scale`. Envelope mode: `scale` = \|c\|, the spectrum's background level relative to the shared background (≈ 1), and `bg_phase_deg` = arg(c) in degrees. Both can be mapped or correlated like any other column. |
| `intfgm_pp` | µV | Peak-to-peak amplitude of each point's baseline-corrected interferogram (the analysed harmonic; the numerator in ratio mode), written to the PDM and standard fit-results CSVs. Map it with `map_parameter` or use it in `correlations_to_check` like any other column, e.g. to check whether a peak amplitude follows laser power. |
| `intfgm_offset` | µV | Mean level (DC offset) of each point's raw interferogram, before baseline correction (same harmonic as `intfgm_pp`), written next to `intfgm_pp` in both fit-results CSVs. It is the part of the demodulated signal that does not interfere with the reference arm: tip-scattered light at that harmonic summed over the whole laser spectrum, so it follows broadband scattering and laser power independently of the fringes. Signed (lock-in X quadrature), so its sign depends on the demodulator phase. Map or correlate it like `intfgm_pp`. |
| `bg_tilt` | % per 100 cm⁻¹ | Spectral tilt of each point's background: a straight line is fitted to ln\|S/R\| (ln\|ratio\| in ratio mode) over the points inside `phase_fit_regions` only, and its slope is given as the relative change of \|S/R\| per 100 cm⁻¹ (logarithmic; +10 means \|S/R\| grows by a factor e^0.1 per 100 cm⁻¹). No fit model is involved, so it is the same for every fit mode (or none), for point spectra and arrays, and it does not trade off against the peaks. It follows the *shape* of the background, which `intfgm_pp` and `intfgm_offset` cannot show: e.g. the laser spectrum drifting away from the one the reference was recorded with, a change of the non-resonant sample response or far-field background. Written to both fit-results CSVs, shown in the bottom panel of the drift plot, and printed (median and range) during the run. Map or correlate it like `intfgm_pp`, e.g. against a fitted center or width to see whether a parameter follows the background. Absorption inside `phase_fit_regions`, band wings included, leaks into it. In per-spectrum envelope mode it measures the same thing as the envelope's relative linear amplitude term, but without the fit. |
| `bg_level` | — | \|S/R\| (or \|ratio\|) of the same straight-line fit, at the middle of the span covered by `phase_fit_regions`: the background amplitude of the normalised spectrum. It mostly follows `intfgm_pp`, but normalised by the reference and taken off-resonance. Written next to `bg_tilt`. |


## 3. Point Dipole Model (PDM) Fitting

### Overview
The Point Dipole Model (PDM) uses a more physically rigorous fitting tecnhique by fitting the physical dielectric function ($\epsilon$) of the material to the complex IR spectrum. It simulates the physical interaction between the AFM tip and the sample, extracting true physical oscillator strengths rather than just near-field amplitudes.


### Basic Instructions
To enable physical modeling, update the PDM block in your `config.json`:
1. Set `"pdm_fit": true`.
2. Choose your oscillator `"pdm_profile"` (`"lorentzian"` for standard physical damping, or `"voigt"` to account for inhomogeneous spatial/Gaussian broadening).
3. Set `"eps_inf"` to the background permittivity of your material (≈ n² in the measured range), and `"bulk_sample"` / `"d_film_nm"` to the sample geometry.
4. Give `"peak_centers"` as oscillator frequencies ω₀ of ε: for strong bands they lie a few cm⁻¹ below the maximum of the phase spectrum.
5. Choose `"pdm_background"` (see below).

### Tuning Considerations
* **Fix the Background Dielectric ($\epsilon_\infty$):** `eps_inf` sets the non-resonant background of the sample. It should approximate the square of the material's refractive index ($n^2$) in the measured range. It is held fixed during the fit, so a wrong value is absorbed by the oscillator strength A.
* **Set the Geometry:** Use the `"bulk_sample"` toggle. Set to `true` for infinitely thick bulk crystals, or `false` if you are measuring a thin film on a highly reflective substrate (like gold), which requires the 3-layer Fresnel reflection model.
* **Computational Cost:** PDM math requires complex integration and vectorized lock-in demodulation. It is significantly slower than simple curve fitting even when parallelized. Test your parameters on a single point spectrum or use a high `"pdm_plot_every"` value before committing to a high-resolution 2D array.

### Tip, tapping and illumination constants (`PDM_main.py`)
These are fixed in the code, not in the config. They set the absolute near-field contrast, so the fitted oscillator strength A depends on them; record them with any PDM result. Changed on 2026-11-17 (previous values in brackets):

| Constant | Value | Meaning |
| :--: | :--: | :--: |
| `TIP_RADIUS_NM` | 30 nm [20] | Tip apex radius. |
| `TIP_AMP_NM` | 50 nm [25] | Tapping amplitude, zero-to-peak: the tip moves 0-100 nm above contact. Set 25 if 50 nm is meant peak-to-peak. |
| `TIP_MATERIAL` | "gold" ["platinum"] | Tip permittivity. "platinum" switches back to the tabulated Pt data. |
| `THETA_DEG` | 60° [30°] | Angle of incidence measured from the surface **normal** (the convention of the Fresnel formulas). |

### Background handling: `pdm_background`
**`"detrend"`** (default). Every spectrum's phase is flattened with a polynomial fitted on `phase_fit_regions`; then a straight line fitted on the same regions is subtracted from Re and a constant from Im. The PDM model goes through the same two steps, and data and model are compared over `fit_wmin`-`fit_wmax`.

Limitations:
* The off-resonance level of S/R is removed, so the oscillator strength A is fixed only through the absolute |S/R| calibration (scale = 1). Freeing the scale is degenerate with A.
* A background that multiplies the spectrum and is curved (e.g. a laser-spectrum change between the reference and the sample measurement) is treated as an additive straight line. The leftover appears under the band and can look like a rotated resonance in the complex plane.
* The results can depend strongly on the phase-correction choices (`phase_correction_order`, the extent of `phase_fit_regions`), because the extrapolated phase under the band changes with them.

**`"envelope"`**. S/R = c · B(ω) · PDM(ω; oscillators), with B(ω) = P_amp(t)·exp(i·P_phase(t)) over the span of `phase_fit_regions`; t = (ω − ω_mid)/ω_half runs from −1 to +1 across that span (ω_mid, ω_half = its centre and half-width), which only keeps the polynomial coefficients well scaled. No detrend and no phase flattening is applied to data or model; the per-spectrum phase correction before the fit stays (it removes phase drift between spectra), and B's phase polynomial takes over the rest.
1. B is fitted once, with the oscillators, on the array-average spectrum.
2. Every spectrum is fitted with B fixed (`envelope_shared: true`): free are a complex scale c and the oscillators. With `envelope_shared: false` every spectrum fits all of B.
3. R² is computed on the resonant part inside `fit_wmin`-`fit_wmax`: data and model divided by c·B·PDM without oscillators, minus 1. The fit plots show this same quantity ("S/R ÷ background − 1").

B multiplies the baseline and the resonance alike, so it cannot tilt or reshape a narrow band, but a too-flexible B (high orders) can absorb a broad wing under it. Because B's phase polynomial takes over the phase background, the oscillator parameters depend much less on the phase-correction choices than in detrend mode.

Extra outputs in envelope mode, next to the fit results:
* `<batch>_PDM_envelope.csv`: B(ω), the array-average spectrum, the background (B·PDM without oscillators) and the fit, as amplitude and Φ_NF.
* `<batch>_PDM_envelope.png`: what was divided out, amplitude and phase; the phase-fit regions are shaded.
* `scale` (= |c|) and `bg_phase_deg` columns in the fit results.

### Interferogram processing (changed 2026-11-17)
* **One baseline subtraction.** Each sweep has exactly one polynomial baseline (cubic in sample index) removed, inside `make_single_spec`. It is fitted on the recorded samples (stage position ≠ 0) without the 10 end samples (`INTFGM_TRIM`), and on the wings only: the centreburst (smoothed fringe envelope above 10 % of its maximum) is left out of the fit. Before, three corrections were stacked, each fitted through the burst. The spectra are computed from the RAW sweeps; `baseline_correct_interferogram` gives the same baseline for `intfgm_pp` and the plots.
* **Apodization window ≤ 1.** The exponential decay is held at 1 before the envelope maximum; it used to keep growing there, which weighted the leading side of the burst up to 30 % more than intended.
* **Interferogram CSV export.** `Interferograms_*.csv` positions are now written in mm, the unit in the header. The .axz stores the sweep in metres while its `SweepUnits` says mm; before, the metre values were written under the "mm" label.
