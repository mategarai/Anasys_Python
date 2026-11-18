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
from scipy.ndimage import shift, uniform_filter1d, zoom, distance_transform_edt
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.special import erf
from phase_utils import baseline_phase_poly

mpl.rcParams["savefig.dpi"] = 300
# ==========================================
# Data Loading & I/O
# ==========================================

def _robust_zero(scans, percentile=1.0):
    """
    Shifts each frame so its `percentile`-th height sits at zero.

    A low percentile instead of np.min makes the offset insensitive to single
    dropout/glitch pixels, so frames stay comparable with each other. Applying
    it more than once is a no-op, so repeated calls down the chain are harmless.
    """
    offsets = np.nanpercentile(scans, percentile, axis=(1, 2), keepdims=True)
    return scans - offsets

def print_warning(msg: str):
    """Prints a yellow warning message to the console."""
    print(f"\033[93m{msg}\033[0m")


def height_map_scan_angle(h_map) -> float:
    """
    Scan angle (degrees) of one AXZ height map: the 'ScanAngle' tag
    (e.g. "-90 deg"), else Rotation/Yaw, else 0.
    """
    tags = (h_map.get("Tags") or {}).get("Tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    for tag in tags:
        attrs = tag.get("Attributes", {}) if isinstance(tag, dict) else {}
        if attrs.get("Name") == "ScanAngle":
            text = str(attrs.get("Value", tag.get("Text", "")))
            num = "".join(ch for ch in text.split("deg")[0] if ch in "+-.0123456789")
            try:
                return float(num)
            except ValueError:
                break
    yaw = (h_map.get("Rotation") or {}).get("Yaw")
    if isinstance(yaw, dict):
        yaw = yaw.get("Text")
    try:
        return float(yaw)
    except (TypeError, ValueError):
        return 0.0


def _axz_text(value):
    """The text of an AXZ XML node (dicts carry it under 'Text')."""
    if isinstance(value, dict):
        if "Text" in value:
            return value["Text"]
        return list(value.values())[0] if value else None
    return value


def _axz_tags(h_map) -> dict:
    """Name -> Value of a height map's Tags."""
    tags = (h_map.get("Tags") or {}).get("Tag", [])
    if isinstance(tags, dict):
        tags = [tags]
    out = {}
    for tag in tags:
        attrs = tag.get("Attributes", {}) if isinstance(tag, dict) else {}
        if "Name" in attrs:
            out[attrs["Name"]] = attrs.get("Value", tag.get("Text"))
    return out


def _axz_time(stamp):
    """AXZ timestamp ('2026-10-14T17:11:22.1372132-06:00') -> seconds since the
    epoch, using its UTC offset (so daylight-saving changes are handled)."""
    import re
    from datetime import datetime
    try:
        text = re.sub(r"(\.\d{6})\d+", r"\1", str(_axz_text(stamp)))
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        return np.nan


def _iso_seconds(duration):
    """ISO-8601 duration ('PT43.333S', 'PT1M5S') -> seconds (nan if unreadable)."""
    import re
    m = re.fullmatch(r"P?T?(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?",
                     str(_axz_text(duration) or "").strip())
    if not m or not any(m.groups()):
        return np.nan
    h, mnt, s = (float(g) if g else 0.0 for g in m.groups())
    return 3600 * h + 60 * mnt + s


def repair_scan_start(z, retrace=None, max_lines=None, k=8.0, max_run=0.4):
    """
    Repair the start-of-scan artefact of one AFM frame.

    The first scan line(s), recorded before the z feedback has settled, can end
    in a short spike or ramp of tens to hundreds of nm. Each of the first
    `max_lines` lines (default ~6% of the rows, at least 3) is compared with the median of the next three lines
    (working upwards, so a bad line is never used as the reference), after
    removing a robust straight-line trend from the difference (the slow creep
    after the scan starts tilts and offsets whole lines; align_rows and the
    flattening deal with that).

    A pixel is flagged when it differs by more than `k` times the normal
    line-to-line variation of its own column (measured in the rest of the
    image, so rough regions get proportionally larger allowances), together
    with the run of neighbouring pixels above k/3. Only runs that reach an end
    of the line and are shorter than `max_run` of it are replaced: the
    artefact is local and sits at the line end, while longer runs or runs in
    the middle are more likely topography. Flagged pixels are filled from the retrace
    where it is clean at that spot, otherwise from the neighbouring lines,
    both brought to the line's own level.

    z       : 2D frame in acquisition order (row 0 = first scan line)
    retrace : the same frame in the other scan direction, aligned to z, or None
    Returns (repaired, mask_of_replaced_pixels, n_filled_from_retrace).
    """
    z = np.array(z, dtype=float, copy=True)
    ny, nx = z.shape
    if max_lines is None:
        max_lines = max(3, int(round(0.06 * ny)))
    max_lines = min(max_lines, ny - 8)
    mask = np.zeros_like(z, dtype=bool)
    if max_lines <= 0:
        return z, mask, 0
    xs = np.arange(nx)

    def detrend(d):
        """d minus a robust straight-line fit along the line."""
        keep = np.ones(nx, dtype=bool)
        for _ in range(3):
            c = np.polyfit(xs[keep], d[keep], 1)
            res = d - np.polyval(c, xs)
            s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) or 1e-12
            keep = np.abs(res - np.median(res[keep])) < 3 * s
            if keep.sum() < nx // 2:
                break
        return res

    # Normal line-to-line variation per column, from the body of the image
    body_rows = range(max_lines + 4, ny - 3)
    steps = np.array([detrend(z[r] - np.median(z[r + 1:r + 4], axis=0)) for r in body_rows])
    col_sigma = 1.4826 * np.median(np.abs(steps - np.median(steps, axis=0)), axis=0)
    floor = 1.4826 * np.median(np.abs(steps - np.median(steps)))
    col_sigma = np.maximum(col_sigma, floor if floor > 0 else 1e-12)

    def flag(d):
        dev = np.abs(detrend(d)) / col_sigma
        bad = dev > k
        if not bad.any():
            return bad
        weak = dev > k / 3.0
        while True:   # grow strong pixels through adjacent weak ones
            grown = bad | (weak & (np.r_[False, bad[:-1]] | np.r_[bad[1:], False]))
            if (grown == bad).all():
                break
            bad = grown
        # keep only short runs that reach an end of the line (where the
        # artefact sits); runs in the middle are more likely topography
        edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            near = max(3, int(0.03 * nx))
            at_end = start <= near or stop >= nx - near
            if stop - start > max_run * nx or not at_end:
                bad[start:stop] = False
        return bad

    n_retrace = 0
    for r in range(max_lines - 1, -1, -1):
        ref = np.median(z[r + 1:r + 4], axis=0)
        d = z[r] - ref
        bad = flag(d)
        if not bad.any():
            continue
        good = ~bad
        # the line's own level relative to the reference, from its good pixels
        c = np.polyfit(xs[good], d[good], 1)
        fill = ref + np.polyval(c, xs)
        if retrace is not None:
            rd = retrace[r] - ref
            clean = ~flag(rd)
            use = bad & clean
            ok = good & clean
            if use.any() and ok.sum() >= nx // 4:
                rc = np.polyfit(xs[ok], rd[ok], 1)
                fill[use] = (retrace[r] - np.polyval(rc, xs) + np.polyval(c, xs))[use]
                n_retrace += int(use.sum())
        z[r, bad] = fill[bad]
        mask[r] = bad
    return z, mask, n_retrace


def _match_retrace(trace, retrace, skip):
    """
    Bring a retrace frame into register with its trace: pick the left-right
    orientation and the lateral (hysteresis) shift that make the rows below
    `skip` agree best. Returns the aligned retrace, or None if it doesn't match
    (or is just a copy of the trace).
    """
    if retrace is None or retrace.shape != trace.shape or np.array_equal(trace, retrace):
        return None
    a = trace[skip:] - np.median(trace[skip:], axis=1, keepdims=True)
    best = (-np.inf, None)
    for cand in (retrace, retrace[:, ::-1]):
        b = cand[skip:] - np.median(cand[skip:], axis=1, keepdims=True)
        for s in range(-10, 11):
            bs = np.roll(b, s, axis=1)[:, 10:-10]
            r = np.corrcoef(a[:, 10:-10].ravel(), bs.ravel())[0, 1]
            if r > best[0]:
                best = (r, np.roll(cand, s, axis=1))
    return best[1] if best[0] > 0.5 else None


def _xcorr_shift(fa, fb, upsample_factor):
    try:
        return np.asarray(phase_cross_correlation(fa, fb, upsample_factor=upsample_factor,
                                                  normalization=None)[0], dtype=float)
    except TypeError:   # scikit-image < 0.19 has no 'normalization'
        return np.asarray(phase_cross_correlation(fa, fb, upsample_factor=upsample_factor)[0],
                          dtype=float)


def _register_pair(a, b, window, upsample_factor=10, iterations=4):
    """
    Shift (rows, cols) that moves frame b onto frame a, plus the quality of the
    match: the correlation of the overlapping pixels after the shift, and the
    overlap fraction.

    Means are removed and the frames tapered, so the frame edges and any
    common offset don't dominate. A taper on its own pulls large shifts towards
    zero, so the estimate is refined: b is moved by the current shift and the
    remaining offset is measured on the common overlap only, until it stops
    changing (the last correction is then tiny, so the taper no longer biases it).
    """
    def overlap_corr(sh):
        moved = shift(b, sh, order=1, mode="constant", cval=np.nan)
        m = np.isfinite(moved)
        if m.sum() < 50:
            return np.nan, m.mean(), moved, m
        return np.corrcoef(a[m], moved[m])[0, 1], m.mean(), moved, m

    s = _xcorr_shift((a - np.mean(a)) * window, (b - np.mean(b)) * window, upsample_factor)
    q, ov, moved, m = overlap_corr(s)
    for _ in range(iterations):
        if not np.isfinite(q):
            break
        fa = np.where(m, a - np.mean(a[m]), 0.0) * window
        fb = np.where(m, moved - np.mean(moved[m]), 0.0) * window
        ds = _xcorr_shift(fa, fb, upsample_factor)
        if np.hypot(*ds) < 0.05 or np.hypot(*ds) > 0.25 * min(a.shape):
            break
        q_new, ov_new, moved_new, m_new = overlap_corr(s + ds)
        if not (np.isfinite(q_new) and q_new >= q):   # keep only steps that improve the match
            break
        s, q, ov, moved, m = s + ds, q_new, ov_new, moved_new, m_new
    return s, q, ov


class AFMArray:

    def __init__(self, axz_dict):
        """Initialize with the dictionary loaded from your AXZ file."""
        self.raw_dict = axz_dict
        self.scans = None  # Will hold the 3D numpy array of images
        self.drift_path = None  # (Y, X) shift in pixels that moves each frame onto frame 0 (nan = not placed)
        self.drift_quality = None  # best match quality of each frame with its neighbours
        self.corrected_scans = None  # Will hold the aligned images
        self.scan_angle = 0.0  # Scan rotation (deg) of the AFM images, from the file
        self.frame_times = None  # mid-scan time of each frame (s since the epoch)
        self.instrument_drift = None  # the instrument's own DriftCorrection X/Y (um, stage frame)
        self.repaired_pixels = None  # pixels fixed by repair_scan_start, per frame

    def extract_scans(self, channel_index=0, flatten=True, fix_scan_start=True):
        """
        Extracts and decodes the 2D AFM scans. Automatically detects how many
        channels are saved per scan based on the XML TimeStamp.
        Filters out scans that have pixel resolution OR physical dimension mismatches.

        fix_scan_start : repair the start-of-scan artefact (spikes/ramps in the
            first scan lines) with repair_scan_start, using the retrace of the
            same channel to fill the gaps when the file has one.
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

            # Rotated scans store their images in the scan's own frame;
            # align_and_plot_spectra uses this to rotate the array to match.
            angles = [height_map_scan_angle(h) for h in filtered_maps]
            self.scan_angle = angles[0] if angles else 0.0
            if angles and max(angles) - min(angles) > 1e-6:
                print_warning(
                    f"Warning: scan angle changes between AFM images "
                    f"({min(angles):g} to {max(angles):g} deg); using {self.scan_angle:g}."
                )

            # The retrace of the same channel, if the file stores one, fills the
            # gaps left by the start-of-scan repair.
            def _channel_id(h):
                attrs = h.get("Attributes", {}) or {}
                return attrs.get("DataChannel"), str(_axz_tags(h).get("TraceRetrace", ""))
            group = height_maps[:channel_stride]
            main_ch, main_dir = _channel_id(group[channel_index])
            retrace_index = next(
                (i for i, h in enumerate(group)
                 if i != channel_index and _channel_id(h)[0] == main_ch
                 and "retrace" in _channel_id(h)[1].lower() and "retrace" not in main_dir.lower()),
                None,
            )
            repair_total, retrace_total, frames_repaired = 0, 0, 0

            def _decode(h):
                ry, rx = int(_axz_text(h["Resolution"]["Y"])), int(_axz_text(h["Resolution"]["X"]))
                return np.frombuffer(base64.b64decode(_axz_text(h["SampleBase64"])),
                                     dtype=np.float32).reshape((ry, rx)).astype(float)

            for g, h_map in enumerate(filtered_maps):
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
                scan_matrix = scan_1d.reshape((res_y, res_x)).astype(float)

                # Start-of-scan artefact: repaired in acquisition order (row 0
                # = first scan line), before flipping and plane fitting
                n_fixed = 0
                if fix_scan_start:
                    retrace = None
                    if retrace_index is not None:
                        try:
                            r_map = height_maps[g * channel_stride + retrace_index]
                            retrace = _match_retrace(scan_matrix, _decode(r_map),
                                                     skip=max(4, int(round(0.08 * res_y))))
                        except (IndexError, KeyError, ValueError):
                            retrace = None
                    scan_matrix, fixed_mask, n_rt = repair_scan_start(scan_matrix, retrace)
                    n_fixed = int(fixed_mask.sum())
                    repair_total += n_fixed
                    retrace_total += n_rt
                    frames_repaired += n_fixed > 0

                scan_matrix = np.flipud(scan_matrix)

                # Mid-scan time (timestamps mark the end of the scan) and the
                # instrument's own drift estimate, for the drift plot
                scan_rate = _axz_tags(h_map).get("ScanRate", "")
                try:
                    duration = res_y / float(str(scan_rate).split()[0])
                except (ValueError, IndexError, ZeroDivisionError):
                    duration = 0.0
                meta = {
                    "time": _axz_time(h_map.get("TimeStamp")) - duration / 2,
                    "dc": (float(_axz_text(h_map.get("DriftCorrectionX")) or 0),
                           float(_axz_text(h_map.get("DriftCorrectionY")) or 0)),
                    "fixed": n_fixed,
                }

                if flatten:
                    Y, X = np.indices(scan_matrix.shape)
                    x_flat = X.flatten()
                    y_flat = Y.flatten()
                    z_flat = scan_matrix.flatten()

                    A = np.c_[x_flat, y_flat, np.ones_like(x_flat)]
                    C, _, _, _ = np.linalg.lstsq(A, z_flat, rcond=None)

                    plane = C[0] * X + C[1] * Y + C[2]
                    scan_matrix = scan_matrix - plane

                # scan_matrix = scan_matrix - np.min(scan_matrix)

                # Store a tuple of the matrix, its physical size and metadata
                scan_list.append((scan_matrix, phys_size, meta))

            if fix_scan_start:
                if repair_total:
                    source = (f", {retrace_total} filled from the retrace" if retrace_index is not None
                              else " (no retrace of this channel in the file; filled from neighbouring lines)")
                    print(f"Start-of-scan repair: fixed {repair_total} pixels in "
                          f"{frames_repaired}/{len(scan_list)} frames{source}.")
                else:
                    print("Start-of-scan repair: nothing to fix.")

            # --- 4. Enforce Uniform Shape AND Dimensions ---
            if not scan_list:
                print("Error: No scans were extracted.")
                return None

            # Create a "signature" for each scan: ((res_y, res_x), (phys_x, phys_y))
            signatures = [(scan.shape, phys) for scan, phys, _ in scan_list]
            most_common_signature = Counter(signatures).most_common(1)[0][0]
            target_shape, target_phys = most_common_signature

            uniform_scans, metas = [], []
            for i, (scan, phys, meta) in enumerate(scan_list):
                if scan.shape == target_shape and phys == target_phys:
                    uniform_scans.append(scan)
                    metas.append(meta)
                else:
                    print_warning(
                        f"Warning: Scan {i} skipped due to mismatch. "
                        f"Expected shape/size {target_shape} / {target_phys}, "
                        f"got {scan.shape} / {phys}."
                    )

            self.scans = _robust_zero(np.array(uniform_scans))
            self.physical_size = target_phys
            self.frame_times = np.array([m["time"] for m in metas], dtype=float)
            self.instrument_drift = np.array([m["dc"] for m in metas], dtype=float)
            self.repaired_pixels = np.array([m["fixed"] for m in metas], dtype=int)

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

    def calculate_and_apply_drift(self, neighbours=3, min_quality=0.5, min_overlap=0.2,
                                  upsample_factor=10, edge_lines=None):
        """
        Measures the drift of every frame relative to frame 0 and aligns the images.

        Each frame is registered against its previous `neighbours` frames
        (tapered, mean-removed cross-correlation, sub-pixel). Every pair gets a
        quality: the correlation of the overlapping pixels after the shift.
        Pairs with quality >= min_quality and overlap >= min_overlap are combined
        in one least-squares solve for all frame positions, with outlying pairs
        down-weighted, so one bad frame or pair cannot shift every later frame.
        Frames with no usable pair are left out (nan) and interpolated in time.
        `edge_lines` rows at the top and bottom (default ~6%) are left out of
        the registration, where start-of-scan problems live.

        Sets self.drift_path (frames x 2, (Y, X) pixels: the shift that moves
        each frame onto frame 0) and self.drift_quality.
        """
        if self.scans is None or len(self.scans) < 2:
            print("Not enough scans extracted to perform drift correction.")
            return

        frames = np.asarray(self.scans, dtype=float)
        n, ny, nx = frames.shape
        e = max(2, int(round(0.06 * ny))) if edge_lines is None else int(edge_lines)
        core = frames[:, e:ny - e, :]
        window = np.outer(np.hanning(core.shape[1]), np.hanning(core.shape[2]))

        pairs = []
        for j in range(1, n):
            for i in range(max(0, j - neighbours), j):
                s, q, ov = _register_pair(core[i], core[j], window, upsample_factor)
                pairs.append((i, j, s, q, ov))
        good = [p for p in pairs if np.isfinite(p[3]) and p[3] >= min_quality and p[4] >= min_overlap]

        # Frames linked to frame 0 through usable pairs
        linked, frontier = {0}, [0]
        while frontier:
            k = frontier.pop()
            for i, j, *_ in good:
                for a, b in ((i, j), (j, i)):
                    if a == k and b not in linked:
                        linked.add(b)
                        frontier.append(b)
        unknown = sorted(linked - {0})
        col = {k: c for c, k in enumerate(unknown)}
        use = [p for p in good if p[0] in linked and p[1] in linked]

        path = np.full((n, 2), np.nan)
        path[0] = 0.0
        if unknown:
            A = np.zeros((len(use), len(unknown)))
            b = np.array([p[2] for p in use])
            for row, (i, j, *_ ) in enumerate(use):
                if j in col:
                    A[row, col[j]] += 1.0
                if i in col:
                    A[row, col[i]] -= 1.0
            w = np.array([p[3] for p in use])
            for _ in range(5):   # iteratively reweighted: Huber, 1 px
                sw = np.sqrt(w)[:, None]
                sol = np.linalg.lstsq(A * sw, b * sw, rcond=None)[0]
                resid = np.hypot(*(A @ sol - b).T)
                w = np.array([p[3] for p in use]) * np.minimum(1.0, 1.0 / np.maximum(resid, 1e-9))
            for k, c in col.items():
                path[k] = sol[c]

        quality = np.full(n, np.nan)
        for i, j, s, q, ov in pairs:
            if np.isfinite(q) and ov >= min_overlap:
                for k in (i, j):
                    quality[k] = q if not np.isfinite(quality[k]) else max(quality[k], q)

        dropped = [k for k in range(n) if not np.isfinite(path[k, 0])]
        if dropped:
            print_warning(f"Warning: frames {dropped} matched no neighbour (quality < {min_quality}); "
                          "their drift is interpolated from the other frames.")
        self.drift_path = path
        self.drift_quality = quality
        self.drift_min_quality = min_quality

        self.corrected_scans = np.array([
            shift(frames[k], path[k], mode="nearest") if np.isfinite(path[k, 0]) else frames[k]
            for k in range(n)
        ])
        print(f"Drift measured for {n - len(dropped)}/{n} frames: up to "
              f"{np.nanmax(np.abs(path[:, 1])):.1f} px in x and {np.nanmax(np.abs(path[:, 0])):.1f} px in y.")
        return self.corrected_scans

    def spectra_times(self):
        """Mid-acquisition time (s since the epoch) of every spectrum, or None."""
        try:
            spectra = self.raw_dict["Document"]["SNOMSpectra"]["AXDSNOMSpectrum"]
        except (KeyError, TypeError):
            return None
        if isinstance(spectra, dict):
            spectra = [spectra]
        times = []
        for sp in spectra:
            sweep = _iso_seconds((sp.get("InterferogramParameters") or {}).get("SweepTime"))
            times.append(_axz_time(sp.get("TimeStamp")) + (sweep / 2 if np.isfinite(sweep) else 0.0))
        times = np.array(times, dtype=float)
        return times if np.isfinite(times).all() else None

    def drift_at(self, times=None, n_points=None):
        """
        Drift (Y, X pixels) at the given times, interpolated linearly between
        the measured frames (held constant beyond the first/last frame). With
        no usable timestamps, spectra are spread evenly over the frames.
        """
        ok = np.isfinite(self.drift_path[:, 0])
        ft = self.frame_times
        if times is not None and ft is not None and np.isfinite(ft[ok]).all() and np.all(np.diff(ft[ok]) > 0):
            x, xp = np.asarray(times, dtype=float), ft[ok]
        else:
            n = len(times) if times is not None else n_points
            x, xp = np.linspace(0, len(self.drift_path) - 1, n), np.flatnonzero(ok).astype(float)
        return np.column_stack([np.interp(x, xp, self.drift_path[ok, 0]),
                                np.interp(x, xp, self.drift_path[ok, 1])])

    def _instrument_drift_image(self):
        """The instrument's DriftCorrection, rotated into the image frame and
        expressed like drift_path (pixels of correction), for comparison."""
        if self.instrument_drift is None or not np.any(self.instrument_drift):
            return None
        theta = np.deg2rad(-self.scan_angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        dc = self.instrument_drift @ rot.T           # (x, y) um in the image frame
        res_y, res_x = self.scans[0].shape
        px, py = self.physical_size[0] / res_x, self.physical_size[1] / res_y
        return np.column_stack([-dc[:, 1] / py, -dc[:, 0] / px])

    def plot_drift(self):
        """Drift correction per frame and per spectrum, with match quality."""
        if self.drift_path is None:
            print("Drift path is empty. Run calculate_and_apply_drift() first.")
            return None

        res_y, res_x = self.scans[0].shape
        px_nm = 1000 * self.physical_size[0] / res_x
        py_nm = 1000 * self.physical_size[1] / res_y
        ok = np.isfinite(self.drift_path[:, 0])
        have_t = self.frame_times is not None and np.isfinite(self.frame_times).all()
        t0 = self.frame_times[0] if have_t else 0.0
        tf = (self.frame_times - t0) / 60 if have_t else np.arange(len(self.drift_path), dtype=float)
        t_spec = self.spectra_times() if have_t else None
        blue, orange, grey = "#2a78d6", "#eb6834", "0.55"

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        inst = self._instrument_drift_image()
        for ax, comp, scale, name in ((axes[0], 1, px_nm, "X"), (axes[1], 0, py_nm, "Y")):
            ax.plot(tf[ok], self.drift_path[ok, comp] * scale, "-", color=blue, lw=1, alpha=0.6)
            ax.scatter(tf[ok], self.drift_path[ok, comp] * scale, s=22, color=blue, zorder=3,
                       label="measured (frames)")
            if t_spec is not None:
                d = self.drift_at(t_spec)
                ax.scatter((t_spec - t0) / 60, d[:, comp] * scale, s=6, color=orange, zorder=4,
                           label="applied to spectra")
            if inst is not None:
                ax.plot(tf, inst[:, comp] * scale, "--", color=grey, lw=1,
                        label="instrument's DriftCorrection")
            ax.set_title(f"{name} position correction")
            ax.set_xlabel("time (min)" if have_t else "frame number")
            ax.set_ylabel("correction (nm)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        ax = axes[2]
        q = self.drift_quality
        ax.scatter(tf[ok], q[ok], s=22, color=blue, label="placed")
        if (~ok).any():
            ax.scatter(tf[~ok], np.nan_to_num(q[~ok], nan=0.0), marker="x", s=40, color="magenta",
                       label="no usable match (interpolated)")
        ax.axhline(getattr(self, "drift_min_quality", 0.5), color=grey, ls="--", lw=1, label="minimum quality")
        ax.set_ylim(min(0, np.nanmin(q) if np.isfinite(q).any() else 0) - 0.05, 1.05)
        ax.set_title("Match quality (overlap correlation)")
        ax.set_xlabel("time (min)" if have_t else "frame number")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        return fig

    def build_mosaic(self):
        """
        Average of all aligned frames on a canvas large enough to hold them,
        each frame levelled to the frames before it where they overlap.
        Returns (mosaic, extent_um) with extent in frame-0 micrometres.
        """
        ok = np.flatnonzero(np.isfinite(self.drift_path[:, 0]))
        ny, nx = self.scans[0].shape
        d = self.drift_path[ok]
        r0, c0 = int(np.floor(d[:, 0].min())), int(np.floor(d[:, 1].min()))
        r1, c1 = int(np.ceil(d[:, 0].max())), int(np.ceil(d[:, 1].max()))
        H, W = ny + r1 - r0, nx + c1 - c0
        acc, cnt = np.zeros((H, W)), np.zeros((H, W))
        for k in ok:
            dy, dx = self.drift_path[k]
            iy, ix = int(np.floor(dy)) - r0, int(np.floor(dx)) - c0
            canvas = np.full((H, W), np.nan)
            canvas[iy:iy + ny, ix:ix + nx] = self.scans[k]
            canvas = shift(canvas, (dy - np.floor(dy), dx - np.floor(dx)), order=1,
                           mode="constant", cval=np.nan)
            valid = np.isfinite(canvas)
            overlap = valid & (cnt > 0)
            if overlap.sum() > 100:
                canvas -= np.median(canvas[overlap] - acc[overlap] / cnt[overlap])
            acc[valid] += canvas[valid]
            cnt[valid] += 1
        mosaic = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        px, py = self.physical_size[0] / nx, self.physical_size[1] / ny
        extent = [c0 * px, (c0 + W) * px, r0 * py, (r0 + H) * py]
        return mosaic, extent

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

        # scan_mins = np.min(scans, axis=(1, 2), keepdims=True)
        # scans -= scan_mins
        scans = _robust_zero(scans)

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
        # scan_mins = np.min(scans, axis=(1, 2), keepdims=True)
        # scans -= scan_mins
        scans = _robust_zero(scans)

        self.scans = scans
        print(f"Successfully applied degree {degree} '{method}' polynomial flattening.")


    def plot_scans(self, show_corrected=False, num_stdev=None, clip_percentile=0.5, cmap="afmhot"):
        """
        Plots all AFM scans from the array in a square grid with a shared color scale.

        Parameters:
        -----------
        show_corrected : bool
            If True, plots the drift-corrected scans instead of the raw ones.
        clip_percentile : float
            Default limit rule. The color scale runs from this percentile of the
            whole stack to (100 - this percentile), so a few dropout or spike
            pixels cannot stretch the scale. 0.5 clips the darkest and brightest
            0.5% of pixels.
        num_stdev : float or None
            Legacy limit rule. If given, the color scale is mean +/- num_stdev
            standard deviations instead, and clip_percentile is ignored. Note
            that single glitch pixels inflate the standard deviation, so this
            usually gives a washed-out scale.
        """

        data_to_plot = self.corrected_scans if show_corrected else self.scans

        if data_to_plot is None:
            print("No data to plot. Run extraction first.")
            return

        num_plots = len(data_to_plot)

        # --- CHANGED: percentile limits by default, std-dev only on request ---
        if num_stdev is not None:
            mean_val = np.nanmean(data_to_plot)
            std_val = np.nanstd(data_to_plot)
            global_vmin = mean_val - (num_stdev * std_val)
            global_vmax = mean_val + (num_stdev * std_val)
        else:
            global_vmin = np.nanpercentile(data_to_plot, clip_percentile)
            global_vmax = np.nanpercentile(data_to_plot, 100.0 - clip_percentile)

        # Calculate grid dimensions to be as close to a square as possible
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        # Create the figure with dynamically scaled sizing
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
                im = ax.imshow(
                    data_to_plot[idx],
                    cmap=cmap,
                    origin="lower",
                    vmin=global_vmin,
                    vmax=global_vmax,
                    **extent_kwargs,
                )
                ax.set_title(f"{title_prefix} Frame {idx}")
                ax.set_xlabel(xaxis_label)
                ax.set_ylabel(yaxis_label)
            else:
                ax.axis("off")

        # Add a single global colorbar for the entire figure
        if im is not None:
            # CHANGED: list() not .tolist() - axes_flat is a plain list when
            # num_plots == 1, and lists have no .tolist()
            cbar = fig.colorbar(im, ax=list(axes_flat), fraction=0.02, pad=0.04)
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

        # 2. Extract the absolute coordinates of all points
        absolute_coords = extract_raw_spectra_coords(self.raw_dict)

        if len(absolute_coords) == 0:
            print("Error: Could not extract coordinates from XML.")
            return

        angle = float(getattr(self, "scan_angle", 0.0) or 0.0)
        if angle == 0:
            # 3. Calculate the global offset using the FIRST point
            offset_x = local_start_x - absolute_coords[0, 0]
            offset_y = local_start_y - absolute_coords[0, 1]

            # 4. Apply this offset to ALL points
            self.spectra_local_coords = absolute_coords + [offset_x, offset_y]
        else:
            # 3-4. Rotated scan: the AFM images are stored in the scan's own
            # frame, but the spectrum locations are stage coordinates. Rotate
            # each point's offset from the first point back by the scan angle
            # so the array lines up with the image (and the Analysis Studio
            # screenshot), then place the first point at the calibrated spot.
            theta = np.deg2rad(-angle)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            offsets = (absolute_coords - absolute_coords[0]) @ rot.T
            self.spectra_local_coords = offsets + [local_start_x, local_start_y]
            print(f"Scan angle {angle:g} deg: array rotated by {-angle:g} deg to match the AFM image.")

        print(f"Aligned {len(self.spectra_local_coords)} points to local coordinates.")

    def apply_drift_to_spectra(
        self,
        fit_results_df=None,
        map_parameter=None,
        bg_scan_idx=0,
        shared_colormap=False,
        interp_resolution=None,
        grid_shape=None,
        fitting_type = None,
        afmcolormap = 'afmhot',
        hyperspectracmap = 'jet',
        keep_mask = None,
    ):
        """
        Shifts every spectrum point by the drift at the moment it was measured
        (interpolated between AFM frames by timestamp) and plots the corrected
        locations onto the first AFM frame, or onto a mosaic of all aligned
        frames when the corrected points fall outside the first frame.
        Supports map_parameter as a string or a list of strings to plot multiple maps.
        If shared_colormap=True, locks the color scale across all generated plots.

        keep_mask : bool array (one entry per point) or None
            Points where keep_mask is False (fit R^2 below threshold) are not
            coloured: they are drawn as an x in the scatter map, left blank in
            the interpolated map, and ignored when setting the colour limits.
            None keeps every point.
        """

        if not hasattr(self, "drift_path") or self.drift_path is None:
            print("Error: No drift path found. Run calculate_and_apply_drift() first.")
            return

        if (
            not hasattr(self, "spectra_local_coords")
            or self.spectra_local_coords is None
        ):
            print(
                "Error: No spectra coordinates found. Run align_and_plot_spectra() first."
            )
            return

        # 1-2. Drift at the time each spectrum was measured
        num_spectra = len(self.spectra_local_coords)
        times = self.spectra_times()
        if times is not None and len(times) != num_spectra:
            times = None
        drift = self.drift_at(times, n_points=num_spectra)
        dy_pix, dx_pix = drift[:, 0], drift[:, 1]

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

        how = "by timestamp" if times is not None else "evenly in time (no timestamps)"
        print(f"Applied drift to {num_spectra} points, interpolated {how} between "
              f"{int(np.isfinite(self.drift_path[:, 0]).sum())} AFM frames.")

        # Ensure we are working with 1D numpy arrays for plotting
        x_coords = self.corrected_spectra_coords[:, 0]
        y_coords = self.corrected_spectra_coords[:, 1]

        # Normalise the R^2 keep mask (True = colour this point)
        if keep_mask is None:
            keep_mask = np.ones(num_spectra, dtype=bool)
        else:
            keep_mask = np.asarray(keep_mask, dtype=bool)
            if keep_mask.shape != (num_spectra,):
                print_warning(
                    f"Warning: keep_mask has {keep_mask.size} entries but there are "
                    f"{num_spectra} points; R^2 filtering is not applied to this map."
                )
                keep_mask = np.ones(num_spectra, dtype=bool)

        # Background: the first AFM frame, or a mosaic of all aligned frames
        # when drift has carried some points outside the first frame
        from matplotlib.patches import Rectangle
        sx, sy = self.physical_size
        outside = (x_coords < 0) | (x_coords > sx) | (y_coords < 0) | (y_coords > sy)
        use_mosaic = bool(outside.any())
        if use_mosaic:
            if getattr(self, "_mosaic", None) is None:
                self._mosaic = self.build_mosaic()
            background, extent = self._mosaic
            print(f"{int(outside.sum())} points fall outside the first AFM frame: "
                  "using a mosaic of all aligned frames as background.")
        else:
            background, extent = self.scans[bg_scan_idx], [0, sx, 0, sy]

        def _frame0_outline(axis):
            if use_mosaic:
                axis.add_patch(Rectangle((0, 0), sx, sy, fill=False, ls="--", ec="white",
                                         lw=1, zorder=4, label="first AFM frame"))

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
                    # Only points that pass the R^2 filter set the scale
                    vals = fit_results_df[param].to_numpy(dtype=float)
                    all_valid_vals.append(vals[keep_mask & np.isfinite(vals)])

            if all_valid_vals:
                combined_vals = np.concatenate(all_valid_vals)
                if combined_vals.size:
                    global_vmin = np.min(combined_vals)
                    global_vmax = np.max(combined_vals)

        # Loop through each requested parameter
        # Calculate limits once for the background scan to ensure they are always bound
        bg_vmin = np.nanpercentile(background, 1)
        bg_vmax = np.nanpercentile(background, 99)
        for param in parameters_to_plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle(fitting_type)
            ax.imshow(
                background,
                extent=extent,
                aspect="equal",
                cmap=afmcolormap,
                origin="lower",
                alpha=1,
                vmin=bg_vmin,
                vmax=bg_vmax
            )
            _frame0_outline(ax)

            if fit_results_df is not None and param is not None:
                if param in fit_results_df.columns:
                    colors = fit_results_df[param].to_numpy(dtype=float)

                    finite = np.isfinite(colors)
                    valid_mask = finite & keep_mask       # coloured
                    invalid_mask = ~finite                # fit failed
                    lowr2_mask = finite & ~keep_mask      # fit ran, R^2 too low

                    # Determine limits for THIS specific plot
                    if shared_colormap and global_vmin is not None:
                        c_min, c_max = global_vmin, global_vmax
                    elif valid_mask.any():
                        c_min, c_max = np.min(colors[valid_mask]), np.max(colors[valid_mask])
                    else:
                        c_min, c_max = None, None

                    if valid_mask.any():
                        # Add vmin and vmax to lock the scale
                        scatter = ax.scatter(
                            x_coords[valid_mask],
                            y_coords[valid_mask],
                            c=colors[valid_mask],
                            cmap=hyperspectracmap,
                            vmin=c_min,
                            vmax=c_max,
                            s=150,
                            edgecolor="k",
                            linewidth=0.5,
                            zorder=5,
                        )

                        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label(param.replace("_", " ").title(), fontsize=12)
                    else:
                        print_warning(
                            f"Warning: no points left to colour for '{param}' "
                            f"({fitting_type}) after the R^2 filter."
                        )

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

                    if lowr2_mask.any():
                        ax.scatter(
                            x_coords[lowr2_mask],
                            y_coords[lowr2_mask],
                            color="magenta",
                            marker="x",
                            s=60,
                            linewidths=1.5,
                            zorder=6,
                            label=r"Low R$^2$ (excluded)",
                        )

                    if invalid_mask.any() or lowr2_mask.any():
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
                        if grid_shape:
                            rows, cols = grid_shape
                            if rows * cols != len(x_coords):
                                print_warning(f"Warning: grid_shape {grid_shape} does not match {len(x_coords)} points. Using scatter plot only.")
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
                                print_warning(f"Warning: Inferred grid {rows}x{cols} does not match {len(x_coords)} total points. Skipping heatmap interpolation.")
                                continue

                        # Linear arrays (1D) cannot generate a 2D mesh surface.
                        # The scatter plot above already handles their visualization.
                        if rows < 2 or cols < 2:
                            continue

                        fig2, ax2 = plt.subplots(figsize=(8, 6))
                        fig2.suptitle(fitting_type)

                        ax2.imshow(
                            background,
                            extent=extent,
                            aspect="equal",
                            cmap=afmcolormap,
                            origin="lower",
                            alpha=1,
                            vmin=bg_vmin,
                            vmax=bg_vmax
                        )
                        _frame0_outline(ax2)

                        # 1. Reshape the 1D arrays into 2D matrices using explicit dimensions
                        X_2D = x_coords.reshape((rows, cols))
                        Y_2D = y_coords.reshape((rows, cols))
                        # Failed and low-R^2 points become holes in the map
                        C_2D = np.where(valid_mask, colors, np.nan).reshape((rows, cols))
                        hole_2D = ~np.isfinite(C_2D)

                        # A NaN passed to a cubic zoom spreads across the whole
                        # image, so fill each hole with its nearest kept value
                        # for the interpolation only, then mask it out again.
                        if hole_2D.any():
                            nearest = distance_transform_edt(
                                hole_2D, return_distances=False, return_indices=True
                            )
                            C_filled = C_2D[tuple(nearest)]
                        else:
                            C_filled = C_2D

                        # 2. Set upsampling factor
                        # Use a tuple to allow independent scaling if necessary in the future
                        zoom_factors = (interp_resolution, interp_resolution) if interp_resolution else (1, 1)

                        # 3. Upsample the deformed coordinate grid and the values
                        # order=1 uses bilinear interpolation, order=3 uses cubic
                        X_highres = zoom(X_2D, zoom_factors, order=1)
                        Y_highres = zoom(Y_2D, zoom_factors, order=1)
                        C_highres = zoom(C_filled, zoom_factors, order=3)

                        # order=0 keeps each hole a sharp block of the upsampled grid
                        hole_highres = zoom(hole_2D.astype(np.uint8), zoom_factors, order=0).astype(bool)
                        C_highres = np.ma.masked_array(C_highres, mask=hole_highres)

                        # 4. Plot the high-resolution deformed map
                        # (masked cells are not drawn, so the AFM shows through)
                        heatmap = ax2.pcolormesh(
                            X_highres,
                            Y_highres,
                            C_highres,
                            cmap=hyperspectracmap,
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
                    print_warning(
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
    clicks = plt.ginput(3, timeout=0, mouse_stop=None, mouse_pop=None)
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

def list_available_signals(spectrum_dict):
    """Returns the set of DataSignal strings present in one AXZ spectrum."""
    ig_list = spectrum_dict.get("Interferograms", {}).get("AXDSNOMInterferogram", [])
    if isinstance(ig_list, dict):
        ig_list = [ig_list]
    signals = set()
    for ig in ig_list:
        if isinstance(ig, dict):
            sig = ig.get("DataSignal", {}).get("Text")
            if sig:
                signals.add(sig)
    return signals

def extract_axz_interferogram(spectrum_dict, target_signal):
    """Finds and decodes the target interferogram from a spectrum dictionary."""

    ig_list = spectrum_dict.get("Interferograms", {}).get("AXDSNOMInterferogram", [])
    if isinstance(ig_list, dict):
        ig_list = [ig_list]

    for ig in ig_list:
        if not isinstance(ig, dict):
            continue
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
    if isinstance(spectra_list, dict):      # single-spectrum files parse as a dict
        spectra_list = [spectra_list]

    all_intensities = []
    all_stage_positions = []
    expected_length = None

    for i, spec in enumerate(spectra_list):
        stage_pos, intensity = extract_axz_interferogram(
            spec, target_signal=target_signal
        )

        if intensity is None or stage_pos is None:
            available = sorted(list_available_signals(spec))
            if expected_length is None:
                # Nothing has parsed yet - almost certainly a wrong target_signal.
                raise ValueError(
                    f"Point {i}: no interferogram matching "
                    f"target_signal={target_signal!r}.\n"
                    f"Signals present in this file: {available}\n"
                    "Check the 'target_signal' entry in your config."
                )
            # An isolated bad point: fill with NaN rather than dropping it, so the
            # point index stays aligned with extract_raw_spectra_coords().
            print_warning(
                f"Point {i}: interferogram missing or corrupt for "
                f"{target_signal!r}; filling with NaN to keep indexing aligned."
            )
            intensity = np.full(expected_length, np.nan, dtype=np.float32)
            stage_pos = all_stage_positions[-1]

        if expected_length is None:
            expected_length = len(intensity)
        elif len(intensity) != expected_length:
            raise ValueError(
                f"Shape mismatch at point {i}: found {len(intensity)} points, "
                f"expected {expected_length}."
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
    correction_order=2,
):
    """
    Corrects instrumental phase twist by fitting a polynomial to specified background regions,
    and applies it over an independent output range.

    The polynomial comes from phase_utils.baseline_phase_poly, the same function
    the PDM model is flattened with (Fitting_module._flatten_phase), so data and
    model are unwrapped and fitted identically.
    """
    # 1-3. Unwrap per region, align the regions by 2*pi, fit ONE polynomial
    p = baseline_phase_poly(wavenumbers, complex_spectrum, fit_regions, correction_order)

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


# Samples dropped at each end of every sweep: the first and last few can be
# damaged. make_single_spec, interferogram_pp/offset and instrumental_lorentz_fwhm
# all use the same selection.
INTFGM_TRIM = 10


def _recorded_trimmed(pos_z, int_z, trim=INTFGM_TRIM):
    """The samples of one sweep that are processed: recorded (stage position
    != 0) and not within `trim` samples of either end. Returns (pos, values)."""
    pos_z = np.asarray(pos_z, dtype=float)
    int_z = np.asarray(int_z, dtype=float)
    valid = pos_z != 0
    pos, val = pos_z[valid], int_z[valid]
    if trim:
        pos, val = pos[trim:-trim], val[trim:-trim]
    return pos, val


def interferogram_baseline(y, deg=3, burst_frac=0.1):
    """
    The ONE baseline subtraction applied to an interferogram.

    y : the processed samples of one sweep (recorded and trimmed, see
        _recorded_trimmed).

    A polynomial of `deg` in sample index, fitted on the wings only. The
    centreburst - where the smoothed fringe envelope exceeds burst_frac of its
    maximum - is left out of the fit, because its large fringes pull a
    polynomial fitted through them. (Before 2026-11-17 three corrections were
    stacked: a cubic through the whole raw sweep, the mean of the trimmed sweep,
    and a second cubic inside apodization(), each fitted through the burst.)

    The burst is located on a rough residual (y minus a polynomial through every
    sample), so a large slow drift cannot be mistaken for fringes. If fewer than
    4*(deg+1) wing samples remain, every sample is used.

    Returns (y - baseline, baseline, coefficients). coefficients are np.polyfit
    coefficients in the index frame of y; all three are NaN/None if y contains
    non-finite values (a corrupt point, see package_axz_interferograms).
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.all(np.isfinite(y)):
        nan = np.full_like(y, np.nan)
        return nan, nan, None
    idx = np.arange(y.size, dtype=float)
    rough = y - np.polyval(np.polyfit(idx, y, deg), idx)
    env = uniform_filter1d(np.abs(hilbert(rough)), size=max(1, y.size // 50))
    wings = env <= burst_frac * env.max()
    if wings.sum() < 4 * (deg + 1):
        wings = np.ones(y.size, dtype=bool)
    coef = np.polyfit(idx[wings], y[wings], deg)
    base = np.polyval(coef, idx)
    return y - base, base, coef


def baseline_correct_interferogram(pos_z, int_z, trim=INTFGM_TRIM, deg=3):
    """
    Full-length copy of one sweep with exactly the baseline make_single_spec
    removes (same samples, same fit), extended over the trimmed end samples.
    Unrecorded samples (stage position 0) are returned unchanged.

    For interferogram_pp and display only: the spectra are computed from the
    RAW sweeps, so the baseline is never subtracted twice.
    """
    pos_z = np.asarray(pos_z, dtype=float)
    out = np.array(int_z, dtype=float, copy=True)
    valid = np.flatnonzero(pos_z != 0)
    keep = valid[trim:-trim] if trim else valid
    _, _, coef = interferogram_baseline(out[keep], deg=deg)
    if coef is None:
        out[valid] = np.nan
        return out
    # index frame of interferogram_baseline: 0 at the first kept sample
    out[valid] -= np.polyval(coef, np.arange(valid.size, dtype=float) - (trim or 0))
    return out


def baseline_correct_sweeps(intfgm_da, trim=INTFGM_TRIM, deg=3):
    """baseline_correct_interferogram for every point of an interferogram
    DataArray (dims point, step, coordinate stage_position_mm)."""
    pos = np.atleast_2d(intfgm_da.stage_position_mm.values)
    vals = np.atleast_2d(intfgm_da.values)
    out = np.array([baseline_correct_interferogram(p, v, trim, deg) for p, v in zip(pos, vals)])
    return intfgm_da.copy(data=out.reshape(intfgm_da.shape))


def apodization(pos, intfgm, decay_frac=0.25):
    """
    Window for ONE interferogram whose baseline has already been removed
    (interferogram_baseline; this function no longer subtracts one itself).

    erf rising edge before the centreburst x exp(-(x - x_max)/tc) after it,
    tc = decay_frac * mirror travel. Before the envelope maximum the exponential
    is held at 1: it used to keep growing there, exp(+(x_max - x)/tc), so the
    window rose above 1 on the leading side of the burst.
    """
    int_corr = np.asarray(intfgm, dtype=float)

    # 1. Smooth the absolute value
    window_size = max(1, len(int_corr) // 50)
    envelope = uniform_filter1d(np.abs(int_corr), size=window_size)

    # 2. Find indices for max and half-max
    max_i = np.argmax(envelope)

    # np.argmax on a boolean array
    # by returning the index of the first True value
    half_i = np.argmax(envelope > (np.max(envelope) / 2))

    # 3. Calculate standard deviation of the Gaussian pulse
    st_dev = np.abs(pos[max_i] - pos[half_i])

    # Prevent division by zero just in case
    if st_dev == 0:
        st_dev = 1e-9

    # 4. Calculate CDF using the error function
    cdf = 0.5 * (1 + erf((pos - pos[half_i] + st_dev) / (np.sqrt(2) * st_dev)))

    # 5. Exponential decay after the envelope maximum, 1 before it
    if decay_frac:
        exp_tc = decay_frac * (np.max(pos) - np.min(pos))     # mm of mirror travel
        exp_decay = np.exp(-np.clip(pos - pos[max_i], 0.0, None) / exp_tc)
    else:
        exp_decay = np.ones_like(pos)

    # 6. Final Apodization function (never above 1)
    apod_function = exp_decay * cdf

    return int_corr * apod_function, apod_function

def instrumental_lorentz_fwhm(pos_mm, decay_frac, trim=INTFGM_TRIM):
    """
    Lorentzian FWHM (cm^-1) that apodization()'s exponential adds to every band.

    exp(-x/tc) on the interferogram (x = mirror travel after ZPD, tc in mm) is a
    convolution of the spectrum with a Lorentzian of FWHM 1/(pi*lambda_OPD),
    lambda_OPD = 2*tc/10 cm, i.e. 5/(pi*tc_mm), with tc = decay_frac * L.
    The point selection mirrors make_single_spec (drop zeros, trim INTFGM_TRIM each end).
    """
    if not decay_frac:
        return 0.0
    p = np.asarray(pos_mm, dtype=float)
    p = p[p != 0][trim:-trim]
    L = float(p.max() - p.min())          # mirror travel actually used, mm
    return 5.0 / (np.pi * decay_frac * L)

def make_single_spec(pos_z, int_z, pad_pow=2, auto_center_intfgm=True, decay_frac=0.25):
    """
    Calculates the spectrum from an interferogram.
    Returns wavenumbers (w_ns) and complex values (x_complex).
    """
    # 1. Recorded samples only (stage position != 0), minus INTFGM_TRIM samples
    #    at each end, which can be damaged
    pos, int_val = _recorded_trimmed(pos_z, int_z)

    # 2. The one baseline subtraction (polynomial fitted on the wings), and
    #    position normalization. Pass RAW sweeps: nothing upstream subtracts a
    #    baseline any more.
    int_val, _, _ = interferogram_baseline(int_val)
    pos = pos - np.min(pos)

    # 3. Calculate and apply filtering (apodization) function
    int_filtered, _ = apodization(pos, int_val, decay_frac=decay_frac)

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
    dx_mm = (pos[-1] - pos[0]) / (len(pos) - 1)               # mirror step, mm
    w_n   = 1.0 / (len(int_pad) * 2.0 * abs(dx_mm) / 10.0)     # cm^-1 per FFT bin

    # Create the wavenumber axis
    w_ns = np.arange(len(int_pad)) * w_n

    return w_ns, x_complex


def process_all_spectra(spectra_da, pad_pow=2, auto_center=True, decay_frac=0.25):
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
        pos_matrix[0], int_matrix[0], pad_pow=pad_pow, auto_center_intfgm=auto_center, decay_frac=decay_frac,
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
            decay_frac=decay_frac,
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


def plot_intfgm(intfgm, title, ax, pad_pow=2, auto_center=True, decay_frac=0.25):
    """
    Loops through an xarray of interferograms, applies pre-FFT processing
    (baseline, apodization, padding, centering), builds a new xarray, and plots it.
    Pass the RAW interferograms: the baseline is removed here, as in
    make_single_spec.
    """
    num_points = intfgm.sizes["point"]

    processed_intfgm = []
    stage_position = []
    target_l = None  # Track padded length

    for i in range(num_points):
        # 1. Extract the raw 1D arrays for this specific point
        pos_1d = intfgm.stage_position_mm.values[i, :]
        int_1d = intfgm.values[i, :]*1e6

        # Same sample selection and the same single baseline as make_single_spec
        # (pass RAW interferograms)
        pos, int_val = _recorded_trimmed(pos_1d, int_1d)
        int_val, _, _ = interferogram_baseline(int_val)
        pos = pos - np.min(pos)

        # Extract the average step size so we can extrapolate the x-axis later
        dx = np.mean(np.diff(pos))

        # 3. Calculate and apply filtering (apodization) function
        # (Assuming apodization is defined elsewhere in your script)
        int_filtered, _ = apodization(pos, int_val, decay_frac=decay_frac)

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
            "Stage position (mm)": (("point", "step"), stage_position_matrix),
        },
        name=r"X2 Signal ($\mu$V)",
    )

    # 8. Plot the interferograms
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # 1. Generate a smooth list of colors from the 'viridis' colormap
    num_lines = processed_interferograms.sizes["point"]
    gradient_colors = plt.cm.viridis(np.linspace(0, 1, num_lines))

    # 2. Force the axis to use this color gradient for the next plot
    ax.set_prop_cycle(color=gradient_colors)
    processed_interferograms.plot.line(
        x="Stage position (mm)", hue="point", ax=ax, add_legend=False, xlim=(-0.1, 0.2)
    )
    ax.set_title(title + " Interferograms")
    plt.tight_layout()


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


def per_peak_tolerance(center_tolerance, n_peaks):
    """
    One centre tolerance (cm^-1) per peak.

    center_tolerance can be
      a number          5.0            -> 5.0 for every peak
      a one-item list   [5.0]          -> 5.0 for every peak
      a full list       [5, 3, 5, 10]  -> one value per peak, in peak_centers order

    A list of any other length, or a value <= 0, raises: a zero tolerance makes
    the lower and upper centre bounds equal, which curve_fit rejects.
    """
    if np.ndim(center_tolerance) > 0:
        tol = [float(t) for t in np.ravel(center_tolerance)]
        if len(tol) == 1:
            tol = tol * n_peaks
        elif len(tol) != n_peaks:
            raise ValueError(
                f"center_tolerance has {len(tol)} values but there are {n_peaks} "
                "peak_centers. Give one value for all peaks, or one per peak."
            )
    else:
        tol = [float(center_tolerance)] * n_peaks
    bad = [f"peak {j+1}: {t:g}" for j, t in enumerate(tol) if not t > 0]
    if bad:
        raise ValueError("center_tolerance must be > 0 (" + ", ".join(bad) + ").")
    return tol


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


        # --- Validate the fit setup once, up front ---
    if n_peaks == 0:
        raise ValueError("peak_centers is empty - nothing to fit.")

    tols = per_peak_tolerance(center_tolerance, n_peaks)

    if len(fwhm_bounds_list) != n_peaks:
        raise ValueError(
            f"fwhm_bounds has {len(fwhm_bounds_list)} entries but there are "
            f"{n_peaks} peak_centers. Give one [min, max] pair per peak, or a "
            "single (min, max) tuple to use for all of them."
        )
    if len(x) < 5:
        raise ValueError(
            f"Only {len(x)} wavenumber points fall inside "
            f"[{w_sorted[0]:.1f}, {w_sorted[1]:.1f}] cm^-1 - nothing to fit."
        )

    # Centre bounds and centre guesses do not depend on the pixel - build once.
    center_bounds, center_p0, problems = [], [], []
    for j, c in enumerate(peak_centers):
        lo = max(w_sorted[0], c - tols[j])
        hi = min(w_sorted[1], c + tols[j])
        if lo >= hi:
            problems.append(
                f"  peak {j+1}: centre {c} +/- {tols[j]:g} does not overlap "
                f"the fit window [{w_sorted[0]:.1f}, {w_sorted[1]:.1f}]"
            )
            lo, hi = w_sorted[0], w_sorted[1]   # placeholder; we raise below anyway
        f_lo, f_hi = fwhm_bounds_list[j]
        if not f_lo < f_hi:
            problems.append(f"  peak {j+1}: fwhm_bounds {(f_lo, f_hi)} needs min < max")

        c_start = float(np.clip(c, lo, hi))
        if c_start != c:
            print_warning(
                f"Peak {j+1}: centre guess {c} is outside the fit window "
                f"[{w_sorted[0]:.1f}, {w_sorted[1]:.1f}]; starting from {c_start:.1f}."
            )
        center_bounds.append((lo, hi))
        center_p0.append(c_start)

    if problems:
        raise ValueError(
            "Cannot build valid fit bounds:\n" + "\n".join(problems) +
            f"\nCheck peak_centers, center_tolerance and fit_wmin/fit_wmax "
            f"(fit window is currently {w_sorted[0]:.1f} to {w_sorted[1]:.1f} cm^-1)."
        )

    fit_failures = []   # (point index, exception type, message)

    if do_plot:
        plots_per_fig = grid_n * grid_n
        axes_flat = None

    last_popt = None  # Tracker for the Warm Start

    update_interval = max(1, num_points // 20)  # Updates every 10%
    bar_length = 20  # How wide the progress bar is in characters

    for i in range(num_points):
        y_raw = y_matrix[i]
        r_squared = np.nan  # stays NaN unless the fit succeeds

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

                p0.extend([amp_guess, center_p0[j], width_guess])
                bounds_lower.extend([0, center_bounds[j][0], w_min])
                bounds_upper.extend([np.inf, center_bounds[j][1], w_max])

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

                # --- R-squared on the points that were fitted ---
                # (same definition as the PDM fit in Fitting_module)
                y_pred = multi_peak_model(x_fit, *popt)
                ss_res = np.sum((y_fit - y_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            except Exception as exc:
                popt = None
                last_popt = None  # Reset if failed
                fit_failures.append((i, type(exc).__name__, str(exc)))

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
                current_ax.set_title(f"Point {i}  (R$^2$={r_squared:.3f})")
            else:
                current_ax.set_title(f"Point {i}")

        # 3. Parameter extraction
        row_data = {
            "point": i,
            "fit_success": popt is not None,
            "r_squared": r_squared,
        }
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
                f"\rFitting Spectra (Standard): [{bar}] {int(percent_float * 100)}% ({i + 1}/{num_points})",
                end="",
                flush=True,
            )

    print()

    if do_plot:
        plt.show()

    print("\nFitting complete!")

    if fit_failures:
        counts = Counter(kind for _, kind, _ in fit_failures)
        print_warning(f"{len(fit_failures)}/{num_points} spectra failed to fit:")
        for kind, n in counts.most_common():
            print_warning(f"    {n} x {kind}")
        first_i, first_kind, first_msg = fit_failures[0]
        print_warning(f"    first: point {first_i} - {first_kind}: {first_msg}")

    return pd.DataFrame(results)


def _recorded_samples(intfgm, trim=INTFGM_TRIM):
    """
    Yields each interferogram's usable samples (in volts): the ones
    make_single_spec keeps. Unrecorded samples (stage position 0) and
    non-finite values are dropped, and `trim` samples at each end are ignored,
    since those can be damaged.
    """
    vals = np.atleast_2d(np.asarray(intfgm.values, dtype=float))
    try:
        pos = np.atleast_2d(np.asarray(intfgm.stage_position_mm.values, dtype=float))
    except AttributeError:
        pos = np.ones_like(vals)
    for p, v in zip(pos, vals):
        v = v[(p != 0) & np.isfinite(v)]
        if trim and v.size > 2 * trim:
            v = v[trim:-trim]
        yield v


def interferogram_pp(intfgm, trim=INTFGM_TRIM):
    """
    Peak-to-peak amplitude of each interferogram, one value per point, in uV
    (recorded volts x 1e6, the scaling plot_intfgm uses).

    Samples as in _recorded_samples. Pass baseline-corrected interferograms,
    so a slow drift of the offset during a sweep does not count as amplitude.
    """
    return np.array([(v.max() - v.min()) * 1e6 if v.size else np.nan
                     for v in _recorded_samples(intfgm, trim)])


def interferogram_offset(intfgm, trim=INTFGM_TRIM):
    """
    Mean level (DC offset) of each interferogram, one value per point, in uV.

    The offset is the part of the demodulated signal that does not interfere
    with the reference arm: tip-scattered light at this harmonic, summed over
    the whole laser spectrum. It tracks the broadband (background) scattering
    and laser power, independent of the fringes. Pass the RAW interferograms -
    baseline correction removes it. Signed: it is the lock-in X quadrature, so
    its sign depends on the demodulator phase. Samples as in _recorded_samples.
    """
    return np.array([v.mean() * 1e6 if v.size else np.nan
                     for v in _recorded_samples(intfgm, trim)])


def background_tilt(wavenumbers, spectra, regions):
    """
    Level and spectral tilt of each spectrum's background, measured on its
    off-resonance points only: no fit model is involved, so the result is the
    same whatever fit (if any) runs afterwards, and nothing trades off against
    the peaks.

    A straight line is fitted to ln|S| against wavenumber over the points that
    fall inside `regions` (phase_fit_regions, which are chosen free of
    absorption):  ln|S(w)| = ln(level) + slope * (w - w_c),  where w_c is the
    middle of the span the regions cover.

    wavenumbers : (W,) axis in cm^-1
    spectra     : (N, W) or (W,) spectra, complex or real (e.g. S/R); one row per point
    regions     : list of [lo, hi] pairs; empty or malformed entries are ignored

    Returns (level, tilt), each of length N:
      level : |S| at w_c - the background amplitude of the normalised spectrum
      tilt  : relative change of |S| in % per 100 cm^-1, i.e. 1e4 * slope
              (logarithmic, so +10 means |S| grows by a factor e^0.1 per
              100 cm^-1). It follows the spectral SHAPE of the background, for
              example the laser spectrum drifting away from the one the
              reference was recorded with, a changing non-resonant response or
              far-field background. Interferogram p-p and offset cannot show
              this.
    A spectrum with fewer than 3 usable points in the regions gives NaN.
    Absorption inside the regions (band wings included) leaks into both values.
    """
    w = np.asarray(wavenumbers, dtype=float)
    S = np.atleast_2d(np.asarray(spectra))
    n = S.shape[0]
    level = np.full(n, np.nan)
    tilt = np.full(n, np.nan)

    in_regions = np.zeros(w.size, dtype=bool)
    edges = []
    for r in regions or []:
        if r is None or len(r) != 2:
            continue
        lo, hi = sorted(float(v) for v in r)
        in_regions |= (w >= lo) & (w <= hi)
        edges += [lo, hi]
    if not edges or not in_regions.any():
        return level, tilt

    x = w[in_regions] - 0.5 * (min(edges) + max(edges))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.log(np.abs(S[:, in_regions]))
    for i in range(n):
        ok = np.isfinite(y[i])
        if ok.sum() < 3 or np.ptp(x[ok]) == 0:
            continue
        slope, intercept = np.polyfit(x[ok], y[i, ok], 1)
        level[i] = np.exp(intercept)
        tilt[i] = 1e4 * slope
    return level, tilt


def _point_series_panel(ax, values, keep, row_length, color, trend_color):
    """One per-point series with row boundaries, hollow R^2 failures and a
    running median. Returns the running median."""
    n = len(values)
    x = np.arange(n)
    if row_length and row_length < n:
        for b in range(row_length, n, row_length):
            ax.axvline(b - 0.5, color="0.88", lw=0.8, zorder=0)
    ax.plot(x, values, color=color, lw=0.8, alpha=0.5, zorder=2)
    ax.scatter(x[keep], values[keep], s=14, color=color, zorder=3, label="point")
    if (~keep).any():
        ax.scatter(x[~keep], values[~keep], s=14, facecolors="none", edgecolors=color,
                   zorder=3, label="point (fails R² filter)")
    window = int(row_length) if row_length else max(3, n // 10)
    trend = pd.Series(values).rolling(window, center=True, min_periods=1).median().to_numpy()
    ax.plot(x, trend, color=trend_color, lw=2, zorder=4, label=f"running median ({window} points)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    return trend


def plot_intfgm_pp(pp, title, ref_pp=None, row_length=None, keep=None,
                   offset=None, ref_offset=None, tilt=None, tilt_label=None):
    """
    Drift monitors against point number: interferogram peak-to-peak amplitude
    and, when given, interferogram offset and background tilt, one panel each.

    pp         : per-point amplitudes from interferogram_pp (uV)
    ref_pp     : the Au reference interferogram(s); their mean is given in the title
    row_length : points per scan row; draws faint row boundaries and sets the
                 running-median window. Sample contrast changes from point to
                 point; a drift shows up as a trend in the running median.
    keep       : boolean mask of points that pass the R^2 filter; the others
                 are drawn hollow.
    offset     : per-point interferogram offsets from interferogram_offset (uV)
    ref_offset : the Au reference interferogram offset(s), given in that panel's title
    tilt       : per-point background tilt from background_tilt (% per 100 cm^-1)
    tilt_label : y label of the tilt panel (names the normalised quantity)
    """
    from matplotlib.ticker import MaxNLocator

    pp = np.asarray(pp, dtype=float)
    n = len(pp)
    keep = np.ones(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    blue, orange, green, violet = "#2a78d6", "#eb6834", "#1f9e76", "#6250d6"
    with_offset = offset is not None and len(offset) == n
    with_tilt = tilt is not None and len(tilt) == n and np.isfinite(np.asarray(tilt, float)).any()

    n_panels = 1 + with_offset + with_tilt
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 0.6 + 3.0 * n_panels), sharex=True, squeeze=False)
    axes = list(axes[:, 0])

    def _with_ref(text, ref):
        # stated, not drawn: the reference is usually several times larger and
        # a line at its level would flatten the drift this plot is for
        if ref is not None and np.isfinite(ref).any():
            return f"{text}   (Au reference {float(np.nanmean(ref)):.0f} µV)"
        return text

    def _zero_line(ax, values):
        if np.nanmin(values) < 0 < np.nanmax(values):   # only when it crosses zero,
            ax.axhline(0, color="0.6", lw=0.8, zorder=1)   # else it squashes the trend

    ax = axes.pop(0)
    _point_series_panel(ax, pp, keep, row_length, blue, orange)
    ax.set_ylabel("interferogram p-p (µV)")
    ax.set_title(_with_ref(f"{title}: interferogram peak-to-peak", ref_pp),
                 fontsize=10, loc="left")
    last = ax

    if with_offset:
        ax = axes.pop(0)
        offset = np.asarray(offset, dtype=float)
        _point_series_panel(ax, offset, keep, row_length, green, orange)
        _zero_line(ax, offset)
        ax.set_ylabel("interferogram offset (µV)")
        ax.set_title(_with_ref("interferogram offset", ref_offset), fontsize=10, loc="left")
        last = ax

    if with_tilt:
        ax = axes.pop(0)
        tilt = np.asarray(tilt, dtype=float)
        _point_series_panel(ax, tilt, keep, row_length, violet, orange)
        _zero_line(ax, tilt)
        ax.set_ylabel(tilt_label or "background tilt (% / 100 cm⁻¹)")
        ax.set_title("background tilt over phase_fit_regions",
                     fontsize=10, loc="left")
        last = ax

    last.set_xlabel("point number")
    last.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def plot_correlations(df, pairs, fitting_type):
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
    fig.suptitle(fitting_type, fontsize=16)

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
# Misc functions
# ==========================================


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
            # keep the row so point i still lines up with fit-result row i
            absolute_coords.append([np.nan, np.nan])
            continue

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
                    print_warning(
                        f"Warning: Raw image '{clean_label}' has unexpected byte ratio ({bytes_per_pixel} bytes/pixel)."
                    )
            else:
                print_warning(f"Warning: Could not determine resolution for '{clean_label}'.")

        except Exception as e:
            print_warning(f"Warning: Failed to decode '{clean_label}'. Error: {e}")


def save_all_interferograms(data_dict, output_folder):
    """Saves spectral data, mapping Y-axis units from DataSources. The sweep
    position is written in mm, converted exactly as extract_axz_interferogram
    reads it (the .axz stores metres)."""
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

                # Units. SweepStart/SweepEnd are stored in METRES even though
                # the file's SweepUnits says "mm" (e.g. 0.0002-0.0015 for a
                # 0.2-1.5 mm sweep). extract_axz_interferogram reads them as
                # metres, so convert the same way and label the column mm.
                # (Before 2026-11-17 the raw metre values were written under the
                # file's "mm" label.)
                z_unit_reference[clean_name] = "mm"
                y_unit_reference[clean_name] = y_units_map.get(sig_name, "a.u.")

                # Construct Z-axis in mm (metres x 1e3)
                sweep_start = float(ig.get("SweepStart", {}).get("Text", 0))
                sweep_end = float(ig.get("SweepEnd", {}).get("Text", 1))
                points = int(ig.get("Points", {}).get("Text", len(data_array)))

                z_axis_reference[clean_name] = np.linspace(
                    sweep_start, sweep_end, points
                ) * 1e3

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
