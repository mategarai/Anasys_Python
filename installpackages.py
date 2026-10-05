#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checks that every package the pipeline needs is installed and new enough.

find_spec() only reports presence, so a minimum version is checked separately
against the installed distribution metadata.
"""

import sys
import subprocess
import importlib.util
from importlib.metadata import version, PackageNotFoundError

# import_name: (pip_name, minimum_version or None)
required_packages = {
    "numpy":      ("numpy", "2.0"),          # np.ptp(); ndarray.ptp removed in 2.0
    "pandas":     ("pandas", "2.1"),         # DataFrame.map (was applymap before 2.1)
    "skimage":    ("scikit-image", "0.19"),  # phase_cross_correlation 3-tuple return
    "matplotlib": ("matplotlib", "3.5"),
    "scipy":      ("scipy", None),
    "xarray":     ("xarray", None),
    "PIL":        ("Pillow", None),
    "gwyfile":    ("gwyfile", None),
    # Optional: only the interactive calibration dialog needs Qt. Once the
    # PyQt5 import in snom_utils is moved inside calibrate_start_point, a
    # machine with no Qt can still run everything by setting
    # "relative_array_coords" in the config, and this line can be deleted.
    "PyQt5":      ("PyQt5", None),
}


def _version_key(v):
    """'2.4.4' -> (2, 4, 4). Non-numeric parts (rc, dev) count as 0."""
    return tuple(int(p) if p.isdigit() else 0 for p in v.split(".")[:3])


for module_name, (package_name, minimum) in required_packages.items():
    needs_install = importlib.util.find_spec(module_name) is None

    if not needs_install and minimum:
        try:
            installed = version(package_name)
        except PackageNotFoundError:
            installed = None
        if installed and _version_key(installed) < _version_key(minimum):
            print(f"{package_name} {installed} is older than the required "
                  f"{minimum}; upgrading...")
            needs_install = True

    if not needs_install:
        print(f"{module_name} up to date!")
        continue

    spec = f"{package_name}>={minimum}" if minimum else package_name
    print(f"Installing {spec}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", spec]
        )
        print(f"Successfully installed {spec}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {spec}. Error: {e}")