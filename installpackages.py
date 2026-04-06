#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:43:07 2026

@author: mategarai
"""

import sys
import subprocess
import importlib.util

# 1. Define the required packages (import_name: pip_install_name)
required_packages = {
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "xarray": "xarray",
    "skimage": "scikit-image",
    "PyQt5": "PyQt5",
    "PIL": "Pillow",
    "pandas": "pandas",
    "gwyfile": "gwyfile",
}

# 2. Check and install missing packages
for module_name, package_name in required_packages.items():
    if importlib.util.find_spec(module_name) is None:
        print(f"Installing missing package: {package_name}...")
        try:
            # sys.executable ensures it uses the pip associated with your current Python environment
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package_name]
            )
            print(f"Successfully installed {package_name}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package_name}. Error: {e}")
