#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:33:47 2026

@author: mategarai
"""

"""
Extracts XML from .axz containers (ZIP/GZIP) and parses it into Python dictionaries.
"""

import base64
import gzip
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Union, Optional, List

# ----------------------- XML name mangling -----------------------


def mangle_xml_name(name: str) -> str:
    return (
        name.replace("-", "_dash_")
        .replace(":", "_colon_")
        .replace(".", "_dot_")
        .replace("_", "u_")
    )


_WHITESPACE_RE = re.compile(r"\s+")


def has_nonwhitespace_text(s: Optional[str]) -> bool:
    """Return True if string contains any non-whitespace characters."""
    return bool(s and _WHITESPACE_RE.sub("", s))


# ----------------------- .axz extraction -----------------------


def extract_main_xml(
    axz_path: Union[str, Path], *, output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Extract the primary XML document from an Anasys `.axz` file.
    Handles both ZIP and GZIP containers.
    """
    axz_path = Path(axz_path)
    if not axz_path.exists():
        raise FileNotFoundError(f"File not found: {axz_path}")

    out_dir = Path(output_dir) if output_dir else axz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = out_dir / f"{axz_path.stem}.xml"
    if xml_path.exists():
        return xml_path

    # Prefer ZIP (most common)
    if zipfile.is_zipfile(axz_path):
        with zipfile.ZipFile(axz_path, "r") as zf:
            xml_members = sorted(
                [m for m in zf.namelist() if m.lower().endswith(".xml")],
                key=lambda m: zf.getinfo(m).file_size,
                reverse=True,
            )
            if not xml_members:
                raise RuntimeError("ZIP .axz contained no .xml members.")

            with zf.open(xml_members[0], "r") as src, open(xml_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return xml_path

    # Fall back to GZIP
    try:
        with gzip.open(axz_path, "rb") as gz, open(xml_path, "wb") as dst:
            shutil.copyfileobj(gz, dst)
        return xml_path
    except OSError as e:
        raise RuntimeError("File is neither a ZIP nor a readable GZIP stream.") from e


# ----------------------- XML -> dict parsing -----------------------


def _append_child(mapping: Dict[str, Any], key: str, value: Dict[str, Any]) -> None:
    """If key exists, convert existing item to list and append"""
    if key in mapping:
        if isinstance(mapping[key], list):
            mapping[key].append(value)
        else:
            mapping[key] = [mapping[key], value]
    else:
        mapping[key] = value


def _parse_element_et(elem: ET.Element) -> Dict[str, Any]:
    """stdlib ElementTree parser."""
    out: Dict[str, Any] = {}

    # Parse children
    for child in list(elem):
        key = mangle_xml_name(child.tag.split("}")[-1])
        _append_child(out, key, _parse_element_et(child))

    # Parse text content
    text_parts: List[str] = []
    if has_nonwhitespace_text(elem.text):
        text_parts.append(elem.text.strip())
    for child in list(elem):
        if has_nonwhitespace_text(child.tail):
            text_parts.append(child.tail.strip())

    if text_parts:
        out["Text"] = "".join(text_parts)

    # Parse attributes
    if elem.attrib:
        out["Attributes"] = {mangle_xml_name(k): v for k, v in elem.attrib.items()}

    return out


def parse_xml_to_dict(xml_input: Union[str, Path]) -> Dict[str, Any]:
    """Parse XML (file path or XML string) into a nested dict."""
    xml_input_str = str(xml_input)

    if Path(xml_input_str).exists():
        root = ET.parse(xml_input_str).getroot()
    else:
        root = ET.fromstring(xml_input_str)

    root_key = mangle_xml_name(root.tag.split("}")[-1])
    return {root_key: _parse_element_et(root)}


# ----------------------- Public API -----------------------


def load_axz_as_dict(
    axz_path: Union[str, Path], *, output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Extract the main XML from an `.axz` and parse it into a dict."""
    xml_path = extract_main_xml(axz_path, output_dir=output_dir)
    return parse_xml_to_dict(xml_path)


def decode_float64_base64(b64_text: str) -> Any:
    """Decode base64 text to a NumPy float64 vector. (Kept for snom_utils compatibility)."""
    import numpy as np

    raw = base64.b64decode(b64_text)
    return np.frombuffer(raw, dtype=np.float64)


if __name__ == "__main__":
    pass
