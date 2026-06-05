"""Shared XML helpers for the steady/transient comparison scripts."""

import math
import xml.etree.ElementTree as ET

NS = {
    "ts": "http://schemas.datacontract.org/2004/07/ThermalSim.Models",
    "a": "http://schemas.microsoft.com/2003/10/Serialization/Arrays",
    "b": "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh",
}


def _local_tag(child):
    tag = child.tag
    return tag.split("}")[-1] if "}" in tag else tag


def find_element(parent, local_tag):
    for child in parent:
        if _local_tag(child) == local_tag:
            return child
    return None


def find_all(parent, local_tag):
    return [child for child in parent if _local_tag(child) == local_tag]


def parse_doubles(elem):
    """Return a list of floats from all <a:double> children. Tolerates NaN text."""
    if elem is None:
        return []
    out = []
    for child in elem:
        if _local_tag(child) != "double":
            continue
        text = (child.text or "").strip()
        if text == "":
            continue
        if text == "NaN" or text.lower() == "nan":
            out.append(float("nan"))
        else:
            out.append(float(text))
    return out


def _read_size(values_elem, tag):
    el = find_element(values_elem, tag)
    return int(el.text) if el is not None else 0


def extract_field(xml_path):
    """Return (values, (sx, sy, sz)) of the first Result3D a:anyType in the XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = find_element(root, "Results")
    if results is None:
        return None
    anytype = find_element(results, "anyType")
    if anytype is None:
        return None
    values = find_element(anytype, "Values")
    if values is None:
        return None
    data = find_element(values, "Data")
    if data is None:
        return None
    size = (_read_size(values, "SizeX"), _read_size(values, "SizeY"), _read_size(values, "SizeZ"))
    return parse_doubles(data), size


def compare_field(ref_vals, out_vals, threshold):
    """Compare two flat value arrays index-by-index. Return (max, mean, over, total) or None on length mismatch."""
    if len(ref_vals) != len(out_vals):
        return None
    errs = []
    over = 0
    for r, o in zip(ref_vals, out_vals):
        if math.isnan(r) and math.isnan(o):
            continue
        if math.isnan(r) or math.isnan(o):
            continue
        e = abs(r - o)
        errs.append(e)
        if e > threshold:
            over += 1
    if not errs:
        return 0.0, 0.0, 0, 0
    return max(errs), sum(errs) / len(errs), over, len(errs)
