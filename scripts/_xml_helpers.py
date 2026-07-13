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
    size = (
        _read_size(values, "SizeX"),
        _read_size(values, "SizeY"),
        _read_size(values, "SizeZ"),
    )
    return parse_doubles(data), size


VALID_ABS_LIMIT = 1e10


def valid_values(vals):
    """Filter out NaN and unphysically-large values; used for diagnostic min/max."""
    return [v for v in vals if not math.isnan(v) and abs(v) < VALID_ABS_LIMIT]


def index_to_position(idx, size):
    """Convert flat index to (vx, vy, vz) for layout index = vz + SizeZ*vy + SizeZ*SizeY*vx."""
    sx, sy, sz = size
    if sy <= 0 or sz <= 0:
        return (idx, 0, 0)
    vx = idx // (sy * sz)
    vy = (idx // sz) % sy
    vz = idx % sz
    return (vx, vy, vz)


def compare_field(ref_vals, out_vals, threshold, keep_worst=False):
    """Compare two flat value arrays index-by-index.

    Returns (max, mean, over, total, worst) on success, or None on length mismatch.
    `worst` is a list of (idx, ref, out, err) tuples for entries exceeding
    threshold, sorted by error descending. Empty when keep_worst=False.
    """
    if len(ref_vals) != len(out_vals):
        return None
    errs = []
    over = 0
    worst = [] if keep_worst else None
    for i, (r, o) in enumerate(zip(ref_vals, out_vals)):
        if math.isnan(r) and math.isnan(o):
            continue
        if math.isnan(r) or math.isnan(o):
            continue
        e = abs(r - o)
        errs.append(e)
        if e > threshold:
            over += 1
            if worst is not None:
                worst.append((i, r, o, e))
    if worst is not None:
        worst.sort(key=lambda x: x[3], reverse=True)
    if not errs:
        return 0.0, 0.0, 0, 0, worst or []
    return max(errs), sum(errs) / len(errs), over, len(errs), worst or []
