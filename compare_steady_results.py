"""Compare simulation results with reference values from original XML files."""

import argparse
import xml.etree.ElementTree as ET
import math

NS = {
    "ts": "http://schemas.datacontract.org/2004/07/ThermalSim.Models",
    "a": "http://schemas.microsoft.com/2003/10/Serialization/Arrays",
    "b": "http://schemas.datacontract.org/2004/07/ThermalSim.Models.Mesh",
}


def find_element(parent, local_tag):
    for child in parent:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == local_tag:
            return child
    return None


def extract_values(xml_path):
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
    size_x_el = find_element(values, "SizeX")
    size_y_el = find_element(values, "SizeY")
    size_z_el = find_element(values, "SizeZ")

    sx = int(size_x_el.text) if size_x_el is not None else 0
    sy = int(size_y_el.text) if size_y_el is not None else 0
    sz = int(size_z_el.text) if size_z_el is not None else 0

    vals = []
    for child in data:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "double":
            text = child.text.strip()
            if text == "NaN" or text.lower() == "nan":
                vals.append(float("nan"))
            else:
                vals.append(float(text))
    return vals, (sx, sy, sz)


def index_to_position(idx, size):
    """Convert flat index to (vx, vy, vz) for layout index = vz + SizeZ*vy + SizeZ*SizeY*vx."""
    sx, sy, sz = size
    if sy <= 0 or sz <= 0:
        return (idx, 0, 0)
    vx = idx // (sy * sz)
    vy = (idx // sz) % sy
    vz = idx % sz
    return (vx, vy, vz)


def main():
    threshold = 5

    for case_num in [1, 2, 3]:
        ref_path = f"cases/original_steady_tests/case{case_num}.xml"
        out_path = f"results/original_steady_tests/case{case_num}_output.xml"

        print(f"\n=== Case {case_num} (threshold = {threshold:g}K) ===")
        ref = extract_values(ref_path)
        out = extract_values(out_path)

        if not ref or not out:
            print("  Failed to extract values")
            continue

        ref_vals, ref_size = ref
        out_vals, out_size = out

        ref_valid = [v for v in ref_vals if not math.isnan(v) and abs(v) < 1e10]
        out_valid = [v for v in out_vals if not math.isnan(v) and abs(v) < 1e10]

        print(
            f"  Ref:  size={ref_size}, total={len(ref_vals)}, valid={len(ref_valid)}, min={min(ref_valid):.2f}, max={max(ref_valid):.2f}"
        )
        print(
            f"  Out:  size={out_size}, total={len(out_vals)}, valid={len(out_valid)}, min={min(out_valid):.2f}, max={max(out_valid):.2f}"
        )

        if len(ref_vals) != len(out_vals):
            print(
                f"  SIZE MISMATCH: ref has {len(ref_vals)} values, out has {len(out_vals)} values"
            )
            continue

        # Prefer reference size for position decoding; fall back to output size.
        pos_size = ref_size if all(d > 0 for d in ref_size) else out_size

        errors = []
        worst = []
        for i in range(len(ref_vals)):
            rv = ref_vals[i]
            ov = out_vals[i]
            if math.isnan(rv) and math.isnan(ov):
                continue
            if math.isnan(rv) or math.isnan(ov):
                continue
            err = abs(rv - ov)
            errors.append(err)
            if err > threshold:
                worst.append((i, rv, ov, err))

        if errors:
            print(f"  Max error: {max(errors):.4f}K")
            print(f"  Mean error: {sum(errors)/len(errors):.4f}K")
            print(f"  Points >{threshold:g}K error: {len(worst)} / {len(errors)}")
            if worst:
                worst.sort(key=lambda x: x[3], reverse=True)
                to_print = worst
                label = "All misaligned"
                print(f"  {label} (position = (vx, vy, vz)):")
                for idx, rv, ov, err in to_print:
                    vx, vy, vz = index_to_position(idx, pos_size)
                    print(
                        f"    idx={idx} pos=({vx},{vy},{vz}): ref={rv:.2f}, out={ov:.2f}, err={err:.4f}"
                    )


if __name__ == "__main__":
    main()
