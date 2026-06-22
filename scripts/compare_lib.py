"""Unified comparison library for steady / transient cases.

A case is identified by its reference XML and output XML paths. This module
owns the per-case comparison work; run_cases.py owns the list of cases.

Thresholds (5K field for steady, 1K field + 5K probe for transient) are
hard-coded by design — they are the regression bar from the original
per-script values, and changing them is an explicit code change here.

Public entry points:
  - compare_field_case: steady field comparison
  - compare_transient_case: transient field + probe comparison
  - main: CLI that discovers case directories and runs the appropriate
    comparator over each XML pair
"""

import math
import os
import xml.etree.ElementTree as ET

from _xml_helpers import (
    compare_field,
    extract_field,
    find_all,
    find_element,
    index_to_position,
    parse_doubles,
    valid_values,
)

# Hard-coded regression thresholds per case kind.
FIELD_THRESHOLD = 5.0  # steady
TRANSIENT_FIELD_THRESHOLD = 1.0  # transient final-step field
PROBE_THRESHOLD = 5.0  # transient Result0DTransient probe traces


def _print_field_summary(ref_vals, ref_size, out_vals, out_size):
    """Print the size / total / valid / min / max header for ref and out fields."""
    ref_valid = valid_values(ref_vals)
    out_valid = valid_values(out_vals)
    print(
        f"  Ref:  size={ref_size}, total={len(ref_vals)}, valid={len(ref_valid)}, "
        f"min={min(ref_valid):.2f}, max={max(ref_valid):.2f}"
    )
    print(
        f"  Out:  size={out_size}, total={len(out_vals)}, valid={len(out_valid)}, "
        f"min={min(out_valid):.2f}, max={max(out_valid):.2f}"
    )


def _print_field_error_stats(mx, mean, over, total, threshold):
    print(f"  Max error:  {mx:.4f}K")
    print(f"  Mean error: {mean:.4f}K")
    print(f"  Points >{threshold:g}K error: {over} / {total}")


def compare_field_case(
    ref_path, out_path, label, threshold=FIELD_THRESHOLD, keep_worst=False
):
    """Compare a single steady case. Returns True if within threshold."""
    print(f"\n=== {label} (threshold = {threshold:g}K) ===")
    ref = extract_field(ref_path)
    out = extract_field(out_path)
    if not ref or not out:
        print("  Failed to extract values")
        return False
    ref_vals, ref_size = ref
    out_vals, out_size = out
    _print_field_summary(ref_vals, ref_size, out_vals, out_size)
    cmp = compare_field(ref_vals, out_vals, threshold, keep_worst=keep_worst)
    if cmp is None:
        print(
            f"  SIZE MISMATCH: ref has {len(ref_vals)} values, out has {len(out_vals)} values"
        )
        return False
    mx, mean, over, total, worst = cmp
    _print_field_error_stats(mx, mean, over, total, threshold)
    if worst:
        pos_size = ref_size if all(d > 0 for d in ref_size) else out_size
        print("  Worst points (position = (vx, vy, vz)):")
        for idx, rv, ov, err in worst:
            vx, vy, vz = index_to_position(idx, pos_size)
            print(
                f"    idx={idx} pos=({vx},{vy},{vz}): ref={rv:.2f}, out={ov:.2f}, err={err:.4f}"
            )
    return over == 0


def extract_probe_traces(xml_path):
    """Return dict {point_name: (times[], values[])} for every Result0DTransient."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    results = find_element(root, "Results")
    if results is None:
        return {}
    traces = {}
    for anytype in find_all(results, "anyType"):
        type_attr = anytype.attrib.get(
            "{http://www.w3.org/2001/XMLSchema-instance}type", ""
        )
        if "Result0DTransient" not in type_attr:
            continue
        pn = find_element(anytype, "PointName")
        if pn is None or not (pn.text or "").strip():
            continue
        name = pn.text.strip()
        times = parse_doubles(find_element(anytype, "Times"))
        values = parse_doubles(find_element(anytype, "Values"))
        traces[name] = (times, values)
    return traces


def _compare_probe_traces(ref_v, out_v):
    """Compare two equal-length value series. Returns (max, mean, n_compared)."""
    errs = [
        abs(r - o) for r, o in zip(ref_v, out_v) if not (math.isnan(r) or math.isnan(o))
    ]
    if not errs:
        return 0.0, 0.0, 0
    return max(errs), sum(errs) / len(errs), len(errs)


def compare_transient_case(ref_path, out_path, label):
    """Compare a transient case: final-step field + probe traces. Returns True if all within threshold."""
    print(
        f"\n=== Transient {label} (field threshold = {TRANSIENT_FIELD_THRESHOLD:g}K) ==="
    )
    overall_ok = True

    # 1) Final-step field
    ref_field = extract_field(ref_path)
    out_field = extract_field(out_path)
    if ref_field is None or out_field is None:
        print("  Failed to extract final-step field")
        overall_ok = False
    else:
        ref_vals, ref_size = ref_field
        out_vals, out_size = out_field
        _print_field_summary(ref_vals, ref_size, out_vals, out_size)
        cmp = compare_field(ref_vals, out_vals, TRANSIENT_FIELD_THRESHOLD)
        if cmp is None:
            print(f"  SIZE MISMATCH: ref={len(ref_vals)} out={len(out_vals)}")
            overall_ok = False
        else:
            mx, mean, over, total, _ = cmp
            print(
                f"  Max error: {mx:.4f}K, mean error: {mean:.4f}K, "
                f"points >{TRANSIENT_FIELD_THRESHOLD:g}K: {over} / {total}"
            )
            if over > 0:
                overall_ok = False

    # 2) Probe traces
    ref_traces = extract_probe_traces(ref_path)
    out_traces = extract_probe_traces(out_path)
    all_probe_names = sorted(set(ref_traces) | set(out_traces))
    if not all_probe_names:
        print("  No Result0DTransient probes found.")
    for name in all_probe_names:
        ref = ref_traces.get(name)
        out = out_traces.get(name)
        if ref is None or out is None:
            print(
                f"  Probe {name!r}: missing in {'reference' if ref is None else 'output'}"
            )
            overall_ok = False
            continue
        _ref_t, ref_v = ref
        _out_t, out_v = out
        if len(ref_v) != len(out_v):
            print(
                f"  Probe {name!r}: length mismatch ref={len(ref_v)} out={len(out_v)}"
            )
            overall_ok = False
            continue
        mx, mean, n = _compare_probe_traces(ref_v, out_v)
        ok_full = "OK" if mx < PROBE_THRESHOLD else "FAIL"
        print(
            f"  Probe {name!r}: n={n}, max_diff={mx:.4f}K, mean_diff={mean:.4f}K  "
            f"[full {ok_full} @ {PROBE_THRESHOLD:g}K]"
        )
        if mx >= PROBE_THRESHOLD:
            overall_ok = False
    return overall_ok


def _discover_cases(case_dir, results_dir):
    """Yield (case_name, ref_path, out_path) for every XML in case_dir that has a matching output."""
    if not os.path.isdir(case_dir):
        return
    for case_file in sorted(f for f in os.listdir(case_dir) if f.endswith(".xml")):
        case_name = case_file[:-4]
        ref_path = os.path.join(case_dir, case_file)
        out_path = os.path.join(results_dir, f"{case_name}_output.xml")
        if not os.path.isfile(out_path):
            print(f"  [{case_name}] missing output {out_path}; run run_cases.py first")
            continue
        yield case_name, ref_path, out_path


def _run_dir_field(case_dir, results_dir, label):
    ok = True
    for case_name, ref_path, out_path in _discover_cases(case_dir, results_dir):
        ok &= compare_field_case(ref_path, out_path, f"{label} {case_name}")
    return ok


def _run_dir_transient(case_dir, results_dir, label):
    ok = True
    for case_name, ref_path, out_path in _discover_cases(case_dir, results_dir):
        ok &= compare_transient_case(ref_path, out_path, f"{label} {case_name}")
    return ok


def main():
    """CLI entry point: compare every XML pair under a case directory.

    Usage: compare_lib.py <case_dir> <results_dir> <steady|transient>
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", help="Directory containing reference XML cases")
    parser.add_argument(
        "results_dir", help="Directory containing <name>_output.xml files"
    )
    parser.add_argument("kind", choices=("steady", "transient"))
    args = parser.parse_args()

    if args.kind == "transient":
        ok = _run_dir_transient(args.case_dir, args.results_dir, "")
    else:
        ok = _run_dir_field(args.case_dir, args.results_dir, "")

    print(
        "\nAll within threshold." if ok else "\nOne or more cases exceeded threshold."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
