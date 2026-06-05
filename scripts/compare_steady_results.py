"""Compare simulation results with reference values from original XML files."""

import math

from _xml_helpers import compare_field, extract_field, find_element


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
        ref_path = f"cases/simple_steady_tests/case{case_num}.xml"
        out_path = f"results/simple_steady_tests/case{case_num}_output.xml"

        print(f"\n=== Case {case_num} (threshold = {threshold:g}K) ===")
        ref = extract_field(ref_path)
        out = extract_field(out_path)

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

        cmp = compare_field(ref_vals, out_vals, threshold)
        if cmp is None:
            print(f"  SIZE MISMATCH: ref has {len(ref_vals)} values, out has {len(out_vals)} values")
            continue
        mx, mean, over, total = cmp
        # Prefer reference size for position decoding; fall back to output size.
        pos_size = ref_size if all(d > 0 for d in ref_size) else out_size

        # Re-walk the inputs to recover worst-point positions, preserving the original diagnostic output.
        worst = []
        for i, (rv, ov) in enumerate(zip(ref_vals, out_vals)):
            if math.isnan(rv) and math.isnan(ov):
                continue
            if math.isnan(rv) or math.isnan(ov):
                continue
            err = abs(rv - ov)
            if err > threshold:
                worst.append((i, rv, ov, err))

        print(f"  Max error: {mx:.4f}K")
        print(f"  Mean error: {mean:.4f}K")
        print(f"  Points >{threshold:g}K error: {over} / {total}")
        if worst:
            worst.sort(key=lambda x: x[3], reverse=True)
            label = "All misaligned"
            print(f"  {label} (position = (vx, vy, vz)):")
            for idx, rv, ov, err in worst:
                vx, vy, vz = index_to_position(idx, pos_size)
                print(
                    f"    idx={idx} pos=({vx},{vy},{vz}): ref={rv:.2f}, out={ov:.2f}, err={err:.4f}"
                )


if __name__ == "__main__":
    main()
