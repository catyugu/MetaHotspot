import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import meshio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_hotspot_series(path: str, trace_index: int) -> List[float]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        return []

    is_grid = any(line.startswith("Layer ") for line in lines)
    has_time_headers = any(line.startswith("t =") for line in lines)

    if not is_grid and not has_time_headers:
        values = []
        for line in lines:
            parts = line.split()
            try:
                values.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
        return values

    # Grid steady or transient files are represented as one or many frames.
    frames: List[List[float]] = []
    current_frame: List[float] = []

    for line in lines:
        if line.startswith("t ="):
            if current_frame:
                frames.append(current_frame)
                current_frame = []
            continue

        if line.startswith("Layer "):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            current_frame.append(float(parts[1]))
        except ValueError:
            continue

    if current_frame:
        frames.append(current_frame)

    if not frames:
        return []

    if trace_index < 0:
        return frames[-1]
    if trace_index >= len(frames):
        raise IndexError(
            f"Requested trace_index={trace_index}, but only {len(frames)} frames exist"
        )
    return frames[trace_index]


def _load_mesh_temperature(path: str) -> List[float]:
    mesh = meshio.read(path)

    field_name = None
    if "Temperature_K" in mesh.cell_data:
        field_name = "Temperature_K"
    else:
        raise KeyError("No Temperature_K cell data found in mesh")

    values: List[float] = []
    for block, block_values in zip(mesh.cells, mesh.cell_data[field_name]):
        if block.type != "hexahedron":
            continue
        values.extend(np.asarray(block_values, dtype=float).tolist())

    return values


def _load_numeric_series(path: str) -> List[float]:
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.replace(",", " ").split()
            try:
                values.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    return values


def _load_metahotspot_series(path: str) -> List[float]:
    extension = os.path.splitext(path)[1].lower()
    if extension in {".vtu", ".vtk", ".msh"}:
        return _load_mesh_temperature(path)
    return _load_numeric_series(path)


def _basic_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _compare_equal_length(
    reference: np.ndarray, candidate: np.ndarray
) -> Dict[str, float]:
    diff = candidate - reference
    return {
        "max_abs_error": float(np.max(np.abs(diff))),
        "mean_abs_error": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
    }


def _compare_distribution(
    reference: np.ndarray, candidate: np.ndarray
) -> Dict[str, float]:
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]
    ref_q = np.quantile(reference, quantiles)
    cand_q = np.quantile(candidate, quantiles)
    delta_q = cand_q - ref_q

    result = {}
    for q, delta in zip(quantiles, delta_q):
        result[f"quantile_delta_{int(q * 100):02d}"] = float(delta)
    result["max_abs_quantile_delta"] = float(np.max(np.abs(delta_q)))
    return result


def compare(
    hotspot_path: str,
    metahotspot_path: str,
    trace_index: int = -1,
    threshold_k: float = 1.0,
) -> Dict[str, object]:
    hotspot_values = np.asarray(
        _load_hotspot_series(hotspot_path, trace_index), dtype=float
    )
    metahotspot_values = np.asarray(
        _load_metahotspot_series(metahotspot_path), dtype=float
    )

    if hotspot_values.size == 0:
        raise ValueError("No usable values read from Hotspot output")
    if metahotspot_values.size == 0:
        raise ValueError("No usable values read from MetaHotspot output")

    summary: Dict[str, object] = {
        "hotspot": _basic_stats(hotspot_values),
        "metahotspot": _basic_stats(metahotspot_values),
        "same_length": bool(hotspot_values.size == metahotspot_values.size),
        "threshold_k": threshold_k,
    }

    if hotspot_values.size == metahotspot_values.size:
        metrics = _compare_equal_length(hotspot_values, metahotspot_values)
        summary["metrics"] = metrics
        summary["pass"] = bool(metrics["max_abs_error"] <= threshold_k)
    else:
        metrics = _compare_distribution(hotspot_values, metahotspot_values)
        summary["metrics"] = metrics
        summary["pass"] = bool(metrics["max_abs_quantile_delta"] <= threshold_k)
        summary["note"] = (
            "Vector lengths differ. Distribution-based comparison was used. "
            "Use matching mesh resolution for strict point-wise validation."
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Hotspot temperature output with MetaHotspot output."
    )
    parser.add_argument(
        "hotspot_output", help="Hotspot output file (.steady/.ttrace/.grid.*)"
    )
    parser.add_argument(
        "metahotspot_output", help="MetaHotspot output (.vtu or numeric text)"
    )
    parser.add_argument(
        "--trace-index",
        type=int,
        default=-1,
        help="Transient frame index for Hotspot grid ttrace files (default: last frame)",
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=1.0,
        help="Pass/fail threshold in Kelvin (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default="",
        help="Optional path to write JSON summary",
    )

    args = parser.parse_args()
    result = compare(
        args.hotspot_output,
        args.metahotspot_output,
        trace_index=args.trace_index,
        threshold_k=args.threshold_k,
    )

    status = "PASS" if result["pass"] else "FAIL"
    print(f"[COMPARE] {status}")
    print(f"[COMPARE] Hotspot cells: {result['hotspot']['count']}")
    print(f"[COMPARE] MetaHotspot cells: {result['metahotspot']['count']}")

    for key, value in result["metrics"].items():
        print(f"[COMPARE] {key} = {value:.6f}")

    if "note" in result:
        print(f"[COMPARE] Note: {result['note']}")

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"[COMPARE] JSON summary written to {args.json_path}")


if __name__ == "__main__":
    main()
