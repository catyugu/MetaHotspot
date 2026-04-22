# Project Source Code: scripts

## Directory Structure
```text
.
├── adapter.py
├── collect_context.py
├── compare_hotspot.py
├── run_example_pipeline.py
├── solver.py
└── visualize.py
```

## File Contents

### File: adapter.py
```py
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.converter import convert_hotspot_with_modes


def _convert_batch_three(
    hotspot_examples_dir: str, output_root: str, mode: str
) -> None:
    for name in ("example1", "example2", "example3"):
        in_dir = os.path.join(hotspot_examples_dir, name)
        out_dir = os.path.join(output_root, name)
        created = convert_hotspot_with_modes(in_dir, out_dir, mode=mode)
        print(f"[CONVERT] {name} -> {out_dir}")
        for config_path in created:
            print(f"[CONVERT]   wrote {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Hotspot example inputs to MetaHotspot solver config and mesh."
    )
    parser.add_argument("input_dir", nargs="?", help="Hotspot example directory")
    parser.add_argument("output_dir", nargs="?", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=("steady", "transient", "both"),
        default="both",
        help="Generate steady, transient, or both solver configs (default: both)",
    )
    parser.add_argument(
        "--batch-three",
        action="store_true",
        help="Convert Hotspot examples example1~example3 in one command",
    )
    parser.add_argument(
        "--hotspot-examples-dir",
        default="Hotspot/examples",
        help="Path to Hotspot examples root (default: Hotspot/examples)",
    )
    parser.add_argument(
        "--output-root",
        default="examples/hotspot_converted",
        help="Batch output root (default: examples/hotspot_converted)",
    )
    args = parser.parse_args()

    if args.batch_three:
        _convert_batch_three(args.hotspot_examples_dir, args.output_root, args.mode)
        return

    if not args.input_dir or not args.output_dir:
        parser.error(
            "input_dir and output_dir are required unless --batch-three is used"
        )

    created = convert_hotspot_with_modes(
        args.input_dir, args.output_dir, mode=args.mode
    )
    for config_path in created:
        print(f"[CONVERT] wrote {config_path}")


if __name__ == "__main__":
    main()

```

### File: collect_context.py
```py
import os
import re
import argparse
from pathlib import Path

# ===================== 配置区 =====================
# 1. 目录忽略正则（匹配文件夹名）
IGNORE_DIR_PATTERNS = [
    r"^\.git$",
    r"^\.vscode$",
    r"^\.idea$",
    r"^__pycache__$",
    r"^node_modules$",
    r"^venv$",
    r"^env$",
    r"^dist$",
    r"^build",
    r"^external$",
    r"^vcpkg_installed$",
    r"^cases$",
    r"^results$",
    r"^\.next$",
    r"^\.pytest_cache$",
]

# 2. 文件忽略正则（匹配文件名）
IGNORE_FILE_PATTERNS = [
    r"^\.DS_Store$",
    r"^package-lock\.json$",
    r"^yarn\.lock$",
    r"^poetry\.lock$",
    r"^pnpm-lock\.yaml$",
    r"^favicon\.ico$",
]

# 3. 允许读取的文本文件后缀
ALLOWED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".c",
    ".cpp",
    ".java",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".h",
    ".hpp",
    ".sql",
    ".yaml",
    ".toml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".html",
    ".css",
    ".sh",
    ".ini",
    ".conf",
    ".cmake",
}
# ==================================================

# 预编译正则表达式（提升性能）
COMPILED_DIR_IGNORES = [re.compile(p) for p in IGNORE_DIR_PATTERNS]
COMPILED_FILE_IGNORES = [re.compile(p) for p in IGNORE_FILE_PATTERNS]


def is_ignored_dir(name: str) -> bool:
    """判断目录是否需要忽略（正则匹配）"""
    return any(pattern.match(name) for pattern in COMPILED_DIR_IGNORES)


def is_ignored_file(name: str) -> bool:
    """判断文件是否需要忽略（正则匹配）"""
    return any(pattern.match(name) for pattern in COMPILED_FILE_IGNORES)


def is_text_file(file_path: Path) -> bool:
    """判断是否为应读取的文本文件"""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def generate_tree(root_dir: Path, prefix: str = "") -> str:
    """递归生成目录树结构字符串"""
    tree_str = ""
    paths = sorted(list(root_dir.iterdir()), key=lambda x: (x.is_file(), x.name))

    # 过滤忽略的目录/文件
    filtered = []
    for p in paths:
        if p.is_dir() and is_ignored_dir(p.name):
            continue
        if p.is_file() and is_ignored_file(p.name):
            continue
        filtered.append(p)

    for i, path in enumerate(filtered):
        connector = "└── " if i == len(filtered) - 1 else "├── "
        tree_str += f"{prefix}{connector}{path.name}\n"
        if path.is_dir():
            extension = "    " if i == len(filtered) - 1 else "│   "
            tree_str += generate_tree(path, prefix + extension)
    return tree_str


def process_repository(repo_path: str, output_file: str):
    root_path = Path(repo_path).resolve()

    with open(output_file, "w", encoding="utf-8") as f:
        # 1. 写入项目标题
        f.write(f"# Project Source Code: {root_path.name}\n\n")

        # 2. 写入目录树
        f.write("## Directory Structure\n")
        f.write("```text\n")
        f.write(".\n")
        f.write(generate_tree(root_path))
        f.write("```\n\n")

        # 3. 递归遍历并写入文件内容
        f.write("## File Contents\n\n")
        for current_path, dirs, files in os.walk(root_path):
            # 跳过忽略的目录（os.walk 会自动不再进入）
            dirs[:] = [d for d in dirs if not is_ignored_dir(d)]

            for file in files:
                if is_ignored_file(file):
                    continue

                file_path = Path(current_path) / file
                if is_text_file(file_path):
                    relative_path = file_path.relative_to(root_path)

                    f.write(f"### File: {relative_path}\n")
                    lang = file_path.suffix.lstrip(".")
                    f.write(f"```{lang}\n")

                    try:
                        with open(
                            file_path, "r", encoding="utf-8", errors="replace"
                        ) as code_f:
                            f.write(code_f.read())
                    except Exception as e:
                        f.write(f"/* Error reading file: {e} */")

                    f.write("\n```\n\n")

    print(f"✅ 处理完成！输出文件已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将代码仓库整理为 AI 友好的 Markdown 格式"
    )
    parser.add_argument("input_folder", help="要读取的文件夹路径")
    parser.add_argument("output_file", help="输出的文本文件名 (例如 output.md)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"❌ 错误: 文件夹 '{args.input_folder}' 不存在")
    else:
        process_repository(args.input_folder, args.output_file)

```

### File: compare_hotspot.py
```py
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

```

### File: run_example_pipeline.py
```py
import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import toml


def _write_solver_configs_from_template(
    template: dict, steady_path: Path, transient_path: Path
) -> None:
    steady = copy.deepcopy(template)
    steady["simulation_type"] = "steady"

    transient = copy.deepcopy(template)
    transient["simulation_type"] = "transient"
    transient["init_temperature_file_path"] = "init.vtu"

    with open(steady_path, "w", encoding="utf-8") as handle:
        toml.dump(steady, handle)
    with open(transient_path, "w", encoding="utf-8") as handle:
        toml.dump(transient, handle)


def _ensure_solver_configs(example_dir: Path) -> tuple[Path, Path]:
    steady_path = example_dir / "solver_config_steady.toml"
    transient_path = example_dir / "solver_config_transient.toml"

    if steady_path.exists() and transient_path.exists():
        return steady_path, transient_path

    template_path = example_dir / "solver_config.toml"
    if template_path.exists():
        template = toml.load(template_path)
        _write_solver_configs_from_template(template, steady_path, transient_path)
        return steady_path, transient_path

    if steady_path.exists() and not transient_path.exists():
        transient = copy.deepcopy(toml.load(steady_path))
        transient["simulation_type"] = "transient"
        transient["init_temperature_file_path"] = "init.vtu"
        with open(transient_path, "w", encoding="utf-8") as handle:
            toml.dump(transient, handle)
        return steady_path, transient_path

    if transient_path.exists() and not steady_path.exists():
        steady = copy.deepcopy(toml.load(transient_path))
        steady["simulation_type"] = "steady"
        with open(steady_path, "w", encoding="utf-8") as handle:
            toml.dump(steady, handle)
        return steady_path, transient_path

    raise FileNotFoundError(
        "No solver configuration found. Expected one of: "
        f"{steady_path}, {transient_path}, or {template_path}."
    )


def _run_solver(project_root: Path, config_path: Path) -> None:
    solver_script = project_root / "scripts" / "solver.py"
    subprocess.run(
        [sys.executable, str(solver_script), str(config_path)],
        check=True,
        cwd=str(project_root),
    )


def _force_transient_init_file(transient_cfg: Path) -> None:
    data = toml.load(transient_cfg)
    data["simulation_type"] = "transient"
    data["init_temperature_file_path"] = "init.vtu"
    with open(transient_cfg, "w", encoding="utf-8") as handle:
        toml.dump(data, handle)


def run_pipeline(example_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = example_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    steady_cfg, transient_cfg = _ensure_solver_configs(example_dir)
    shutil.copy2(steady_cfg, outputs_dir / steady_cfg.name)
    shutil.copy2(transient_cfg, outputs_dir / transient_cfg.name)

    print(f"[PIPELINE] steady solve: {steady_cfg}")
    _run_solver(project_root, steady_cfg)

    steady_result = example_dir / "result.vtu"
    if not steady_result.exists():
        raise FileNotFoundError(f"Steady result not found: {steady_result}")

    init_vtu = example_dir / "init.vtu"
    shutil.copy2(steady_result, init_vtu)
    shutil.copy2(steady_result, outputs_dir / "steady_result.vtu")
    shutil.copy2(init_vtu, outputs_dir / "init.vtu")

    # Always run transient with the latest steady result as initial field.
    _force_transient_init_file(transient_cfg)

    print(f"[PIPELINE] transient solve: {transient_cfg}")
    _run_solver(project_root, transient_cfg)

    transient_result = example_dir / "transient_result.vtu"
    if not transient_result.exists():
        raise FileNotFoundError(f"Transient result not found: {transient_result}")

    shutil.copy2(transient_result, outputs_dir / "transient_result.vtu")
    print(f"[PIPELINE] outputs saved to {outputs_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate steady/transient configs and run steady->init->transient pipeline."
    )
    parser.add_argument("example_dir", help="Path to one converted example directory")
    args = parser.parse_args()

    run_pipeline(Path(args.example_dir).resolve())


if __name__ == "__main__":
    main()

```

### File: solver.py
```py
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.fvm_solver import FVMSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MetaHotspot finite-volume solver")
    parser.add_argument("config", help="Path to solver_config.toml")
    args = parser.parse_args()

    FVMSolver(args.config).solve()


if __name__ == "__main__":
    main()

```

### File: visualize.py
```py
import argparse

import pyvista as pv


def _pick_scalar(mesh: pv.DataSet) -> str:
    if "Temperature_K" in mesh.cell_data:
        return "Temperature_K"
    raise KeyError("No Temperature_K cell data found")


def visualize(vtu_path: str) -> None:
    mesh = pv.read(vtu_path)
    scalar_name = _pick_scalar(mesh)

    print(f"Loading mesh from {vtu_path}")
    print(mesh)

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=scalar_name, cmap="hot", show_edges=True)
    plotter.add_scalar_bar("Temperature (K)")
    plotter.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize temperature field in VTU mesh"
    )
    parser.add_argument("vtu_path", help="Path to result VTU file")
    args = parser.parse_args()
    visualize(args.vtu_path)


if __name__ == "__main__":
    main()

```

