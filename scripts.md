# Project Source Code: scripts

## Directory Structure
```text
.
├── adapter.py
├── collect_context.py
└── run_example_pipeline.py
```

## File Contents

### File: adapter.py
```py
import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.legacy.converter import convert_hotspot_with_modes

_logger = logging.getLogger(__name__)


def _convert_batch(hotspot_examples_dir: str, output_root: str, mode: str) -> None:
    """Convert HotSpot examples example1~example4."""
    for name in ("example1", "example2", "example3", "example4"):
        in_dir = os.path.join(hotspot_examples_dir, name)
        out_dir = os.path.join(output_root, name)
        created = convert_hotspot_with_modes(in_dir, out_dir, mode=mode)
        _logger.info(f"{name} -> {out_dir}")
        for config_path in created:
            _logger.info(f"  wrote {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HotSpot example inputs to MetaHotspot solver config."
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
        "--batch-four",
        action="store_true",
        help="Convert HotSpot examples example1~example4 in one command",
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

    if args.batch_four:
        _convert_batch(args.hotspot_examples_dir, args.output_root, args.mode)
        return

    if not args.input_dir or not args.output_dir:
        parser.error(
            "input_dir and output_dir are required unless --batch-four is used"
        )

    created = convert_hotspot_with_modes(
        args.input_dir, args.output_dir, mode=args.mode
    )
    for config_path in created:
        _logger.info(f"wrote {config_path}")


if __name__ == "__main__":
    main()

```

### File: collect_context.py
```py
import logging
import os
import re
import argparse
from pathlib import Path

_logger = logging.getLogger(__name__)

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

    _logger.info(f"处理完成！输出文件已保存至: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将代码仓库整理为 AI 友好的 Markdown 格式"
    )
    parser.add_argument("input_folder", help="要读取的文件夹路径")
    parser.add_argument("output_file", help="输出的文本文件名 (例如 output.md)")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        _logger.error(f"错误: 文件夹 '{args.input_folder}' 不存在")
    else:
        process_repository(args.input_folder, args.output_file)

```

### File: run_example_pipeline.py
```py
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.metahotspot_solver import MetaHotspotSolver


def _write_solver_configs_from_template(
    template: dict, steady_path: Path, transient_path: Path
) -> None:
    steady = copy.deepcopy(template)
    steady["simulation_type"] = "steady"

    transient = copy.deepcopy(template)
    transient["simulation_type"] = "transient"
    transient["init_temperature_file_path"] = "init.vtu"

    with open(steady_path, "w", encoding="utf-8") as handle:
        json.dump(steady, handle, indent=4)
    with open(transient_path, "w", encoding="utf-8") as handle:
        json.dump(transient, handle, indent=4)


def _ensure_solver_configs(example_dir: Path) -> tuple[Path, Path]:
    steady_path = example_dir / "solver_config_steady.json"
    transient_path = example_dir / "solver_config_transient.json"

    if steady_path.exists() and transient_path.exists():
        return steady_path, transient_path

    template_path = example_dir / "solver_config.json"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            template = json.load(f)
        _write_solver_configs_from_template(template, steady_path, transient_path)
        return steady_path, transient_path

    if steady_path.exists() and not transient_path.exists():
        with open(steady_path, "r", encoding="utf-8") as f:
            transient = copy.deepcopy(json.load(f))
        transient["simulation_type"] = "transient"
        transient["init_temperature_file_path"] = "init.vtu"
        with open(transient_path, "w", encoding="utf-8") as handle:
            json.dump(transient, handle, indent=4)
        return steady_path, transient_path

    if transient_path.exists() and not steady_path.exists():
        with open(transient_path, "r", encoding="utf-8") as f:
            steady = copy.deepcopy(json.load(f))
        steady["simulation_type"] = "steady"
        with open(steady_path, "w", encoding="utf-8") as handle:
            json.dump(steady, handle, indent=4)
        return steady_path, transient_path

    raise FileNotFoundError(
        "No solver configuration found. Expected one of: "
        f"{steady_path}, {transient_path}, or {template_path}."
    )


def _run_solver(config_path: Path) -> None:
    MetaHotspotSolver(str(config_path)).run()


def _force_transient_init_file(transient_cfg: Path) -> None:
    with open(transient_cfg, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    data["simulation_type"] = "transient"
    data["init_temperature_file_path"] = "init.vtu"

    with open(transient_cfg, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def run_pipeline(example_dir: Path) -> None:
    outputs_dir = example_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    steady_cfg, transient_cfg = _ensure_solver_configs(example_dir)
    shutil.copy2(steady_cfg, outputs_dir / steady_cfg.name)
    shutil.copy2(transient_cfg, outputs_dir / transient_cfg.name)

    _run_solver(steady_cfg)

    steady_result = example_dir / "result.vtu"
    if not steady_result.exists():
        raise FileNotFoundError(f"Steady result not found: {steady_result}")

    init_vtu = example_dir / "init.vtu"
    shutil.copy2(steady_result, init_vtu)
    shutil.copy2(steady_result, outputs_dir / "steady_result.vtu")
    shutil.copy2(init_vtu, outputs_dir / "init.vtu")

    # Always run transient with the latest steady result as initial field.
    _force_transient_init_file(transient_cfg)
    _run_solver(transient_cfg)

    transient_result = example_dir / "transient_result.vtu"
    if not transient_result.exists():
        raise FileNotFoundError(f"Transient result not found: {transient_result}")

    shutil.copy2(transient_result, outputs_dir / "transient_result.vtu")


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

