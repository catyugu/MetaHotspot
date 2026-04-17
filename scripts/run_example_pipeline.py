import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import toml


def _write_solver_configs(example_dir: Path) -> tuple[Path, Path]:
    template_path = example_dir / "solver_config.toml"
    if not template_path.exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")

    base = toml.load(template_path)

    steady = copy.deepcopy(base)
    steady["simulation_type"] = "steady"

    transient = copy.deepcopy(base)
    transient["simulation_type"] = "transient"
    transient["init_temperature_file_path"] = "init.vtu"

    steady_path = example_dir / "solver_config_steady.toml"
    transient_path = example_dir / "solver_config_transient.toml"

    with open(steady_path, "w", encoding="utf-8") as handle:
        toml.dump(steady, handle)
    with open(transient_path, "w", encoding="utf-8") as handle:
        toml.dump(transient, handle)

    return steady_path, transient_path


def _run_solver(project_root: Path, config_path: Path) -> None:
    solver_script = project_root / "scripts" / "solver.py"
    subprocess.run(
        [sys.executable, str(solver_script), str(config_path)],
        check=True,
        cwd=str(project_root),
    )


def run_pipeline(example_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    outputs_dir = example_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    steady_cfg, transient_cfg = _write_solver_configs(example_dir)
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
