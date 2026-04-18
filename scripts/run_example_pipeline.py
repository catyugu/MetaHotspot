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
