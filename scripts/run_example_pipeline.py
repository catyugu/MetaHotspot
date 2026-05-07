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

    print(f"[PIPELINE] steady solve: {steady_cfg}")
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

    print(f"[PIPELINE] transient solve: {transient_cfg}")
    _run_solver(transient_cfg)

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
