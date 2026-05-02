import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metahotspot.fvm_solver import FVMSolver


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MetaHotspot finite-volume solver")
    parser.add_argument("config", help="Path to solver_config.json")
    args = parser.parse_args()

    FVMSolver(args.config).solve()


if __name__ == "__main__":
    main()
