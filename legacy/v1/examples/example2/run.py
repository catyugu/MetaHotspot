import subprocess
import sys
from pathlib import Path


def main() -> None:
    example_dir = Path(__file__).resolve().parent
    project_root = example_dir.parents[1]
    script = project_root / "scripts" / "run_example_pipeline.py"
    subprocess.run([sys.executable, str(script), str(example_dir)], check=True)


if __name__ == "__main__":
    main()
