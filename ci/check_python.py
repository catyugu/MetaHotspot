"""Build an sdist, then build, install and test its wheel."""

import sys
from pathlib import Path

from check_build import run


def main():
    root = Path(__file__).resolve().parents[1]
    dist = root / "dist"
    run(sys.executable, "-m", "build", root, "--outdir", dist)
    (wheel,) = dist.glob("*.whl")
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", wheel)
    run(sys.executable, "-c", "import metahotspot; print(metahotspot.__version__)")
    run(sys.executable, "-m", "pytest", root / "python" / "tests")


if __name__ == "__main__":
    main()
