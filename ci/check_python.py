"""Build and repair one wheel, or install and test a previously repaired wheel."""

import argparse
import os
import sys
from pathlib import Path

from check_build import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel", type=Path, help="Test this existing wheel; do not build"
    )
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.wheel:
        wheel = args.wheel.resolve()
    else:
        dist = root / "dist"
        run(sys.executable, "-m", "build", root, "--outdir", dist)
        (raw,) = dist.glob("*.whl")
        repaired = root / "wheelhouse"
        if sys.platform == "linux":
            run(
                "auditwheel",
                "repair",
                "--plat",
                "manylinux_2_28_x86_64",
                "-w",
                repaired,
                raw,
            )
        elif sys.platform == "darwin":
            run(
                "delocate-wheel",
                "--require-archs",
                "arm64",
                "--require-target-macos-version",
                "11.0",
                "-w",
                repaired,
                raw,
            )
        else:
            run(
                sys.executable,
                "-m",
                "delvewheel",
                "repair",
                "--analyze-existing",
                "--no-mangle",
                "mhs_c_api.dll",
                "-w",
                repaired,
                raw,
            )
        (wheel,) = repaired.glob("*.whl")
        run(sys.executable, root / "ci/check_wheel.py", wheel)
    if args.build_only:
        return
    run(sys.executable, "-m", "pip", "install", "--force-reinstall", wheel)
    # A build-tree override would invalidate the installed-wheel test.
    os.environ.pop("MHS_CAPI_PATH", None)
    run(sys.executable, "-c", "import metahotspot; print(metahotspot.__version__)")
    run(sys.executable, "-m", "pytest", root / "python/tests")


if __name__ == "__main__":
    main()
