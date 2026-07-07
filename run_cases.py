"""Run all registered test cases and compare results against their references.

Cases are registered as a list of (case_dir, results_subdir, kind) tuples,
where kind is "steady" or "transient".
Thresholds live in compare_lib.py — they are the regression bar, not a
per-invocation knob.

The case group list is the single source of truth for what exists; both the
metahotspot runs and the comparisons iterate it.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
EXE = os.path.join(REPO_ROOT, "build", "bin", "metahotspot")

# (case_dir, results_subdir, kind)
CASE_GROUPS = [
    ("cases/simple_steady_tests", "simple_steady_tests", "steady"),
    ("cases/nonlinear_steady_tests", "nonlinear_steady_tests", "steady"),
    ("cases/simple_transient_tests", "simple_transient_tests", "transient"),
]


def _run_one(input_path, vtu_path, xml_path):
    print(f">>> Running {os.path.basename(input_path)}", flush=True)
    subprocess.run(
        [
            EXE,
            "--input",
            input_path,
            "--output-vtu",
            vtu_path,
            "--output-xml",
            xml_path,
        ],
        check=False,
    )


def _run_group(case_dir, results_subdir, _kind=""):
    if not os.path.isdir(case_dir):
        return
    os.makedirs(os.path.join("results", results_subdir), exist_ok=True)
    for case_file in sorted(f for f in os.listdir(case_dir) if f.endswith(".xml")):
        case_name = case_file[:-4]
        _run_one(
            os.path.join(case_dir, case_file),
            os.path.join("results", results_subdir, f"{case_name}_output.vtu"),
            os.path.join("results", results_subdir, f"{case_name}_output.xml"),
        )


def _compare_group(case_dir, results_subdir, kind):
    if not os.path.isdir(case_dir):
        return True
    return (
        subprocess.run(
            [
                sys.executable,
                os.path.join(SCRIPTS_DIR, "compare_lib.py"),
                os.path.join(REPO_ROOT, case_dir),
                os.path.join(REPO_ROOT, "results", results_subdir),
                kind,
            ],
            check=False,
        ).returncode
        == 0
    )


def main():
    for group in CASE_GROUPS:
        _run_group(*group)

    all_ok = True
    for group in CASE_GROUPS:
        all_ok &= _compare_group(*group)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
