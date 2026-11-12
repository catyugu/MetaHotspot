"""Resume the paired three-seed experiment, keeping large artifacts in results/."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def command(script, args, log):
    with log.open("w") as stream:
        subprocess.run(
            [sys.executable, "-u", str(HERE / script), *args],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=HERE / "results")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    runs = []
    for seed in (20260805, 20260806, 20260807):
        for method in ("baseline", "global"):
            folder = a.out / f"{method}_{seed}"
            runs.append(str(folder))
            if not (folder / "training.json").exists():
                print("extracting", folder.name, flush=True)
                command(
                    "train.py",
                    ["--method", method, "--seed", str(seed), "--out", str(folder)],
                    a.out / f"{folder.name}.log",
                )
    print("evaluating paired runs", flush=True)
    command("evaluate.py", [*runs, "--out", str(a.out)], a.out / "evaluation_all.log")
    first = [str(a.out / f"{m}_20260805") for m in ("baseline", "global")]
    print("compression ablations for first seed", flush=True)
    command(
        "evaluate.py",
        [*first, "--out", str(a.out), "--ablate"],
        a.out / "evaluation_ablation.log",
    )
    print("DONE", flush=True)
