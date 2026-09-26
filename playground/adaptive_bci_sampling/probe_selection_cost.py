"""Where does selection spend its time at 1 mm?"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "bci_rom_testcase1")]

import deterministic_design as dd  # noqa: E402
import residual_certificate as rc  # noqa: E402
from model_case1 import Case1Config, Case1Model  # noqa: E402

mesh = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
maximum_points = int(sys.argv[2]) if len(sys.argv) > 2 else 3
grid = int(sys.argv[3]) if len(sys.argv) > 3 else 41

model = Case1Model(Case1Config(max_xy_cell_mm=mesh, max_z_cell_mm=mesh))
kernel = model.core.K.tocsc()
source = np.asarray(model.source_shape, dtype=np.float64)
terms = [term.tocsc() for term in model.boundary_terms]
ranges = np.asarray(model.h_ranges(), dtype=np.float64)
power = np.asarray(model.nominal_power(), dtype=np.float64)
print(f"mesh={mesh} n={kernel.shape[0]} grid={grid} points<={maximum_points}", flush=True)

timers = {}
calls = {}


def wrap(module, name, label):
    original = getattr(module, name)

    def timed(*args, **kwargs):
        clock = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timers[label] = timers.get(label, 0.0) + time.perf_counter() - clock
            calls[label] = calls.get(label, 0) + 1

    setattr(module, name, timed)


wrap(dd, "zolotarev_seed", "zolotarev_seed")
wrap(dd, "coordinate_spectral_enclosures", "spectral_enclosures")
wrap(dd, "prepare_residual_certificate", "prepare_certificate")
wrap(dd, "orthonormalize_block", "orthonormalize")
wrap(rc, "prepare_residual_certificate", "prepare_certificate_rc")

original_evaluate = rc.ResidualCertificate.evaluate_entrywise


def timed_evaluate(self, parameter):
    clock = time.perf_counter()
    try:
        return original_evaluate(self, parameter)
    finally:
        timers["evaluate_entrywise"] = (
            timers.get("evaluate_entrywise", 0.0) + time.perf_counter() - clock
        )
        calls["evaluate_entrywise"] = calls.get("evaluate_entrywise", 0) + 1


rc.ResidualCertificate.evaluate_entrywise = timed_evaluate

clock = time.perf_counter()
points, certificate, selection = dd.certified_greedy_points(
    kernel, terms, source, ranges, None, tolerance=1e-5,
    maximum_points=maximum_points, grid=grid, power=power, metric="entrywise",
)
total = time.perf_counter() - clock
print(f"selection total {total:.1f}s   certificate {certificate:.3e}", flush=True)
for label in sorted(timers, key=lambda name: -timers[name]):
    print(f"  {label:26s} {timers[label]:8.2f}s  x{calls[label]}", flush=True)
accounted = timers.get("zolotarev_seed", 0.0) + timers.get("prepare_certificate", 0.0)
print(f"  {'candidate loop (residual)':26s} {total - accounted:8.2f}s", flush=True)
print("points", points.tolist(), flush=True)
