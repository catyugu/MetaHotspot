#!/usr/bin/env python3
"""Extract and evaluate a paper-style BCI ROM for simple_case1.

This is an isolated experiment based on Codecasa et al., THERMINIC 2015,
Algorithm 1: random admissible HTC samples, elliptic frequency shifts,
residual-driven enrichment, and final SVD reduction. It uses the existing
model-agnostic implementation only as an experimental backend and does not
modify production macromodel code.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "playground" / "simple_erom_case1"
sys.path[:0] = [str(ROOT / "python"), str(CASE)]

from experiment_simple_case1_integrated import (  # noqa: E402
    DT,
    DURATION,
    AMBIENT_K,
    build_lower,
    read_links,
    read_matrix,
    read_diag,
)
from metahotspot.macromodel.utils import (  # noqa: E402
    build_parametric_basis,
    normalized_operators,
)

DATA = CASE / "simple_case1_EROM"
OUT = CASE / "results" / "simple_erom_paper2015_extraction.json"
HTC_VALUES = (1.0, 10.0, 100.0, 1000.0, 10000.0)
TARGET_ORDER = 14
EXTRACTION_MAX_ORDER = 64
TOLERANCE = 1.0e-3
SEED = 20260825


def solve_transient(K, C, rhs):
    steady = spla.spsolve(K.tocsc(), rhs)
    times = np.arange(0.0, DURATION + 0.5 * DT, DT)
    factor = spla.factorized((K + C / DT).tocsc())
    state = np.zeros(K.shape[0])
    history = [state.copy()]
    for _ in times[1:]:
        state = factor(C @ state / DT + rhs)
        history.append(state.copy())
    return steady, np.asarray(history), times


def detailed_at_h(ops, compiled, top, top_area, h):
    material = compiled.eval_materials()
    k = np.asarray(material["conductivity_z"])[top]
    half = compiled.cells.cell_sizes[top, 2] / 2.0
    # The detailed experiment uses the physical surface coefficient directly.
    H = sp.csc_matrix((h * top_area, (top, top)), shape=ops.K.shape)
    source = ops.f.reshape(-1)
    source = source / source.sum()
    ss, hist, times = solve_transient(ops.K + H, ops.C, source)
    return ss, hist, times, source


def paper_rom_at_h(basis, ops, source, top, top_area, h):
    C_hat = sp.csc_matrix(basis.T @ ops.C @ basis)
    K_hat = sp.csc_matrix(basis.T @ ops.K @ basis)
    F_hat = np.asarray(basis.T @ source.reshape(-1, 1), dtype=np.float64)
    H = sp.csc_matrix((top_area, (top, top)), shape=ops.K.shape)
    H_hat = basis.T @ H @ basis
    K_h = K_hat + h * sp.csc_matrix(H_hat)
    rhs = F_hat[:, 0]
    ss_q = spla.spsolve(K_h.tocsc(), rhs)
    times = np.arange(0.0, DURATION + 0.5 * DT, DT)
    lhs = (K_h + C_hat / DT).tocsc()
    factor = spla.factorized(lhs)
    q = np.zeros(basis.shape[1])
    hist_q = [q.copy()]
    for _ in times[1:]:
        q = factor(C_hat @ q / DT + rhs)
        hist_q.append(q.copy())
    hist_q = np.asarray(hist_q)
    return ss_q, hist_q, times, basis


def summarize(ops, compiled, top, top_area, basis, source):
    results = []
    for h in HTC_VALUES:
        dss, dh, times, source = detailed_at_h(ops, compiled, top, top_area, h)
        rss, rh, times_r, basis = paper_rom_at_h(basis, ops, source, top, top_area, h)
        assert np.array_equal(times, times_r)
        detail_j = source / source.sum()
        d_j_ss = float(detail_j @ dss)
        r_j_ss = float(detail_j @ (basis @ rss))
        d_j_hist = dh @ detail_j
        r_j_hist = (rh @ basis.T) @ detail_j
        d_trace = dss[top]
        r_trace = (basis @ rss)[top]
        d_trace_hist = dh[:, top]
        r_trace_hist = (rh @ basis.T)[:, top]
        results.append(
            {
                "h_W_m2K": h,
                "source_junction_steady_error_K": abs(r_j_ss - d_j_ss),
                "source_junction_transient_max_error_K": float(
                    np.max(np.abs(r_j_hist - d_j_hist))
                ),
                "interface_trace_steady_max_error_K": float(
                    np.max(np.abs(r_trace - d_trace))
                ),
                "interface_trace_transient_max_error_K": float(
                    np.max(np.abs(r_trace_hist - d_trace_hist))
                ),
                "detail_source_steady_rise_K": d_j_ss,
                "paper_rom_source_steady_rise_K": r_j_ss,
                "detail_source_final_rise_K": float(d_j_hist[-1]),
                "paper_rom_source_final_rise_K": float(r_j_hist[-1]),
                "detail_trace_max_K": float(np.max(d_trace)),
                "paper_rom_trace_max_K": float(np.max(r_trace)),
            }
        )
    return results


def main():
    started = time.perf_counter()
    links = read_links()
    x = np.unique(np.r_[links[:, 5], links[:, 6]])
    y = np.unique(np.r_[links[:, 7], links[:, 8]])
    target_v = read_matrix("Vb")
    target_k = read_diag("K_bci_hat")
    ops, compiled = build_lower(x, y)
    source = ops.f.reshape(-1)
    source = source / source.sum()
    top = np.flatnonzero(compiled.cells.ijk[:, 2] == compiled.cells.nz - 1)
    top = top[
        np.lexsort((compiled.cells.centers[top, 1], compiled.cells.centers[top, 0]))
    ]
    top_area = compiled.cells.cell_sizes[top, 0] * compiled.cells.cell_sizes[top, 1]
    H_top = sp.csc_matrix((top_area, (top, top)), shape=ops.K.shape)
    operators = normalized_operators(ops.K, ops.C, np.zeros((ops.K.shape[0], 1)))

    extract_started = time.perf_counter()
    basis, summary = build_parametric_basis(
        operators,
        source.reshape(-1, 1),
        [H_top],
        [(1.0, 1.0e4)],
        tolerance=TOLERANCE,
        max_order=EXTRACTION_MAX_ORDER,
        probe_rounds=2,
        seed=SEED,
    )
    extract_seconds = time.perf_counter() - extract_started
    modal_k, modal_q = scipy.linalg.eigh(
        basis.T @ ops.K.toarray() @ basis,
        basis.T @ ops.C.toarray() @ basis,
        check_finite=False,
    )
    basis_modal = basis @ modal_q
    # Compare only if the experiment produced the requested order; this is a
    # diagnostic against the private export, not a claim of identical extraction.
    trace_rel = (
        float(np.linalg.norm(basis_modal[top] - target_v) / np.linalg.norm(target_v))
        if basis_modal.shape[1] == target_k.size
        else None
    )
    spectral = {
        "paper_rom_modal_eigenvalues_descending": np.sort(modal_k)[::-1].tolist(),
        "exported_erom_eigenvalues": target_k.tolist(),
        "order_match": int(basis_modal.shape[1]) == int(target_k.size),
        "trace_relative_error_to_exported_Vb": trace_rel,
    }
    errors = summarize(ops, compiled, top, top_area, basis_modal, source)
    result = {
        "case": "simple_case1",
        "algorithm": "Codecasa THERMINIC 2015 Algorithm 1 (experimental model-agnostic realization)",
        "target_export": str(DATA),
        "detailed_cells": int(ops.K.shape[0]),
        "source_power_W": float(source.sum()),
        "target_order": int(target_k.size),
        "extracted_order": int(basis_modal.shape[1]),
        "tolerance": TOLERANCE,
        "seed": SEED,
        "extraction_seconds": extract_seconds,
        "total_seconds": time.perf_counter() - started,
        "basis_summary": summary,
        "spectral_comparison": spectral,
        "errors": errors,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
