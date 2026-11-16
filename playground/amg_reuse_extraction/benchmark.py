#!/usr/bin/env python3
"""Instrument the released extractor without changing its response-space logic."""
from __future__ import annotations
import argparse
import inspect
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import pyamg
import scipy
import scipy.sparse.linalg as spla
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'playground/bci_rom_testcase1'))
from model_case1 import Case1Model, Case1Config
from metahotspot.macromodel import utils as u
from extractor import build_frequency_major_basis


def run(method, seed, output, cell_mm=0.5):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = Case1Model(Case1Config(max_cell_mm=cell_mm))
    core, G, terms, ranges = model.core, model.source_shape, model.boundary_terms, model.h_ranges()
    assembly_s = time.perf_counter() - started
    Kmin = core.K.copy()
    for H, (lo, _) in zip(terms, ranges, strict=True):
        Kmin = Kmin + lo * H
    originals = {name: getattr(u, name) for name in ('port_eigenvalue_bounds', 'spd_solve', '_snapshot_svd_basis')}
    cache_key, cache_M = None, None
    events, bounds = [], []
    max_cached_hierarchies = 0
    times = dict(spectrum=0., setup=0., cg=0., svd=0., projection=0.)

    def shared_bounds(Klo, Khi, C, g):
        if not bounds:
            t = time.perf_counter()
            # Positive full-support starting vector for the global smallest mode.
            bounds.extend(originals['port_eigenvalue_bounds'](Klo, Khi, C, np.sqrt(C.diagonal())))
            times['spectrum'] += time.perf_counter() - t
            print('GLOBAL_BOUNDS', bounds, flush=True)
        return tuple(bounds)

    def solve(A, b, x0=None, rtol=u.ENRICH_RTOL):
        nonlocal cache_key, cache_M, max_cached_hierarchies
        frame = inspect.currentframe().f_back
        shift, port = float(frame.f_locals['shift']), int(frame.f_locals['port'])
        h = list(frame.f_locals['h_vec'])
        del frame
        key = (port, shift) if method == 'port_cache' else shift
        setup_s = 0.
        if method == 'fresh' or key != cache_key:
            # Release the completed frequency before constructing its successor.
            cache_M = None
            t = time.perf_counter()
            reference = A.tocsr() if method == 'fresh' else (Kmin + shift * core.C).tocsr()
            M = u._rs_preconditioner(reference)
            setup_s = time.perf_counter() - t
            times['setup'] += setup_s
            if method != 'fresh':
                cache_key, cache_M = key, M
                max_cached_hierarchies = max(max_cached_hierarchies, 1)
        else:
            M = cache_M
        iterations = 0
        def callback(x):
            nonlocal iterations
            iterations += 1
        t = time.perf_counter()
        x, info = spla.cg(A, np.asarray(b).ravel(), x0=x0, rtol=rtol, atol=0., maxiter=2000, M=M, callback=callback)
        cg_s = time.perf_counter() - t
        times['cg'] += cg_s
        residual = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
        event = dict(port=port, shift=shift, h=h, setup_s=setup_s, cg_s=cg_s, iterations=iterations, info=int(info), relative_residual=residual)
        events.append(event)
        if info or residual > rtol * 1.02:
            raise RuntimeError(f'CG failed: {event}')
        if len(events) % 20 == 0:
            print('RESPONSES', len(events), 'CG', sum(e['iterations'] for e in events), 'elapsed', time.perf_counter()-extract_started, flush=True)
        return x

    def svd(*args):
        t = time.perf_counter()
        value = originals['_snapshot_svd_basis'](*args)
        times['svd'] += time.perf_counter() - t
        return value

    u.port_eigenvalue_bounds, u.spd_solve, u._snapshot_svd_basis = shared_bounds, solve, svd
    extract_started = time.perf_counter()
    try:
        V, summary = build_frequency_major_basis(core, G, terms, ranges, tolerance=1e-3, max_order=1024, probe_rounds=10, seed=seed)
        t = time.perf_counter()
        reduced = u.project_bci(core, G, terms, V)
        times['projection'] = time.perf_counter() - t
        total = time.perf_counter() - extract_started
    finally:
        for name, original in originals.items():
            setattr(u, name, original)
    times['probes_basis_other'] = total - sum(times.values())
    tag = f'{method}_{seed}'
    # Kept outside version control and omitted from the deliverable ZIP.
    np.save(output / f'{tag}_basis.npy', V)
    result = dict(method=method, seed=seed, cell_mm=cell_mm, cells=V.shape[0], total_s=total,
                  assembly_s=assembly_s, times=times, loop_order="frequency_major", max_cached_hierarchies=max_cached_hierarchies, setup_count=sum(e['setup_s'] > 0 for e in events),
                  response_count=len(events), cg_iterations=sum(e['iterations'] for e in events),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  max_linear_residual=max(e['relative_residual'] for e in events), summary=summary,
                  events=events, environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, pyamg=pyamg.__version__,
                  platform=platform.platform(), threads=threadpool_info(), git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  cpu=next(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))))
    (output / f'{tag}.json').write_text(json.dumps(result, indent=2))
    print('COMPLETED', tag, json.dumps({k:result[k] for k in ('total_s','times','setup_count','response_count','cg_iterations')}), 'order',V.shape[1], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fresh','port_cache','shift_cache'], required=True)
    p.add_argument('--seed', type=int, default=20260805)
    p.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    p.add_argument('--cell-mm', type=float, default=.5)
    args=p.parse_args()
    with threadpool_limits(limits=1):
        run(args.method,args.seed,args.output,args.cell_mm)
