#!/usr/bin/env python3
"""Compare prediction, enclosure, correction and honest history replay costs."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from bounds import Certificate, decide, replay, solve_checked
from models import Forcings, native_device, powers, smoke_device


SHIFTS = (0., .1, 1., 10., 100., 1000.)


def preconditioner(A, smoke=False):
    if smoke:
        diagonal = A.diagonal()
        return sla.LinearOperator(A.shape, matvec=lambda x: x / diagonal, dtype=float)
    import pyamg
    hierarchy = pyamg.ruge_stuben_solver(A.tocsr(), interpolation='direct',
                                        presmoother=('gauss_seidel', {'sweep': 'symmetric'}),
                                        postsmoother=('gauss_seidel', {'sweep': 'symmetric'}))
    return hierarchy.aspreconditioner(cycle='V')


def cg(A, rhs, M, guess=None, rtol=1e-11):
    count = [0]
    def callback(_):
        count[0] += 1
    x, info = sla.cg(A, rhs, M=M, x0=guess, rtol=rtol, atol=0.,
                     maxiter=2000, callback=callback)
    if info != 0:
        raise RuntimeError(f'AMG-CG failed: info={info}')
    return x, count[0]


class Predictor:
    """Fixed 12-vector Ritz predictor; no training on held-out time traces."""
    def __init__(self, device, A, smoke):
        start = time.perf_counter()
        snapshots = []
        self.training_iterations = 0
        for shift in SHIFTS:
            matrix = device.K + sp.diags(shift * device.C)
            M = preconditioner(matrix, smoke)
            for source in (0, 1):
                x, count = cg(matrix, device.B[:, source], M)
                self.training_iterations += count
                snapshots.append(x / np.linalg.norm(x))
        U, sigma, _ = la.svd(np.column_stack(snapshots), full_matrices=False,
                             check_finite=False)
        self.V = np.ascontiguousarray(U[:, sigma > 1e-10 * sigma[0]])
        self.K = device.K
        self.chol = la.cho_factor(self.V.T @ (A @ self.V), check_finite=False)
        self.seconds = time.perf_counter() - start
        self.order = self.V.shape[1]
        self.training_solves = len(snapshots)

    def predict(self, previous, forcing):
        residual = forcing - self.K @ previous
        update = la.cho_solve(self.chol, self.V.T @ residual, check_finite=False)
        return previous + self.V @ update


def full_trajectory(A, D, forcing, M, *, rtol, cert=None, store=False):
    x = np.zeros(len(D))
    b = x.copy()
    states, peaks, reference_radii = [], [], []
    elapsed, iterations = 0., 0
    for step in range(len(forcing)):
        tick = time.perf_counter()
        f = forcing[step]
        previous = x
        x, count = cg(A, f + D * previous, M, guess=previous, rtol=rtol)
        peak = float(x.max())
        elapsed += time.perf_counter() - tick
        iterations += count
        peaks.append(peak)
        if store:
            states.append(x.copy())
        if cert is not None:
            # Validation only: excluded from the un-certified FOM timing.
            b = cert.bound(x, previous, b, f)
            reference_radii.append(float(b.max()))
    return {'seconds': elapsed, 'iterations': iterations, 'states': states,
            'peaks': peaks, 'reference_radii': reference_radii}


def evaluate(method, cert, predictor, M, forcing, reference, *, width, threshold):
    """Reference values are read ONLY after algorithm timing/decisions."""
    n = len(cert.D)
    x, b = np.zeros(n), np.zeros(n)
    checkpoint_index = 0
    checkpoint_state, checkpoint_radius = x.copy(), b.copy()
    elapsed = 0.
    iterations = 0
    refined_steps, replay_events, replayed_steps = 0, 0, 0
    zero_correction_steps = 0
    rows = []
    max_field_violation = 0.
    for step in range(len(forcing)):
        tick = time.perf_counter()
        f = forcing[step]
        previous, prior = x, b
        guess = predictor.predict(previous, f)
        count, n_replayed = 0, 0
        if method == 'rom_unchecked':
            x = guess
            lo = hi = float(x.max())
            b = np.zeros(n)
        elif method == 'rom_enclosed':
            x = guess
            b = cert.bound(x, previous, prior, f)
            lo, hi = cert.peak(x, b)
        else:
            target_threshold = threshold if method == 'decision_adaptive' else None
            checked = solve_checked(cert, previous, prior, f, guess, M.matvec,
                                    width=width, threshold=target_threshold)
            x, b = checked.state, checked.radius
            lo, hi = checked.lower, checked.upper
            count = checked.iterations
            if not checked.accepted:
                # Solving only the current equation cannot erase history error.
                x, b, n_replayed, replay_count = replay(
                    cert, forcing, checkpoint_index, step + 1,
                    checkpoint_state, checkpoint_radius, M.matvec)
                count += replay_count
                lo, hi = cert.peak(x, b)
                replay_events += 1
                replayed_steps += n_replayed
                checkpoint_index = step + 1
                checkpoint_state, checkpoint_radius = x.copy(), b.copy()
            if count:
                refined_steps += 1
            else:
                zero_correction_steps += 1
        prediction = float(x.max())
        status = decide(lo, hi, threshold)
        elapsed += time.perf_counter() - tick
        iterations += count
        # Audit code below is deliberately outside the algorithm's timed region.
        ref = reference['peaks'][step]
        ref_radius = reference['reference_radii'][step]
        violation = max(0., lo - ref - ref_radius, ref - ref_radius - hi)
        field_violation = float(np.maximum(np.abs(x - reference['states'][step])
                                           - b - ref_radius, 0.).max())
        if method != 'rom_unchecked':
            max_field_violation = max(max_field_violation, field_violation)
            if max(violation, field_violation) > 1e-6:
                raise AssertionError(f'{method}: enclosure failed at step {step}: '
                                     f'peak={violation}, field={field_violation}')
        rows.append({
            'step': step + 1, 'lower_rise_K': lo, 'upper_rise_K': hi,
            'reference_rise_K': ref, 'prediction_rise_K': prediction,
            'reference_radius_K': ref_radius, 'width_K': hi - lo,
            'status': status, 'iterations': count, 'replayed_steps': n_replayed,
            'peak_error_K': abs(prediction - ref),
            'peak_enclosure_violation_K': violation,
            'false_safe': int(status == 'safe' and ref - ref_radius > threshold),
            'false_unsafe': int(status == 'unsafe' and ref + ref_radius < threshold),
        })
    widths = np.array([r['width_K'] for r in rows])
    return {
        'method': method, 'seconds': elapsed, 'iterations': iterations,
        'refined_steps': refined_steps, 'zero_correction_steps': zero_correction_steps,
        'replay_events': replay_events, 'replayed_steps': replayed_steps,
        'width_median_K': float(np.median(widths)),
        'width_p95_K': float(np.quantile(widths, .95)), 'width_max_K': float(widths.max()),
        'peak_error_max_K': max(r['peak_error_K'] for r in rows),
        'peak_enclosure_violations': sum(r['peak_enclosure_violation_K'] > 1e-6 for r in rows),
        'field_enclosure_violation_max_K': max_field_violation,
        'false_safe': sum(r['false_safe'] for r in rows),
        'false_unsafe': sum(r['false_unsafe'] for r in rows),
        'safe_steps': sum(r['status'] == 'safe' for r in rows),
        'unsafe_steps': sum(r['status'] == 'unsafe' for r in rows),
        'unknown_steps': sum(r['status'] == 'unknown' for r in rows),
        'trace': rows,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_device(output, nx, nz, args):
    tick = time.perf_counter()
    device = smoke_device() if args.smoke else native_device(nx, nz, args.bottleneck)
    assembly_seconds = time.perf_counter() - tick
    D = device.C / args.dt
    A = (device.K + sp.diags(D)).tocsr()
    tick = time.perf_counter()
    M = preconditioner(A, args.smoke)
    common_setup_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    # One positive Poisson barrier plus the free constant barrier.
    MK = preconditioner(device.K, args.smoke)
    weight, barrier_iterations = cg(device.K, device.C, MK)
    cert = Certificate(A, D, np.column_stack((np.ones(len(D)), weight)))
    barrier_seconds = time.perf_counter() - tick
    del MK
    predictor = Predictor(device, A, args.smoke)
    offline = predictor.seconds + barrier_seconds
    setup = {
        'assembly_seconds': assembly_seconds, 'common_AMG_setup_seconds': common_setup_seconds,
        'barrier_seconds': barrier_seconds, 'barrier_solves': 1,
        'barrier_iterations': barrier_iterations, 'predictor_seconds': predictor.seconds,
        'predictor_solves': predictor.training_solves, 'order': predictor.order,
        'predictor_iterations': predictor.training_iterations,
        'prediction_basis_bytes': predictor.V.nbytes,
        'certificate_array_bytes': cert.W.nbytes + cert.AW_lower.nbytes + len(D) * 8,
        'offline_extra_seconds': offline,
    }
    results = []
    for held_out in (False, True):
        label = ('smoke' if args.smoke else f'n{nx}_z{4*nz}') + ('_unseen_hotspot' if held_out else '_switching')
        seed = args.seed
        P = powers(args.steps, seed, held_out)
        forcing = Forcings(device.B, P)
        write_csv(output / f'{label}-power.csv', [
            {'step': i+1, 'time_s': (i+1)*args.dt, 'source0_W': p[0],
             'source1_W': p[1], 'held_out_W': p[2]} for i, p in enumerate(P)])
        reference = full_trajectory(A, D, forcing, M, rtol=1e-11, cert=cert, store=True)
        if max(reference['reference_radii']) > 1e-4:
            raise AssertionError('reference trajectory is not sufficiently resolved')
        repeat = full_trajectory(A, D, forcing, M, rtol=1e-11)
        fom_times = [reference['seconds'], repeat['seconds']]
        fom_seconds = float(np.median(fom_times))
        practical_times, practical = [], None
        for _ in range(args.repeats):
            practical = full_trajectory(A, D, forcing, M, rtol=1e-6)
            practical_times.append(practical['seconds'])
        practical_seconds = float(np.median(practical_times))
        table = []
        methods = ['rom_unchecked', 'rom_enclosed', 'width_adaptive', 'decision_adaptive']
        trials = {method: [] for method in methods}
        for trial in range(args.repeats):
            for method in (methods if trial % 2 == 0 else methods[::-1]):
                result = evaluate(method, cert, predictor, M, forcing, reference,
                                  width=(.1 if method == 'decision_adaptive' else 1.),
                                  threshold=args.threshold_K - 300.)
                trace = result.pop('trace')
                if trial == 0:
                    for row in trace:
                        row['time_s'] = row['step'] * args.dt
                    write_csv(output / f'{label}-{method}.csv', trace)
                trials[method].append(result)
        for method in methods:
            rows = trials[method]
            result = rows[0]
            times = [row['seconds'] for row in rows]
            result['seconds'] = float(np.median(times))
            result['timing_samples_seconds'] = times
            result['speedup_vs_tight_FOM'] = fom_seconds / result['seconds']
            result['speedup_vs_practical_FOM'] = practical_seconds / result['seconds']
            cost = predictor.seconds + (0 if method == 'rom_unchecked' else barrier_seconds)
            result['single_query_speedup_including_setup'] = (common_setup_seconds + fom_seconds) / (
                common_setup_seconds + cost + result['seconds'])
            gain = fom_seconds - result['seconds']
            result['break_even_queries_vs_tight_FOM'] = int(np.ceil(cost / gain)) if gain > 0 else None
            result['requested_width_K'] = .1 if method == 'decision_adaptive' else (1. if method == 'width_adaptive' else None)
            table.append(result)
        report = {
            'label': label, 'device': device.metadata, 'setup': setup, 'seed': seed,
            'dt_s': args.dt, 'steps': args.steps, 'threshold_K': args.threshold_K,
            'reference_FOM_seconds': fom_seconds, 'reference_timing_samples_seconds': fom_times,
            'reference_FOM_iterations': reference['iterations'],
            'reference_radius_max_K': max(reference['reference_radii']),
            'reference_peak_range_K': [300.+min(reference['peaks']), 300.+max(reference['peaks'])],
            'practical_FOM_seconds': practical_seconds,
            'practical_timing_samples_seconds': practical_times,
            'practical_FOM_iterations': practical['iterations'],
            'practical_peak_error_max_K': float(np.max(np.abs(np.array(practical['peaks']) - reference['peaks']))),
            'methods': table,
        }
        (output / f'{label}.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: report[k] for k in ('label', 'reference_FOM_seconds', 'practical_FOM_seconds', 'reference_peak_range_K')}) , flush=True)
        for row in table:
            print(json.dumps({k: row[k] for k in ('method', 'seconds', 'speedup_vs_practical_FOM', 'width_max_K', 'replay_events', 'replayed_steps', 'unknown_steps')}), flush=True)
        results.append(report)
        del reference
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--dt', type=float, default=.025)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--threshold-K', type=float, default=350.)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--nx', type=int, nargs='+', default=[24, 64])
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 4 or args.dt <= 0 or args.repeats < 1:
        parser.error('steps must be a positive multiple of four; dt/repeats positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for nx in ([0] if args.smoke else args.nx):
        reports.extend(run_device(args.output_dir, nx, 3 if nx <= 24 else 6, args))
    summary = {'python': platform.python_version(), 'numpy': np.__version__,
               'scipy': scipy.__version__, 'platform': platform.platform(),
               'peak_process_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.,
               'smoke_only': args.smoke,
               'guarantee_scope': 'stored linear FVM/backward-Euler system at sample times; guarded floating point, not formally verified arithmetic; excludes PDE, time-discretization, parameter and model error',
               'reports': reports}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    flat = []
    for report in reports:
        for method in report['methods']:
            flat.append({'case': report['label'], 'dof': report['device']['dof'],
                         **{k: v for k, v in method.items() if k != 'timing_samples_seconds'}})
    write_csv(args.output_dir / 'summary.csv', flat)


if __name__ == '__main__':
    main()
