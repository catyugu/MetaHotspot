#!/usr/bin/env python3
"""Controlled follow-up after CI 35303895189 exposed history overestimation.

Keep the model, loads, thresholds, solver settings and predictor fixed. Compare
shape-preserving approximate-inverse supersolutions against the original bound,
and include a no-ROM warm-start ablation and a looser fixed-tolerance FOM.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from bounds import Certificate
from models import Forcings, native_device, powers, smoke_device
from run import Predictor, cg, evaluate, full_trajectory, preconditioner, write_csv
from transport import TransportCertificate


class PreviousStatePredictor:
    """No ROM, no extraction cost: propose the preceding full-length state."""
    def predict(self, previous, forcing):
        del forcing
        return previous.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 4 or args.steps % 4 or args.repeats < 1:
        parser.error('steps must be a positive multiple of four; repeats positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = smoke_device() if args.smoke else native_device(64, 6, False)
    D = device.C / .025
    A = (device.K + sp.diags(D)).tocsr()
    tick = time.perf_counter()
    M = preconditioner(A, args.smoke)
    common_setup = time.perf_counter() - tick
    tick = time.perf_counter()
    MK = preconditioner(device.K, args.smoke)
    w, _ = cg(device.K, device.C, MK)
    weights = np.column_stack((np.ones(len(D)), w))
    original = Certificate(A, D, weights)
    barrier_seconds = time.perf_counter() - tick
    del MK
    predictor = Predictor(device, A, args.smoke)
    warm = PreviousStatePredictor()
    tick = time.perf_counter()
    transport1 = TransportCertificate(A, D, weights, M.matvec, cycles=1)
    transport1_setup = time.perf_counter() - tick
    tick = time.perf_counter()
    transport2 = TransportCertificate(A, D, weights, M.matvec, cycles=2)
    transport2_setup = time.perf_counter() - tick
    cases = []
    configurations = [
        ('original_width', 'width_adaptive', original, predictor, 1),
        ('original_decision', 'decision_adaptive', original, predictor, .1),
        ('no_rom_width', 'width_adaptive', original, warm, 1),
        ('transport1_width', 'width_adaptive', transport1, predictor, 1),
        ('transport1_decision', 'decision_adaptive', transport1, predictor, .1),
        ('transport2_width', 'width_adaptive', transport2, predictor, 1),
        ('transport2_decision', 'decision_adaptive', transport2, predictor, .1),
    ]
    for unseen in (False, True):
        label = 'unseen_hotspot' if unseen else 'switching'
        forcing = Forcings(device.B, powers(args.steps, 20260918, unseen))
        reference = full_trajectory(A, D, forcing, M, rtol=1e-11, cert=original, store=True)
        if max(reference['reference_radii']) > 1e-4:
            raise AssertionError('reference uncertainty too large')
        baselines = []
        for tolerance in (1e-6, 1e-3):
            samples, peak_errors, work = [], [], []
            for _ in range(args.repeats):
                result = full_trajectory(A, D, forcing, M, rtol=tolerance)
                samples.append(result['seconds'])
                peak_errors.append(float(np.max(np.abs(np.array(result['peaks']) - reference['peaks']))))
                work.append(result['iterations'])
            baselines.append({'rtol': tolerance, 'seconds': float(np.median(samples)),
                              'timing_samples_seconds': samples, 'iterations': work[0],
                              'peak_error_max_K': max(peak_errors)})
        measurements = {entry[0]: [] for entry in configurations}
        for trial in range(args.repeats):
            ordered = configurations if trial % 2 == 0 else configurations[::-1]
            for name, policy, cert, predict, width in ordered:
                if hasattr(cert, 'inverse_calls'):
                    cert.inverse_calls = 0
                result = evaluate(policy, cert, predict, M, forcing, reference,
                                  width=width, threshold=50.)
                trace = result.pop('trace')
                if trial == 0:
                    for row in trace:
                        row['time_s'] = .025 * row['step']
                    write_csv(args.output_dir / f'{label}-{name}.csv', trace)
                result['method'] = name
                result['certificate_AMG_cycles'] = getattr(cert, 'inverse_calls', 0)
                measurements[name].append(result)
        table = []
        for name, _, chosen_cert, predict, _ in configurations:
            trials = measurements[name]
            result = trials[0]
            result['timing_samples_seconds'] = [r['seconds'] for r in trials]
            result['seconds'] = float(np.median(result['timing_samples_seconds']))
            extra_cert = transport1_setup if chosen_cert is transport1 else (transport2_setup if chosen_cert is transport2 else 0.)
            result['extra_setup_seconds'] = barrier_seconds + extra_cert + (predictor.seconds if predict is predictor else 0.)
            result['speedup_vs_FOM_1e6'] = baselines[0]['seconds'] / result['seconds']
            result['speedup_vs_FOM_1e3'] = baselines[1]['seconds'] / result['seconds']
            result['cold_speedup_vs_FOM_1e6'] = (common_setup + baselines[0]['seconds']) / (
                common_setup + result['extra_setup_seconds'] + result['seconds'])
            result['cold_speedup_vs_FOM_1e3'] = (common_setup + baselines[1]['seconds']) / (
                common_setup + result['extra_setup_seconds'] + result['seconds'])
            table.append(result)
        record = {'label': label, 'baselines': baselines, 'methods': table,
                  'reference_radius_max_K': max(reference['reference_radii'])}
        cases.append(record)
        (args.output_dir / f'{label}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)
    summary = {'predecessor_run': 35303895189, 'smoke_only': args.smoke,
               'device': device.metadata, 'steps': args.steps, 'dt_s': .025,
               'threshold_K': 350., 'common_AMG_setup_seconds': common_setup,
               'barrier_setup_seconds': barrier_seconds,
               'predictor_setup_seconds': predictor.seconds, 'predictor_order': predictor.order,
               'cases': cases}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    write_csv(args.output_dir / 'summary.csv', [
        {'case': case['label'], **{k: v for k, v in row.items() if k != 'timing_samples_seconds'}}
        for case in cases for row in case['methods']])


if __name__ == '__main__':
    main()
