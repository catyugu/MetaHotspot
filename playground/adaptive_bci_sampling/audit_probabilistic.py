"""Independent signed-input / time-limit audits, not a unit-test suite.

Small FOM: dense exact-arithmetic identities evaluated with ordinary floating
point, sampled time axis. Large FOM: optional BE references with analytic
time-discretization and actual CG-defect bounds. No continuum acceptance.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la

from case1_system import case1_reconstruction
from numerics import Solver, operator, sym, relative, f, counts


def run(a):
    start = time.perf_counter()
    data = np.load(a.candidate)
    metadata = json.loads(a.candidate.with_suffix('.json').read_text())
    K, C, _, H, ranges, meta = case1_reconstruction(a.mesh_mm)
    c, sc = C.diagonal(), np.sqrt(C.diagonal())
    G = data['G']
    bases = {name: data[key] for name, key in [('pool_pre', 'V'), ('stock_pre', 'stock_pre'), ('stock_post', 'stock_post')]}
    F0 = G / c[:, None]
    initial = {name: relative(F0-V@la.solve(sym(V.T@(C@V)), V.T@G, assume_a='pos'), F0, C)
               for name, V in bases.items()}
    hp = np.exp(np.log(ranges[:, 0])+np.asarray(list(itertools.product(np.linspace(0, 1, a.grid), repeat=len(ranges))))*
                np.log(ranges[:, 1]/ranges[:, 0]))
    alpha = metadata['witness']['alpha']
    rows, cost = [], []
    for h in hp:
        A = operator(K, H, h)
        solver = Solver(A)
        Xsteady = solver.solve(G)
        cost.append(solver.counts())
        reduced = {}
        steady = {}
        for name, V in bases.items():
            l, U = la.eigh(sym(V.T@(A@V)), sym(V.T@(C@V)))
            B = U.T@(V.T@G)
            reduced[name] = (V@U, l, B)
            steady[name] = relative(Xsteady-(V@U)@(B/l[:, None]), Xsteady, C)
        row = dict(h=h.tolist(), steady_C_relative=steady)
        if len(c) <= 2000:
            l, U = la.eigh(A.toarray()/sc[:, None]/sc[None, :])
            B = U.T@(G/sc[:, None])
            times = np.geomspace(1e-6/l[-1], 20/l[0], a.times)
            maxima = {name: initial[name] for name in bases}
            maximum_time = {name: 0. for name in bases}
            for t in times:
                X = (U@(f(t, l)[:, None]*B))/sc[:, None]
                for name, (VU, lr, Br) in reduced.items():
                    error = relative(X-VU@(f(t, lr)[:, None]*Br), X, C)
                    if error > maxima[name]:
                        maxima[name], maximum_time[name] = error, float(t)
            row.update(sampled_time_maxima=maxima, sampled_time_argmax=maximum_time,
                       time_interval=[float(times[0]), float(times[-1])], time_count=len(times))
        elif a.steps:
            samples = []
            for t in [.01, 1.]:
                dt = t/a.steps
                solver = Solver(C+dt*A)
                X = np.zeros_like(G)
                cg = 0.
                rho = 1/(1+dt*alpha)
                for _ in range(a.steps):
                    rhs = c[:, None]*X+dt*G
                    X = solver.solve(rhs)
                    cg = rho*(cg+la.norm((rhs-solver.A@X)/sc[:, None], 2))
                be = t/(2*a.steps)*((a.steps-1)/a.steps)**(a.steps-1)*la.norm(G/sc[:, None], 2)
                den = la.svdvals(sc[:, None]*X)[-1]
                uncertainty = (be+cg)/den
                sample = dict(t=t, BE_CG_relative_uncertainty=uncertainty, approximations={})
                for name, (VU, lr, Br) in reduced.items():
                    error = relative(X-VU@(f(t, lr)[:, None]*Br), X, C)
                    upper = (error+uncertainty)/(1-uncertainty) if uncertainty < 1 else None
                    lower = max(0., error-uncertainty)/(1+uncertainty)
                    sample['approximations'][name] = dict(BE_relative=error, true_relative_upper=upper,
                        true_relative_lower=lower, certified_point_pass=bool(upper is not None and upper <= np.sqrt(.001)))
                samples.append(sample)
                cost.append(solver.counts())
            row['BE_samples'] = samples
        rows.append(row)
    out = dict(n=len(c), metadata=meta, initial_step_limit=initial, rows=rows, costs=counts(cost),
               seconds=time.perf_counter()-start, tolerance=np.sqrt(.001),
               scope='arbitrary signed input combinations; fixed HTC points, sampled times / analytic limits; NOT all-time or continuous-HTC certification')
    a.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(dict(n=len(c), initial=initial,
        steady_C_max={name:max(r['steady_C_relative'][name] for r in rows) for name in bases},
        sampled_step_max={name:max(r['sampled_time_maxima'][name] for r in rows) for name in bases} if len(c)<=2000 else None,
        seconds=out['seconds'])), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mesh-mm', type=float, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--grid', type=int, default=3)
    p.add_argument('--times', type=int, default=60)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
