"""Standalone checks for the inexact-moment dynamic bridge."""

import math
import unittest
import numpy as np


def orth(a):
    q, r = np.linalg.qr(a, mode="reduced")
    d = np.abs(np.diag(r))
    rank = int(np.count_nonzero(d > 1e-12 * max(float(d.max(initial=0.0)), 1.0)))
    return q[:, :rank]


def h2_sq(b, c):
    lam, u = np.linalg.eigh(b)
    z = u.T @ c
    g = z @ z.T
    return float(np.sum(g * g / (lam[:, None] + lam[None, :])))


def h2_cross(b1, c1, b2, c2):
    l1, u1 = np.linalg.eigh(b1)
    l2, u2 = np.linalg.eigh(b2)
    g = (u1.T @ c1) @ (u2.T @ c2).T
    return float(np.sum(g * g / (l1[:, None] + l2[None, :])))


def h2_diff(b1, c1, b2, c2):
    q = h2_sq(b1, c1) + h2_sq(b2, c2) - 2 * h2_cross(b1, c1, b2, c2)
    return math.sqrt(max(q, 0.0))


def make_case(seed=1, angle=5e-4):
    rng = np.random.default_rng(seed)
    n, ports = 18, 2
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    b = u @ np.diag(np.geomspace(0.8, 18.0, n)) @ u.T
    c = rng.standard_normal((n, ports))
    shifts = np.array([0.4, 1.5, 6.0])

    exact = np.column_stack([
        np.linalg.solve(b + s*np.eye(n), c[:, j])
        for s in shifts for j in range(ports)
    ])
    q = orth(exact)
    z = rng.standard_normal((n, q.shape[1] + 3))
    z -= q @ (q.T @ z)
    qp = orth(z)
    v = orth(np.column_stack([
        math.cos(angle)*q + math.sin(angle)*qp[:, :q.shape[1]],
        qp[:, q.shape[1]:q.shape[1]+2],
    ]))

    xs, rs, ys = [], [], []
    for s in shifts:
        a = b + s*np.eye(n)
        ar = v.T @ a @ v
        for j in range(ports):
            y = np.linalg.solve(ar, v.T @ c[:, j])
            x = v @ y
            xs.append(x)
            rs.append(c[:, j] - a @ x)
            ys.append(y)
    x = np.column_stack(xs)
    r = np.column_stack(rs)
    y = np.column_stack(ys)
    ab = r @ np.linalg.pinv(x)
    f = ab + ab.T
    return b, c, shifts, v, x, r, y, f


class TheoryTests(unittest.TestCase):
    def test_node_identity_and_derivative_bound(self):
        b, c, shifts, v, *_ = make_case()
        rng = np.random.default_rng(9)
        for s in shifts:
            a = b + s*np.eye(len(b))
            x = np.linalg.solve(a, c)
            xv = v @ np.linalg.solve(v.T @ a @ v, v.T @ c)
            r = c - a @ xv
            e = x - xv
            d = c.T @ x - c.T @ xv
            np.testing.assert_allclose(d, r.T @ np.linalg.solve(a, r), rtol=2e-8, atol=1e-12)
            np.testing.assert_allclose(d, e.T @ a @ e, rtol=2e-8, atol=1e-12)
            for _ in range(4):
                p = rng.standard_normal(c.shape[1])
                xp, vp, ep = x@p, xv@p, e@p
                den = float(xp.T @ a @ xp)
                eta = math.sqrt(max(float(ep.T @ a @ ep) / den, 0.0))
                lhs = abs(float(xp.T@xp - vp.T@vp))
                self.assertLessEqual(lhs, (2*eta + eta*eta)*den/s * (1 + 1e-12))

    def test_minimum_symmetric_backward_error(self):
        b, c, shifts, v, x, r, y, f = make_case()
        np.testing.assert_allclose(x.T @ r, 0.0, atol=4e-12)
        np.testing.assert_allclose(f @ x, r, rtol=2e-9, atol=2e-11)
        np.testing.assert_allclose(v.T @ f @ v, 0.0, atol=4e-11)
        one = r @ np.linalg.pinv(x)
        self.assertAlmostEqual(np.linalg.norm(f, 2), np.linalg.norm(one, 2), delta=3e-10)
        _, sr = np.linalg.qr(r, mode="reduced")
        self.assertAlmostEqual(np.linalg.norm(f, 2), np.linalg.norm(sr @ np.linalg.pinv(y), 2), delta=3e-10)

        bt = b + f
        br = v.T @ b @ v
        for s in shifts:
            near = np.linalg.solve(bt + s*np.eye(len(b)), c)
            rom = v @ np.linalg.solve(br + s*np.eye(v.shape[1]), v.T @ c)
            np.testing.assert_allclose(near, rom, rtol=2e-9, atol=2e-11)
            np.testing.assert_allclose(near.T@near, rom.T@rom, rtol=2e-9, atol=2e-11)

    def test_h2_perturbation_bound_stress(self):
        worst = 0.0
        for seed in range(16):
            b, c, _, v, _, _, _, f = make_case(100+seed)
            bt = b + f
            delta = float(np.linalg.norm(f, 2))
            lam = float(np.linalg.eigvalsh(b)[0])
            self.assertLess(delta, lam)
            bound = (np.linalg.norm(c, 2)*np.linalg.norm(c, "fro")*delta /
                     math.sqrt(2*lam*(lam-delta)*(2*lam-delta)))
            actual = h2_diff(b, c, bt, c)
            self.assertLessEqual(actual, bound*(1+1e-10))
            worst = max(worst, actual/bound)
            hv = math.sqrt(h2_sq(v.T@b@v, v.T@c))
            hb = math.sqrt(h2_sq(b, c))
            self.assertLessEqual(hv, hb*(1+1e-12))
        self.assertLess(worst, 1.0)

    def test_small_scalar_residuals_can_hide_large_block_error(self):
        tau = 1e-6
        vals = []
        for k in (1e-1, 1e-2, 1e-3):
            x = np.array([[1.,1.],[0.,k],[0.,0.],[0.,0.]])
            r = np.array([[0.,0.],[0.,0.],[tau,-tau],[0.,0.]])
            self.assertLessEqual(np.linalg.norm(r, axis=0).max(), tau)
            vals.append(np.linalg.norm(r @ np.linalg.pinv(x), 2))
        self.assertGreater(vals[1], 9*vals[0])
        self.assertGreater(vals[2], 9*vals[1])

    def test_nested_galerkin_h2_error_is_not_monotone(self):
        b = np.diag([0.5, 2., 10., 50.])
        c = np.array([[1.], [.8], [.4], [.2]])
        q = np.array([
            [-.90613417, .04103754, .35412967, -.22765974],
            [.00748377, -.96430087, .22278921, .14294329],
            [-.41602417, -.06016822, -.65159468, .63144917],
            [.07608379, .25459679, .63275708, .72733088],
        ])
        q, _ = np.linalg.qr(q)

        def err(v):
            br, cr = v.T@b@v, v.T@c
            return math.sqrt(max(h2_sq(b,c) + h2_sq(br,cr) - 2*h2_cross(b,c,br,cr), 0.0))
        self.assertGreater(err(q[:,:2]), err(q[:,:1]) + 1e-4)


if __name__ == "__main__":
    unittest.main()
