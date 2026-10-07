import numpy as np
import scipy.linalg as la
from steady_step_audit import step_all_time_bound, steady_cell_bound
from rational_dynamic import AffineSystem


def test_step_certificate_covers_initial_and_steady_limits_and_interior():
    K, C = np.diag([.1, 3., 100.]), np.eye(3)
    G = np.array([[1., 0.], [0., 1.], [.01, -.02]])
    V = np.eye(3)[:, :2]
    cert = step_all_time_bound(K, C, G, V, .1)
    assert cert['accepted_analytic']
    for t in np.geomspace(1e-12, 1e5, 100):
        X = (-np.expm1(-np.diag(K)*t)/np.diag(K))[:, None]*G
        Y = X.copy(); Y[2] = 0
        E = X-Y
        assert np.sqrt(la.eigvalsh(E.T@E, X.T@X)[-1]) <= cert['bound']*(1+1e-10)


def test_step_certificate_rejects_missing_input_direction():
    cert = step_all_time_bound(np.diag([1., 2.]), np.eye(2), np.eye(2),
                               np.eye(2)[:, :1], .1)
    assert not cert['accepted_analytic']
    assert cert['bound'] >= 1.


def test_steady_cell_certificate_covers_interior_all_input_errors():
    K = np.diag([1., 2., 4.]); C = np.eye(3)
    G = np.array([[1., 0.], [0., 1.], [.001, -.001]])
    V = np.eye(3)[:, :2]
    system = AffineSystem(K, C, G, [np.diag([.1, .2, .3])], V)
    cert = steady_cell_bound(system, np.array([1.]), np.array([2.]))
    assert cert['counts']['riesz_rhs'] == 6
    for h in np.linspace(1., 2., 11):
        A = system.operator([h]); X = la.solve(A, G)
        E = X-V@la.solve(V.T@A@V, V.T@G)
        error = np.sqrt(la.eigvalsh(E.T@A@E, X.T@A@X)[-1])
        assert error <= cert['bound']*(1+1e-9)
