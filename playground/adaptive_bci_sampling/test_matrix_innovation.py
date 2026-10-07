"""Tests for signed temporal Gram / continuous-spectrum matrix certificates."""
import numpy as np
import scipy.linalg as la
from scipy.integrate import quad_vec
from matrix_innovation import innovation_gram, spectral_majorant, joint_control_norm
from rational_dynamic import AffineSystem, cell_certificate, innovation_energy_bound


def test_poisson_kernel_identity_against_independent_frequency_integral():
    shifts = np.array([.07, 2., .3, 4.])
    for rate in [.001, .2, 8.]:
        def integrand(omega):
            s, product = 1j*omega, 1.+0j
            psi = []
            for sigma in shifts:
                psi.append(2*sigma/(s+sigma)*product)
                product *= (s-sigma)/(s+sigma)
            return rate/(rate**2+omega**2)*np.real(np.outer(np.conj(psi), psi))/np.pi
        gram, _ = quad_vec(integrand, 0., np.inf, epsabs=1e-10, epsrel=1e-10)
        np.testing.assert_allclose(innovation_gram(rate, shifts), gram, atol=1e-10, rtol=1e-9)


def test_matrix_majorant_covers_continuous_spectral_interval():
    shifts = np.geomspace(.02, 30., 16)
    M, metadata = spectral_majorant(shifts, (.001, 80.))
    assert metadata['construction'] == 'poisson_loewner'
    for rate in np.geomspace(.001, 80., 401):
        assert la.eigvalsh(M-innovation_gram(rate, shifts))[0] >= -2e-10
    assert la.eigvalsh(M)[0] >= -1e-12


def test_signed_matrix_bound_can_resolve_cancellation_lost_by_scalar_norms():
    shifts, rate = np.ones(8), 1e-5
    M, _ = spectral_majorant(shifts, (rate, 2*rate))
    F = np.full((1, 8, 1, 1), np.sqrt(rate))
    norm = joint_control_norm(F, F/rate, 1, np.eye(1), M)
    old = innovation_energy_bound(np.ones(8), shifts, (rate, 2*rate))
    assert norm < old/100


def test_matrix_bound_covers_forced_generalized_system_and_riesz_monotonicity():
    rng = np.random.default_rng(728)
    U = la.qr(rng.normal(size=(7, 7)))[0]
    Ka = U @ np.diag(np.geomspace(.02, 12., 7)) @ U.T
    K = Ka+np.diag(rng.uniform(.01, .3, 7))
    C = np.diag(np.geomspace(.1, 3., 7))
    shifts = np.geomspace(.03, 20., 15)
    F = rng.normal(size=(1, len(shifts), 7, 2))
    actions = la.solve(Ka, F[0].transpose(1, 0, 2).reshape(7, -1), assume_a='pos')
    actions = actions.reshape(7, len(shifts), 2).transpose(1, 0, 2)[None]
    rates = la.eigvalsh(K, C)
    M, _ = spectral_majorant(shifts, (rates[0], rates[-1]))
    bound = joint_control_norm(F, actions, 1, np.eye(2), M)
    q, gram = np.zeros((7, 2)), np.zeros((2, 2))
    for sigma, f in zip(shifts, F[0]):
        x = np.sqrt(2*sigma)*la.solve(K+sigma*C, q+f, assume_a='pos')
        gram += x.T @ C @ x
        q -= np.sqrt(2*sigma)*C @ x
    gram += .5*q.T @ la.solve(K, q, assume_a='pos')
    assert np.sqrt(la.eigvalsh(gram)[-1]) <= bound*(1+1e-12)


def test_matrix_cell_bound_covers_continuous_points_and_preserves_trial_cost():
    from run_rational_dynamic import synthetic
    K, C, G, H, _, _ = synthetic()
    V = np.eye(32)[:, :25]
    system = AffineSystem(K, C, G, H, V)
    args = ([1., 1.], [1.2, 1.2], np.geomspace(.01, 20., 24), 2)
    old = cell_certificate(system, *args, propagation_model='energy')
    new = cell_certificate(system, *args, propagation_model='matrix')
    assert new['bound'] <= old['bound']*(1+1e-12)
    assert new['counts']['trial_rhs'] == old['counts']['trial_rhs']
    assert new['counts']['riesz_rhs'] == old['counts']['riesz_rhs']
    from rational_dynamic import exact_impulse_gram
    for h in np.random.default_rng(11).uniform([1., 1.], [1.2, 1.2], (10, 2)):
        J, Q = exact_impulse_gram(system.operator(h), C, G, V)
        assert np.sqrt(la.eigvalsh(J, Q)[-1]) <= new['bound']*(1+1e-9)


def test_final_svd_acceptance_uses_twice_extraction_tolerance():
    from run_rational_dynamic import acceptance_threshold
    assert acceptance_threshold(.001, 'final') == .002
    assert acceptance_threshold(.001, 'raw') == .001


def test_dense_small_model_spectral_interval_is_reported_and_valid():
    rng = np.random.default_rng(219)
    U = la.qr(rng.normal(size=(5, 5)))[0]
    K = U @ np.diag([.01, .2, 1., 2., 5.]) @ U.T
    C = np.diag([.2, .5, 1., 2., 3.])
    system = AffineSystem(K, C, rng.normal(size=(5, 2)), [np.eye(5)], U[:, :3])
    cert = cell_certificate(system, [.1], [.4], [1., .2, 3.], 1,
                            propagation_model='matrix', spectral_contraction='dense')
    alpha, beta = cert['spectral_interval']
    np.testing.assert_allclose([alpha, beta], [la.eigvalsh(K+.1*np.eye(5), C)[0],
                                             la.eigvalsh(K+.4*np.eye(5), C)[-1]])
    assert cert['counts']['dense_spectral_eigendecompositions'] == 2


def test_svd_candidate_is_nested_and_retains_the_uniform_mode():
    from svd_dynamic_guard import svd_candidate_coordinates
    rng = np.random.default_rng(773)
    snapshots = rng.normal(size=(9, 5))
    U = la.svd(snapshots, full_matrices=False)[0]
    W = la.qr(np.column_stack([U, np.ones(9)]), mode='economic')[0]
    T = svd_candidate_coordinates(U, W, 3)
    V = W @ T
    np.testing.assert_allclose(T.T @ T, np.eye(4), atol=1e-13)
    np.testing.assert_allclose(V @ V.T @ U[:, :3], U[:, :3], atol=1e-13)
    np.testing.assert_allclose(V @ V.T @ np.ones(9), np.ones(9), atol=1e-13)


def test_nested_dynamic_triangle_in_full_input_denominator():
    rng = np.random.default_rng(71)
    C = np.diag(np.geomspace(.2, 2., 9))
    U = la.qr(rng.normal(size=(9, 9)))[0]
    K = U @ np.diag(np.geomspace(.01, 10., 9)) @ U.T
    G = rng.normal(size=(9, 2))
    W, T = U[:, :7], np.eye(7)[:, :4]
    from rational_dynamic import exact_impulse_gram
    Jw, Q = exact_impulse_gram(K, C, G, W)
    Jv, _ = exact_impulse_gram(K, C, G, W @ T)
    Jr, Qr = exact_impulse_gram(W.T @ K @ W, W.T @ C @ W, W.T @ G, T)
    full = np.sqrt(la.eigvalsh(Jv, Q)[-1])
    upper = np.sqrt(la.eigvalsh(Jw, Q)[-1])+np.sqrt(la.eigvalsh(Jr, Qr)[-1])
    assert full <= upper*(1+1e-12)


def test_poisson_majorant_is_invariant_under_common_rate_shift_scaling():
    shifts = np.geomspace(.01, 20., 12)
    original, _ = spectral_majorant(shifts, (.001, 50.))
    scaled, _ = spectral_majorant(1e-8*shifts, (1e-11, 5e-7))
    np.testing.assert_allclose(scaled, original, atol=1e-11, rtol=1e-10)


def test_matrix_refinement_preserves_full_inverse_counts_and_covers_points():
    from run_rational_dynamic import synthetic
    from rational_dynamic import exact_impulse_gram
    K, C, G, H, _, _ = synthetic()
    system = AffineSystem(K, C, G, H, np.eye(32)[:, :25])
    args = ([1., 1.], [1.2, 1.2], np.geomspace(.01, 20., 24), 2)
    old = cell_certificate(system, *args, propagation_model='matrix')
    new = cell_certificate(system, *args, propagation_model='matrix', envelope_depth=1)
    assert new['bound'] <= old['bound']*(1+1e-12)
    assert new['counts']['riesz_rhs'] == old['counts']['riesz_rhs']
    assert new['counts']['trial_rhs'] == old['counts']['trial_rhs']
    for h in [[1., 1.], [1.2, 1.2], [1.07, 1.14]]:
        J, Q = exact_impulse_gram(system.operator(h), C, G, system.V)
        assert np.sqrt(la.eigvalsh(J, Q)[-1]) <= new['bound']*(1+1e-9)


def test_signed_matrix_certificate_keeps_worst_nearly_parallel_input():
    K, C = np.diag([1., 2.]), np.eye(2)
    G = np.array([[1., 1.], [1e-4, -1e-4]])
    V = np.array([[1.], [0.]])
    system = AffineSystem(K, C, G, [np.eye(2)], V)
    cert = cell_certificate(system, [1.], [1.], np.geomspace(.1, 10., 40), 0,
                            propagation_model='matrix')
    assert cert['bound'] >= .999


def test_guard_experiment_controls_preserve_default_and_allow_native_model():
    from svd_dynamic_guard import parse_arguments
    default = parse_arguments(['--output', 'unused.json'])
    assert (default.seed, default.mesh_mm, default.model) == (20260805, 10., 'case1')
    varied = parse_arguments(['--output', 'unused.json', '--seed', '20261007',
                              '--mesh-mm', '8', '--model', 'native-case1'])
    assert (varied.seed, varied.mesh_mm, varied.model) == (20261007, 8., 'native-case1')


def test_saved_validation_uses_pack_matrices_and_threshold():
    from verify_guarded_svd_cells import load_saved_system
    pack = {'K': np.diag([2., 3., 4.]), 'C': np.eye(3),
            'G': np.ones((3, 1)), 'H': np.array([np.eye(3)]),
            'final_basis': np.eye(3)[:, :2], 'ranges': np.array([[1., 9.]]),
            'tolerance': np.array(.01)}
    system, ranges, threshold = load_saved_system(pack)
    np.testing.assert_array_equal(system.K.toarray(), pack['K'])
    np.testing.assert_array_equal(ranges, pack['ranges'])
    assert threshold == .02
