"""Algebraic tests independent of the held-out research seeds."""
import unittest
import numpy as np
import scipy.linalg as la
from operators import assemble, coefficients, Backbone, build_backbone
from compression import exact_oracle, spectral_inverse, refine


class OperatorTests(unittest.TestCase):
    def test_constant_fvm_entries(self):
        a = assemble(np.ones((3, 3)), 2.0).toarray()
        self.assertEqual(a[0, 0], 56.0)
        self.assertEqual(a[4, 4], 38.0)
        self.assertEqual(a[0, 1], -9.0)
        np.testing.assert_allclose(a, a.T)
        self.assertGreater(la.eigvalsh(a)[0], 0)

    def test_reject_invalid_coefficients(self):
        for k in [np.zeros((3, 3)), np.ones((2, 3)), np.full((3, 3), np.nan)]:
            with self.assertRaises(ValueError):
                assemble(k, 0.0)
        with self.assertRaises(ValueError):
            assemble(np.ones((3, 3)), -1.0)

    def test_backbone_inverse_is_exact_for_layers(self):
        n = 6
        layer = np.array([1, 1, 30, 30, 3, 3.0])
        scale = np.ones((n, n))
        backbone = Backbone(layer, scale, 7.0)
        a = assemble(np.broadcast_to(layer, (n, n)), 7.0).toarray()
        p = backbone.apply(np.eye(n*n))
        np.testing.assert_allclose(a @ p, np.eye(n*n), atol=2e-13)

    def test_q_and_qt_are_adjoints(self):
        rng = np.random.default_rng(42)
        bb = Backbone(np.exp(rng.normal(size=6)), np.exp(rng.normal(size=(6,6))), 4.)
        q = bb.q(np.eye(36))
        np.testing.assert_allclose(bb.qt(np.eye(36)), q.T, atol=2e-14)
        np.testing.assert_allclose(bb.apply(np.eye(36)), q @ q.T, atol=2e-14)

    def test_scaled_backbone_corresponds_to_declared_matrix(self):
        rng = np.random.default_rng(42)
        layer = np.exp(rng.normal(size=6))
        scale = np.exp(.2*rng.normal(size=(6,6)))
        bb = Backbone(layer, scale, 13.)
        base = assemble(np.broadcast_to(layer,(6,6)),0.).toarray()
        mass = np.tile(np.mean(1/scale**2,axis=0),6)
        d = np.diag(scale.ravel())
        expected = d @ (base + 13*np.diag(mass)) @ d
        np.testing.assert_allclose(expected @ bb.apply(np.eye(36)),np.eye(36),atol=3e-13)

    def test_coefficient_family_is_mesh_consistent(self):
        # Analytic fields and fixed aligned layers, no per-cell random draws.
        k, layer = coefficients(12, 'layered', 42)
        self.assertEqual(k.shape,(12,12))
        np.testing.assert_array_equal(layer,[1]*4+[30]*4+[3]*4)
        k2, _ = coefficients(12, 'layered', 42)
        np.testing.assert_array_equal(k,k2)
        self.assertGreater(k.min(),0.)

    def test_matching_variants_agree_without_jumps(self):
        k, layer = coefficients(6,'smooth',42)
        x = np.arange(36.)
        p = build_backbone(k,layer,7.,'scaled_uniform')
        q = build_backbone(k,layer,7.,'jump_aware')
        np.testing.assert_allclose(p.apply(x),q.apply(x),atol=1e-14)

    def test_liouville_face_coefficient_identity(self):
        ki,kj=3.7,12.1
        g=2*ki*kj/(ki+kj)
        self.assertAlmostEqual(g/np.sqrt(ki*kj),1/np.cosh(.5*np.log(ki/kj)))


class CompressionTests(unittest.TestCase):
    def setUp(self):
        self.k,self.layer=coefficients(6,'smooth',42)
        self.a=assemble(self.k,5.)
        self.bb=build_backbone(self.k,self.layer,5.,'jump_aware')

    def test_full_rank_correction_recovers_inverse(self):
        values,u=exact_oracle(self.a,self.bb)
        v=self.bb.q(u)
        p=self.bb.apply(np.eye(36))+ (v*(1/values-1))@v.T
        np.testing.assert_allclose(self.a@p,np.eye(36),atol=3e-12)

    def test_exact_energy_error_equals_tail(self):
        values,u=exact_oracle(self.a,self.bb)
        order=np.argsort(abs(values-1))[::-1]
        keep=order[:5]; tail=np.max(abs(values[order[5:]]-1))
        v=self.bb.q(u[:,keep])
        p=self.bb.apply(np.eye(36))+(v*(1/values[keep]-1))@v.T
        eig,z=la.eigh(self.a.toarray()); root=(z*np.sqrt(eig))@z.T
        actual=la.norm(np.eye(36)-root@p@root,2)
        self.assertAlmostEqual(actual,tail,places=11)

    def test_refinement_squares_energy_error(self):
        values,u=exact_oracle(self.a,self.bb)
        keep=np.argsort(abs(values-1))[::-1][:5]
        v=self.bb.q(u[:,keep]); p=self.bb.apply(np.eye(36))+(v*(1/values[keep]-1))@v.T
        f=lambda b:p@b
        p2=refine(self.a,f,np.eye(36),2)
        np.testing.assert_allclose(p2,2*p-p@self.a@p,atol=1e-13)
        eig,z=la.eigh(self.a.toarray()); root=(z*np.sqrt(eig))@z.T
        e=np.eye(36)-root@p@root; e2=np.eye(36)-root@p2@root
        np.testing.assert_allclose(e2,e@e,atol=3e-12)

    def test_matrix_free_subspace_matches_oracle(self):
        inv,info=spectral_inverse(self.a,self.bb,rank=5,tolerance=1e-10)
        values,_=exact_oracle(self.a,self.bb)
        target=np.sort(abs(values-1))[::-1][:5]
        np.testing.assert_allclose(np.sort(abs(np.array(info['ritz_values'])-1))[::-1],target,rtol=1e-7,atol=1e-10)
        p=inv(np.eye(36))
        np.testing.assert_allclose(p,p.T,atol=2e-13)
        self.assertGreater(la.eigvalsh(p)[0],0.)
        self.assertGreater(info['operator_columns'],5)

    def test_zero_rank_is_backbone(self):
        inv,info=spectral_inverse(self.a,self.bb,0)
        np.testing.assert_allclose(inv(np.ones(36)),self.bb.apply(np.ones(36)))
        self.assertEqual(info['operator_columns'],0)

    def test_no_physical_rhs_in_construction(self):
        # API consumes only the operator and backbone, not source snapshots.
        import inspect
        self.assertNotIn('sources',inspect.signature(spectral_inverse).parameters)
        self.assertNotIn('rhs',inspect.signature(spectral_inverse).parameters)

    def test_refinement_validation(self):
        with self.assertRaises(ValueError):
            refine(self.a,self.bb.apply,np.ones(36),0)


class AdaptiveTests(unittest.TestCase):
    def test_adaptive_matches_small_exact_threshold(self):
        from compression import adaptive_inverse
        k,l=coefficients(6,'smooth',42); a=assemble(k,5); bb=build_backbone(k,l,5,'jump_aware')
        inv,info=adaptive_inverse(a,bb,tail_target=.1,cap=24)
        ev,_=exact_oracle(a,bb)
        self.assertEqual(info['rank'],int(np.sum(abs(ev-1)>.1)))
        self.assertTrue(info['tail_observed'])
        self.assertLess(info['invariance_residual'],1e-7)
        x=inv(np.ones(36)); self.assertTrue(np.all(np.isfinite(x)))


if __name__=='__main__':
    unittest.main()
