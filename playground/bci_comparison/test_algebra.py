"""Algebraic guards; no native DLL or locally installed AMG is required."""
import unittest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from algorithms import orthogonal_block, direction, cg_directions, response_compress, dense_be

class AlgebraTests(unittest.TestCase):
    def test_orthogonal_block_removes_existing_space(self):
        V=np.eye(6)[:,:2]
        Q=orthogonal_block(V,np.eye(6)[:,1:4])
        self.assertEqual(Q.shape,(6,2))
        np.testing.assert_allclose(V.T@Q,0,atol=1e-14)
        np.testing.assert_allclose(Q.T@Q,np.eye(2),atol=1e-14)

    def test_dependent_correction_is_not_enriched(self):
        V=np.eye(6)[:,:2]
        self.assertEqual(orthogonal_block(V,V).shape[1],0)

    def test_tangent_is_dominant_normalized_residual(self):
        R=np.array([[1.,3.],[2.,1.],[0.,2.]])
        d=direction(R,'tangent')
        self.assertAlmostEqual(np.linalg.norm(d),1.)
        self.assertAlmostEqual(np.linalg.norm(R@d),la.svdvals(R)[0])

    def test_column_is_worst_port(self):
        d=direction(np.diag([1.,3.,2.]),'column')
        np.testing.assert_array_equal(d,[0.,1.,0.])

    def test_cg_directions_solve_two_dimensional_problem(self):
        A=np.diag([1.,4.]); b=np.array([2.,3.])
        X,info=cg_directions(A,b,lambda r:r,4)
        Q=orthogonal_block(np.empty((2,0)),X)
        x=Q@la.solve(Q.T@A@Q,Q.T@b)
        np.testing.assert_allclose(A@x,b,atol=1e-13)
        self.assertEqual(info['iterations'],2)

    def test_exact_preconditioner_needs_one_direction(self):
        A=np.diag([1.,4.,9.]); b=np.ones(3)
        X,info=cg_directions(A,b,lambda r:la.solve(A,r),4)
        self.assertEqual(info['iterations'],1)
        self.assertEqual(X.shape[1],1)

    def test_cg_directions_are_positive_energy(self):
        A=np.diag(np.arange(1.,11.)); X,_=cg_directions(A,np.ones(10),lambda r:r,4)
        self.assertTrue(np.all(np.diag(X.T@A@X)>0))

    def test_cg_rejects_indefinite_curvature(self):
        with self.assertRaises(ValueError):
            cg_directions(-np.eye(2),np.ones(2),lambda r:r,2)

    def test_response_compression_preserves_constant(self):
        V=np.eye(4); As=[np.diag([1.,2.,3.,4.])]; G=np.eye(4)[:,:1]
        Q,info=response_compress(V,As,G,[(1.,)],1e-3)
        c=np.ones(4)/2
        self.assertLess(np.linalg.norm(c-Q@(Q.T@c)),1e-12)

    def test_full_basis_galerkin_equals_direct(self):
        A=np.diag([1.,2.,3.]); C=np.diag([2.,3.,4.]); B=np.eye(3)[:,:2]
        t=np.arange(6)*.2; U=np.ones((6,2))
        X=dense_be(C,A,B,U,.2)
        L=C/.2+A; x=np.zeros(3)
        for i in range(1,6):
            x=la.solve(L,C@x/.2+B@U[i])
            np.testing.assert_allclose(X[i],x,rtol=1e-13,atol=1e-13)

    def test_backward_euler_uses_right_endpoint_input(self):
        X=dense_be(np.eye(1),np.eye(1),np.eye(1),np.array([[10.],[2.],[0.]]),1.)
        np.testing.assert_allclose(X[:,0],[0.,1.,.5])

    def test_zero_input_stays_zero(self):
        X=dense_be(np.eye(2),np.eye(2),np.ones((2,1)),np.zeros((6,1)),.1)
        np.testing.assert_array_equal(X,0)

class MetricTests(unittest.TestCase):
    def test_split_ports_preserves_nominal_load(self):
        from measures import split_ports
        centers=np.array([[x,y,0.] for x in (-1.,1.) for y in (-1.,1.)])
        G=np.ones((4,1))/4; power=np.array([2.])
        B,P=split_ports(G,power,centers)
        np.testing.assert_allclose(B@P,G@power)
        np.testing.assert_allclose(B.sum(axis=0),1.)

    def test_absolute_temperature_does_not_dilute_error(self):
        from measures import errors
        X=np.ones((2,3,1))*2; Y=X+1
        v=errors(X,Y,np.ones((3,1))/3,np.ones(3),np.ones((3,1))*2)
        self.assertAlmostEqual(v['field_relative'],.5)
        self.assertAlmostEqual(v['junction_relative'],.5)
        self.assertAlmostEqual(v['peak_absolute_K'],1.)

    def test_identical_fields_have_zero_error(self):
        from measures import errors
        X=np.ones((4,3,2)); G=np.ones((3,2))/3
        self.assertEqual(errors(X,X,G,np.ones(3),X[-1])['field_relative'],0.)

    def test_eff_htc_inverse(self):
        from measures import effective_h
        h=np.array([50.,1000.]); r=np.array([1e-5,1e-3])
        p=effective_h(h,r)
        np.testing.assert_allclose(p/(1-r*p),h)

class IntegrationTests(unittest.TestCase):
    def test_incremental_projection_matches_dense(self):
        from algorithms import SharedSpace
        rng=np.random.default_rng(42); X=rng.normal(size=(12,12))
        A=X@X.T+np.eye(12); C=np.diag(np.arange(1.,13.)); G=rng.normal(size=(12,2))
        s=SharedSpace([sp.csc_matrix(A),sp.csc_matrix(C)],G)
        s.append(rng.normal(size=(12,3)))
        Y,R=s.response((1.,.2)); V=s.V
        np.testing.assert_allclose(R,G-(A+.2*C)@V@Y,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(s.small[0],V.T@A@V,rtol=1e-12,atol=1e-12)

    def test_native_dependency_extractors_on_small_matrix(self):
        import importlib.util
        if importlib.util.find_spec('pyamg') is None:
            self.skipTest('AMG intentionally not installed locally; required in CI')
        from algorithms import extract
        from metahotspot.macromodel import utils as stock
        n=16
        K=sp.diags([-np.ones(n-1),np.r_[1.,np.full(n-2,2.),1.],-np.ones(n-1)],[-1,0,1]).tocsc()
        C=sp.eye(n,format='csc'); G=np.eye(n)[:,[3,12]]
        H=[sp.diags(np.r_[1.,np.zeros(n-1)]),sp.diags(np.r_[np.zeros(n-1),1.])]
        core=stock.normalized_operators(K,C,G.sum(axis=1))
        for method in ('shared_column','shared_tangent','shared_tangent_cached','krylov_tangent'):
            V,_=extract(core,G,H,np.array([[.1,2.],[.1,2.]]),tolerance=1e-4,seed=42,method=method,probe_rounds=2,max_order=32)
            A=K+.37*H[0]+1.13*H[1]+.45*C
            exact=la.solve(A.toarray(),G)
            reduced=V@la.solve(V.T@A@V,V.T@G)
            self.assertLess(la.norm(exact-reduced)/la.norm(exact),.02)

if __name__=='__main__': unittest.main()
