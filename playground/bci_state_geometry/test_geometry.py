"""Small algebra tests; no native library or AMG substitute is used."""
import unittest
import numpy as np
import scipy.linalg as la
from geometry import (orth, whiten, make_bank, pod_basis, loss_gradient,
                      optimize_space, query_space, balanced_space, project)

class GeometryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42); self.n=12; self.p=2
        R=rng.normal(size=(12,12)); self.K=R@R.T+np.eye(12)
        self.F=rng.normal(size=(12,2)); self.C=np.diag(np.linspace(.3,3,12))
        self.c=np.ones((12,1))/np.sqrt(12)
        self.bank=make_bank([self.K,self.K+np.diag(np.linspace(0,2,12))],self.F,[0.,.4,4.])
        self.P=pod_basis(self.bank,self.c)
    def test_orth(self):
        Q=orth(np.column_stack((self.F,self.F)))
        self.assertEqual(Q.shape[1],2); np.testing.assert_allclose(Q.T@Q,np.eye(2),atol=2e-14)
    def test_whiten(self):
        T=whiten(self.C); np.testing.assert_allclose(T.T@self.C@T,np.eye(12),atol=2e-14)
    def test_pod_fixed(self):
        np.testing.assert_allclose(self.P[:,0],self.c[:,0],atol=2e-14)
        np.testing.assert_allclose(self.P.T@self.P,np.eye(self.P.shape[1]),atol=1e-12)
    def test_full_space_error(self):
        _,_,worst=loss_gradient(np.eye(12),self.bank,.02)
        self.assertLess(abs(worst),1e-12)
    def test_gradient(self):
        W=self.P[:,:5]; D=np.random.default_rng(54).normal(size=W.shape)
        f,g,_=loss_gradient(W,self.bank,.04); eps=1e-6
        fd=(loss_gradient(W+eps*D,self.bank,.04)[0]-loss_gradient(W-eps*D,self.bank,.04)[0])/(2*eps)
        self.assertAlmostEqual(fd,float(np.sum(g*D)),delta=2e-6)
    def test_rotational_invariance(self):
        W=self.P[:,:5]; O=la.qr(np.random.default_rng(3).normal(size=(5,5)))[0]
        f,g,w=loss_gradient(W,self.bank,.01); f2,g2,w2=loss_gradient(W@O,self.bank,.01)
        self.assertAlmostEqual(f,f2,places=12); np.testing.assert_allclose(g@O,g2,atol=1e-11)
    def test_optimization_fixed(self):
        W,log=optimize_space(self.P[:,:5],self.bank,self.c,iterations=3,temperatures=(.03,))
        self.assertLessEqual(log[-1]['loss'],log[0]['loss']+1e-13)
        np.testing.assert_allclose(W[:,0],self.c[:,0],atol=1e-13)
        np.testing.assert_allclose(W.T@W,np.eye(5),atol=2e-13)
    def check_dc(self,kind):
        W,stats=query_space(self.K,self.F,self.P,7,self.c,kind)
        self.assertLessEqual(W.shape[1],7)
        app=W@la.solve(W.T@self.K@W,W.T@self.F,assume_a='pos')
        np.testing.assert_allclose(app,la.solve(self.K,self.F),rtol=3e-11,atol=3e-12)
        np.testing.assert_allclose(W@W.T@self.c,self.c,atol=1e-12)
    def test_dc_augment(self): self.check_dc('dc_augment')
    def test_harmonic_dc(self): self.check_dc('harmonic')
    def test_harmonic_not_static_feedthrough(self):
        W,_=query_space(self.K,self.F,self.P,7,self.c,'harmonic')
        c,k,f=project(self.C,self.K,self.F,W)
        self.assertEqual(f.shape,(W.shape[1],2))
        self.assertGreater(la.eigvalsh(c).min(),0)
        self.assertGreater(la.eigvalsh(k).min(),0)
        np.testing.assert_array_equal(W@np.zeros(W.shape[1]),np.zeros(12))
    def test_harmonic_full_rank(self):
        W,_=query_space(self.K,self.F,self.P,12,self.c,'harmonic')
        self.assertEqual(W.shape,(12,12))
    def test_budget_guard(self):
        with self.assertRaises(ValueError):query_space(self.K,self.F,self.P,2,self.c,'harmonic')
    def test_balanced_positive(self):
        W=balanced_space(self.K,self.F,6,self.c)
        np.testing.assert_allclose(W.T@W,np.eye(6),atol=1e-12)
        self.assertGreater(la.eigvalsh(W.T@self.K@W).min(),0)
    def test_nested_projection(self):
        D=orth(np.random.default_rng(13).normal(size=(20,12)))
        A=np.diag(np.linspace(.1,5,20)); G=np.random.default_rng(12).normal(size=(20,2))
        W=self.P[:,:6]
        np.testing.assert_allclose((D@W).T@A@(D@W),W.T@(D.T@A@D)@W,atol=1e-13)
        np.testing.assert_allclose((D@W).T@G,W.T@(D.T@G),atol=1e-13)
    def test_schur_identity(self):
        W=self.P[:,:5]; A=self.K; B=self.F
        X=la.solve(A,B,assume_a='pos'); Y=la.solve(W.T@A@W,W.T@B,assume_a='pos'); E=X-W@Y
        np.testing.assert_allclose(B.T@X-B.T@W@Y,E.T@A@E,atol=1e-12)
    def test_bank_has_no_validation_argument(self):
        self.assertEqual(len(self.bank),6)
        for b in self.bank:
            np.testing.assert_allclose(np.diag(b['Z']),np.ones(2),atol=2e-13)


    def test_pod_keeps_ordered_singular_subspaces(self):
        # Re-SVD of an orthonormal block must not destroy POD ordering.
        c=np.eye(5)[:,0:1]
        rng=np.random.default_rng(412)
        U,_=la.qr(rng.normal(size=(4,4)))
        X=np.vstack((np.zeros((1,12)),U@np.diag([10.,3.,.3,.01])@rng.normal(size=(4,12))))
        normalized=X/np.linalg.norm(X,axis=0)
        expected=la.svd(normalized,full_matrices=False)[0][:,:2]
        actual=pod_basis([{'X':X}],c)[:,1:3]
        self.assertLess(la.norm(expected-actual@(actual.T@expected)),1e-10)

if __name__=='__main__': unittest.main()
