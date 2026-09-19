import unittest
import numpy as np
import scipy.linalg as la
from numerics import dense_be, modal_be, recover, effective_h, split_ports, errors

class NumericsTests(unittest.TestCase):
    def test_backends(self):
        rng=np.random.default_rng(51); M=rng.normal(size=(9,9)); K=M@M.T+np.eye(9)
        C=np.diag(np.linspace(.4,3,9)); F=rng.normal(size=(9,3))
        P=rng.normal(size=(15,3)); P[0]=0
        Zd=dense_be(C,K,F,P,.1)
        Y,U,Fm=modal_be(C,K,F,P,.1)
        np.testing.assert_allclose(Zd,Y@U.T,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(Zd@F,Y@Fm,rtol=1e-11,atol=1e-12)
    def test_unit_backends(self):
        K=np.diag([1.,2.,3.]); C=np.eye(3); F=np.eye(3)[:,:2]
        Zd=dense_be(C,K,F,None,.5,steps=6)
        Y,U,Fm=modal_be(C,K,F,None,.5,steps=6)
        np.testing.assert_allclose(recover(np.eye(3),Zd),recover(U,Y),atol=1e-14)
    def test_effective(self):
        np.testing.assert_allclose(effective_h([10.,100.],[.01,.02]),[10/1.1,100/3])
    def test_split_power(self):
        G=np.ones((4,1))/4; centers=np.array([[-1,-1,0],[-1,1,0],[1,-1,0],[1,1,0]])
        B,p=split_ports(G,np.array([3.]),centers)
        np.testing.assert_allclose(B@p,G[:,0]*3)
        self.assertEqual(B.shape,(4,4))
    def test_weak_source_error_not_hidden(self):
        X=np.array([[[10.,.01],[5.,.02]]]); app=X.copy(); app[0,0,1]+=.001
        d=errors(X,app,np.eye(2),np.ones(2),X[0])
        self.assertAlmostEqual(d['field_relative'],.05)
    def test_recover_shape(self):
        V=np.arange(12.).reshape(4,3); Z=np.arange(30.).reshape(5,3,2)
        R=recover(V,Z); self.assertEqual(R.shape,(5,4,2))
        for i in range(5):np.testing.assert_allclose(R[i],V@Z[i])
    def test_zero_initial(self):
        C=np.eye(2); K=np.eye(2); F=np.ones((2,1)); P=np.ones((8,1))
        Z=dense_be(C,K,F,P,.1)
        np.testing.assert_array_equal(Z[0],np.zeros(2))

    def test_stricter_stock_control_preserves_original_grid(self):
        from run import TOLS
        self.assertTrue(all(t in TOLS for t in (.01,.001,.0001)))
        self.assertIn(.00001,TOLS)

if __name__=='__main__': unittest.main()
