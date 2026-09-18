"""Tests before the two-limit-matched inverse representation is implemented."""
import unittest
import numpy as np
import scipy.linalg as la
from blend import weights, scalar_bound, Blend
from operators import assemble, coefficients

class BlendTests(unittest.TestCase):
    def test_harmonic_moments(self):
        a=np.exp(np.linspace(-1,1,36)).reshape(6,6)
        nodes,w=weights(a,5,'harmonic')
        np.testing.assert_allclose(w.sum(axis=0),1,atol=2e-15)
        np.testing.assert_allclose(np.einsum('j,jxy->xy',1/nodes,w),1/a,atol=2e-15)
        self.assertGreaterEqual(w.min(),0)

    def test_scalar_uniform_relative_error(self):
        a=np.exp(np.linspace(-1,1,36)).reshape(6,6)
        nodes,w=weights(a,5,'harmonic')
        bound=scalar_bound(nodes)
        for lam in np.logspace(-5,5,30):
            prediction=np.einsum('j,jxy->xy',1/(nodes*lam+1),w)
            truth=1/(a*lam+1)
            self.assertGreaterEqual(np.min(truth-prediction),-1e-14)
            self.assertLessEqual(np.max((truth-prediction)/truth),bound+1e-13)

    def test_constant_field_inverse_exact(self):
        n=6; layer=np.array([1,1,30,30,3,3.]); a=np.full((n,n),2.7)
        inv=Blend(a,layer,17.,5,'harmonic')
        matrix=assemble(a*layer[None,:],17.)
        np.testing.assert_allclose(matrix@inv.apply(np.eye(n*n)),np.eye(n*n),atol=2e-13)

    def test_symmetric_positive_without_claiming_spectral_bound(self):
        k,l=coefficients(6,'layered',42)
        inv=Blend(k/l[None,:],l,2000.,5,'harmonic')
        p=inv.apply(np.eye(36))
        np.testing.assert_allclose(p,p.T,atol=1e-13)
        self.assertGreater(la.eigvalsh(p)[0],0)

    def test_high_shift_limit(self):
        k,l=coefficients(6,'layered',42)
        inv=Blend(k/l[None,:],l,1e12,5,'harmonic')
        np.testing.assert_allclose(1e12*inv.apply(np.eye(36)),np.eye(36),atol=2e-8)

    def test_invalid_nodes(self):
        with self.assertRaises(ValueError):weights(np.ones((3,3)),1,'harmonic')
        with self.assertRaises(ValueError):weights(np.zeros((3,3)),5,'harmonic')

if __name__=='__main__':unittest.main()
