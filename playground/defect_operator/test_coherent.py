"""Algebraic checks for coherent factor interpolation, preceding implementation."""
import unittest
import numpy as np
from operators import coefficients,build_backbone
from coherent import Coherent
from blend import Blend

class CoherentTests(unittest.TestCase):
    def test_zero_shift_retains_full_scaled_inverse(self):
        k,l=coefficients(6,'layered',42);a=k/l[None,:]
        for count in [3,5,9]:
            p=Coherent(a,l,0.,count)
            expected=build_backbone(k,l,0.,'jump_aware').apply(np.eye(36))
            np.testing.assert_allclose(p.apply(np.eye(36)),expected,atol=5e-15,rtol=5e-13)

    def test_adjoint(self):
        k,l=coefficients(6,'layered',42);p=Coherent(k/l[None,:],l,2000.,5)
        q=p.q(np.eye(36)); np.testing.assert_allclose(p.qt(np.eye(36)),q.T,atol=2e-15)
        np.testing.assert_allclose(p.apply(np.eye(36)),q@q.T,atol=2e-15)

    def test_mass_limit(self):
        k,l=coefficients(6,'layered',42);p=Coherent(k/l[None,:],l,1e12,5)
        np.testing.assert_allclose(1e12*p.apply(np.eye(36)),np.eye(36),atol=2e-8)

    def test_factor_moment(self):
        a=np.exp(np.linspace(-1,1,36)).reshape(6,6);p=Coherent(a,np.ones(6),20.,5)
        np.testing.assert_allclose(p.w.sum(axis=0),1.,atol=2e-15)
        np.testing.assert_allclose(np.einsum('j,jxy->xy',1/np.sqrt(p.nodes),p.w),1/np.sqrt(a),atol=2e-15)

    def test_sum_of_inverses_kills_disjoint_coefficient_coupling(self):
        a=np.ones((6,6));a[3:]=9; layer=np.ones(6)
        incoherent=Blend(a,layer,0.,5,'harmonic').apply(np.eye(36))
        coherent=Coherent(a,layer,0.,5).apply(np.eye(36))
        self.assertEqual(incoherent[0,-1],0.)
        self.assertGreater(coherent[0,-1],0.)

if __name__=='__main__':unittest.main()
