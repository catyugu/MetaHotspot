"""Comparison contracts, independent of the unavailable native backend."""
import unittest
import numpy as np
from contracts import measure, from_steps, reduced_history, recover


class ContractTests(unittest.TestCase):
    def test_weak_junction_channel_not_hidden(self):
        G=np.eye(2); ref=np.array([[[1.,.01],[.01,2.]]]); app=ref.copy()
        app[0,0,1]+=.001
        result=measure(ref,app,G,np.ones(2),ref[0])
        self.assertAlmostEqual(result['junction_channel_relative'],.1)

    def test_mixed_average_peak_not_average_of_peaks(self):
        ref=np.array([[[2.],[0.]],[[0.],[2.]]]); G=np.ones((2,1))/2
        result=measure(ref,ref+1.,G,np.ones(2),None)
        self.assertAlmostEqual(result['junction_channel_relative'],1.)

    def test_temperature_rise_scale(self):
        ref=np.array([[[2.],[4.]]]); result=measure(ref,ref+.4,np.eye(2),np.ones(2),ref[0])
        self.assertAlmostEqual(result['field_relative'],.1)
        self.assertAlmostEqual(result['peak_absolute_K'],.4)

    def test_step_superposition_matches_direct_be(self):
        C=np.diag([2.,3.]); K=np.array([[2.,-1.],[-1.,4.]]); F=np.eye(2)
        P=np.random.default_rng(42).normal(size=(11,2)); P[0]=0.
        S=reduced_history(C,K,F,.1,10)
        X=reduced_history(C,K,F,.1,10,P)
        np.testing.assert_allclose(from_steps(S,P),X,rtol=1e-12,atol=1e-14)

    def test_nonzero_unused_power_at_initial_time(self):
        S=np.arange(4.)[:,None,None]; P=np.ones((4,1))*2
        np.testing.assert_allclose(from_steps(S,P)[:,0,0],[0.,2.,4.,6.])

    def test_reconstruction_and_output_layout(self):
        V=np.array([[1.,2.],[3.,4.],[5.,6.]])
        Z=np.arange(16.).reshape(4,2,2)
        np.testing.assert_allclose(recover(V,Z),np.stack([V@z for z in Z]))

    def test_identical_fields_have_no_error(self):
        ref=np.arange(18.).reshape(3,3,2); result=measure(ref,ref,np.eye(3),np.ones(3),ref[-1])
        self.assertEqual(result['field_relative'],0.)
        self.assertEqual(result['capacity_l2_relative'],0.)

if __name__=='__main__': unittest.main()
