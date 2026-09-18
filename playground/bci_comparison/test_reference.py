"""Fine-grid FOM reference backend checks against direct algebra."""
import unittest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from reference_backend import cg_solve

class ReferenceTests(unittest.TestCase):
    def test_multiple_rhs_matches_direct(self):
        A=sp.diags([-np.ones(9),np.full(10,4.),-np.ones(9)],[-1,0,1]).tocsc()
        B=np.column_stack([np.ones(10),np.arange(10.)]);M=sla.LinearOperator(A.shape,matvec=lambda x:x/4)
        X,stats=cg_solve(A,B,M)
        np.testing.assert_allclose(X,np.linalg.solve(A.toarray(),B),rtol=1e-9,atol=1e-11)
        self.assertLess(stats['relative_residual'],1e-9)

    def test_zero_rhs_preserves_zero(self):
        A=sp.eye(5,format='csc');M=sla.aslinearoperator(A)
        X,stats=cg_solve(A,np.zeros((5,2)),M,x0=np.ones((5,2)))
        np.testing.assert_array_equal(X,0)
        self.assertEqual(stats['iterations'],0)

    def test_warm_start_is_not_redefined_as_truth(self):
        A=sp.diags([1.,2.,3.]); B=np.ones((3,1));M=sla.aslinearoperator(sp.eye(3))
        X,stats=cg_solve(A,B,M,x0=np.ones((3,1))*123.)
        np.testing.assert_allclose(X[:,0],[1.,.5,1/3],rtol=1e-9)

class MetricTimingTests(unittest.TestCase):
    def test_mixed_junction_uses_peak_of_average_not_average_of_peaks(self):
        from measures import errors
        X=np.array([[[2.],[0.]],[[0.],[2.]]]); G=np.ones((2,1))/2
        Y=X+1.
        result=errors(X,Y,G,np.ones(2),X.max(axis=0),trajectory_normalization=True)
        self.assertAlmostEqual(result['junction_relative'],1.)

if __name__=='__main__': unittest.main()
