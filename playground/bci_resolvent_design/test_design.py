"""Algebra, dependency budgets, source normalization, and interpolation tests."""
import unittest
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp

from design import (effective, log_derivative, trilinear_lift, weighted_features,
                    select_agenda, random_agenda, close_basis, bump_parameter,
                    Sample, Family, coarse_bank, make_pool, discrepancy_features)
from evaluation import errors, march_dense, march_modal


class DesignTests(unittest.TestCase):
    def test_physical_effective_map_and_log_derivative(self):
        h=np.array([30.,900.]); beta=np.array([.01,.002]); d=1e-5
        np.testing.assert_allclose(effective(h,beta),h/(1+h*beta))
        fd=(effective(h*np.exp(d),beta)-effective(h*np.exp(-d),beta))/(2*d)
        np.testing.assert_allclose(fd,log_derivative(h,beta),rtol=1e-9)

    def test_invalid_physical_coefficient_rejected(self):
        with self.assertRaises(ValueError): effective(np.array([-1.]),np.array([1.]))

    def test_lift_preserves_constants_and_interior_linear_fields(self):
        c=np.array([(x,y,z) for x in [0.,1.] for y in [0.,1.] for z in [0.,1.]])
        f=np.array([[.2,.3,.4],[.8,.5,.1],[-.5,.5,1.5]])
        P=trilinear_lift(c,f)
        np.testing.assert_allclose(P@np.ones(8),1.)
        np.testing.assert_allclose((P@c)[:2],f[:2])
        self.assertGreaterEqual(P.data.min(),0.)

    def test_incomplete_coarse_grid_rejected(self):
        with self.assertRaises(ValueError):
            trilinear_lift(np.array([[0,0,0],[1,0,0],[0,1,0]]),np.zeros((1,3)))

    def test_weighted_constant_mode_is_removed(self):
        X=np.column_stack((np.ones(3)*2.,[1.,2.,4.]))
        F,norm=weighted_features(X,np.array([1.,2.,3.]))
        np.testing.assert_allclose(F[:,0],0.,atol=1e-15)
        self.assertTrue(np.all(norm>0))
        self.assertAlmostEqual(np.sqrt([1.,2.,3.])@F[:,1],0.,places=14)

    def test_budget_counts_parent_solves(self):
        val=np.eye(4); jets=np.array([val[:,::-1]*3.])
        actions=select_agenda(val,jets,7,[0])
        seen=set()
        self.assertEqual(len(actions),7)
        self.assertEqual(len(set(actions)),7)
        for a in actions:
            if a.derivative>=0: self.assertIn(a.sample,seen)
            else: seen.add(a.sample)

    def test_value_selection_is_invariant_under_orthogonal_row_rotation(self):
        rng=np.random.default_rng(123); F=rng.normal(size=(6,14)); Q=la.qr(rng.normal(size=(6,6)))[0]
        self.assertEqual(select_agenda(F,None,6,[0]),select_agenda(Q@F,None,6,[0]))

    def test_random_agenda_is_reproducible_unique_and_contains_initial_sources(self):
        a=random_agenda(20,12,[1,3,5,7],42)
        self.assertEqual(a,random_agenda(20,12,[1,3,5,7],42))
        self.assertEqual([x.sample for x in a[:4]],[1,3,5,7])
        self.assertEqual(len(set(a)),12)

    def test_closing_keeps_uniform_and_span(self):
        X=np.array([[1.,2.],[0.,1.],[3.,2.],[4.,1.]])
        V=close_basis(X,1e-9)
        np.testing.assert_allclose(V.T@V,np.eye(V.shape[1]),atol=1e-12)
        np.testing.assert_allclose(V@(V.T@X),X,atol=1e-12)
        np.testing.assert_allclose(V@(V.T@np.ones(4)),1.,atol=1e-12)

    def test_zero_snapshot_rejected(self):
        with self.assertRaises(ValueError): close_basis(np.zeros((3,2)),1e-3)

    def test_secants_stay_inside_parameter_box(self):
        for x in [1.,100.,1e4]:
            h=np.array([x,100.]); q,d=bump_parameter(h,0)
            self.assertTrue(1.<=q[0]<=1e4)
            self.assertAlmostEqual(np.log(q[0]/h[0]),d)
            self.assertEqual(q[1],h[1])

    @staticmethod
    def family():
        K=sp.csc_matrix([[1.,-1.],[-1.,1.]])
        C=sp.diags([2.,3.],format='csc'); G=np.eye(2)
        H=[sp.diags([1.,0.],format='csc'),sp.diags([0.,2.],format='csc')]
        return Family(K,C,G,H,np.array([.01,.02]),np.array([[0.,0.,0.],[1.,0.,0.]]))

    def test_coarse_jet_matches_finite_difference_of_resolvent(self):
        f=self.family(); pool=[Sample((30.,90.),.7,0),Sample((30.,90.),.7,1)]
        X,D,stats=coarse_bank(f,pool,True)
        delta=1e-5
        for j in range(2):
            hp=np.array([30.,90.]); hm=hp.copy(); hp[j]*=np.exp(delta); hm[j]*=np.exp(-delta)
            xp=la.solve(f.operator(hp,.7).toarray(),f.G)
            xm=la.solve(f.operator(hm,.7).toarray(),f.G)
            np.testing.assert_allclose(D[j],(xp-xm)/(2*delta),rtol=1e-9,atol=1e-12)
        self.assertEqual(stats['factorizations'],1)
        self.assertEqual(stats['rhs_solves'],6)

    def test_identical_models_have_zero_discrepancy(self):
        f=self.family(); pool=[Sample((30.,90.),.7,0),Sample((30.,90.),.7,1)]
        X,_,_=coarse_bank(f,pool,False)
        D,_=discrepancy_features(f,f,pool,X,123,sketch_rows=8)
        np.testing.assert_allclose(D,0.,atol=1e-14)

    def test_pool_is_deterministic_and_covers_all_sources_at_initial_sample(self):
        pool,initial=make_pool(4,100.,42)
        self.assertEqual(pool,make_pool(4,100.,42)[0])
        self.assertEqual([pool[i].port for i in initial],list(range(4)))
        self.assertTrue(all(pool[i].shift==0 for i in initial))


class EvaluationTests(unittest.TestCase):
    def test_weak_sources_are_not_hidden_by_strong_sources(self):
        X=np.array([[[100.,0.],[0.,1.]]]); Y=X.copy(); Y[0,1,1]=.5
        e=errors(X,Y,np.eye(2),np.ones(2),X[0],False)
        self.assertAlmostEqual(e['field_relative'],.5)
        self.assertAlmostEqual(e['junction_relative'],.5)

    def test_mixed_junction_denominator_is_peak_of_actual_average(self):
        X=np.array([[[2.],[0.]],[[0.],[2.]]]); G=np.ones((2,1))/2
        e=errors(X,X+1.,G,np.ones(2),X.max(axis=0),True)
        self.assertAlmostEqual(e['junction_relative'],1.)

    def test_dense_and_modal_bdf1_are_same_equations(self):
        C=np.array([[2.,.1],[.1,1.]]); K=np.array([[3.,-1.],[-1.,2.]])
        F=np.eye(2); P=np.tile(np.eye(2),(12,1,1))
        Z=march_dense(C,K,F,P,.1)
        Q,Zm=march_modal(C,K,F,P,.1)
        np.testing.assert_allclose(Z,np.einsum('ij,tjk->tik',Q,Zm),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(Z[0],0.)



class StockAndContractTests(unittest.TestCase):
    def test_repository_library_source_is_exact(self):
        import hashlib
        from pathlib import Path
        from run import UTILS_SHA256
        p=Path(__file__).resolve().parents[2]/'python/metahotspot/macromodel/utils.py'
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),UTILS_SHA256)

    def test_qualifying_stock_is_not_replaced_by_expensive_parent(self):
        from run import summarize
        rec=[]; metrics=[]; timing=[]
        for ident,method,cost,rank,err in [('stock_t0.001','stock',1.,10,.009),('stock_t1e-05','stock',100.,50,.0001),('coarse_qr_b24_t0.001','coarse_qr',5.,10,.001)]:
            rec.append(dict(id=ident,method=method,offline_seconds=cost,rank=rank,budget=24,tolerance=.001,repeat=0))
            metrics.append(dict(id=ident,field_relative=err,junction_relative=err,peak_relative=err,peak_error_K=err,capacity_L2_relative=err))
            for full in (False,True):
                for p in ('unit_steps','fast_mixed'):
                    for backend in ('dense','modal'):
                        timing.append(dict(id=ident,full=full,profile=p,backend=backend,seconds=1.))
        out=summarize(rec,metrics,timing)['comparisons']['field']
        self.assertEqual(out['stock_offline_id'],'stock_t0.001')
        self.assertAlmostEqual(out['methods']['coarse_qr']['offline_speedup'],.2)
        self.assertFalse(out['methods']['coarse_qr']['practical_gate'])

    def test_ci_actual_fine_solve_and_library_projection(self):
        import importlib.util
        if importlib.util.find_spec('pyamg') is None:
            self.skipTest('real PyAMG unavailable locally; exercised by CI, not stubbed')
        from metahotspot.macromodel import utils as u
        from run import extract_candidate
        f=DesignTests.family()
        for method in ('coarse_qr','defect_qr','jet','secant'):
            models,records,plan=extract_candidate(f,f,method,77,u,budgets=(6,),tolerances=(1e-6,))
            self.assertEqual(records[0]['full_solves'],6)
            b=next(iter(models.values())); V=b['V']
            np.testing.assert_allclose(V@V.T,np.eye(2),atol=1e-11)
            np.testing.assert_allclose(b['C'],V.T@f.C@V,atol=1e-11)


if __name__=='__main__': unittest.main()
