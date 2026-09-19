"""Coverage and comparator controls added after the first scientific screen."""
import unittest
import numpy as np
from design import select_agenda


class FollowupTests(unittest.TestCase):
    def test_quota_really_exercises_derivatives_with_paid_parents(self):
        rng=np.random.default_rng(11)
        F=rng.normal(size=(12,40)); D=np.array([F*1e-5,F*2e-5])
        a=select_agenda(F,D,48,[0,1,2,3],derivative_fraction=.25)
        seen=set()
        for b in (24,48): self.assertEqual(sum(x.derivative>=0 for x in a[:b]),b//4)
        for x in a:
            if x.derivative>=0:self.assertIn(x.sample,seen)
            else:seen.add(x.sample)
        self.assertEqual(len(set(a)),48)

    def test_default_agenda_unchanged_and_invalid_quotas_rejected(self):
        F=np.eye(8); D=np.array([F])
        self.assertEqual(select_agenda(F,D,8,[0]),select_agenda(F,D,8,[0],derivative_fraction=0.))
        with self.assertRaises(ValueError):select_agenda(F,None,8,[0],derivative_fraction=.25)
        with self.assertRaises(ValueError):select_agenda(F,D,8,[0],derivative_fraction=1.1)

    def test_followup_strengthens_stock_without_dropping_original_controls(self):
        from run import study_settings
        initial,methods=study_settings(False); follow,expanded=study_settings(True)
        self.assertEqual(initial,(1e-2,1e-3,1e-4,1e-5))
        self.assertTrue(set(initial).issubset(follow))
        self.assertTrue(set(methods).issubset(expanded))
        self.assertIn('quota_jet',expanded);self.assertIn('quota_secant',expanded)
        self.assertEqual(len(follow),10)


if __name__=='__main__':unittest.main()
