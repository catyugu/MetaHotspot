"""Regression for integer-valued shift arguments; scientific shifts are floats."""
import unittest
import numpy as np
from operators import assemble


class NumericTypesTests(unittest.TestCase):
    def test_integer_shift_does_not_truncate_conductances(self):
        k=np.array([[1.3,2.9],[5.1,8.7]])
        np.testing.assert_allclose(assemble(k,5).toarray(),assemble(k,5.).toarray(),atol=0,rtol=0)
