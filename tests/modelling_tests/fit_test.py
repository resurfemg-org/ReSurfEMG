"""
Sanity tests for the modelling.fit module of the ReSurfEMG library.
"""

import unittest
import numpy as np

from resurfemg.modelling import fit



class TestExplainedVariance(unittest.TestCase):
    def test_explained_variance(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        result = fit.explained_variance(y_true, y_pred)
        # ss_tot = 2^2 + 1^2 + 0 + 1^2 + 1^2 = 10.0
        # ss_res = 5 * 0.01 = 0.05
        # explained_variance = 1 - (0.05 / 10.0) = 0.995
        self.assertEqual(result, 0.995)


class TestResidualStandardError(unittest.TestCase):
    def test_residual_standard_error(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true + 0.2
        result = fit.residual_standard_error(y_true, y_pred, p=2)
        # ss_res = 4 * 0.2 ** 2 = 4 * 0.04 = 0.16
        # rse = sqrt(0.16 / (4 - 2)) = 0.4 / sqrt(2) = 0.2828427124746184
        self.assertAlmostEqual(result, 0.4/np.sqrt(2), places=12)


class TestCheckModelFit(unittest.TestCase):
    def test_check_model_fit(self):
        # Create a simple linear model
        # y = 2x + 1
        x_input = np.array([[1], [2], [3], [4], [5]])
        y_true = np.array([3, 5, 7, 9, 11])
        mdl = fit.sm.OLS(y_true, fit.sm.add_constant(x_input)).fit()

        # Check the model fit
        result = {}
        result = fit.check_model_fit(
            mdl,
            x_input,
            y_true,
            include_constant_term=True,
            verbose=True)
        # R2
        self.assertAlmostEqual(result[0], 1.0, places=12)
        # CoD
        self.assertAlmostEqual(result[1], 1.0, places=12)
        # MAE
        self.assertAlmostEqual(result[2], 0.0, places=12)
        # Residuals
        np.testing.assert_array_almost_equal(
            result[3], np.zeros(result[3].shape), decimal=12)
