import unittest
import numpy as np
from unittest.mock import MagicMock

from app.utils.psd import (
    Palasantzas_2_beta,
    Palasantzas_2_minimize,
    Palasantzas_2,
    Palasantzas_2b,
    Palasantzas_1,
    Gaussian,
    NoWhiteNoise
)


class TestModels(unittest.TestCase):
    """
    Unit tests for PSD model functions.
    """

    def setUp(self):
        """
        Set up the test environment.

        Initializes mock image parameters and sets up numpy arrays for PSD and frequency data.
        """
        self.mock_image = MagicMock()
        self.mock_image.image.parameters = {
            "High_frequency_cut": 10,
            "High_frequency_average": 5,
            "Low_frequency_cut": 1,
            "Low_frequency_average": 2,
            "Correlation_length": 1.5,
            "Alpha": 0.5
        }

        self.PSD = np.random.rand(20).astype(np.float32)
        self.freq = np.linspace(0.1, 10, 20).astype(np.float32)
        self.beta = [1.0, 2.0, 0.1, 0.5]

    def test_Palasantzas_2_beta(self):
        """
        Test the Palasantzas_2_beta function.

        Verifies that the function returns the correct length of beta, beta_min, and beta_max lists,
        and checks specific values or ranges.
        """
        beta, beta_min, beta_max = Palasantzas_2_beta(self.mock_image, self.PSD)
        self.assertEqual(len(beta), 4)
        self.assertEqual(len(beta_min), 4)
        self.assertEqual(len(beta_max), 4)
        self.assertGreater(beta[0], 0)
        self.assertEqual(beta[1], self.mock_image.image.parameters["Correlation_length"])

    def test_Palasantzas_2_minimize(self):
        """
        Test the Palasantzas_2_minimize function.

        Verifies that the function returns a floating-point value and checks that it is grater or equal to zero.
        """
        result = Palasantzas_2_minimize(self.beta, self.freq, self.PSD)
        self.assertIsInstance(result, np.floating)
        self.assertGreaterEqual(result, 0)

    def test_Palasantzas_2(self):
        """
        Test the Palasantzas_2 function.

        Verifies that the function returns an array with the correct shape and that all values are non-negative.
        """
        result = Palasantzas_2(self.freq, *self.beta)
        self.assertEqual(result.shape, self.freq.shape)
        self.assertTrue(np.all(result >= 0))

    def test_Palasantzas_2b(self):
        """
        Test the Palasantzas_2b function.

        Verifies that the function returns an array with the correct shape and that all values are non-negative.
        """
        result = Palasantzas_2b(self.freq, self.beta)
        self.assertEqual(result.shape, self.freq.shape)
        self.assertTrue(np.all(result >= 0))

    def test_Palasantzas_1(self):
        """
        Test the Palasantzas_1 function.

        Verifies that the function returns an array with the correct shape and that all values are non-negative.
        """
        beta = [1.0, 2.0, 0.1, 0.5, 0.2]
        result = Palasantzas_1(self.freq, *beta)
        self.assertEqual(result.shape, self.freq.shape)
        self.assertTrue(np.all(result >= 0))

    def test_Gaussian(self):
        """
        Test the Gaussian function.

        Verifies that the function returns an array with the correct shape and that all values are non-negative.
        """
        beta = [1.0, 2.0, 0.1]
        result = Gaussian(self.freq, *beta)
        self.assertEqual(result.shape, self.freq.shape)
        self.assertTrue(np.all(result >= 0))

    def test_NoWhiteNoise(self):
        """
        Test the NoWhiteNoise function.

        Verifies that the function returns an array with the correct shape and that all values are non-negative.
        """
        beta = [1.0, 2.0, 0.5]
        result = NoWhiteNoise(self.freq, *beta)
        self.assertEqual(result.shape, self.freq.shape)
        self.assertTrue(np.all(result >= 0))


if __name__ == '__main__':
    unittest.main()
