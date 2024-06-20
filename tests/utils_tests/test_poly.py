import unittest
import numpy as np
from app.utils.poly import _poly11, _poly22, _poly33, poly11, poly22, poly33, binary_image_histogram_model, \
    gaussian_profile


class TestPolynomials(unittest.TestCase):
    def test_poly11(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0)
        result = _poly11(M, *args)
        expected = np.array(1.0 * 1.0 + 2.0 * 2.0 + 3.0)
        np.testing.assert_array_equal(result, expected)

    def test_poly22(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        result = _poly22(M, *args)
        expected = np.array(
            1.0 * 1.0 +
            2.0 * 2.0 +
            3.0 +
            4.0 * 1.0 ** 2 +
            5.0 * 1.0 * 2.0 +
            6.0 * 2.0 ** 2
        )
        np.testing.assert_array_equal(result, expected)

    def test_poly33(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        result = _poly33(M, *args)
        expected = np.array(
            1.0 * 1.0 +
            2.0 * 2.0 +
            3.0 +
            4.0 * 1.0 ** 2 +
            5.0 * 1.0 * 2.0 +
            6.0 * 2.0 ** 2 +
            7.0 * 1.0 ** 3 +
            8.0 * 2.0 * 1.0 ** 2 +
            9.0 * 1.0 * 2.0 ** 2 +
            10.0 * 2.0 ** 3
        )
        np.testing.assert_array_equal(result, expected)

    def test_poly11_alternate(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0)
        result = poly11(M, args)
        expected = np.array(1.0 * 1.0 + 2.0 * 2.0 + 3.0)
        np.testing.assert_array_equal(result, expected)

    def test_poly22_alternate(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        result = poly22(M, args)
        expected = np.array(
            1.0 * 1.0 +
            2.0 * 2.0 +
            3.0 +
            4.0 * 1.0 ** 2 +
            5.0 * 1.0 * 2.0 +
            6.0 * 2.0 ** 2
        )
        np.testing.assert_array_equal(result, expected)

    def test_poly33_alternate(self):
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        result = poly33(M, args)
        expected = np.array(
            1.0 * 1.0 +
            2.0 * 2.0 +
            3.0 +
            4.0 * 1.0 ** 2 +
            5.0 * 1.0 * 2.0 +
            6.0 * 2.0 ** 2 +
            7.0 * 1.0 ** 3 +
            8.0 * 2.0 * 1.0 ** 2 +
            9.0 * 1.0 * 2.0 ** 2 +
            10.0 * 2.0 ** 3
        )
        np.testing.assert_array_equal(result, expected)


class TestOtherFunctions(unittest.TestCase):

    def test_binary_image_histogram_model(self):
        x = 1.0
        beta = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        result = binary_image_histogram_model(x, *beta)
        # Compute expected result based on the function definition
        expected = np.array(
            1.0 * np.exp(-(((x - 2.0) / 3.0) ** 2)) +
            4.0 * np.exp(-(((x - 5.0) / 6.0) ** 2)) +
            7.0 * np.exp(-(((x - 8.0) / 9.0) ** 2))
        )
        np.testing.assert_array_equal(result, expected)

    def test_gaussian_profile(self):
        x = 1.0
        beta = (1.0, 2.0, 3.0)
        result = gaussian_profile(x, *beta)
        # Compute expected result based on the function definition
        expected = np.array(
            1.0 * np.exp(-(((x - 2.0) / 3.0) ** 2))
        )
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
