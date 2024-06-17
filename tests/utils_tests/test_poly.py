import unittest
import numpy as np
from app.utils.poly import _poly11, _poly22, _poly33, poly11, poly22, poly33, binary_image_histogram_model, \
    gaussian_profile


class TestPolynomials(unittest.TestCase):
    """
    Unit tests for polynomial functions of degrees 1, 2, and 3 in two variables (x and y).
    """

    def test_poly11(self):
        """
        Test case for _poly11 function, computing a degree 1 polynomial.
        """
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0)
        result = _poly11(M, *args)
        expected = np.array(1.0 * 1.0 + 2.0 * 2.0 + 3.0)
        np.testing.assert_array_equal(result, expected)

    def test_poly22(self):
        """
        Test case for _poly22 function, computing a degree 2 polynomial.
        """
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
        """
        Test case for _poly33 function, computing a degree 3 polynomial.
        """
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
        """
        Test case for poly11 function (alternative implementation), computing a degree 1 polynomial.
        """
        M = (1.0, 2.0)
        args = (1.0, 2.0, 3.0)
        result = poly11(M, args)
        expected = np.array(1.0 * 1.0 + 2.0 * 2.0 + 3.0)
        np.testing.assert_array_equal(result, expected)

    def test_poly22_alternate(self):
        """
        Test case for poly22 function (alternative implementation), computing a degree 2 polynomial.
        """
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
        """
        Test case for poly33 function (alternative implementation), computing a degree 3 polynomial.
        """
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
    """
    Unit tests for other utility functions.
    """

    def test_binary_image_histogram_model(self):
        """
        Test case for binary_image_histogram_model function, computing a binary image histogram model.
        """
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
        """
        Test case for gaussian_profile function, computing a Gaussian profile.
        """
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
