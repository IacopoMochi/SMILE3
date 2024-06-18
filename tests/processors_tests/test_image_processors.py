import unittest
import numpy as np
from unittest.mock import MagicMock
from app.utils.poly import _poly11, poly11, binary_image_histogram_model, gaussian_profile
from app.utils.psd import Palasantzas_2_minimize, Palasantzas_2_beta, Palasantzas_2b
from app.processors.image_processors import PreProcessor


class MockImage:
    def __init__(self, image, parameters):
        self.image = image
        self.parameters = parameters
        self.processed_image = None
        self.intensity_histogram = None
        self.intensity_histogram_gaussian_fit_parameters = None
        self.intensity_histogram_low = None
        self.intensity_histogram_high = None
        self.intensity_histogram_medium = None
        self.lines_snr = None


class TestPreProcessor(unittest.TestCase):
    def setUp(self):
        """
        Set up a mock image and PreProcessor instance before each test.

        This method initializes a mock image with random data and sets up
        a PreProcessor instance to be tested.
        """
        image_data = np.random.rand(100, 100)
        parameters = {
            "X1": 10,
            "X2": 90,
            "Y1": 10,
            "Y2": 90
        }
        self.image = MockImage(image_data, parameters)
        self.preprocessor = PreProcessor(self.image)

    def test_crop_and_rotate_image(self):
        """
        Test the crop_and_rotate_image method for correctness.

        This test verifies that the crop_and_rotate_image method correctly
        crops and rotates the image, ensuring the resulting image shape is
        (80, 80).
        """
        cropped_rotated_image = self.preprocessor.crop_and_rotate_image()
        self.assertEqual(cropped_rotated_image.shape, (80, 80))

    def test_remove_brightness_gradient(self):
        """
        Test the remove_brightness_gradient method for correctness.

        This test checks that the remove_brightness_gradient method correctly
        removes the brightness gradient from the image. It verifies that the
        resulting flattened image has a shape of (80, 80), the mean value is
        approximately 0, and the standard deviation is less than that of the
        original cropped image.
        """
        flattened_image = self.preprocessor.remove_brightness_gradient()

        self.assertEqual(flattened_image.shape, (80, 80))
        self.assertAlmostEqual(np.mean(flattened_image), 0, delta=0.3)
        self.assertLess(np.std(flattened_image), np.std(self.image.image[10:90, 10:90]))

        original_cropped_image = self.image.image[10:90, 10:90]
        differences = np.sum(flattened_image != original_cropped_image)
        self.assertGreater(differences, 0)

    def test_normalize_image(self):
        """
        Test the normalize_image method to ensure the image is normalized.

        This test verifies that the normalize_image method correctly normalizes
        the image, ensuring that the processed image has values between 0 and 1.
        It checks the maximum and minimum values of the processed image with
        precision up to 5 decimal places.
        """
        self.preprocessor.normalize_image()
        self.assertIsNotNone(self.image.processed_image)
        self.assertAlmostEqual(np.max(self.image.processed_image), 1.0, places=5)
        self.assertAlmostEqual(np.min(self.image.processed_image), 0.0, places=5)

    def test_calculate_histogram_parameters(self):
        """
        Test the calculate_histogram_parameters method for correctness.

        This test validates that the calculate_histogram_parameters method
        correctly computes histogram parameters after normalizing the image.
        It ensures that the intensity histogram and its Gaussian fit parameters,
        as well as other related attributes, are set and not None.
        """
        self.preprocessor.normalize_image()
        self.preprocessor.calculate_histogram_parameters()

        self.assertIsNotNone(self.image.intensity_histogram)
        self.assertIsNotNone(self.image.intensity_histogram_gaussian_fit_parameters)
        self.assertIsNotNone(self.image.intensity_histogram_low)
        self.assertIsNotNone(self.image.intensity_histogram_high)
        self.assertIsNotNone(self.image.intensity_histogram_medium)
        self.assertIsNotNone(self.image.lines_snr)


if __name__ == '__main__':
    unittest.main()
