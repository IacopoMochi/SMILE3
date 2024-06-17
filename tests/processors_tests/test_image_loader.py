import unittest
import os
from PIL import Image
import numpy as np

from app.processors.image_loader import ImageLoader


class TestImageLoader(unittest.TestCase):
    """
    Test case for the ImageLoader class.

    This test case ensures that the ImageLoader class correctly loads and processes images.
    It includes setup and teardown methods to create and clean up a test image, and a test method
    to verify the image loading and rotation functionality.
    """

    def setUp(self):
        """
        Set up the test environment.

        Creates a test folder and a test image file before each test.
        The test image is a simple 3x3 NumPy array saved as a PNG file.
        """

        self.test_folder = 'test_images'
        self.test_file_name = 'test_image.png'
        self.test_file_path = os.path.join(self.test_folder, self.test_file_name)

        os.makedirs(self.test_folder, exist_ok=True)

        self.test_image = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ], dtype=np.uint8)

        pil_image = Image.fromarray(self.test_image)
        pil_image.save(self.test_file_path)

    def tearDown(self):
        """
        Tear down the test environment.

        Removes the test image file and the test folder after each test.
        """

        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_folder):
            os.rmdir(self.test_folder)

    def test_load_image(self):
        """
        Test the load_image method of the ImageLoader class.

        Verifies that the image is loaded correctly and rotated by 270 degrees counterclockwise.
        """

        image_loader = ImageLoader(self.test_folder, self.test_file_name)
        loaded_image = image_loader.load_image()
        expected_image = np.rot90(self.test_image, 3)

        np.testing.assert_array_equal(loaded_image, expected_image)


if __name__ == '__main__':
    unittest.main()
