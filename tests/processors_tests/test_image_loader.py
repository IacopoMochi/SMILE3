import unittest
import os
from PIL import Image
import numpy as np

from app.processors.image_loader import ImageLoader


class TestImageLoader(unittest.TestCase):
    def setUp(self):
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
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_folder):
            os.rmdir(self.test_folder)

    def test_load_image(self):
        image_loader = ImageLoader(self.test_folder, self.test_file_name)
        loaded_image = image_loader.load_image()
        expected_image = np.rot90(self.test_image, 3)

        np.testing.assert_array_equal(loaded_image, expected_image)


if __name__ == '__main__':
    unittest.main()
