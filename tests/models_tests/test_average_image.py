import unittest
import numpy as np
import copy

from app.models.images_list import ImagesList
from app.models.image_container import Image
from app.models.average_image import AverageImage


class TestAverageImage(unittest.TestCase):
    """
    Unit tests for the AverageImage class.
    """

    def setUp(self):
        self.images_list = ImagesList()
        self.image1 = Image(id=1, path="/path/to/image1", file_name="image1.jpg")
        self.image2 = Image(id=2, path="/path/to/image2", file_name="image2.jpg")

        self.image1.consolidated_leading_edges = np.array([1, 2])
        self.image1.consolidated_trailing_edges = np.array([3, 4])
        self.image1.frequency = np.linspace(0.1, 10, 20).astype(np.float32)
        self.image1.zero_mean_leading_edge_profiles = np.array([9, 10])
        self.image1.zero_mean_trailing_edge_profiles = np.array([11, 12])

        self.image2.consolidated_leading_edges = np.array([5, 6])
        self.image2.consolidated_trailing_edges = np.array([7, 8])
        self.image2.frequency = np.linspace(0.1, 10, 20).astype(np.float32)
        self.image2.zero_mean_leading_edge_profiles = np.array([13, 14])
        self.image2.zero_mean_trailing_edge_profiles = np.array([15, 16])

        self.images_list.images_list = [self.image1, self.image2]

    def test_average_image_gather_edges(self):
        """
        Test that the gather_edges method correctly concatenates leading edges and ensures trailing edges are not None.
        """

        avg_image = AverageImage(self.images_list)
        avg_image.gather_edges()

        expected_leading_edges = np.concatenate([
            self.image1.consolidated_leading_edges,
            self.image2.consolidated_leading_edges
        ])
        np.testing.assert_array_equal(avg_image.image.consolidated_leading_edges, expected_leading_edges)
        self.assertIsNotNone(avg_image.image.consolidated_trailing_edges)

    def test_frequency_not_consistent(self):
        """
        Test that ValueError is raised when images have inconsistent frequencies.
        """

        self.image2.frequency = np.linspace(0.1, 15, 20).astype(np.float32)

        with self.assertRaises(ValueError) as context:
            avg_image = AverageImage(self.images_list)
            avg_image.gather_edges()

        self.assertEqual(str(context.exception), 'Frequency not consistent')


if __name__ == '__main__':
    unittest.main()
