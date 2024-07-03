import unittest
import numpy as np
from app.utils.processors_service import edge_consolidation, edge_mean_subtraction


class TestProcessorService(unittest.TestCase):

    def test_edge_consolidation(self):
        raw_edge_profiles = np.array([
            [2, 3, np.nan, 4],
            [np.nan, np.nan, 3, 4],
            [5, 6, 7, 8]
        ])

        expected_result = np.array([
            [2, 3, 3, 4],
            [3.5, 3.5, 3, 4],
            [5, 6, 7, 8]
        ])

        result = edge_consolidation(raw_edge_profiles)

        np.testing.assert_array_almost_equal(result, expected_result)

    def test_edge_mean_subtraction(self):
        absolute_edge_profiles = np.random.rand(3, 100)
        zero_mean_profiles = edge_mean_subtraction(absolute_edge_profiles)
        self.assertAlmostEqual(np.mean(zero_mean_profiles), 0, places=5)


if __name__ == '__main__':
    unittest.main()