import unittest
from copy import copy
import numpy as np
from unittest.mock import patch, MagicMock

from app.models.image_container import Image
from app.utils.processors_service import edge_consolidation, edge_mean_subtraction
from app.utils.psd import Palasantzas_2_minimize, Palasantzas_2_beta, Palasantzas_2b
from app.processors.image_processors import PreProcessor, EdgeDetector, MetricCalculator, PostProcessor


class MockImagePreProcessor:
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
        image_data = np.random.rand(100, 100)
        parameters = {
            "X1": 10,
            "X2": 90,
            "Y1": 10,
            "Y2": 90
        }
        self.image = MockImagePreProcessor(image_data, parameters)
        self.preprocessor = PreProcessor(self.image)

    def test_crop_and_rotate_image(self):
        cropped_rotated_image = self.preprocessor.crop_and_rotate_image()
        self.assertEqual(cropped_rotated_image.shape, (80, 80))

    def test_remove_brightness_gradient(self):
        flattened_image = self.preprocessor.remove_brightness_gradient()

        self.assertEqual(flattened_image.shape, (80, 80))
        self.assertAlmostEqual(np.mean(flattened_image), 0, delta=0.3)
        self.assertNotEqual(np.std(flattened_image), np.std(self.image.image[10:90, 10:90]))

        original_cropped_image = self.image.image[10:90, 10:90]
        differences = np.sum(flattened_image != original_cropped_image)
        self.assertGreater(differences, 0)

    def test_normalize_image(self):
        self.preprocessor.normalize_image()
        self.assertIsNotNone(self.image.processed_image)
        self.assertAlmostEqual(np.max(self.image.processed_image), 1.0, places=5)
        self.assertAlmostEqual(np.min(self.image.processed_image), 0.0, places=5)

    def test_calculate_histogram_parameters(self):
        self.preprocessor.normalize_image()
        self.preprocessor.calculate_histogram_parameters()

        self.assertIsNotNone(self.image.intensity_histogram)
        self.assertIsNotNone(self.image.intensity_histogram_gaussian_fit_parameters)
        self.assertIsNotNone(self.image.intensity_histogram_low)
        self.assertIsNotNone(self.image.intensity_histogram_high)
        self.assertIsNotNone(self.image.intensity_histogram_medium)
        self.assertIsNotNone(self.image.lines_snr)


class MockImageEdgeDetector:
    def __init__(self):
        self.processed_image = np.random.rand(100, 100)
        self.parameters = {
            "EdgeRange": 5,
            "Edge_fit_function": "polynomial",
            "Threshold": 0.5,
            "brightEdge": False,
            "tone_positive_radiobutton": True
        }
        self.leading_edges = None
        self.trailing_edges = None
        self.consolidated_leading_edges = None
        self.consolidated_trailing_edges = None
        self.zero_mean_leading_edge_profiles = None
        self.zero_mean_trailing_edge_profiles = None
        self.number_of_lines = None
        self.critical_dimension = None
        self.critical_dimension_std_estimate = None
        self.critical_dimension_estimate = None
        self.pitch_estimate = None


class TestEdgeDetector(unittest.TestCase):

    def setUp(self):
        self.image = MockImageEdgeDetector()
        self.edge_detector = EdgeDetector(self.image)

    def test_edge_detection(self):
        new_edges = np.array([10, 20, 30])
        edges_profiles = np.nan * np.zeros([len(new_edges), self.image.processed_image.shape[1]])
        result = self.edge_detector.edge_detection(new_edges, edges_profiles)

        self.assertEqual(result.shape, edges_profiles.shape)
        self.assertFalse(np.isnan(result).all())

    def test_filter_and_reduce_noise(self):
        image_sum, image_sum_filtered = self.edge_detector.filter_and_reduce_noise()

        self.assertEqual(image_sum.shape[0], self.image.processed_image.shape[0])
        self.assertEqual(image_sum_filtered.shape[0], self.image.processed_image.shape[0])
        self.assertFalse(np.allclose(image_sum, image_sum_filtered))

    def test_detect_peaks(self):
        image_sum_derivative, peaks = self.edge_detector.detect_peaks()

        self.assertEqual(len(peaks), 2)
        self.assertIsInstance(peaks[0], np.ndarray)
        self.assertIsInstance(peaks[1], dict)

    @patch.object(EdgeDetector, 'detect_peaks')
    def test_classify_edges(self, mock_detect_peaks):
        mock_detect_peaks.return_value = (np.array([0, 2, 1, 2]), (np.array([0, 2]), {}))

        leading_edges, trailing_edges = self.edge_detector.classify_edges()

        expected_leading_edges = np.array([2.0])
        expected_trailing_edges = np.array([0.0])

        np.testing.assert_array_equal(leading_edges, expected_leading_edges)
        np.testing.assert_array_equal(trailing_edges, expected_trailing_edges)

    @patch.object(EdgeDetector, 'classify_edges')
    def test_find_leading_and_trailing_edges(self, mock_classify_edges):
        mock_classify_edges.return_value = (np.array([2.0, 3.0]), np.array([1.0, 3.0]))

        new_leading_edges, new_trailing_edges = self.edge_detector.find_leading_and_trailing_edges()

        self.assertEqual(new_leading_edges, [2.0])
        self.assertEqual(new_trailing_edges, [3.0])

    @patch.object(EdgeDetector, 'find_leading_and_trailing_edges')
    def test_determine_edge_profiles(self, mock_find_leading_and_trailing_edges):
        mock_find_leading_and_trailing_edges.return_value = (np.array([2.0, 3.0]), np.array([1.0, 3.0]))

        new_leading_edges, new_trailing_edges = self.edge_detector.determine_edge_profiles()
        self.assertEqual(new_leading_edges.shape, (2, self.edge_detector.image.processed_image.shape[1]))
        self.assertEqual(new_trailing_edges.shape, (2, self.edge_detector.image.processed_image.shape[1]))

    @patch.object(EdgeDetector, 'determine_edge_profiles')
    def test_find_edges(self, mock_determine_edge_profiles):
        mock_determine_edge_profiles.return_value = (
            np.array([[1.3, 1.4], [2.0, 2.2]]),
            np.array([[1.8, 2.0], [3.0, 4.1]])
        )
        self.edge_detector.find_edges()

        self.assertEqual(self.image.consolidated_leading_edges.shape, (2, 2))
        self.assertEqual(self.image.consolidated_trailing_edges.shape, (2, 2))
        self.assertEqual(self.image.zero_mean_leading_edge_profiles.shape, (2, 2))
        self.assertEqual(self.image.zero_mean_trailing_edge_profiles.shape, (2, 2))
        self.assertEqual(self.image.critical_dimension_std_estimate, 0.44999999999999984)
        self.assertEqual(self.image.critical_dimension_estimate, 0.9999999999999999)
        self.assertEqual(self.image.pitch_estimate, 1.2)


class MockImagePostProcessorAndMultiTaper:
    def __init__(self):
        self.processed_image = np.array([[21, 22, 23, 24], [25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]])

        self.consolidated_leading_edges = np.array([[5, 6], [7, 8]])
        self.consolidated_trailing_edges = np.array([[9, 10], [11, 12]])
        self.zero_mean_leading_edge_profiles = np.array([[13, 14], [15, 16]])
        self.zero_mean_trailing_edge_profiles = np.array([[17, 18], [19, 20]])

        self.basic_consolidated_leading_edges = np.array([[5, 6], [7, 8]])
        self.basic_consolidated_trailing_edges = np.array([[9, 10], [11, 12]])
        self.basic_zero_mean_leading_edge_profiles = np.array([[13, 14], [15, 16]])
        self.basic_zero_mean_trailing_edge_profiles = np.array([[17, 18], [19, 20]])

        self.post_processing_cache = None
        self.multi_taper_cache = None


class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.image = MockImagePostProcessorAndMultiTaper()
        self.processor = PostProcessor(self.image)

    def test_post_processing_with_cache(self):
        self.image.post_processing_cache = (
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
            np.array([[13, 14], [15, 16]])
        )

        with patch.object(self.processor, 'restore_cache', wraps=self.processor.restore_cache) as mock_restore_cache:
            self.processor.post_processing(True)

            mock_restore_cache.assert_called_once()

            np.testing.assert_array_equal(self.image.consolidated_leading_edges, np.array([[1, 2], [3, 4]]))
            np.testing.assert_array_equal(self.image.consolidated_trailing_edges, np.array([[5, 6], [7, 8]]))
            np.testing.assert_array_equal(self.image.zero_mean_leading_edge_profiles, np.array([[9, 10], [11, 12]]))
            np.testing.assert_array_equal(self.image.zero_mean_trailing_edge_profiles, np.array([[13, 14], [15, 16]]))

    def test_post_processing_without_cache(self):
        with patch.object(self.processor,
                          'calculate_new_post_processed_consolidated_edges') as mock_calculate_consolidated, \
                patch.object(self.processor, 'calculate_new_post_processed_zero_mean_edges',
                             wraps=self.processor.calculate_new_post_processed_zero_mean_edges) as mock_calculate_zero_mean, \
                patch.object(self.processor, 'store_cache', wraps=self.processor.store_cache) as mock_store_cache:
            self.processor.post_processing(True)

            mock_calculate_consolidated.assert_called_once()
            mock_calculate_zero_mean.assert_called_once()
            mock_store_cache.assert_called_once()

    def test_post_processing_without_use_post_processing(self):
        with patch.object(self.processor, 'restore_base_attributes',
                          wraps=self.processor.restore_base_attributes) as mock_restore_base:
            self.processor.post_processing(False)

            mock_restore_base.assert_called_once()


class MockImageMetricCalculator:
    def __init__(self):
        self.parameters = {
            "PixelSize": 0.1,
        }
        self.consolidated_leading_edges = None
        self.consolidated_trailing_edges = None
        self.zero_mean_leading_edge_profiles = None
        self.zero_mean_trailing_edge_profiles = None
        self.pixel_size = None
        self.frequency = None
        self.LWR_PSD = None
        self.LER_PSD = None
        self.LER_Leading_PSD = None
        self.LER_Trailing_PSD = None


class TestMetricCalculator(unittest.TestCase):

    def setUp(self):
        self.image = MockImageMetricCalculator()
        self.metric_calculator = MetricCalculator(self.image)

    def test_setup_frequency(self):
        self.image.consolidated_leading_edges = np.zeros((10, 10))

        self.metric_calculator.setup_frequency()

        self.assertEqual(self.image.pixel_size, 0.1)
        expected_frequency = np.array([0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        np.testing.assert_almost_equal(self.image.frequency, expected_frequency)

    def test_select_psd_model(self):
        self.image.parameters["PSD_model"] = "Palasantzas 2"

        model, model_beta, model_2 = self.metric_calculator.select_psd_model()

        self.assertIs(model, Palasantzas_2_minimize)
        self.assertIs(model_beta, Palasantzas_2_beta)
        self.assertIs(model_2, Palasantzas_2b)

        self.image.parameters["PSD_model"] = "Invalid Model"
        with self.assertRaises(ValueError) as context:
            self.metric_calculator.select_psd_model()
        self.assertEqual(str(context.exception), 'Please select valid model')

    def test_calculate_and_fit_psd(self):
        input_data = np.array([
            [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0]
        ])

        self.image.parameters = {
            "High_frequency_cut": 2,
            "High_frequency_average": 3,
            "Low_frequency_cut": 0,
            "Low_frequency_average": 5,
            "Correlation_length": 1.5,
            "Alpha": 0.5,
            "PSD_model": "Palasantzas 2"
        }

        self.image.frequency = np.linspace(0.1, 10, 4).astype(np.float32)

        PSD, PSD_fit_parameters, PSD_fit, PSD_unbiased, PSD_fit_unbiased = self.metric_calculator.calculate_and_fit_psd(
            input_data)

        self.assertEqual(PSD_fit_parameters.shape, (4,))
        self.assertEqual(PSD_fit.shape, (4,))
        self.assertEqual(PSD_unbiased.shape, (4,))
        self.assertEqual(PSD_fit_unbiased.shape, (4,))
        self.assertEqual(PSD_fit[0], 2250000)
        self.assertAlmostEqual(PSD_fit_unbiased[1], 260119.6, places=1)
        self.assertAlmostEqual(PSD[2], 750000, 1)


if __name__ == '__main__':
    unittest.main()
