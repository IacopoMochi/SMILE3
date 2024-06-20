import unittest
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow
from pyqtgraph import PlotWidget
from unittest.mock import MagicMock

from app.models.image_container import Image
from app.view.display_calculation_result_images import ResultImagesManager


class TestResultImagesManager(unittest.TestCase):

    def setUp(self):
        self.app = QApplication([])

        self.mock_plot_widget_parameters_tab = MagicMock(spec=PlotWidget)
        self.mock_plot_widget_lines_tab = MagicMock(spec=PlotWidget)
        self.mock_widget_metric_tab = MagicMock(spec=PlotWidget)
        self.window = MagicMock()

        self.result_manager = ResultImagesManager(
            self.mock_plot_widget_parameters_tab,
            self.mock_plot_widget_lines_tab,
            self.mock_widget_metric_tab,
            self.window
        )

        self.mock_image = MagicMock(spec=Image)
        self.mock_image.leading_edges = np.array([[1, 2, 3], [4, 5, 6]])
        self.mock_image.trailing_edges = np.array([[7, 8, 9], [10, 11, 12]])
        self.mock_image.intensity_histogram = np.random.rand(256)
        self.mock_image.intensity_histogram_low = np.random.rand(256)
        self.mock_image.intensity_histogram_medium = np.random.rand(256)
        self.mock_image.intensity_histogram_high = np.random.rand(256)
        self.mock_image.frequency = np.linspace(0, 100, 100)
        self.mock_image.LWR_PSD = np.random.rand(100)
        self.mock_image.LWR_PSD_fit = np.random.rand(100)
        self.mock_image.LWR_PSD_unbiased = np.random.rand(100)
        self.mock_image.LWR_PSD_fit_unbiased = np.random.rand(100)

        self.mock_image.LER_PSD = np.array([1, 2, 3, 4])
        self.mock_image.LER_PSD_fit = np.array([1.1, 2.1, 3.1, 4.1])
        self.mock_image.LER_PSD_unbiased = np.array([0.9, 1.9, 2.9, 3.9])
        self.mock_image.LER_PSD_fit_unbiased = np.array([1.05, 2.05, 3.05, 4.05])
        self.mock_image.LER_Leading_PSD = np.array([1, 2, 3, 4])
        self.mock_image.LER_Leading_PSD_fit = np.array([1.1, 2.1, 3.1, 4.1])
        self.mock_image.LER_Leading_PSD_unbiased = np.array([0.9, 1.9, 2.9, 3.9])
        self.mock_image.LER_Leading_PSD_fit_unbiased = np.array([1.05, 2.05, 3.05, 4.05])
        self.mock_image.LER_Trailing_PSD = np.array([1, 2, 3, 4])
        self.mock_image.LER_Trailing_PSD_fit = np.array([1.1, 2.1, 3.1, 4.1])
        self.mock_image.LER_Trailing_PSD_unbiased = np.array([0.9, 1.9, 2.9, 3.9])
        self.mock_image.LER_Trailing_PSD_fit_unbiased = np.array([1.05, 2.05, 3.05, 4.05])

        self.mock_widget_metric_tab.clear = MagicMock()
        self.mock_widget_metric_tab.addItem = MagicMock()

    def tearDown(self):
        self.app.quit()

    def test_display_profiles_on_lines_tab(self):

        self.result_manager.display_profiles_on_lines_tab(self.mock_image)

        self.mock_plot_widget_lines_tab.addItem.assert_called()
        self.assertEqual(len(self.mock_plot_widget_lines_tab.addItem.call_args_list),
                         len(self.mock_image.leading_edges) + len(self.mock_image.trailing_edges))

    def test_display_histogram_on_metric_tab(self):
        self.result_manager.window.histogram.isChecked.return_value = True
        self.result_manager.display_plot_on_metric_tab(self.mock_image)

        self.mock_widget_metric_tab.clear.assert_called_once()
        self.mock_widget_metric_tab.addItem.assert_called()
        self.assertEqual(self.mock_widget_metric_tab.addItem.call_count, 5)

    def test_display_psd_on_metric_tab(self):
        self.result_manager.window.lineWidthPSD.isChecked.return_value = True

        self.result_manager.display_plot_on_metric_tab(self.mock_image)

        self.mock_widget_metric_tab.clear.assert_called_once()
        self.mock_widget_metric_tab.addItem.assert_called()
        self.assertEqual(self.mock_widget_metric_tab.addItem.call_count, 5)


if __name__ == '__main__':
    unittest.main()
