import unittest
from unittest.mock import MagicMock
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication
from app.view.display_image import ImageDisplayManager
from app.models.image_container import Image


class TestImageDisplayManager(unittest.TestCase):
    """
    Test case for the ImageDisplayManager class.

    This test case ensures that the ImageDisplayManager class correctly handles the display of images
    on different tabs in the GUI.
    """

    def setUp(self):
        """
        Set up the test environment.

        Creates mock objects for PlotWidget and Image. Initializes the ImageDisplayManager with these mocks.
        """
        self.app = QApplication([])
        self.mock_widget_parameters_tab = MagicMock(spec=pg.PlotWidget)
        self.mock_widget_lines_tab = MagicMock(spec=pg.PlotWidget)
        self.mock_widget_parameters_tab.clear = MagicMock()
        self.mock_widget_parameters_tab.addItem = MagicMock()
        self.mock_widget_lines_tab.clear = MagicMock()
        self.mock_widget_lines_tab.addItem = MagicMock()

        self.image_display_manager = ImageDisplayManager(self.mock_widget_parameters_tab, self.mock_widget_lines_tab)

        self.mock_image = MagicMock(spec=Image)
        self.mock_image.image = np.array([[0, 1], [2, 3]])
        self.mock_image.processed_image = np.array([[3, 2], [1, 0]])

    def test_display_image_on_parameters_tab(self):
        """
        Test the display_image_on_parameters_tab method.

        Verifies that the image is correctly displayed on the parameters tab.
        """
        self.image_display_manager.display_image_on_parameters_tab(self.mock_image)
        self.mock_widget_parameters_tab.clear.assert_called_once()
        self.mock_widget_parameters_tab.addItem.assert_called()

    def test_display_image_on_lines_tab_with_processed_image(self):
        """
        Test the display_image_on_lines_tab method with a processed image.

        Verifies that the processed image is correctly displayed on the lines tab.
        """
        self.image_display_manager.display_image_on_lines_tab(self.mock_image)
        self.mock_widget_lines_tab.clear.assert_called_once()
        self.mock_widget_lines_tab.addItem.assert_called()

    def test_display_image_on_lines_tab_without_processed_image(self):
        """
        Test the display_image_on_lines_tab method without a processed image.

        Verifies that the original image is correctly displayed on the lines tab when the processed image is not available.
        """
        self.mock_image.processed_image = None
        self.image_display_manager.display_image_on_lines_tab(self.mock_image)
        self.mock_widget_lines_tab.clear.assert_called_once()
        self.mock_widget_lines_tab.addItem.assert_called()


if __name__ == '__main__':
    unittest.main()
