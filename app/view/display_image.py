from PyQt6 import QtWidgets
import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import pyqtSignal
from pyqtgraph import PlotWidget

from app.models.image_container import Image


class ImageDisplayManager(QtWidgets.QWidget):
    """
    Manages the display of images on various tabs in the GUI.

    Attributes:
        error_signal (pyqtSignal): Signal to emit error messages.
        widget_parameters_tab (PlotWidget): The widget for displaying parameter images.
        widget_lines_tab (PlotWidget): The widget for displaying line images.
    """
    error_signal = pyqtSignal(str)

    def __init__(self, widget_parameters_tab: PlotWidget, widget_lines_tab: PlotWidget):
        super().__init__()
        self.widget_parameters_tab = widget_parameters_tab
        self.widget_lines_tab = widget_lines_tab
        self.roi = None

    def display_image_on_parameters_tab(self, image: Image) -> None:
        """
        Displays the image on the parameters tab.

        Args:
            image (Image): The image object to display.
        """
        try:
            self.widget_parameters_tab.clear()
            if image is not None:
                image_item = pg.ImageItem(np.array(image.image))
                self.widget_parameters_tab.addItem(image_item)

        except Exception as e:
            self.error_signal.emit(f"Error occurred while displaying the image: {str(e)}")

    def set_roi(self, x1_widget, x2_widget, y1_widget, y2_widget):
        self.roi = pg.RectROI((0, 0), (100, 100))
        self.widget_parameters_tab.addItem(self.roi)
        self.roi.sigRegionChanged.connect(lambda: self.update_roi_coordinates(x1_widget, x2_widget, y1_widget, y2_widget))

    def update_roi_coordinates(self, x1_widget, x2_widget, y1_widget, y2_widget):
        """
        Update the coordinates of the ROI to the corresponding text fields.
        """

        pos = self.roi.pos()
        size = self.roi.size()

        x1, y1 = pos.x(), pos.y()
        x2, y2 = x1 + size.x(), y1 + size.y()

        x1_widget.setText(str(int(x1)))
        x2_widget.setText(str(int(x2)))
        y1_widget.setText(str(int(y1)))
        y2_widget.setText(str(int(y2)))

    def display_image_on_lines_tab(self, image: Image) -> None:
        """
        Displays the image on the lines tab.

        Args:
            image (Image): The image object to display.
        """
        try:
            self.widget_lines_tab.clear()
            if image.processed_image is not None:
                image_item = pg.ImageItem(np.array(image.processed_image))
            else:
                image_item = pg.ImageItem(np.array(image.image))
            self.widget_lines_tab.addItem(image_item)
        except Exception as e:
            self.error_signal.emit(f"Error occurred while displaying the image: {str(e)}")
