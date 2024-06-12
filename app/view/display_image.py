from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import pyqtSignal


class ImageDisplayManager(QtWidgets.QWidget):
    error_signal = pyqtSignal(str)

    def __init__(self, widget_parameters_tab, widget_lines_tab):
        super().__init__()
        self.widget_parameters_tab = widget_parameters_tab
        self.widget_lines_tab = widget_lines_tab

    def display_image_on_parameters_tab(self, image):
        try:
            self.widget_parameters_tab.clear()
            if image is not None:
                image_item = pg.ImageItem(np.array(image.image))
                self.widget_parameters_tab.addItem(image_item)
            ROI = pg.RectROI((0, 0), (100, 100))
            self.widget_parameters_tab.addItem(ROI)
        except Exception as e:
            self.error_signal.emit(f"Error occurred while displaying the image: {str(e)}")

    def display_image_on_lines_tab(self, image):
        try:
            self.widget_lines_tab.clear()
            if image.processed_image is not None:
                image_item = pg.ImageItem(np.array(image.processed_image))
            else:
                image_item = pg.ImageItem(np.array(image.image))
            self.widget_lines_tab.addItem(image_item)
        except Exception as e:
            self.error_signal.emit(f"Error occurred while displaying the image: {str(e)}")
