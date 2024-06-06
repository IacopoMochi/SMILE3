from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import numpy as np


class ImageDisplayManager(QtWidgets.QWidget):
    def __init__(self, widget_parameters_tab, widget_lines_tab):
        self.widget_parameters_tab = widget_parameters_tab
        self.widget_lines_tab = widget_lines_tab

    def display_image_on_parameters_tab(self, image):
        self.widget_parameters_tab.clear()
        if image is not None:
            image_item = pg.ImageItem(np.array(image.image))
            self.widget_parameters_tab.addItem(image_item)
        ROI = pg.RectROI((0, 0), (100, 100))
        self.widget_parameters_tab.addItem(ROI)

    def display_image_on_lines_tab(self, image):
        self.widget_lines_tab.clear()
        if image.processed_image is not None:
            image_item = pg.ImageItem(np.array(image.processed_image))
        else:
            image_item = pg.ImageItem(np.array(image.image))
        self.widget_lines_tab.addItem(image_item)
