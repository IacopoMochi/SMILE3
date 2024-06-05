from PyQt6 import QtWidgets, uic
import pyqtgraph as pg
import numpy as np


class ImageDisplayManager(QtWidgets.QWidget):
    def __init__(self, plot_widget_parameters_tab, plot_widget_lines_tab):
        self.plot_widget_parameters_tab = plot_widget_parameters_tab
        self.plot_widget_lines_tab = plot_widget_lines_tab

    def display_image_on_parameters_tab(self, image):
        self.plot_widget_parameters_tab.clear()
        if image is not None:
            image_item = pg.ImageItem(np.array(image.image))
            self.plot_widget_parameters_tab.addItem(image_item)
        ROI = pg.RectROI((0, 0), (100, 100))
        self.plot_widget_parameters_tab.addItem(ROI)

    def display_image_on_lines_tab(self, image):
        self.plot_widget_lines_tab.clear()
        if image.processed_image is not None:
            image_item = pg.ImageItem(np.array(image.processed_image))
        else:
            image_item = pg.ImageItem(np.array(image.image))
        self.plot_widget_lines_tab.addItem(image_item)
