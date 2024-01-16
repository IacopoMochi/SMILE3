import sys
import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
import pyqtgraph as pg

from smile_lines_image_class import SmileLinesImage
from smile_image_list_class import LineImageList


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("SMILE3.ui", self)
        self.line_image_list = LineImageList()
        self.pushButton_ImageFolder.pressed.connect(self.load_lines_image)
        self.process_lines_button.pressed.connect(self.process_line_images)
        self.linesTable.cellClicked.connect(self.navigateLinesTable)

    def display_lines_data(self, Image):

        # Display lines image on the top axis of the lines tab
        # Clean display
        plot_item = self.line_image_view.getPlotItem()
        plot_item.clear()
        # Check if a processed image exists
        if (Image.processed == True) and not (Image.processed_image is None):
            # Display image
            image_item = pg.ImageItem(np.array(Image.processed_image))
            self.line_image_view.addItem(image_item)
            # Display profiles
            # Display additional stuff (errors, defects, etc)

        else:
            image_item = pg.ImageItem(np.array(Image.image))
            self.line_image_view.addItem(image_item)

        # Display selected metric on the bottom axis of the lines tab
        # Check if the image has been processed
        if Image.processed and not (Image.processed_image is None):
            if Image.parameters["Histogram"]:
                histogram_plot = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram)
                histogram_plot_low = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_low)
                histogram_plot_medium = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_medium)
                histogram_plot_high = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_high)
                self.metric_plot.clear()
                self.metric_plot.addItem(histogram_plot)
                self.metric_plot.addItem(histogram_plot_low)
                self.metric_plot.addItem(histogram_plot_medium)
                self.metric_plot.addItem(histogram_plot_high)

        # image_view = self.line_image_view.getView()
        # pci = pg.PlotCurveItem(x=[1, 50, 100, 150, 200], y=[1, 50, 100, 150, 200])
        # image_view.addItem(pci)

    def process_line_images(self):
        for lines_image in self.line_image_list.lineImages:
            if self.linesTable.item(lines_image.id,0).checkState() == Qt.CheckState.Checked:
                self.gather_parameters(lines_image)
                self.line_image_list.current_image = lines_image.id
                lines_image.pre_processing
                lines_image.find_edges

            lines_image.processed = True

            item_processed = QtWidgets.QTableWidgetItem()
            item_processed.setCheckState(Qt.CheckState.Checked)
            self.linesTable.setItem(lines_image.id, 1, item_processed)
            self.display_lines_data(lines_image)

    def gather_parameters(self, Image):
        parameters = {'Threshold': np.double(window.threshold_line_edit.text()),
                      'MinPeakDistance': np.double(window.minPeakDistance_line_edit.text()),
                      'MinPeakProminence': np.double(window.minPeakProminence_line_edit.text()),
                      'PixelSize': np.double(window.pixelSize_line_edit.text()),
                      'X1': np.double(window.X1.text()),
                      'X2': np.double(window.X2.text()),
                      'Y1': np.double(window.Y1.text()),
                      'Y2': np.double(window.Y2.text()),
                      'tone_positive_radiobutton': window.tone_positive_radiobutton.isChecked(),
                      'brightEdge': window.brightEdge.isChecked(),
                      'Histogram': window.histogram.isChecked()
                      }
        Image.parameters = parameters

    def navigateLinesTable(self, nrow, ncol):
        self.line_image_list.current_image = nrow
        self.display_lines_data(self.line_image_list.lineImages[nrow])

    def load_lines_image(self):
        self.line_image_list = LineImageList()
        select_folder_dialog = QtWidgets.QFileDialog()
        select_folder_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        if select_folder_dialog.exec():
            file_names = select_folder_dialog.selectedFiles()
            cnt = -1
            for root, dirs, files in os.walk(file_names[0], topdown=True):
                for name in files:
                    if (
                            name.endswith(".tif")
                            | name.endswith(".jpg")
                            | name.endswith(".png")
                    ):
                        cnt += 1
                        # self.linesTable.setRowCount (cnt+2)
                        image_object = SmileLinesImage(cnt, name, root, "lines")
                        self.gather_parameters(image_object)
                        self.line_image_list.lineImages.append(
                            image_object
                        )
                        item_name = QtWidgets.QTableWidgetItem(image_object.file_name)

                        item_selected = QtWidgets.QTableWidgetItem()
                        item_selected.setFlags(
                            Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                        )
                        item_selected.setCheckState(Qt.CheckState.Checked)

                        item_processed = QtWidgets.QTableWidgetItem()
                        item_processed.setFlags(
                            Qt.ItemFlag.ItemIsEnabled
                        )
                        item_processed.setCheckState(Qt.CheckState.Unchecked)
                        self.linesTable.setRowCount(cnt + 2)
                        self.linesTable.setItem(cnt, 0, item_selected)
                        self.linesTable.setItem(cnt, 1, item_processed)
                        self.linesTable.setItem(cnt, 2, item_name)
                        self.display_lines_data(image_object)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
