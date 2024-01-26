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
        if Image.processed and not (Image.processed_image is None):
            # Display image
            image_item = pg.ImageItem(np.array(Image.processed_image))
            self.line_image_view.addItem(image_item)
            # Display profiles
            edge_color = pg.mkColor(0, 200, 0)
            edge_pen = pg.mkPen(edge_color, width=3)

            if not (Image.leading_edges is None):
                for edge in Image.leading_edges:
                    leading_edge_plot = pg.PlotDataItem(edge, np.arange(0, Image.profiles_length), pen=edge_pen)
                    self.line_image_view.addItem(leading_edge_plot)
            if not (Image.trailing_edges is None):
                for edge in Image.trailing_edges:
                    trailing_edge_plot = pg.PlotDataItem(edge, np.arange(0,Image.profiles_length), pen=edge_pen)
                    self.line_image_view.addItem(trailing_edge_plot)
            # Display additional stuff (errors, defects, etc)

        else:
            image_item = pg.ImageItem(np.array(Image.image))
            self.line_image_view.addItem(image_item)

        # Display selected metric on the bottom axis of the lines tab
        # Check if the image has been processed
        if Image.processed and not (Image.processed_image is None):
            histogram_color = pg.mkColor(200, 200, 200)
            histogram_pen = pg.mkPen(histogram_color, width=3)
            histogram_fit_color = pg.mkColor(0, 20, 200)
            histogram_fit_pen = pg.mkPen(histogram_fit_color, width=3)
            histogram_curves_color = pg.mkColor(200, 0, 0)
            histogram_curves_pen = pg.mkPen(histogram_curves_color, width=3)

            if self.histogram.isChecked():
                histogram_plot = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram, pen=histogram_pen)
                histogram_plot_low = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_low, pen=histogram_curves_pen)
                histogram_plot_medium = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_medium, pen=histogram_curves_pen)
                histogram_plot_high = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_high, pen=histogram_curves_pen)
                histogram_plot_fit = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_high+Image.intensity_histogram_low+Image.intensity_histogram_medium,
                                                      pen=histogram_fit_pen)
                self.metric_plot.clear()
                self.metric_plot.addItem(histogram_plot)
                self.metric_plot.addItem(histogram_plot_low)
                self.metric_plot.addItem(histogram_plot_medium)
                self.metric_plot.addItem(histogram_plot_high)
                self.metric_plot.addItem(histogram_plot_fit)
                self.metric_plot.setLogMode(False, False)

            elif self.lineWidthPSD.isChecked():
                PSD_color = pg.mkColor(200, 200, 200)
                PSD_fit_color = pg.mkColor(0, 200, 200)
                PSD_unbiased_color = pg.mkColor(200, 0, 0)
                PSD_fit_unbiased_color = pg.mkColor(0, 200, 0)
                PSD_pen = pg.mkPen(PSD_color, width=3)
                PSD_fit_pen = pg.mkPen(PSD_fit_color, width=3)
                PSD_unbiased_pen = pg.mkPen(PSD_unbiased_color, width=3)
                PSD_fit_unbiased_pen = pg.mkPen(PSD_fit_unbiased_color, width=3)
                LW_PSD_plot = pg.PlotDataItem(Image.frequency, Image.LWR_PSD[0:len(Image.frequency)], pen=PSD_pen)
                LW_PSD_fit_plot = pg.PlotDataItem(Image.frequency, Image.LWR_PSD_fit[0:len(Image.frequency)], pen=PSD_fit_pen)
                LW_PSD_fit_plot0 = pg.PlotDataItem(Image.frequency, Image.LWR_PSD_model[0:len(Image.frequency)], pen=PSD_unbiased_pen)
                self.metric_plot.clear()
                self.metric_plot.addItem(LW_PSD_plot)
                self.metric_plot.addItem(LW_PSD_fit_plot)
                self.metric_plot.addItem(LW_PSD_fit_plot0)
                self.metric_plot.setLogMode(True, True)
                self.metric_plot.setAutoVisible(y=True)


            elif self.LineEdge_PSD.isChecked():
                LER_PSD_plot = pg.PlotDataItem(Image.frequency, Image.LER_PSD[0:len(Image.frequency)])
                self.metric_plot.clear()
                self.metric_plot.addItem(LER_PSD_plot)
                self.metric_plot.setLogMode(True, True)


        # image_view = self.line_image_view.getView()
        # pci = pg.PlotCurveItem(x=[1, 50, 100, 150, 200], y=[1, 50, 100, 150, 200])
        # image_view.addItem(pci)

    def process_line_images(self):
        # Count how many images have been selected for processing
        number_of_selected_images = 0
        for lines_image in self.line_image_list.lineImages:
            if self.linesTable.item(lines_image.id,0).checkState() == Qt.CheckState.Checked:
                number_of_selected_images += 1
        number_of_processed_images = 0
        self.image_progressBar.setMinimum(0)
        self.image_progressBar.setMaximum(number_of_selected_images)
        self.image_progressBar.setValue(0)

        for lines_image in self.line_image_list.lineImages:
            self.status_label.setText("Processing " + str(number_of_processed_images+1) + " of " + str(number_of_selected_images))
            if self.linesTable.item(lines_image.id,0).checkState() == Qt.CheckState.Checked:
                self.gather_parameters(lines_image)
                self.line_image_list.current_image = lines_image.id
                lines_image.pre_processing()
                lines_image.find_edges()
                lines_image.calculate_metrics()

                item_averageCD = QtWidgets.QTableWidgetItem(f"{lines_image.critical_dimension_estimate:{0}.{5}}")
                item_number_of_lines = QtWidgets.QTableWidgetItem(str(lines_image.number_of_lines))
                item_averageCDstd = QtWidgets.QTableWidgetItem(f"{lines_image.critical_dimension_std_estimate:{0}.{5}}")
                item_pitchEstimate = QtWidgets.QTableWidgetItem(f"{lines_image.pitch_estimate:{0}.{5}}")
                self.linesTable.setItem(lines_image.id, 3, item_number_of_lines)
                self.linesTable.setItem(lines_image.id, 4, item_pitchEstimate)
                self.linesTable.setItem(lines_image.id, 5, item_averageCD)
                self.linesTable.setItem(lines_image.id, 6, item_averageCDstd)
                lines_image.processed = True
                number_of_processed_images += 1
                self.image_progressBar.setValue(number_of_processed_images)
                QtWidgets.QApplication.processEvents()

            item_processed = QtWidgets.QTableWidgetItem()
            item_processed.setCheckState(Qt.CheckState.Checked)
            self.linesTable.setItem(lines_image.id, 1, item_processed)
            self.display_lines_data(lines_image)

            # Update the GUI for each processed image
            QtWidgets.QApplication.processEvents()
        self.status_label.setText("Ready")
    def gather_parameters(self, Image):

        if self.Polynomial.isChecked():
            edge_fit_function = 'polynomial'
        elif self.Linear.isChecked():
            edge_fit_function = 'linear'
        elif self.ThresholdEdge.isChecked():
            edge_fit_function = 'threshold'
        else:
            edge_fit_function = 'bright_edge'

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
                      'Edge_fit_function': edge_fit_function,
                      'CDFraction': np.double(window.CDFraction.text()),
                      'EdgeRange': np.double(window.EdgeRange.text()),
                      'High_frequency_cut': int(window.high_freq_cut.text()),
                      'Low_frequency_cut': int(window.low_freq_cut.text()),
                      'Low_frequency_average': int(window.low_freq_average.text()),
                      'High_frequency_average': int(window.high_freq_average.text()),
                      'Correlation_length': np.double(window.correlation_length.text()),
                      'Alpha': np.double(window.alpha.text()),
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
