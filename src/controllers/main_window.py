import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from openpyxl import Workbook

from src.models.smile_image_list_class import LineImageList
from src.models.image_container import SmileLinesImage



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("ui/SMILE3.ui", self)
        self.line_image_list = LineImageList(-1, 'Average', 'Empty')
        self.pushButton_ImageFolder.pressed.connect(self.load_lines_image)
        self.process_lines_button.pressed.connect(self.process_line_images)
        self.linesTable.cellClicked.connect(self.navigateLinesTable)
        self.metric_selection.clicked.connect(self.display_lines_data)
        self.export_data.clicked.connect(self.export_lines_data)

    # saves data to excel file
    def export_lines_data(self):
        wb = Workbook()
        wb.save('test.xlsx')

    # display image data on GUI chart, plot -> smaller functions
    def clear_plot_items(self):
        self.line_image_view_parameters.getPlotItem().clear()
        self.line_image_view.getPlotItem().clear()

    def display_image_on_parameters_tab(self, Image):
        if self.line_image_list.current_image != -2:
            image_item_parameters = pg.ImageItem(np.array(Image.image))
            ROI = pg.RectROI((0, 0), (100, 100))
            self.line_image_view_parameters.addItem(image_item_parameters)
            self.line_image_view_parameters.addItem(ROI)

    def display_image_on_lines_tab(self, Image):
        if Image.processed and Image.processed_image is not None and self.line_image_list.current_image != -2:
            image_item = pg.ImageItem(np.array(Image.processed_image))
        else:
            image_item = pg.ImageItem(np.array(Image.image))
        self.line_image_view.addItem(image_item)

    def display_profiles(self, Image):
        edge_color = pg.mkColor(0, 200, 0)
        edge_pen = pg.mkPen(edge_color, width=3)

        if Image.leading_edges is not None:
            profiles_length = np.shape(Image.leading_edges)[1]
            for edge in Image.leading_edges:
                leading_edge_plot = pg.PlotDataItem(edge, np.arange(0, profiles_length), pen=edge_pen)
                self.line_image_view.addItem(leading_edge_plot)

        if Image.trailing_edges is not None:
            for edge in Image.trailing_edges:
                trailing_edge_plot = pg.PlotDataItem(edge, np.arange(0, profiles_length), pen=edge_pen)
                self.line_image_view.addItem(trailing_edge_plot)

    def display_histogram(self, Image):
        histogram_color = pg.mkColor(200, 200, 200)
        histogram_pen = pg.mkPen(histogram_color, width=3)
        histogram_curves_color = pg.mkColor(200, 0, 0)
        histogram_curves_pen = pg.mkPen(histogram_curves_color, width=3)
        histogram_fit_color = pg.mkColor(0, 20, 200)
        histogram_fit_pen = pg.mkPen(histogram_fit_color, width=3)

        histogram_plot = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram, pen=histogram_pen)
        histogram_plot_low = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_low,
                                             pen=histogram_curves_pen)
        histogram_plot_medium = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_medium,
                                                pen=histogram_curves_pen)
        histogram_plot_high = pg.PlotDataItem(np.linspace(0, 255, 256), Image.intensity_histogram_high,
                                              pen=histogram_curves_pen)
        histogram_plot_fit = pg.PlotDataItem(np.linspace(0, 255, 256),
                                             Image.intensity_histogram_high + Image.intensity_histogram_low + Image.intensity_histogram_medium,
                                             pen=histogram_fit_pen)

        self.metric_plot.clear()
        self.metric_plot.addItem(histogram_plot)
        self.metric_plot.addItem(histogram_plot_low)
        self.metric_plot.addItem(histogram_plot_medium)
        self.metric_plot.addItem(histogram_plot_high)
        self.metric_plot.addItem(histogram_plot_fit)
        self.metric_plot.setLogMode(False, False)

    def display_psd(self, Image, plot_type):
        PSD_color = pg.mkColor(200, 200, 200)
        PSD_fit_color = pg.mkColor(0, 200, 200)
        PSD_unbiased_color = pg.mkColor(200, 0, 0)
        PSD_fit_unbiased_color = pg.mkColor(0, 200, 0)
        PSD_pen = pg.mkPen(PSD_color, width=3)
        PSD_fit_pen = pg.mkPen(PSD_fit_color, width=3)
        PSD_unbiased_pen = pg.mkPen(PSD_unbiased_color, width=3)
        PSD_fit_unbiased_pen = pg.mkPen(PSD_fit_unbiased_color, width=3)

        psd_plots = {
            "LW_PSD": (Image.LWR_PSD, Image.LWR_PSD_fit, Image.LWR_PSD_unbiased, Image.LWR_PSD_fit_unbiased),
            "LER_PSD": (Image.LER_PSD, Image.LER_PSD_fit, Image.LER_PSD_unbiased, Image.LER_PSD_fit_unbiased),
            "leading_LER_PSD": (Image.LER_Leading_PSD, Image.LER_Leading_PSD_fit, Image.LER_Leading_PSD_unbiased,
                                Image.LER_Leading_PSD_fit_unbiased),
            "trailing_LER_PSD": (Image.LER_Trailing_PSD, Image.LER_Trailing_PSD_fit, Image.LER_Trailing_PSD_unbiased,
                                 Image.LER_Trailing_PSD_fit_unbiased)
        }

        plots = psd_plots.get(plot_type)
        if plots:
            PSD_plot, PSD_fit_plot, PSD_unbiased_plot, PSD_fit_unbiased_plot = plots

            self.metric_plot.clear()
            if self.metric_original_data.isChecked():
                self.metric_plot.addItem(
                    pg.PlotDataItem(Image.frequency, PSD_plot[0:len(Image.frequency)], pen=PSD_pen))
            if self.metric_model_fit.isChecked():
                self.metric_plot.addItem(
                    pg.PlotDataItem(Image.frequency, PSD_fit_plot[0:len(Image.frequency)], pen=PSD_fit_pen))
            if self.metric_data_unbiased.isChecked():
                self.metric_plot.addItem(
                    pg.PlotDataItem(Image.frequency, PSD_unbiased_plot[0:len(Image.frequency)], pen=PSD_unbiased_pen))
            if self.metric_model_fit_unbiased.isChecked():
                self.metric_plot.addItem(pg.PlotDataItem(Image.frequency, PSD_fit_unbiased_plot[0:len(Image.frequency)],
                                                         pen=PSD_fit_unbiased_pen))

            self.metric_plot.setLogMode(True, True)
            self.metric_plot.setAutoVisible(y=True)

    def display_lines_data(self, Image):
        self.clear_plot_items()
        self.display_image_on_parameters_tab(Image)
        self.display_image_on_lines_tab(Image)
        self.display_profiles(Image)

        if Image.processed and Image.processed_image is not None or self.line_image_list.current_image == -2:
            if self.histogram.isChecked():
                self.display_histogram(Image)
            elif self.lineWidthPSD.isChecked():
                self.display_psd(Image, "LW_PSD")
            elif self.LineEdgePSD.isChecked():
                self.display_psd(Image, "LER_PSD")
            elif self.LeadingEdgePSD.isChecked():
                self.display_psd(Image, "leading_LER_PSD")
            elif self.TrailingEdgePSD.isChecked():
                self.display_psd(Image, "trailing_LER_PSD")

        # image_view = self.line_image_view.getView()
        # pci = pg.PlotCurveItem(x=[1, 50, 100, 150, 200], y=[1, 50, 100, 150, 200])
        # image_view.addItem(pci)

    # process each image from LineImageList (smile_image_list_class) and updates GUI with metrics
    # check how meny images has been selected
    # call gather_parameters() along with pre_processing(), find_edges(), calculate_metrics() (smile_lines_image_class) for each image
    # update the parameters in table
    # use gather_edges() from mile_image_list_class()
    def process_line_images(self):
        number_of_selected_images = self.get_number_selected_images()
        number_of_processed_images = 0

        # set up for progress bar
        self.image_progressBar.setMinimum(0)
        self.image_progressBar.setMaximum(number_of_selected_images)
        self.image_progressBar.setValue(0)

        for lines_image in self.line_image_list.lineImages:
            self.status_label.setText(
                "Processing " + str(number_of_processed_images + 1) + " of " + str(number_of_selected_images))
            if self.linesTable.item(lines_image.id, 0).checkState() == Qt.CheckState.Checked:

                self.gather_parameters(lines_image)
                self.line_image_list.current_image = lines_image.id
                lines_image.pre_processing()
                lines_image.find_edges()
                lines_image.calculate_metrics()

                self.update_table_with_processed_image(lines_image)

                # update progress bar
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
        self.line_image_list.gather_edges()
        self.line_image_list.parameters = self.line_image_list.lineImages[0].parameters
        self.line_image_list.calculate_metrics()
        self.status_label.setText("Ready")

    def update_table_with_processed_image(self, lines_image):
        item_averageCD = QtWidgets.QTableWidgetItem(f"{lines_image.critical_dimension_estimate:{0}.{5}}")
        item_number_of_lines = QtWidgets.QTableWidgetItem(str(lines_image.number_of_lines))
        item_averageCDstd = QtWidgets.QTableWidgetItem(f"{lines_image.critical_dimension_std_estimate:{0}.{5}}")
        item_pitchEstimate = QtWidgets.QTableWidgetItem(f"{lines_image.pitch_estimate:{0}.{5}}")
        self.linesTable.setItem(lines_image.id, 3, item_number_of_lines)
        self.linesTable.setItem(lines_image.id, 4, item_pitchEstimate)
        self.linesTable.setItem(lines_image.id, 5, item_averageCD)
        self.linesTable.setItem(lines_image.id, 6, item_averageCDstd)

    def get_number_selected_images(self):
        # Count how many images have been selected for processing
        number_of_selected_images = 0
        for lines_image in self.line_image_list.lineImages:
            if self.linesTable.item(lines_image.id, 0).checkState() == Qt.CheckState.Checked:
                number_of_selected_images += 1
        return number_of_selected_images

    # helper function to prepare image parameters needed by SmileLinesImage (smile_lines_image_class)
    # parameters collected from UI elements
    def gather_parameters(self, Image):

        edge_fit_function, window = self.get_fit_function()

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
                      'PSD_model': window.PSD_model.currentText(),
                      }
        Image.parameters = parameters

    def get_fit_function(self):
        if self.Polynomial.isChecked():
            edge_fit_function = 'polynomial'
        elif self.Linear.isChecked():
            edge_fit_function = 'linear'
        elif self.ThresholdEdge.isChecked():
            edge_fit_function = 'threshold'
        else:
            edge_fit_function = 'bright_edge'
        window = MainWindow()
        return edge_fit_function, window

    # helper function, handles the navigation in display selected line image data when a cell in the lines table is clicked
    def navigateLinesTable(self, nrow, ncol):
        self.line_image_list.current_image = nrow
        if nrow < len(self.line_image_list.lineImages):
            self.display_lines_data(self.line_image_list.lineImages[nrow])
        else:
            self.line_image_list.current_image = -2
            self.display_lines_data(self.line_image_list)

    # open a dialog to select an image folder, loads image
    # call function gather parameters (main_controller)
    # append image object (in loop) to LineImageList (smile_image_list_class)
    def load_lines_image(self):
        # Initialize image list
        self.line_image_list = LineImageList('-1', 'imageList', 'Empty')

        file_names = folder_loader()
        image_id = -1
        for root, dirs, files in os.walk(file_names[0], topdown=True):
            for name in files:
                if (
                        name.endswith(".tif")
                        | name.endswith(".jpg")
                        | name.endswith(".png")
                ):
                    image_id += 1
                    self.add_new_row(image_id, name, root)
                else:
                    raise ValueError('Invalid file extension')

    def add_new_row(self, image_id, name, root):
        image_object = SmileLinesImage(image_id, name, root)
        print(image_object)
        image_object.load_image()
        # TODO why second time call gather_parameters if already is called in process_line_image
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

        # self.linesTable.setRowCount (image_id+2)
        self.linesTable.setRowCount(image_id + 2)
        self.linesTable.setItem(image_id, 0, item_selected)
        self.linesTable.setItem(image_id, 1, item_processed)
        self.linesTable.setItem(image_id, 2, item_name)
        self.display_lines_data(image_object)


def folder_loader():
    select_folder_dialog = QtWidgets.QFileDialog()
    select_folder_dialog.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
    if select_folder_dialog.exec():
        file_names = select_folder_dialog.selectedFiles()
        return file_names
    else:
        raise ValueError('Cannot open this folder')
