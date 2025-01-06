from pyqtgraph import PlotWidget
from PyQt6 import QtWidgets
import pyqtgraph as pg
import numpy as np

from app.view.display_image import ImageDisplayManager
from app.models.image_container import Image


class ResultImagesManager(QtWidgets.QWidget):
    """
    Manages the display of result images on various tabs in the GUI.

    Attributes:
        plot_widget_parameters_tab (PlotWidget): The widget for displaying image in parameters tab.
        plot_widget_lines_tab (PlotWidget): The widget for displaying image in line tab.
        widget_metric_tab (PlotWidget): The widget for displaying metric plots.
        window: The main window of the application.
        base_images (ImageDisplayManager): Manages the display of base images.
    """

    def __init__(self, plot_widget_parameters_tab: PlotWidget, plot_widget_lines_tab: PlotWidget,
                 widget_metric_tab: PlotWidget, window) -> None:
        self.plot_widget_parameters_tab = plot_widget_parameters_tab
        self.plot_widget_lines_tab = plot_widget_lines_tab
        self.widget_metric_tab = widget_metric_tab
        self.window = window

        self.base_images = ImageDisplayManager(self.plot_widget_parameters_tab, self.plot_widget_lines_tab)

    def display_profiles_on_lines_tab(self, image: Image) -> None:
        """
        Marks profiles on the image.

        Args:
            image (Image): The image object, contain profiles attributes to display.
        """

        self.base_images.display_image_on_lines_tab(image)
        self._display_profiles(image, self.plot_widget_lines_tab)

    def display_profiles_on_parameters_tab(self, image: Image) -> None:

        self.base_images.display_image_on_parameters_tab(image)
        self._display_profiles(image, self.plot_widget_parameters_tab)

    def _display_profiles(self, image, tab: PlotWidget) -> None:
        edge_color = pg.mkColor(0, 200, 0)
        edge_pen = pg.mkPen(edge_color, width=3)
        self._display_edges(image.leading_edges, "Leading edges were not found", edge_pen, tab)
        self._display_edges(image.trailing_edges, "Trailing edges were not found", edge_pen, tab)

    def _display_edges(self, edges: np.ndarray, error_message: str, pen: str, tab: PlotWidget) -> None:
        """
        Helper function to add edges on the image in lines tab.

        Args:
            edges (np.ndarray): The edges to display.
            error_message (str): The error message to show if edges are not found.
            pen (str): The pen for drawing the edges.
        """

        if edges is not None:
            profiles_length = np.shape(edges)[1]
            for edge in edges:
                edge_plot = pg.PlotDataItem(edge, np.arange(0, profiles_length), pen=pen)
                tab.addItem(edge_plot)
        else:
            self.window.show_error_message(error_message)

    def display_plot_on_metric_tab(self, image: Image) -> None:
        """
        Displays the appropriate plot on the metric tab based on user selection.

        Args:
            image (Image): The image object containing the data to plot.
        """

        try:
            if self.window.histogram.isChecked():
                self._display_histogram(image)
            elif self.window.LineWidthPSD.isChecked():
                self._display_metric(image)
            elif self.window.LineEdgePSD.isChecked():
                self._display_metric(image)
            elif self.window.LeadingEdgePSD.isChecked():
                self._display_metric(image)
            elif self.window.TrailingEdgePSD.isChecked():
                self._display_metric(image)
            elif self.window.LineWidthHHCF.isChecked():
                self._display_metric(image)
            elif self.window.LineEdgeHHCF.isChecked():
                self._display_metric(image)
        except Exception as e:
            self.window.show_error_message(f"Plot can not be display. {str(e)}")

    def _display_histogram(self, image: Image) -> None:
        """
        Helper method to display the histogram on the metric tab.

        Args:
            image (Image): The image object containing the histogram data.
        """

        histogram_color = pg.mkColor(200, 200, 200)
        histogram_pen = pg.mkPen(histogram_color, width=3)
        histogram_curves_color = pg.mkColor(200, 0, 0)
        histogram_curves_pen = pg.mkPen(histogram_curves_color, width=3)
        histogram_fit_color = pg.mkColor(0, 20, 200)
        histogram_fit_pen = pg.mkPen(histogram_fit_color, width=3)

        histogram_plot = pg.PlotDataItem(np.linspace(0, 255, 256), image.intensity_histogram, pen=histogram_pen)
        histogram_plot_low = pg.PlotDataItem(np.linspace(0, 255, 256), image.intensity_histogram_low,
                                             pen=histogram_curves_pen)
        histogram_plot_medium = pg.PlotDataItem(np.linspace(0, 255, 256), image.intensity_histogram_medium,
                                                pen=histogram_curves_pen)
        histogram_plot_high = pg.PlotDataItem(np.linspace(0, 255, 256), image.intensity_histogram_high,
                                              pen=histogram_curves_pen)
        histogram_plot_fit = pg.PlotDataItem(np.linspace(0, 255, 256),
                                             image.intensity_histogram_high + image.intensity_histogram_low + image.intensity_histogram_medium,
                                             pen=histogram_fit_pen)

        self.widget_metric_tab.clear()
        self.widget_metric_tab.addItem(histogram_plot)
        self.widget_metric_tab.addItem(histogram_plot_low)
        self.widget_metric_tab.addItem(histogram_plot_medium)
        self.widget_metric_tab.addItem(histogram_plot_high)
        self.widget_metric_tab.addItem(histogram_plot_fit)
        self.widget_metric_tab.setLogMode(False, False)

    def _display_metric(self, image: Image) -> None:

        self.widget_metric_tab.clear()

        PSD_color = pg.mkColor(200, 200, 200)
        PSD_fit_color = pg.mkColor(0, 200, 200)
        PSD_unbiased_color = pg.mkColor(200, 0, 0)
        PSD_fit_unbiased_color = pg.mkColor(0, 200, 0)
        PSD_pen = pg.mkPen(PSD_color, width=3)
        PSD_fit_pen = pg.mkPen(PSD_fit_color, width=3)
        PSD_unbiased_pen = pg.mkPen(PSD_unbiased_color, width=3)
        PSD_fit_unbiased_pen = pg.mkPen(PSD_fit_unbiased_color, width=3)

        if self.window.LineWidthPSD.isChecked():

            data = pg.PlotDataItem(image.frequency, image.LWR_PSD[0:len(image.frequency)], pen=PSD_pen)
            fit = pg.PlotDataItem(image.frequency, image.LWR_PSD_fit[0:len(image.frequency)], pen=PSD_fit_pen)
            data_unbiased = pg.PlotDataItem(image.frequency, image.LWR_PSD_unbiased[0:len(image.frequency)], pen=PSD_unbiased_pen)
            fit_unbiased = pg.PlotDataItem(image.frequency, image.LWR_PSD_fit_unbiased[0:len(image.frequency)], pen=PSD_fit_unbiased_pen)
            self.widget_metric_tab.setLabel('bottom', "Frequency", units='nm^-1')
            self.widget_metric_tab.setLabel('left', "Power spectral density", units='nm^3')

        elif self.window.LineEdgePSD.isChecked():
            data = pg.PlotDataItem(image.frequency, image.LER_PSD[0:len(image.frequency)], pen=PSD_pen)
            fit = pg.PlotDataItem(image.frequency, image.LER_PSD_fit[0:len(image.frequency)], pen=PSD_fit_pen)
            data_unbiased = pg.PlotDataItem(image.frequency, image.LER_PSD_unbiased[0:len(image.frequency)], pen=PSD_unbiased_pen)
            fit_unbiased = pg.PlotDataItem(image.frequency, image.LER_PSD_fit_unbiased[0:len(image.frequency)], pen=PSD_fit_unbiased_pen)
            self.widget_metric_tab.setLabel('bottom', "Frequency", units='nm^-1')
            self.widget_metric_tab.setLabel('left', "Power spectral density", units='nm^3')

        elif self.window.LeadingEdgePSD.isChecked():
            data = pg.PlotDataItem(image.frequency, image.LER_Leading_PSD[0:len(image.frequency)], pen=PSD_pen)
            fit = pg.PlotDataItem(image.frequency, image.LER_Leading_PSD_fit[0:len(image.frequency)], pen=PSD_fit_pen)
            data_unbiased = pg.PlotDataItem(image.frequency, image.LER_Leading_PSD_unbiased[0:len(image.frequency)],
                                            pen=PSD_unbiased_pen)
            fit_unbiased = pg.PlotDataItem(image.frequency, image.LER_Leading_PSD_fit_unbiased[0:len(image.frequency)],
                                           pen=PSD_fit_unbiased_pen)
            self.widget_metric_tab.setLabel('bottom', "Frequency", units='nm^-1')
            self.widget_metric_tab.setLabel('left', "Power spectral density", units='nm^3')

        elif self.window.TrailingEdgePSD.isChecked():
            data = pg.PlotDataItem(image.frequency, image.LER_Trailing_PSD[0:len(image.frequency)], pen=PSD_pen)
            fit = pg.PlotDataItem(image.frequency, image.LER_Trailing_PSD_fit[0:len(image.frequency)], pen=PSD_fit_pen)
            data_unbiased = pg.PlotDataItem(image.frequency, image.LER_Trailing_PSD_unbiased[0:len(image.frequency)],
                                            pen=PSD_unbiased_pen)
            fit_unbiased = pg.PlotDataItem(image.frequency, image.LER_Trailing_PSD_fit_unbiased[0:len(image.frequency)],
                                           pen=PSD_fit_unbiased_pen)
            self.widget_metric_tab.setLabel('bottom', "Frequency", units='nm^-1')
            self.widget_metric_tab.setLabel('left', "Power spectral density", units='nm^3')

        elif self.window.LineWidthHHCF.isChecked():
            profile_length = np.arange(0,len(image.LW_HHCF)) * image.pixel_size
            data = pg.PlotDataItem(profile_length, image.LW_HHCF, pen=PSD_pen)
            fit = pg.PlotDataItem(profile_length, image.LW_HHCF_fit, pen=PSD_fit_pen)
            self.widget_metric_tab.setLabel('bottom', "Distance", units='nm')
            self.widget_metric_tab.setLabel('left', "Correlation", units='nm^2')

        elif self.window.LineEdgeHHCF.isChecked():
            profile_length = np.arange(0, len(image.Lines_edge_HHCF)) * image.pixel_size
            data = pg.PlotDataItem(profile_length, image.Lines_edge_HHCF, pen=PSD_pen)
            fit = pg.PlotDataItem(profile_length, image.Lines_edge_HHCF_fit, pen=PSD_fit_pen)
            self.widget_metric_tab.setLabel('bottom', "Distance", units='nm')
            self.widget_metric_tab.setLabel('left', "Correlation", units='nm^2')
            
        if self.window.metric_original_data.isChecked():
            self.widget_metric_tab.addItem(data)
        if self.window.metric_model_fit.isChecked():
            self.widget_metric_tab.addItem(fit)
        if self.window.metric_model_fit_unbiased.isChecked():
            if 'fit_unbiased' in locals():
                self.widget_metric_tab.addItem(fit_unbiased)
        if self.window.metric_data_unbiased.isChecked():
            if 'data_unbiased' in locals():
                self.widget_metric_tab.addItem(data_unbiased)

        self.widget_metric_tab.setLogMode(True, True)
        self.widget_metric_tab.setAutoVisible(y=True)


    # def _display_psd(self, image: Image) -> None:
    #     """
    #     Helper method to display the PSD plot on the metric tab.
    #
    #     Args:
    #         image (Image): The image object containing the PSD data.
    #     """
    #
    #     PSD_color = pg.mkColor(200, 200, 200)
    #     PSD_fit_color = pg.mkColor(0, 200, 200)
    #     PSD_unbiased_color = pg.mkColor(200, 0, 0)
    #     PSD_fit_unbiased_color = pg.mkColor(0, 200, 0)
    #     PSD_pen = pg.mkPen(PSD_color, width=3)
    #     PSD_fit_pen = pg.mkPen(PSD_fit_color, width=3)
    #     PSD_unbiased_pen = pg.mkPen(PSD_unbiased_color, width=3)
    #     PSD_fit_unbiased_pen = pg.mkPen(PSD_fit_unbiased_color, width=3)
    #
    #     self.widget_metric_tab.clear()
    #
    #     plot_LWR_PSD_data = pg.PlotDataItem(image.frequency, image.LWR_PSD[0:len(image.frequency)], pen=PSD_pen)
    #     plot_LWR_PSD_fit = pg.PlotDataItem(image.frequency, image.LWR_PSD_fit[0:len(image.frequency)], pen=PSD_pen)
    #     plot_LWR_PSD_unbiased = pg.PlotDataItem(image.frequency, image.LWR_PSD_unbiased[0:len(image.frequency)], pen=PSD_pen)
    #     plot_LWR_PSD_fit_unbiased = pg.PlotDataItem(image.frequency, image.LWR_PSD_fit_unbiased[0:len(image.frequency)],
    #                                             pen=PSD_pen)
    #
    #     if self.window.metric_original_data.isChecked():
    #         self.widget_metric_tab.addItem(
    #             pg.PlotDataItem(image.frequency, image.LWR_PSD[0:len(image.frequency)], pen=PSD_pen))
    #     if self.window.metric_model_fit.isChecked():
    #         self.widget_metric_tab.addItem(
    #             pg.PlotDataItem(image.frequency, image.LWR_PSD_fit[0:len(image.frequency)], pen=PSD_fit_pen))
    #     if self.window.metric_data_unbiased.isChecked():
    #         self.widget_metric_tab.addItem(
    #             pg.PlotDataItem(image.frequency, image.LWR_PSD_unbiased [0:len(image.frequency)],
    #                             pen=PSD_unbiased_pen))
    #     if self.window.metric_model_fit_unbiased.isChecked():
    #         self.widget_metric_tab.addItem(
    #             pg.PlotDataItem(image.frequency, image.LWR_PSD_fit_unbiased[0:len(image.frequency)],
    #                             pen=PSD_fit_unbiased_pen))
    #
    #     self.widget_metric_tab.setLogMode(True, True)
    #     self.widget_metric_tab.setAutoVisible(y=True)
