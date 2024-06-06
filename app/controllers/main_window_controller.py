from PyQt6 import QtWidgets, uic
from pyqtgraph import PlotWidget

from app.models.images_list import ImagesList
from app.controllers.table_controller import TableController
from app.view.display_calculation_result_images import ResultImagesManager
from app.view.display_image import ImageDisplayManager
from app.controllers.folder_image_loader import FolderImageLoader


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("ui/window.ui", self)
        self.images_list = ImagesList()

        self.init_ui()
        self.ui_checkboxes()
        self.setup_connections()

        self.image_loader = FolderImageLoader(self.images_list, self)
        self.table_controller = TableController(self.table)
        self.image_display_manager = ImageDisplayManager(self.widget_parameters_tab, self.widget_lines_tab)
        self.result_images_manager = ResultImagesManager(self.widget_parameters_tab, self.widget_lines_tab,
                                                         self.widget_metric_tab, self)

    def init_ui(self):
        self.push_button_image_folder = self.findChild(QtWidgets.QPushButton, "pushButton_ImageFolder")
        self.table = self.findChild(QtWidgets.QTableWidget, "linesTable")
        self.widget_parameters_tab = self.findChild(PlotWidget, "line_image_view_parameters")
        self.widget_lines_tab = self.findChild(PlotWidget, "line_image_view")
        self.widget_metric_tab = self.findChild(PlotWidget, "metric_plot")

    # def ui_checkboxes(self):
    #     self.histogram_radiobutton = self.findChild(QtWidgets.QRadioButton, "histogram")
    #     self.lineWidthPSD_radiobutton = self.findChild(QtWidgets.QRadioButton, "lineWidthPSD")
    #     self.LineEdgePSD_radiobutton = self.findChild(QtWidgets.QRadioButton, "LineEdgePSD")
    #     self.LeadingEdgePSD_radiobutton = self.findChild(QtWidgets.QRadioButton, "LeadingEdgePSD")
    #     self.TrailingEdgePSD_radiobutton = self.findChild(QtWidgets.QRadioButton, "TrailingEdgePSD")
    #     self.metric_original_data_checkbox = self.findChild(QtWidgets.QCheckBox, "metric_original_data")
    #     self.metric_model_fit_checkbox = self.findChild(QtWidgets.QCheckBox, "metric_model_fit")
    #     self.metric_data_unbiased_checkbox = self.findChild(QtWidgets.QCheckBox, "metric_data_unbiased")
    #     self.metric_data_unbiasedic_model_fit_unbiased_checkbox = self.findChild(QtWidgets.QCheckBox, "metric_model_fit_unbiased")


    def setup_connections(self):
        self.push_button_image_folder.pressed.connect(self.prepare_image)

    def prepare_image(self):
        self.image_loader.load_images_from_folder()
        for image in self.images_list.images_list:
            self.image_display_manager.display_image_on_parameters_tab(image)
            self.image_display_manager.display_image_on_lines_tab(image)
        self.table_controller.update_with_image(self.images_list.images_list)
