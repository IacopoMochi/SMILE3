from functools import partial
from pathlib import Path

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from pyqtgraph import PlotWidget
from PyQt6.QtWidgets import QMessageBox, QTableWidgetItem
from PyQt6.QtGui import QIntValidator, QDoubleValidator

from app.models.images_list import ImagesList
from app.controllers.table_controller import TableController
from app.utils.display_calculation_result_images import ResultImagesManager
from app.view.display_image import ImageDisplayManager
from app.controllers.folder_image_loader import FolderImageLoader
from app.controllers.processing_controller import ProcessingController
from app.models.average_image import AverageImage
from app.controllers.data_export_controller import DataExporter



class MainWindow(QtWidgets.QMainWindow):
    """
    Main window class for the application that handles UI setup, image loading,
    processing, and displaying results.
    """

    def __init__(self):
        super(MainWindow, self).__init__()
        uiPath = str(Path(__file__).parent.parent / "ui/window.ui")
        uic.loadUi(uiPath, self)

        self.init_ui()
        self.init_ui_for_roi()
        self.init_classes()
        self.setup_connections()

    def init_classes(self):
        """
        Initialize the main window class.
        """

        self.images_list = ImagesList()
        self.average_image = AverageImage(self.images_list, self.table)
        self.image_loader = FolderImageLoader(self.images_list, self)
        self.table_controller = TableController(self.table)
        self.image_display_manager = ImageDisplayManager(self.widget_parameters_tab, self.widget_lines_tab)
        self.result_images_manager = ResultImagesManager(self.widget_parameters_tab, self.widget_lines_tab,
                                                         self.widget_metric_tab, self)
        self.processing_controller = ProcessingController(self, self.images_list, self.table)
        self.average_image = None
        self.data_exporter = DataExporter(self.images_list, self)

    def init_ui(self) -> None:
        """
        Finds and initializes UI components.
        """

        self.push_button_image_folder = self.findChild(QtWidgets.QPushButton, "pushButton_ImageFolder")
        self.push_button_process_images = self.findChild(QtWidgets.QPushButton, "process_lines_button")
        self.push_button_recalculate_metrics = self.findChild(QtWidgets.QPushButton, "recalculate_metrics_button")
        self.push_button_process_images.setEnabled(False)
        self.push_button_recalculate_metrics.setEnabled(False)
        self.table = self.findChild(QtWidgets.QTableWidget, "linesTable")
        self.widget_parameters_tab = self.findChild(PlotWidget, "line_image_view_parameters")
        self.widget_lines_tab = self.findChild(PlotWidget, "line_image_view")
        self.widget_metric_tab = self.findChild(PlotWidget, "metric_plot")
        self.push_button_export_data = self.findChild(QtWidgets.QPushButton, "export_data")
        self.push_button_select_all = self.findChild(QtWidgets.QPushButton, "pushButton_selectAll")
        self.push_button_select_none = self.findChild(QtWidgets.QPushButton, "pushButton_selectNone")
        self.edge_search_range = self.findChild(QtWidgets.QRadioButton, "EdgeRange_button")
        self.edge_search_CD_fraction = self.findChild(QtWidgets.QRadioButton, "CDfraction_button")
        self.pixel_size = self.findChild(QtWidgets.QLineEdit, "pixelSize_line_edit")
        self.pixel_size.setValidator(QDoubleValidator())

        self.LineEdgePSD = self.findChild(QtWidgets.QRadioButton, "LineEdgePSD")
        self.LineWidthPSD = self.findChild(QtWidgets.QRadioButton, "LineWidthPSD")
        self.LeadingEdgePSD = self.findChild(QtWidgets.QRadioButton, "LeadingEdgePSD")
        self.TrailingEdgePSD = self.findChild(QtWidgets.QRadioButton, "TrailingEdgePSD")
        self.Histogram = self.findChild(QtWidgets.QRadioButton, "histogram")
        self.LineWidthHHCF = self.findChild(QtWidgets.QRadioButton, "LineWidthHHCF")
        self.LineEdgeHHCF = self.findChild(QtWidgets.QRadioButton, "LineEdgeHHCF")
        self.LeadingEdgeHHCF = self.findChild(QtWidgets.QRadioButton, "LeadingEdgeHHCF")
        self.TrailingEdgeHHCF = self.findChild(QtWidgets.QRadioButton, "TrailingEdgeHHCF")
        self.DisplayData = self.findChild(QtWidgets.QCheckBox,"metric_original_data")
        self.DisplayFit = self.findChild(QtWidgets.QCheckBox, "metric_model_fit")
        self.DisplayDataUnbiased = self.findChild(QtWidgets.QCheckBox,"metric_data_unbiased")
        self.DisplayFitUnbiased = self.findChild(QtWidgets.QCheckBox,"metric_model_fit_unbiased")
        self.MaxEdgeSpike = self.findChild(QtWidgets.QLineEdit,"MaxSpike")

    def init_ui_for_roi(self) -> None:

        self.x1_widget = self.findChild(QtWidgets.QLineEdit, "X1")
        self.x2_widget = self.findChild(QtWidgets.QLineEdit, "X2")
        self.y1_widget = self.findChild(QtWidgets.QLineEdit, "Y1")
        self.y2_widget = self.findChild(QtWidgets.QLineEdit, "Y2")


    def setup_connections(self) -> None:
        """
        Sets up push-buttons and signal-slot connections for UI components.
        """

        self.push_button_image_folder.pressed.connect(self.prepare_image)
        self.push_button_process_images.pressed.connect(partial(self.process_image, recalculate_metrics=False))
        self.push_button_recalculate_metrics.pressed.connect(partial(self.process_image, recalculate_metrics=True))

        #self.table.cell.connect(self.display_corresponding_images)
        self.table.itemSelectionChanged.connect(self.display_corresponding_images)
        self.LineEdgePSD.clicked.connect(self.display_corresponding_images)
        self.LineWidthPSD.clicked.connect(self.display_corresponding_images)
        self.LeadingEdgePSD.clicked.connect(self.display_corresponding_images)
        self.TrailingEdgePSD.clicked.connect(self.display_corresponding_images)
        self.Histogram.clicked.connect(self.display_corresponding_images)
        self.LineWidthHHCF.clicked.connect(self.display_corresponding_images)
        self.LineEdgeHHCF.clicked.connect(self.display_corresponding_images)
        self.LeadingEdgeHHCF.clicked.connect(self.display_corresponding_images)
        self.TrailingEdgeHHCF.clicked.connect(self.display_corresponding_images)
        self.edge_search_CD_fraction.clicked.connect(self.switch_edge_search_method_range2CD)
        self.edge_search_range.clicked.connect(self.switch_edge_search_method_CD2range)
        self.pixel_size.textEdited.connect(self.store_pixel_size)

        self.table.itemChanged.connect(self.check_selection)

        self.push_button_export_data.pressed.connect(self.data_exporter.export_data)

        self.push_button_select_all.pressed.connect(self.select_all_images)
        self.push_button_select_none.pressed.connect(self.unselect_all_images)

        self.table_controller.error_signal.connect(self.show_error_message)
        self.image_display_manager.error_signal.connect(self.show_error_message)

    def store_pixel_size(self) -> None:
        print('Update the current image Pixel size')
        # print("f{self.pixel_size.text + 10}")
        # self.images_list.images_list[self.images_list.active_image].pixel_size = self.pixel_size.text()

    def switch_edge_search_method_CD2range(self):
        """
        Sets the edge search method to "range"
        """
        self.edge_search_CD_fraction.setChecked(False)
        self.edge_search_range.setChecked(True)

    def switch_edge_search_method_range2CD(self):
        """
        Sets the edge search method to "CD fraction"
        """
        self.edge_search_CD_fraction.setChecked(True)
        self.edge_search_range.setChecked(False)


    def prepare_image(self) -> None:
        """
        Loads images from a folder and updates the UI with loaded images.
        """

        self.clean_tab()
        self.image_loader.load_images_from_folder()
        for image in self.images_list.images_list:
            self.image_display_manager.display_image_on_parameters_tab(image)
            self.image_display_manager.set_roi(self.x1_widget, self.x2_widget, self.y1_widget, self.y2_widget)
            self.image_display_manager.display_image_on_lines_tab(image)
        self.table_controller.update_with_image(self.images_list)

        if self.images_list.images_list:
            self.push_button_process_images.setEnabled(True)

    def process_image(self, recalculate_metrics=False) -> None:
        """
        Processes selected images and updates the UI with results.
        """

        self.push_button_process_images.setEnabled(False)
        self.processing_controller.get_number_selected_images()
        self.processing_controller.set_up_progress_bar()

        number_processed_images = 0

        for row in range(len(self.images_list.images_list) - 1):
            if self.table.item(row, 0).checkState() != Qt.CheckState.Checked:
                [self.table.setItem(row, column, QTableWidgetItem("")) for column in range(3, self.table.columnCount())]
                self.table.setItem(row, 1, QTableWidgetItem("No"))

        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:

                if recalculate_metrics:
                    if image.processed:
                        self.processing_controller.recalculate_metrics(image)
                    else:
                        self.table.item(image.id, 0).setCheckState(Qt.CheckState.Unchecked)
                        self.processing_controller.number_selected_images -= 1
                        self.processing_controller.set_up_progress_bar()

                else:
                    self.processing_controller.process_image(image)

                if image.processed:
                    self.table_controller.update_with_processed_image(image)
                    number_processed_images += 1
                    self.processing_controller.update_progress_bar(number_processed_images)
                    self.table_controller.mark_image_as_processed(image.id)
                    self.result_images_manager.display_profiles_on_lines_tab(image)
                    self.result_images_manager.display_profiles_on_parameters_tab(image)
                    self.result_images_manager.display_plot_on_metric_tab(image)
                    QtWidgets.QApplication.processEvents()

        self.average_image = AverageImage(self.images_list, self.table)
        self.average_image.prepare_average_image()
        self.table_controller.add_average_image(self.average_image)
        self.result_images_manager.display_plot_on_metric_tab(self.average_image.image)
        QtWidgets.QApplication.processEvents()
        self.push_button_process_images.setEnabled(True)
        self.push_button_recalculate_metrics.setEnabled(True)
        self.status_label.setText('Ready')

    def display_corresponding_images(self) -> None:
        """
        Displays images and metrics corresponding to the selected table row.
        """
        row = self.table.currentRow()
        if row == -1:
            print("No image selected")
        elif row <= len(self.images_list.images_list) - 1:
            image = self.images_list.images_list[row]
            self.pixel_size.setText(f"{image.pixel_size}")

            if image.processed:

                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    self.image_display_manager.display_image_on_lines_tab(image)
                    self.image_display_manager.display_image_on_parameters_tab(image)
                    self.result_images_manager.display_profiles_on_lines_tab(image)
                    self.result_images_manager.display_profiles_on_parameters_tab(image)
                    self.result_images_manager.display_plot_on_metric_tab(image)
                else:
                    self.widget_parameters_tab.clear()
                    self.widget_lines_tab.clear()
                    self.widget_metric_tab.clear()

            else:
                self.image_display_manager.display_image_on_lines_tab(image)
                self.image_display_manager.display_image_on_parameters_tab(image)
                self.image_display_manager.set_roi(self.x1_widget, self.x2_widget, self.y1_widget, self.y2_widget)
                self.widget_metric_tab.clear()

        else:
            average_image = AverageImage(self.images_list, self.table)
            average_image.prepare_average_image()
            self.widget_parameters_tab.clear()
            self.widget_lines_tab.clear()
            if hasattr(average_image,"image"):
                self.result_images_manager.display_plot_on_metric_tab(average_image.image)

    def check_selection(self):
        """
        Check if any image is selected. If not, disable the process button.
        """

        any_selected = False
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
                any_selected = True
                break
        self.push_button_process_images.setEnabled(any_selected)

    def select_all_images(self):
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Unchecked:
                self.table.item(image.id, 0).setCheckState(Qt.CheckState.Checked)
                image.selected = True

    def unselect_all_images(self):
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
                self.table.item(image.id, 0).setCheckState(Qt.CheckState.Unchecked)
                image.selected = False

    def show_error_message(self, message: str) -> None:
        """
        Displays an error message in a message box.
        """

        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec()

    def clean_tab(self) -> None:
        """
        Clears the images list and table contents.
        """

        self.images_list.images_list = []
        self.table.clearContents()
        self.widget_lines_tab.clear()
        self.widget_parameters_tab.clear()
        self.widget_metric_tab.clear()
        self.processing_controller.update_progress_bar(0)
        self.push_button_recalculate_metrics.setEnabled(False)





