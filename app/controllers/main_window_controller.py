from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from pyqtgraph import PlotWidget
from PyQt6.QtWidgets import QMessageBox

from app.models.images_list import ImagesList
from app.controllers.table_controller import TableController
from app.view.display_calculation_result_images import ResultImagesManager
from app.view.display_image import ImageDisplayManager
from app.controllers.folder_image_loader import FolderImageLoader
from app.controllers.processing_controller import ProcessingController
from app.models.average_image import AverageImage


class MainWindow(QtWidgets.QMainWindow):
    """
    Main window class for the application that handles UI setup, image loading,
    processing, and displaying results.

    """
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("ui/window.ui", self)
        self.images_list = ImagesList()

        self.init_ui()

        self.image_loader = FolderImageLoader(self.images_list, self)
        self.table_controller = TableController(self.table)
        self.image_display_manager = ImageDisplayManager(self.widget_parameters_tab, self.widget_lines_tab)
        self.result_images_manager = ResultImagesManager(self.widget_parameters_tab, self.widget_lines_tab,
                                                         self.widget_metric_tab, self)
        self.processing_controller = ProcessingController(self, self.images_list, self.table)
        self.average_image = None

        self.setup_connections()

    def init_ui(self) -> None:
        """
        Finds and initializes UI components.
        """

        self.push_button_image_folder = self.findChild(QtWidgets.QPushButton, "pushButton_ImageFolder")
        self.push_button_process_images = self.findChild(QtWidgets.QPushButton, "process_lines_button")
        self.table = self.findChild(QtWidgets.QTableWidget, "linesTable")
        self.widget_parameters_tab = self.findChild(PlotWidget, "line_image_view_parameters")
        self.widget_lines_tab = self.findChild(PlotWidget, "line_image_view")
        self.widget_metric_tab = self.findChild(PlotWidget, "metric_plot")

    def setup_connections(self) -> None:
        """
        Sets up push-buttons and signal-slot connections for UI components.
        """

        self.push_button_image_folder.pressed.connect(self.prepare_image)
        self.push_button_process_images.pressed.connect(self.process_image)
        self.table.cellClicked.connect(self.display_corresponding_images)

        self.table_controller.error_signal.connect(self.show_error_message)
        self.image_display_manager.error_signal.connect(self.show_error_message)

    def prepare_image(self) -> None:
        """
        Loads images from a folder and updates the UI with loaded images.
        """

        # self.clean_tab()
        self.image_loader.load_images_from_folder()
        for image in self.images_list.images_list:
            self.image_display_manager.display_image_on_parameters_tab(image)
            self.image_display_manager.display_image_on_lines_tab(image)
        self.table_controller.update_with_image(self.images_list)

    def clean_tab(self) -> None:
        """
        Clears the images list and table contents.
        """

        self.images_list.images_list = []
        self.table.clearContents()

    def process_image(self) -> None:
        """
        Processes selected images and updates the UI with results.
        """

        self.processing_controller.get_number_selected_images()
        self.processing_controller.set_up_progress_bar()

        number_processed_images = 0
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
                self.processing_controller.process_image(image)
                self.table_controller.update_with_processed_image(image)
                number_processed_images += 1
                self.processing_controller.update_progress_bar(number_processed_images)
                self.table_controller.mark_image_as_processed(image.id)
                self.result_images_manager.display_profiles_on_lines_tab(image)
                self.result_images_manager.display_plot_on_metric_tab(image)
                QtWidgets.QApplication.processEvents()

        self.average_image = AverageImage(self.images_list)
        self.table_controller.add_average_image(self.average_image)
        self.result_images_manager.display_plot_on_metric_tab(self.average_image.image)
        QtWidgets.QApplication.processEvents()

    def display_corresponding_images(self, row: int) -> None:
        """
        Displays images and metrics corresponding to the selected table row.
        """

        if row <= len(self.images_list.images_list) - 1:
            image = self.images_list.images_list[row]
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                self.image_display_manager.display_image_on_lines_tab(image)
                self.image_display_manager.display_image_on_parameters_tab(image)
                self.result_images_manager.display_profiles_on_lines_tab(image)
                self.result_images_manager.display_plot_on_metric_tab(image)
            else:
                self.widget_parameters_tab.clear()
                self.widget_lines_tab.clear()
                self.widget_metric_tab.clear()
        else:
            image = AverageImage(self.images_list).image
            self.widget_parameters_tab.clear()
            self.widget_lines_tab.clear()
            self.result_images_manager.display_plot_on_metric_tab(image)

    def show_error_message(self, message: str) -> None:
        """
        Displays an error message in a message box.
        """

        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec()




