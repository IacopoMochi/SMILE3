from PyQt6 import QtWidgets, uic
from pyqtgraph import PlotWidget

from app.models.images_list import ImagesList
from app.controllers.table_controller import TableController
from app.view.display_image import ImageDisplayManager
from app.controllers.folder_image_loader import FolderImageLoader


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("ui/window.ui", self)
        self.images_list = ImagesList()

        self.init_ui()
        self.setup_connections()

        self.image_loader = FolderImageLoader(self.images_list, self)
        self.table_controller = TableController(self.table)
        self.image_display_manager = ImageDisplayManager(self.plot_widget_parameters_tab, self.plot_widget_lines_tab)

    def init_ui(self):
        self.pushButton_ImageFolder = self.findChild(QtWidgets.QPushButton, "pushButton_ImageFolder")
        self.table = self.findChild(QtWidgets.QTableWidget, "linesTable")
        self.plot_widget_parameters_tab = self.findChild(PlotWidget, "line_image_view_parameters")
        self.plot_widget_lines_tab = self.findChild(PlotWidget, "line_image_view")

    def setup_connections(self):
        self.pushButton_ImageFolder.pressed.connect(self.prepare_image)

    def prepare_image(self):
        self.image_loader.load_images_from_folder()
        for image in self.images_list.images_list:
            self.image_display_manager.display_image_on_parameters_tab(image)
            self.image_display_manager.display_image_on_lines_tab(image)
        self.table_controller.update_with_image(self.images_list.images_list)
