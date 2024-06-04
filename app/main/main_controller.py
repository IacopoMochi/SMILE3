import os
import sys
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from openpyxl import Workbook

from app.image.image_controller import Image
from app.image.images_list import ImagesList
from app.main.table_controller import TableController


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("view/window.ui", self)
        self.images_list = ImagesList()
        self.setup_connections()

        self.linesTable = self.findChild(QtWidgets.QTableWidget, "linesTable")
        self.table_controller = TableController(self.linesTable)

    def setup_connections(self):
        self.pushButton_ImageFolder.pressed.connect(self.load_images_from_folder)

    def load_images_from_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select images folder')
        if folder_path:
            for root, dirs, files in os.walk(folder_path):
                image_id = 0
                for file_name in files:
                    if file_name.lower().endswith(('.jpg', '.tiff', '.png', '.tif', '.jpeg')):
                        image_object = Image(image_id, root, file_name)
                        try:
                            image_object.load_image()
                            self.images_list.add_image_to_list(image_object)
                            image_id += 1
                        except PermissionError as e:
                            print(f"PermissionError: {e}")
                        except Exception as e:
                            print(f"An unexpected error occurred: {e}")

            self.table_controller.update_with_image(self.images_list.images_list)