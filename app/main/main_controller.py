import os
import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
import pyqtgraph as pg
from openpyxl import Workbook

from app.image.image_controller import Image
from app.image.images_list import ImagesList


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi("view/window.ui", self)
        self.setup_connections()
        self.images_list = ImagesList()

    def setup_connections(self):
        self.pushButton_ImageFolder.clicked.connect(self.load_images_from_folder)

    def load_images_from_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select images folder')
        if folder_path:
            for root, dirs, files in os.walk(folder_path):
                image_id = 0
                for file_name in files:
                    # what about .jpeg and .tif
                    if file_name.lower().endswith(('.jpg', '.tiff', '.png')):
                        image_object = Image(image_id, root, file_name)
                        image_object.load_image()
                        self.images_list.add_image_to_list(image_object)
                    image_id += 1


