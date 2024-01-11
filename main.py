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
        self.process_lines_button.pressed.connect(self.load_lines_image)

    def process_line_images(self, Image):
        Image.pre_processing()
    def display_lines_data(self, Image):

        # Display lines image on the top axis of the lines tab
        # Check if a processed image exists
        if (Image.processed == True) and not(Image.processed_image is None):
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
        if (Image.processed == True) and not (Image.processed_image is None):
            print("Work to do")



        #image_view = self.line_image_view.getView()
        #pci = pg.PlotCurveItem(x=[1, 50, 100, 150, 200], y=[1, 50, 100, 150, 200])
        #image_view.addItem(pci)

    def gather_parameters(self, Image):
        parameters = {'Threshold': np.double(window.threshold_line_edit.text()),
                      'MinPeakDistance': np.double(window.minPeakDistance_line_edit.text()),
                      'MinPeakProminence': np.double(window.minPeakProminence_line_edit.text()),
                      'PixelSize': np.double(window.pixelSize_line_edit.text()),
                      'X1': np.double(window.X1.text()),
                      'X2': np.double(window.X2.text()),
                      'Y1': np.double(window.Y1.text()),
                      'Y2': np.double(window.Y2.text())
                      }
        Image.parameters = parameters

    def load_lines_image(self):
        self.linesTable.setRowCount(4)
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
                        #self.linesTable.setRowCount (cnt+2)
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

                        self.linesTable.setItem(cnt, 0, item_selected)
                        self.linesTable.setItem(cnt, 1, item_processed)
                        self.linesTable.setItem(cnt, 2, item_name)
                        self.display_lines_data(image_object)




app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
