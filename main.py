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

    def display_lines_data(self, Image):
        self.line_image_view.setImage(np.array(Image.image))

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
                        Image = SmileLinesImage(cnt, name, root, "lines")
                        self.gather_parameters(Image)
                        self.line_image_list.lineImages.append(
                            Image
                        )
                        item_name = QtWidgets.QTableWidgetItem(Image.file_name)

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
                        self.display_lines_data(Image)




app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
