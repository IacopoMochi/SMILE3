import sys
import os
import numpy as np
from PyQt6 import QtWidgets, uic

from smile_lines_image_class import SmileLinesImage
from smile_image_list_class import LineImageList

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("SMILE3.ui", self)
        self.line_image_list = LineImageList()
        self.pushButton_ImageFolder.pressed.connect(self.load_lines_image)

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
                        #SmileLinesImage.gather_parameters(self)
                        Image = SmileLinesImage(cnt, name, root, "lines")
                        self.gather_parameters(Image)
                        self.line_image_list.lineImages.append(
                            Image
                        )


                        print(self.line_image_list.lineImages[cnt].name)


    #def populate_lines_table(self):


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
