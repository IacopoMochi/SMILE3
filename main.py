import sys
import os
from PyQt6 import QtWidgets, uic

from smile_lines_image_class import SmileLinesImage
from smile_image_list_class import LineImageList

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("SMILE3.ui", self)
        self.line_images = LineImageList()
        self.pushButton_ImageFolder.pressed.connect(self.load_lines_image)

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
                        self.line_images.lineImages.append(
                            SmileLinesImage(cnt, name, root, "lines")
                        )
                        SmileLinesImage.gather_parameters(self)
                        print(self.parameters["Threshold"])
                        print(self.parameters)

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
