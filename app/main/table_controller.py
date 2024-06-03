from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt


class TableView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TableView, self).__init__(parent)
        uic.loadUi("view/table.ui", self)
        self.tableWidget = self.findChild(QtWidgets.QTableWidget, "tableWidget")
        self.configure_table()

    def configure_table(self):
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setHorizontalHeaderLabels(["Selected", "Processed", "Name", "N of Lines", "Average pitch", "Average CD",
                                                    "CD std", "Unbiased LWR", "Unbiased LWR fit"])

    def update_with_image(self, images_list):
        self.tableWidget.setRowCount(len(images_list) + 2)
        for idx, image in enumerate(images_list):
            check_box = QtWidgets.QTableWidgetItem()
            check_box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_box.setCheckState(Qt.CheckState.Unchecked if not image.selected else Qt.CheckState.Checked)

            self.tableWidget.setItem(idx, 0, check_box)
            self.tableWidget.setItem(idx, 1, QtWidgets.QTableWidgetItem(image.file_name))
            self.tableWidget.setItem(idx, 2, QtWidgets.QTableWidgetItem('Yes' if image.processed else 'No'))



