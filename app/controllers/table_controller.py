from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtCore import Qt
from app.models.image_container import Image


class TableController(QtWidgets.QWidget):
    def __init__(self, table_widget: QTableWidget):
        self.table_widget = table_widget
        self.configure_table()

    def configure_table(self):
        self.table_widget.setColumnCount(9)
        self.table_widget.setHorizontalHeaderLabels(["Selected", "Processed", "Name", "N of Lines", "Average pitch",
                                                     "Average CD", "CD std", "Unbiased LWR", "Unbiased LWR fit"])

    def update_with_image(self, images_list):
        self.table_widget.setRowCount(len(images_list) + 1)
        for idx, image in enumerate(images_list):
            check_box = QTableWidgetItem()
            check_box.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_box.setCheckState(Qt.CheckState.Unchecked if not image.selected else Qt.CheckState.Checked)

            image_name = image.file_name.split('.')[0]

            self.table_widget.setItem(idx, 0, check_box)
            self.table_widget.setItem(idx, 1, QTableWidgetItem('Yes' if image.processed else 'No'))
            self.table_widget.setItem(idx, 2, QTableWidgetItem(image_name))

    def mark_image_as_processed(self, image_id):
        self.table_widget.setItem(image_id, 1, QTableWidgetItem("Yes"))

    def update_with_processed_image(self, image: Image):
        if image.critical_dimension_estimate is not None:
            item_averageCD = QtWidgets.QTableWidgetItem(f"{image.critical_dimension_estimate:.5f}")
        item_number_of_lines = QtWidgets.QTableWidgetItem(str(image.number_of_lines))
        if image.critical_dimension_std_estimate is not None:
            item_averageCDstd = QtWidgets.QTableWidgetItem(f"{image.critical_dimension_std_estimate:.5f}")
        if image.pitch_estimate is not None:
            item_pitchEstimate = QtWidgets.QTableWidgetItem(f"{image.pitch_estimate:.5f}")

        self.table_widget.setItem(image.id, 3, item_number_of_lines)
        self.table_widget.setItem(image.id, 4, item_pitchEstimate)
        self.table_widget.setItem(image.id, 5, item_averageCD)
        self.table_widget.setItem(image.id, 6, item_averageCDstd)
