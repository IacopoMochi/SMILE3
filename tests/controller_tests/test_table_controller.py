import unittest
from PyQt6.QtWidgets import QApplication, QTableWidget
from PyQt6.QtCore import Qt
from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.models.average_image import AverageImage
from app.controllers.table_controller import TableController


class TestTableController(unittest.TestCase):

    def setUp(self):
        self.app = QApplication([])

        self.table_widget = QTableWidget()
        self.table_controller = TableController(self.table_widget)

        self.image1 = Image(id=0, path="/path/to/image1", file_name="image1.jpg")
        self.image2 = Image(id=1, path="/path/to/image2", file_name="image2.jpg")

        self.image1.number_of_lines = 10
        self.image1.pitch_estimate = 1.23
        self.image1.critical_dimension_estimate = 2.34
        self.image1.critical_dimension_std_estimate = 0.45

        self.image2.number_of_lines = 15
        self.image2.pitch_estimate = 2.34
        self.image2.critical_dimension_estimate = 3.45
        self.image2.critical_dimension_std_estimate = 0.56

        self.images_list = ImagesList()
        self.images_list.images_list = [self.image1, self.image2]

    def test_configure_table(self):
        self.table_controller.configure_table()
        self.assertEqual(self.table_widget.columnCount(), 9)
        self.assertEqual(self.table_widget.horizontalHeaderItem(0).text(), "Selected")
        self.assertEqual(self.table_widget.horizontalHeaderItem(4).text(), "Average pitch")

    def test_update_with_image(self):
        self.table_controller.update_with_image(self.images_list)
        self.assertEqual(self.table_widget.rowCount(), len(self.images_list.images_list) + 2)

        for idx, image in enumerate(self.images_list.images_list):
            check_box = self.table_widget.item(idx, 0)
            self.assertTrue(check_box.flags() & Qt.ItemFlag.ItemIsUserCheckable)
            expected_check_state = Qt.CheckState.Checked if image.selected else Qt.CheckState.Unchecked
            self.assertEqual(check_box.checkState(), expected_check_state)
            self.assertEqual(self.table_widget.item(idx, 1).text(), 'No')
            self.assertEqual(self.table_widget.item(idx, 2).text(), image.file_name.split('.')[0])

    def test_mark_image_as_processed(self):
        self.table_controller.update_with_image(self.images_list)
        self.table_controller.mark_image_as_processed(0)
        self.assertEqual(self.table_widget.item(0, 1).text(), "Yes")

    def test_update_with_processed_image(self):
        self.table_controller.update_with_image(self.images_list)
        self.table_controller.update_with_processed_image(self.image1)

        self.assertEqual(self.table_widget.item(0, 3).text(), "10")
        self.assertEqual(self.table_widget.item(0, 4).text(), "1.23000")
        self.assertEqual(self.table_widget.item(0, 5).text(), "2.34000")
        self.assertEqual(self.table_widget.item(0, 6).text(), "0.45000")

    def test_add_average_image(self):
        average_image = AverageImage(self.images_list)

        row_count_needed = average_image.image.id + 1
        if self.table_widget.rowCount() < row_count_needed:
            self.table_widget.setRowCount(row_count_needed)

        self.table_controller.add_average_image(average_image)

        self.assertEqual(self.table_widget.item(average_image.image.id, 2).text(), "average")


if __name__ == '__main__':
    unittest.main()
