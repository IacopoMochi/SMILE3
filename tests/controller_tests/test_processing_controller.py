import unittest
from unittest.mock import Mock, patch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QProgressBar, QLabel
from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.controllers.processing_controller import ProcessingController


class TestProcessingController(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.main_window = Mock(QMainWindow)
        self.main_window.image_progressBar = Mock(QProgressBar)
        self.main_window.status_label = Mock(QLabel)
        self.main_window.show_error_message = Mock()
        self.images_list = ImagesList()
        self.table = QTableWidget(5, 1)  # Assuming 5 images for the test
        self.controller = ProcessingController(self.main_window, self.images_list, self.table)

        for i in range(5):
            image = Image(id=i, path=f"/mock/path/image_{i}.png", file_name=f"image_{i}.png")
            self.images_list.images_list.append(image)
            item = QTableWidgetItem()
            item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(i, 0, item)

    def tearDown(self):
        patch.stopall()

    def test_set_up_progress_bar(self):
        self.controller.number_selected_images = 3
        self.controller.set_up_progress_bar()
        self.main_window.image_progressBar.setMinimum.assert_called_with(0)
        self.main_window.image_progressBar.setMaximum.assert_called_with(3)
        self.main_window.image_progressBar.setValue.assert_called_with(0)

    @patch('app.controllers.processing_controller.gather_parameters')
    @patch.object(Image, 'pre_processing')
    @patch.object(Image, 'find_edges')
    @patch.object(Image, 'calculate_metrics')
    def test_process_image(self, mock_calculate_metrics, mock_find_edges, mock_pre_processing, mock_gather_parameters):
        image = self.images_list.images_list[0]
        self.table.item(image.id, 0).setCheckState(Qt.CheckState.Checked)

        self.controller.process_image(image)

        mock_gather_parameters.assert_called_once_with(self.main_window, image)
        mock_pre_processing.assert_called_once()
        mock_find_edges.assert_called_once()
        mock_calculate_metrics.assert_called_once()
        self.assertTrue(image.processed)

    @patch('app.controllers.processing_controller.gather_parameters')
    @patch.object(Image, 'pre_processing')
    @patch.object(Image, 'find_edges')
    @patch.object(Image, 'calculate_metrics')
    def test_process_image_with_exception(self, mock_calculate_metrics, mock_find_edges, mock_pre_processing,
                                          mock_gather_parameters):
        image = self.images_list.images_list[0]
        self.table.item(image.id, 0).setCheckState(Qt.CheckState.Checked)

        mock_pre_processing.side_effect = Exception('Test Exception')
        self.controller.process_image(image)

        mock_gather_parameters.assert_called_once_with(self.main_window, image)
        self.main_window.show_error_message.assert_called_once_with(
            "Error has occurred while processing image: Test Exception")

    def test_update_progress_bar(self):
        self.controller.update_progress_bar(2)
        self.main_window.image_progressBar.setValue.assert_called_with(2)
        self.main_window.status_label.setText.assert_called_with('Processing 2 of None')

    def test_get_number_selected_images(self):
        self.table.item(0, 0).setCheckState(Qt.CheckState.Checked)
        self.table.item(2, 0).setCheckState(Qt.CheckState.Checked)
        self.table.item(4, 0).setCheckState(Qt.CheckState.Checked)

        self.controller.get_number_selected_images()

        self.assertEqual(self.controller.number_selected_images, 3)


if __name__ == '__main__':
    unittest.main()
