from PyQt6.QtCore import Qt
from app.models.image_container import Image
from app.processors.parameters_collector import gather_parameters
from PyQt6 import QtWidgets


class ProcessingController:
    def __init__(self, window, images_list, table):
        self.window = window
        self.images_list = images_list
        self.table = table
        self.number_selected_images = None

    def set_up_progress_bar(self):
        self.window.image_progressBar.setMinimum(0)
        self.window.image_progressBar.setMaximum(self.number_selected_images)
        self.window.image_progressBar.setValue(0)

    def process_image(self, image: Image):
        if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
            gather_parameters(self.window, image)
            image.pre_processing()
            image.find_edges()
            image.calculate_metrics()
            image.processed = True

    def update_progress_bar(self, number_processed_images):
        self.window.image_progressBar.setValue(number_processed_images)
        self.window.status_label.setText(f'Processing {number_processed_images + 1} of {self.number_selected_images}')
        QtWidgets.QApplication.processEvents()

    def get_number_selected_images(self):
        number_selected_images = 0
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
                number_selected_images += 1
        self.number_selected_images = number_selected_images
