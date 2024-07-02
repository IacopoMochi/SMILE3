from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QTableWidget

from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.processors.parameters_collector import gather_parameters


class ProcessingController:
    """
    A controller class to manage the processing of images in the application.

    Attributes:
        window: The main application window for user interaction.
        images_list (ImagesList): The list of images to be processed.
        table (QTableWidget): The table widget displaying the images.
        number_selected_images (int): The number of selected images to be processed.
    """

    def __init__(self, window, images_list: ImagesList, table: QTableWidget):
        self.window = window
        self.images_list = images_list
        self.table = table
        self.number_selected_images = None

    def set_up_progress_bar(self) -> None:
        """
        Sets up the progress bar in the main window for tracking image processing progress.
        """

        self.window.image_progressBar.setMinimum(0)
        self.window.image_progressBar.setMaximum(self.number_selected_images)
        self.window.image_progressBar.setValue(0)

    def process_image(self, image: Image) -> None:
        """
        Processes a single image if it is selected in the table.

        The image goes through several processing steps from the Image class, which in turn
        utilizes methods from the ProcessingImage class: gathering parameters, pre-processing, finding edges,
        and calculating metrics. If an error occurs during processing, an error message is displayed.

        Args:
            image (Image): The image to be processed.
        """

        if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
            try:
                gather_parameters(self.window, image)
                image.pre_processing()
                image.find_edges()
                image.post_processing(True if self.window.checkBox_9.isChecked() else False)
                image.multi_taper(True if self.window.radioButton_26.isChecked() else False)
                image.calculate_metrics()
                image.processed = True
            except Exception as e:
                # print(e)
                self.window.show_error_message(f"Error has occurred while processing image: {e}")

    def update_progress_bar(self, number_processed_images: int) -> None:
        """
        Updates the progress bar and status label to reflect the number of processed images.

        Args:
            number_processed_images (int): The number of images that have been processed so far.
        """

        self.window.image_progressBar.setValue(number_processed_images)
        self.window.status_label.setText(f'Processing {number_processed_images} of {self.number_selected_images}')
        QtWidgets.QApplication.processEvents()

    def get_number_selected_images(self) -> None:
        """
        Iterates through the images list and counts how many images are selected to be processed in the table.
        """

        number_selected_images = 0
        for image in self.images_list.images_list:
            if self.table.item(image.id, 0).checkState() == Qt.CheckState.Checked:
                number_selected_images += 1
        self.number_selected_images = number_selected_images
