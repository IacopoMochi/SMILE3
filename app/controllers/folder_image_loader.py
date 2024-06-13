import os
from PyQt6 import QtWidgets
from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.processors.parameters_collector import gather_parameters


class FolderImageLoader:
    def __init__(self, images_list: ImagesList, window):
        self.images_list = images_list
        self.window = window

    def load_images_from_folder(self) -> None:
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Select images folder')
        if not folder_path:
            return

        for root, dirs, files in os.walk(folder_path):
            image_id = len(self.images_list.images_list)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.tiff', '.png', '.tif', '.jpeg')):
                    image_object = Image(image_id, root, file_name)
                    try:
                        image_object.load_image()
                        gather_parameters(self.window, image_object)
                        self.images_list.add_image_to_list(image_object)
                        image_id += 1
                    except PermissionError as e:
                        self.window.show_error_message(f"PermissionError: {str(e)}")
                    except Exception as e:
                        self.window.show_error_message(f"An unexpected error occurred: {str(e)}."
                                                       f"Make sure that you have selected at least one image to process")
