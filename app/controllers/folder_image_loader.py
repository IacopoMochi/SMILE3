import os
from PyQt6 import QtWidgets
from app.models.image_container import Image
from app.processors.parameters_collector import gather_parameters


class FolderImageLoader:
    def __init__(self, images_list, parent):
        self.images_list = images_list
        self.parent = parent

    def load_images_from_folder(self):
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.parent, 'Select images folder')
        if not folder_path:
            return

        for root, dirs, files in os.walk(folder_path):
            image_id = 0
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.tiff', '.png', '.tif', '.jpeg')):
                    image_object = Image(image_id, root, file_name)
                    try:
                        image_object.load_image()
                        gather_parameters(self.parent, image_object)
                        self.images_list.add_image_to_list(image_object)
                        image_id += 1
                    except PermissionError as e:
                        print(f"PermissionError: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
