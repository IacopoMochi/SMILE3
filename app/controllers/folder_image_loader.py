import os
from PyQt6 import QtWidgets
from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.processors.parameters_collector import gather_parameters


class FolderImageLoader:
    """
    A class to load images from a specified folder into an ImagesList.

    Attributes:
        images_list (ImagesList): The list to store loaded images.
        window: The main application window for user interaction.
    """
    def __init__(self, images_list: ImagesList, window):
        self.images_list = images_list
        self.window = window

    def load_images_from_folder(self) -> None:
        """
       Loads images from a user-specified folder into the images list.

       Opens a dialog to select a folder, iterates through the files in the folder,
       and loads images with supported formats (.jpg, .tiff, .png, .tif, .jpeg).
       Gather parameters for each image and add them to the images list.

       Raises:
           PermissionError: If there is a permission error when accessing the files.
           Exception: For any unexpected errors that occur during image loading.
       """

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
