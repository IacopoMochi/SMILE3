import os
from PyQt6 import QtWidgets
from app.models.image_container import Image
from app.models.images_list import ImagesList
from app.processors.parameters_collector import gather_parameters
from PyQt6.QtCore import QSettings


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
        self.settings = QSettings("PSI", "SMILE3")

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
        folder_path = self.get_folder_path()
        self.save_last_used_directory(folder_path)

        for root, dirs, files in os.walk(folder_path):
            image_id = len(self.images_list.images_list)
            for file_name in files:
                if file_name.lower().endswith(('.jpg', '.tiff', '.png', '.tif', '.jpeg')):
                    image_object = Image(image_id, root, file_name)
                    try:
                        image_object.load_image()
                        gather_parameters(self.window, image_object)
                        self.images_list.add_image_to_list(image_object)
                        self.window.pixelSize_line_edit.setText(f"{image_object.pixel_size}")
                        
                        image_id += 1
                    except PermissionError as e:
                        self.window.show_error_message(f"PermissionError: {str(e)}")
                    except Exception as e:
                        self.window.show_error_message(f"An unexpected error occurred: {str(e)}."
                                                       f"Make sure that you have selected at least one image to process")

    def get_folder_path(self) -> str:
        """
        Opens a QFileDialog to select a folder and retrieves its path.

        This method uses the last accessed directory from settings to open the dialog
        in the parent directory of the last accessed folder. If there is no last accessed
        directory, it opens in the default directory.

        Returns:
            str: The path to the selected folder. Returns an empty string if no folder is selected.
        """

        try:
            last_directory = self.settings.value("last_directory", "")

            if last_directory:
                parent_directory = os.path.dirname(last_directory)
            else:
                parent_directory = ""

            folder_path = QtWidgets.QFileDialog.getExistingDirectory(self.window, 'Select images folder', parent_directory)
            return folder_path
        except Exception as e:
            self.window.show_error_message(f"An error occurred while selecting folder: {str(e)}")

    def save_last_used_directory(self, folder_path: str) -> None:
        """
        Saves the directory path of the last accessed folder to settings.

        Args:
            folder_path (str): The path of the folder to be saved.
        """

        self.settings.setValue("last_directory", folder_path)



