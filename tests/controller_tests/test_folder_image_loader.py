import unittest
from unittest.mock import patch, MagicMock
from app.controllers.folder_image_loader import FolderImageLoader


class TestFolderImageLoader(unittest.TestCase):
    @patch('app.controllers.folder_image_loader.QtWidgets.QFileDialog.getExistingDirectory')
    @patch('app.controllers.folder_image_loader.os.walk')
    @patch('app.controllers.folder_image_loader.Image.load_image')
    @patch('app.controllers.folder_image_loader.gather_parameters')
    def test_load_images_from_folder(self, mock_gather_parameters, mock_load_image, mock_os_walk,
                                     mock_get_existing_directory):
        """
        Tests the load_images_from_folder method of the FolderImageLoader class.

        This test checks the following:
        1. The method correctly opens a file dialog to select a folder.
        2. It iterates over the files in the selected folder, filtering for supported image formats.
        3. It loads each valid image, gathers parameters, and adds it to the images list.
        4. It handles PermissionError and generic exceptions properly, showing error messages when necessary.

        The test uses mocks to simulate:
        - The folder selection dialog returning a predefined folder path.
        - The os.walk function returning a predefined directory structure with a mix of image and non-image files.
        - The Image.load_image method to avoid actual image loading.
        - The gather_parameters function to avoid actual parameter gathering.

        Assertions check:
        - The number of images added to the images list matches the expected count.
        - The filenames of the images added to the list are as expected.
        - The load_image and gather_parameters functions are called appropriately.
        - The correct error messages are shown if exceptions occur.
        """

        mock_get_existing_directory.return_value = '/mocked/folder'
        mock_os_walk.return_value = [
            ('/mocked/folder', ('subdir',), ('image1.jpg', 'image2.png', 'document.txt'))
        ]

        mock_images_list = MagicMock()
        mock_images_list.images_list = []

        mock_window = MagicMock()

        folder_image_loader = FolderImageLoader(mock_images_list, mock_window)

        folder_image_loader.load_images_from_folder()

        self.assertEqual(len(mock_images_list.add_image_to_list.call_args_list), 2)
        self.assertEqual(mock_images_list.add_image_to_list.call_args_list[0][0][0].file_name, 'image1.jpg')
        self.assertEqual(mock_images_list.add_image_to_list.call_args_list[1][0][0].file_name, 'image2.png')
        mock_load_image.assert_called()
        mock_gather_parameters.assert_called()

        mock_load_image.side_effect = PermissionError("Mocked permission error")
        folder_image_loader.load_images_from_folder()
        mock_window.show_error_message.assert_called_with("PermissionError: Mocked permission error")

        mock_load_image.side_effect = Exception("Mocked generic error")
        folder_image_loader.load_images_from_folder()
        mock_window.show_error_message.assert_called_with(
            "An unexpected error occurred: Mocked generic error."
            "Make sure that you have selected at least one image to process"
        )

if __name__ == '__main__':
    unittest.main()
