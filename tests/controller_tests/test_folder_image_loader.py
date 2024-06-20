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
