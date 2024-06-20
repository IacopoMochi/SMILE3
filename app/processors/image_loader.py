import os
from PIL import Image

import numpy as np


class ImageLoader:
    """
    A class to load single image from file path.

    Attributes:
        folder (str): The folder where the image is located.
        file_name (str): The name of the image file.
    """

    def __init__(self, folder: str, file_name: str):
        self.folder = folder
        self.file_name = file_name

    def load_image(self) -> np.ndarray:
        """
        Loads and rotates an image by 270 degrees counterclockwise.

        Returns:
            np.ndarray: The loaded and rotated image as a NumPy array.
        """

        file_path = os.path.join(self.folder, self.file_name)
        with Image.open(file_path) as img:
            img = np.rot90(img, 3)
            return img
