import os
from PIL import Image

import numpy as np


class ImageLoader:
    def __init__(self, folder, file_name):
        self.folder = folder
        self.file_name = file_name

    # construct the path, open image, rotate to correct position
    def load_image(self):
        # Load the image
        file_path = os.path.join(self.file_name, self.folder)
        with Image.open(file_path) as img:
            img = np.rot90(img, 3)
            return img
