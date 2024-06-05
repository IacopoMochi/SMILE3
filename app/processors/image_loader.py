import os
from PIL import Image

import numpy as np


class ImageLoader:
    def __init__(self, folder, file_name):
        self.folder = folder
        self.file_name = file_name

    # construct the path, open models, rotate to correct position
    def load_image(self):
        # Load the models
        file_path = os.path.join(self.folder, self.file_name)
        with Image.open(file_path) as img:
            img = np.rot90(img, 3)
            return img
