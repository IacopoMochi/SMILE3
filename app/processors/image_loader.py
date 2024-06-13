import os
from PIL import Image

import numpy as np


class ImageLoader:
    def __init__(self, folder: str, file_name: str):
        self.folder = folder
        self.file_name = file_name

    def load_image(self) -> np.ndarray:
        file_path = os.path.join(self.folder, self.file_name)
        with Image.open(file_path) as img:
            img = np.rot90(img, 3)
            return img
