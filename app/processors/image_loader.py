import os
from PIL import Image
from PIL.ExifTags import TAGS

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

            info_dict = {
                "Filename": img.filename,
                "Image Size": img.size,
                "Image Height": img.height,
                "Image Width": img.width,
                "Image Format": img.format,
                "Image Mode": img.mode,
                "Image is Animated": getattr(img, "is_animated", False),
                "Frames in Image": getattr(img, "n_frames", 1)
            }

            for label, value in info_dict.items():
                print(f"{label:25}: {value}")

            metadata = img.getexif()
            for tag_id in metadata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                # decode bytes
                if isinstance(data, bytes):
                    data = data.decode()
                print(f"{tag:25}: {data}")
            return img
