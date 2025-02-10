import os
from PIL import Image
from PIL.ExifTags import TAGS

import numpy as np


class ImageLoader:
    """
    A class to load single image from file path and read the pixel size from the file header if available

    Attributes:
        folder (str): The folder where the image is located.
        file_name (str): The name of the image file.
    """

    def __init__(self, folder: str, file_name: str):
        self.folder = folder
        self.file_name = file_name

    def load_image(self) -> tuple[np.ndarray, np.double]:
        """
        Loads and rotates an image by 270 degrees counterclockwise.

        Returns:
            np.ndarray: The loaded and rotated image as a NumPy array.
        """

        file_path = os.path.join(self.folder, self.file_name)
        pixel_size = 1.0  # Assign the default pixel size value in nanometers

        with Image.open(file_path) as img:
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

            # Procedure to read the pixel size from a Zeiss SEM

            metadata = img.getexif()
            for tag_id in metadata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = metadata.get(tag_id)
                # decode bytes
                if isinstance(data, bytes):
                    data = data.decode()
                S = f"{data}"
                idx = S.find("Image Pixel Size = ")
                if idx > 0:
                    SC = S[idx:idx+35]
                    end_of_line = (SC.find('\n'))
                    equal = (SC.find('='))
                    S = S[idx:idx + end_of_line - 1]
                    if S[-2:] == "pm":
                        scaling_factor = 0.001
                    elif S[-2:] == "nm":
                        scaling_factor = 1.0
                    pixel_size = np.double(SC[equal+2: end_of_line-3])*scaling_factor

            # Procedure to read the pixel size from a Hitachi SEM
            for tag, value in img.tag_v2.items():
                stringa = ""
                if tag == 40092:
                    for byte in value:
                        if byte > 0:
                            stringa = stringa + chr(byte)
                    index_PixelSize = stringa.find("PixelSize")
                    stringa = stringa[index_PixelSize+10:-1]
                    index_return = stringa.find("\n")
                    stringa = stringa[0:index_return-1]
                    pixel_size = np.double(stringa)


            img = np.rot90(img, 3)

        return img, pixel_size
