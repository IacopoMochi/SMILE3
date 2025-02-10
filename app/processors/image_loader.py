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
            # for label, value in info_dict.items():
            #     print(f"{label:25}: {value}")

            metadata = img.getexif()
            for tag_id in metadata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = metadata.get(tag_id)
                # decode bytes
                if isinstance(data, bytes):
                    data = data.decode()
                S = (f"{data}")
                idx = S.find("Image Pixel Size = ")
                if idx > 0:
                    SC = S[idx:idx+35]
                    end_of_line = (SC.find('\n'))
                    equal = (SC.find('='))
                    S = S[idx:idx + end_of_line - 1]
                    if S[-2:] == "pm":
                        scaling_factor = 0.001
                    elif S[-2:] == "nm":
                        scaling_factor = 1
                    pixel_size = np.double(SC[equal+2: end_of_line-3])
                    print(pixel_size*scaling_factor)

            for tag, value in img.tag_v2.items():
                stringa = ""
                if tag == 40092:
                    for byte in value:
                        if byte > 0:
                            stringa = stringa + chr(byte)
                    index_PixelSize = stringa.find("PixelSize")
                    stringa = stringa[index_PixelSize+10:-1]
                    index_return = stringa.find("\n")
                    stringa=stringa[0:index_return-1]
                    print(stringa)




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
