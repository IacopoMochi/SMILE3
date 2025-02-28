from app.models.image_container import Image


class ImagesList:
    """
    A class to manage a list of images.

    Attributes:
        images_list (list[Image]): A list to store image objects.
        active_image (int): The index of the currently selected image.
    """
    def __init__(self):
        self.images_list = []
        self.active_image = -1

    def add_image_to_list(self, image: Image) -> None:
        """
        Adds an image to the list if it is an instance of the Image class.

        Args:
            image (Image): The image object to add.

        Raises:
            ValueError: If the provided image is not an instance of the Image class.
        """

        if isinstance(image, Image):
            self.images_list.append(image)
            self.active_image = len(self.images_list)-1
        else:
            raise ValueError('Invalid models type')
