from app.models.image_container import Image


class ImagesList:
    """
    A class to manage a list of images.

    Attributes:
        images_list (list[Image]): A list to store image objects.
    """
    def __init__(self):
        self.images_list = []

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
        else:
            raise ValueError('Invalid models type')
