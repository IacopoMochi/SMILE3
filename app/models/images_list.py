from app.models.image_container import Image


class ImagesList:
    def __init__(self):
        self.images_list = []

    def add_image_to_list(self, image: Image):
        if isinstance(image, Image):
            self.images_list.append(image)
        else:
            raise ValueError('Invalid models type')
