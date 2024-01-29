
from smile_lines_image_class import SmileLinesImage
class LineImageList:
    lineImages: list
    current_image: int
    def __init__(self):
        self.lineImages = list()
        self.current_image = -1


class ContactImageList:
    contactImages: list
    current_image: int
    def __init__(self):
        self.contactImages = list()
        self.current_image = -1
