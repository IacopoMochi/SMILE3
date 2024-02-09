
from smile_lines_image_class import SmileLinesImage
class LineImageList(SmileLinesImage):
    lineImages: list
    current_image: int

    def __init__(self, id, file_name, path, feature):
        super().__init__(id, file_name, path, feature)
        self.lineImages = list()
        self.current_image = -1
    def gather_edges(self):
        self.leading_edges = []
        self.trailing_edges = []
        for image in self.lineImages:
            self.leading_edges.append(image.zero_mean_leading_edge_profiles)
            self.trailing_edges.append(image.zero_mean_trailing_edge_profiles)


class ContactImageList:
    contactImages: list
    current_image: int
    def __init__(self):
        self.contactImages = list()
        self.current_image = -1
