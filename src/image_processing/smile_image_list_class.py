import numpy as np
from src.image_processing.image_models import SmileLinesImage


class LineImageList(SmileLinesImage):
    lineImages: list
    current_image: int

    def __init__(self, id, file_name, path, feature):
        super().__init__(id, file_name, path, feature)
        self.lineImages = list()
        self.current_image = -1
        self.frequency = None
        self.image = 'average'
        self.processed_image = 'average'

# gather edges data from multiple images into a single dataset
    def gather_edges(self):
        self.consolidated_leading_edges = np.empty_like(self.lineImages[0].consolidated_leading_edges)
        self.consolidated_trailing_edges = np.empty_like(self.lineImages[0].consolidated_leading_edges)
        self.zero_mean_leading_edge_profiles = np.empty_like(self.lineImages[0].consolidated_leading_edges)
        self.zero_mean_trailing_edge_profiles = np.empty_like(self.lineImages[0].consolidated_leading_edges)
        for image in self.lineImages:
            self.consolidated_leading_edges = np.concatenate(
                (self.consolidated_leading_edges, image.consolidated_leading_edges))
            self.consolidated_trailing_edges = np.concatenate(
                (self.consolidated_trailing_edges, image.consolidated_trailing_edges))
            self.zero_mean_leading_edge_profiles = np.concatenate(
                (self.zero_mean_leading_edge_profiles, image.zero_mean_leading_edge_profiles))
            self.zero_mean_trailing_edge_profiles = np.concatenate(
                (self.zero_mean_trailing_edge_profiles, image.zero_mean_trailing_edge_profiles))
            self.frequency = image.frequency


class ContactImageList:
    contactImages: list
    current_image: int

    def __init__(self):
        self.contactImages = list()
        self.current_image = -1
