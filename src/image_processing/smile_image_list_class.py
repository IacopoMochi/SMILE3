import numpy as np
from src.image_processing.image_container import SmileLinesImage


# class LineImageList(SmileLinesImage):
#     # TODO change inheritance to composition
#     lineImages: list
#     current_image: int
#
#     def __init__(self, id, file_name, path, feature):
#         # TODO why do we have second time attributes, why arbitrary attributes like 'average'
#         super().__init__(id, file_name, path, feature)
#         self.lineImages = list()
#         self.current_image = -1
#         self.frequency = None
#         self.image = 'average'
#         self.processed_image = 'average'
#
#     # gather edges data from multiple images into a single dataset
#     def gather_edges(self):
#         # TODO hardcoded 0 element?
#         self.consolidated_leading_edges = np.empty_like(self.lineImages[0].consolidated_leading_edges)
#         self.consolidated_trailing_edges = np.empty_like(self.lineImages[0].consolidated_leading_edges)
#         self.zero_mean_leading_edge_profiles = np.empty_like(self.lineImages[0].consolidated_leading_edges)
#         self.zero_mean_trailing_edge_profiles = np.empty_like(self.lineImages[0].consolidated_leading_edges)
#         for image in self.lineImages:
#             self.consolidated_leading_edges = np.concatenate(
#                 (self.consolidated_leading_edges, image.consolidated_leading_edges))
#             self.consolidated_trailing_edges = np.concatenate(
#                 (self.consolidated_trailing_edges, image.consolidated_trailing_edges))
#             self.zero_mean_leading_edge_profiles = np.concatenate(
#                 (self.zero_mean_leading_edge_profiles, image.zero_mean_leading_edge_profiles))
#             self.zero_mean_trailing_edge_profiles = np.concatenate(
#                 (self.zero_mean_trailing_edge_profiles, image.zero_mean_trailing_edge_profiles))
#             self.frequency = image.frequency


class LineImageList:
    def __init__(self):
        self.lineImages = []
        self.consolidated_leading_edges = None
        self.consolidated_trailing_edges = None
        self.zero_mean_leading_edge_profiles = None
        self.zero_mean_trailing_edge_profiles = None
        self.average_frequency = None
        self.current_image = -1
        self.image = 'average'
        self.processed_image = 'average'

    def add_image(self, image: SmileLinesImage):
        if isinstance(image, SmileLinesImage):
            self.lineImages.append(image)
        else:
            raise ValueError('Invalid image type')

    def gather_edges(self):
        if not self.lineImages:
            return None

        leading_edges = []
        trailing_edges = []
        zero_mean_leading_edge_profiles = []
        zero_mean_trailing_edge_profiles = []
        frequency_set = set()

        for image in self.lineImages:
            leading_edges.append(image.consolidated_leading_edges)
            trailing_edges.append(image.consolidated_trailing_edges)
            zero_mean_leading_edge_profiles.append(image.zero_mean_leading_edge_profiles)
            zero_mean_trailing_edge_profiles.append(image.zero_mean_trailing_edge_profiles)
            frequency_set.add(image.frequency)

        self.consolidated_leading_edges = np.concatenate(leading_edges)
        self.consolidated_trailing_edges = np.concatenate(trailing_edges)
        self.zero_mean_leading_edge_profiles = np.concatenate(zero_mean_leading_edge_profiles)
        self.zero_mean_trailing_edge_profiles = np.concatenate(zero_mean_trailing_edge_profiles)
        if len(frequency_set) == 1:
            self.average_frequency = frequency_set.pop()
        else:
            raise ValueError('Frequency not consistent')


class ContactImageList:
    contactImages: list
    current_image: int

    def __init__(self):
        self.contactImages = list()
        self.current_image = -1
