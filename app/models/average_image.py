import numpy as np
import copy

from app.models.images_list import ImagesList
from app.models.image_container import Image
from app.processors.image_processors import MetricCalculator


class AverageImage:
    """
    A class to create and manage an average image from a list of images.

    Attributes:
        image (Image): A deep copy of the first image in the images list, used to fill missing data.
        images_list (ImagesList): The list of images to average.
    """
    def __init__(self, images_list: ImagesList) -> None:
        self.image: Image = copy.deepcopy(images_list.images_list[0])
        self.image.id = len(images_list.images_list) + 1
        self.images_list = images_list

    def gather_edges(self) -> None:
        """
        Gathers and concatenates edges and zero-mean edge profiles from all images in the list.
        Ensures frequency consistency across all images.

        Raises:
            ValueError: If frequencies are not consistent across all images.
        """

        if not self.image:
            return None

        leading_edges = []
        trailing_edges = []
        zero_mean_leading_edge_profiles = []
        zero_mean_trailing_edge_profiles = []
        frequency_set = set()

        for image in self.images_list.images_list:
            leading_edges.append(image.consolidated_leading_edges)
            trailing_edges.append(image.consolidated_trailing_edges)
            zero_mean_leading_edge_profiles.append(image.zero_mean_leading_edge_profiles)
            zero_mean_trailing_edge_profiles.append(image.zero_mean_trailing_edge_profiles)
            frequency_set.add(tuple(image.frequency))

        self.image.consolidated_leading_edges = np.concatenate(leading_edges)
        self.image.consolidated_trailing_edges = np.concatenate(trailing_edges)
        self.image.zero_mean_leading_edge_profiles = np.concatenate(zero_mean_leading_edge_profiles)
        self.image.zero_mean_trailing_edge_profiles = np.concatenate(zero_mean_trailing_edge_profiles)
        if len(frequency_set) == 1:
            self.image.average_frequency = frequency_set.pop()
        else:
            raise ValueError('Frequency not consistent')

    def calculate_metrics(self) -> None:
        """
        Calculates metrics for the average image using the MetricCalculator class.
        """

        MetricCalculator(self.image).calculate_metrics()

    def prepare_average_image(self) -> None:
        """
        Prepares the average image by calling gathering edges and calculating metrics functions.
        """

        self.gather_edges()
        self.calculate_metrics()

