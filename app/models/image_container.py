from app.processors.image_loader import ImageLoader
from app.processors.image_processors import PreProcessor, EdgeDetector, MetricCalculator, PostProcessor, MultiTaper


class Image:
    """
    A container class to represent and process an image with methods for loading, preprocessing,
    edge detection, and metric calculation. The actual functionality is delegated to various
    helper classes.
    """

    def __init__(self, id, path, file_name):
        self.zero_mean_leading_edge_profiles = None
        self.zero_mean_trailing_edge_profiles = None
        self.LWR_PSD = None
        self.LWR_PSD_fit_parameters = None
        self.LWR_PSD_fit = None
        self.LWR_PSD_fit_unbiased = None
        self.LWR_PSD_unbiased = None
        self.LWR_3s = None
        self.LWR = None
        self.LWR_fit_parameters = None
        self.LWR_fit = None
        self.LWR_fit_unbiased = None
        self.LWR_unbiased = None

        self.LER_PSD = None
        self.LER_PSD_fit_parameters = None
        self.LER_PSD_fit = None
        self.LER_PSD_fit_unbiased = None
        self.LER_PSD_unbiased = None

        self.LER_Leading_PSD = None
        self.LER_Leading_PSD_fit_parameters = None
        self.LER_Leading_PSD_fit = None
        self.LER_Leading_PSD_fit_unbiased = None
        self.LER_Leading_PSD_unbiased = None

        self.LER_Trailing_PSD = None
        self.LER_Trailing_PSD_fit_parameters = None
        self.LER_Trailing_PSD_fit = None
        self.LER_Trailing_PSD_fit_unbiased = None
        self.LER_Trailing_PSD_unbiased = None

        self.consolidated_leading_edges = None
        self.consolidated_trailing_edges = None
        self.pitch_estimate = None
        self.parameters = None
        self.leading_edges = None
        self.trailing_edges = None
        self.number_of_lines = None
        self.critical_dimension = None
        self.critical_dimension_std_estimate = None
        self.critical_dimension_estimate = None
        self.pixel_size = None
        self.metrics = None
        self.processed_image = None
        self.selected_image = True
        self.intensity_histogram = None
        self.intensity_histogram_low = None
        self.intensity_histogram_high = None
        self.intensity_histogram_medium = None

        self.file_name = file_name
        self.folder = path
        self.frequency = None
        self.id = id
        self.selected = True
        self.processed = False
        self.image = None

        # store original values to restore after additional operations on consolidated edges
        self.basic_consolidated_leading_edges = None
        self.basic_consolidated_trailing_edges = None
        self.basic_zero_mean_leading_edge_profiles = None
        self.basic_zero_mean_trailing_edge_profiles = None

        # caches
        self.post_processing_cache = None
        self.multi_taper_cache = None

    def load_image(self) -> None:
        """
        A method that calls ImageLoader class for loading the image and rotates it.
        """

        image_loader = ImageLoader(self.folder, self.file_name)
        self.image = image_loader.load_image()

    def pre_processing(self) -> None:
        """
        A method that calls PreProcessing class for preprocessing images, including cropping, rotating, and normalizing.
        """

        pre_processor = PreProcessor(self)
        pre_processor.normalize_image()
        pre_processor.calculate_histogram_parameters()

    def find_edges(self) -> None:
        """
        A method that calls EdgeDetector class for detecting and analyzing edges in an image.
        """

        edge_detector = EdgeDetector(self)
        edge_detector.find_edges()

    def calculate_metrics(self) -> None:
        """
        A method that calls MetricCalculator for calculating the metrics of the image.
        """

        metric_calculator = MetricCalculator(self)
        metric_calculator.setup_frequency()
        metric_calculator.calculate_metrics()

    def post_processing(self, use_post_processing=False):
        """
        A method that calls PostProcessing class for calculating post-process consolidated edges.

        Args:
            use_post_processing (bool): Flag to determine if post-processing should be applied.
                                    Defaults to False.
        """

        post_processor = PostProcessor(self)
        post_processor.post_processing(use_post_processing)

    def multi_taper(self, use_multi_taper=False):
        """
        A method that calls MultiTaper class for calculating consolidated edges in multi-taper process.

        Args:
            use_multi_taper (bool): Flag to determine if multi-taper processing should be applied.
                                    Defaults to False.
        """

        multi_taper = MultiTaper(self)
        multi_taper.multi_taper(use_multi_taper)
