# from src.image_processing.processors import pre_processing, find_edges, post_processing, calculate_metrics
from src.image_processing.image_loader import ImageLoader
from src.image_processing.processors import PreProcessor, EdgeDetector, MetricCalculator


# HELPER FUNCTIONS USED IN process_line_images() FUNCTION

class SmileLinesImage:
    def __init__(self, id, file_name, path, feature):
        # TODO ask where those atribiutes are uesed
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
        #s = os.path.join(path, file_name)
        #img = Image.open(s)
        #img = np.rot90(img, 3)
        #self.image = img
        self.feature = feature
        self.frequency = None
        self.id = id
        self.selected = True
        self.processed = False
        self.image = None

    def load_image(self):
        image_loader = ImageLoader(self.folder, self.file_name)
        self.image = image_loader.load_image()


    def pre_processing(self):
        histogram_params = PreProcessor().calculate_histogram_parameters(self.image, self.parameters)
        self.processed_image = histogram_params.processed_image
        self.intensity_histogram_medium = histogram_params.intensity_histogram_medium
        self.intensity_histogram_low = histogram_params.intensity_histogram_low
        self.intensity_histogram_high = histogram_params.intensity_histogram_high
        self.intensity_histogram = histogram_params.intensity_histogram

    def find_edges(self):
        edge_detector = EdgeDetector().find_edges(self.processed_image, self.parameters)
        self.pitch_estimate = edge_detector.pitch_estimate
        self.critical_dimension = edge_detector.critical_dimension
        self.critical_dimension_std_estimate = edge_detector.critical_dimension_std_estimate
        self.critical_dimension_estimate = edge_detector.critical_dimension_estimate
        self.number_of_lines = edge_detector.number_of_lines
        self.zero_mean_leading_edge_profiles = edge_detector.zero_mean_leading_edge_profiles
        self.zero_mean_trailing_edge_profiles = edge_detector.zero_mean_trailing_edge_profiles
        self.consolidated_leading_edges = edge_detector.consolidated_leading_edges
        self.consolidated_trailing_edges = edge_detector.consolidated_trailing_edges
        self.trailing_edges = edge_detector.trailing_edges
        self.leading_edges = edge_detector.leading_edges

    def calculate_metrics(self):
        metric_calculator = MetricCalculator().calculate_metrics(self.parameters, self.consolidated_leading_edges,
                                                                 self.consolidated_trailing_edges,
                                                                 self.zero_mean_leading_edge_profiles,
                                                                 self.zero_mean_trailing_edge_profiles)
        self.LWR_PSD = metric_calculator.LWR_PSD
        self.LWR_PSD_fit_parameters = metric_calculator.LWR_PSD_fit_parameters
        self.LWR_PSD_fit = metric_calculator.LWR_PSD_fit
        self.LWR_PSD_fit_unbiased = metric_calculator.LWR_PSD_fit_unbiased
        self.LWR_PSD_unbiased = metric_calculator.LWR_PSD_unbiased

        self.LER_PSD = metric_calculator.LER_PSD
        self.LER_PSD_fit_parameters = metric_calculator.LER_PSD_fit_parameters
        self.LER_PSD_fit = metric_calculator.LER_PSD_fit
        self.LER_PSD_unbiased = metric_calculator.LER_PSD_unbiased
        self.LER_PSD_fit_unbiased = metric_calculator.LER_PSD_fit_unbiased

        self.LER_Leading_PSD = metric_calculator.LER_Leading_PSD
        self.LER_Leading_PSD_fit_parameters = metric_calculator.LER_Leading_PSD_fit_parameters
        self.LER_Leading_PSD_fit = metric_calculator.LER_Leading_PSD_fit
        self.LER_Leading_PSD_unbiased = metric_calculator.LER_Leading_PSD_unbiased
        self.LER_Leading_PSD_fit_unbiased = metric_calculator.LER_Leading_PSD_fit_unbiased

        self.LER_Trailing_PSD = metric_calculator.LER_Trailing_PSD
        self.LER_Trailing_PSD_fit_parameters = metric_calculator.LER_Trailing_PSD_fit_parameters
        self.LER_Trailing_PSD_fit = metric_calculator.LER_Trailing_PSD_fit
        self.LER_Trailing_PSD_unbiased = metric_calculator.LER_Trailing_PSD_unbiased
        self.LER_Trailing_PSD_fit_unbiased = metric_calculator.LER_Trailing_PSD_fit_unbiased
