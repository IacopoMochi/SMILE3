# from src.image_processing.processors import pre_processing, find_edges, post_processing, calculate_metrics
from src.image_processing.image_loader import ImageLoader


# HELPER FUNCTIONS USED IN process_line_images() FUNCTION

class SmileLinesImage:
    def __init__(self, id, file_name, path, feature):
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
        self.image = None
        self.feature = feature
        self.frequency = None
        self.id = id
        self.selected = True
        self.processed = False

        self.image_loader = ImageLoader(self.folder, self.file_name)


        # self.image = self.image_loader.load_image()