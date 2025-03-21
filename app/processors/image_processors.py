from typing import Callable, Union, Tuple, Any, Iterable

from numpy import ndarray
from scipy.optimize import minimize, Bounds
from scipy.signal import medfilt2d, filtfilt, butter, find_peaks
import numpy as np
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit
from scipy.ndimage import histogram
from copy import deepcopy, copy

from app.utils.poly import _poly11, poly11, binary_image_histogram_model, gaussian_profile
from app.utils.psd import Palasantzas_2_minimize, Palasantzas_2_beta, Palasantzas_2b
from app.utils.hhcf import hhcf_, hhcf_minimize

from app.utils.processors_service import edge_consolidation, edge_mean_subtraction

class PreProcessor:
    """
    Class for preprocessing images, including cropping, rotating, and normalizing.
    """

    def __init__(self, image):
        self.image = image

    def crop_and_rotate_image(self) -> np.ndarray:
        """
        Crops and rotates the image to align it for further processing.
        """
        x1, x2 = int(self.image.parameters["X1"]), int(self.image.parameters["X2"])
        y1, y2 = int(self.image.parameters["Y1"]), int(self.image.parameters["Y2"])

        image_cropped = self.image.image[x1:x2, y1:y2]
        theta = np.linspace(85.0, 95.0, 50, endpoint=False)
        radius = np.min(np.array(image_cropped.shape)) / 2 - 2
        dim1, dim2 = image_cropped.shape
        x, y = np.meshgrid(
            np.linspace(-dim2 / 2 + 1, dim2 / 2 - 1, dim2),
            np.linspace(-dim1 / 2 + 1, dim1 / 2 - 1, dim1)
        )
        circle = np.array((x * x + y * y) < (radius * radius))
        sinogram = radon(image_cropped * circle.astype(int), theta=theta)
        R = np.sum(np.power(sinogram, 2), 0)
        max_id = np.argmax(R)

        rotated_image = rotate(self.image.image, -float(theta[max_id]) + 90, order=0)
        image_rotated_cropped = rotated_image[x1:x2, y1:y2]
        return image_rotated_cropped

    def remove_brightness_gradient(self) -> np.ndarray:
        """
        Removes the brightness gradient from the image.
        """
        image = self.crop_and_rotate_image()
        brightness_map = image > np.median(image)
        x, y = np.meshgrid(
            np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[0])
        )
        image_array = image[brightness_map]
        xdata = (x[brightness_map], y[brightness_map])
        parameters_initial_guess = [0, 0, np.median(image_array)]

        optimized_parameters, _ = curve_fit(
            _poly11, xdata, image_array, parameters_initial_guess
        )

        brightness = poly11((x, y), optimized_parameters)
        image_flattened = image - brightness
        image_flattened[image_flattened < 0] = 0

        return image_flattened

    def normalize_image(self) -> None:
        """
        Normalizes the image.
        """
        #image = self.remove_brightness_gradient()
        image = self.crop_and_rotate_image()
        filtered_image = medfilt2d(image, [5, 5])
        image_max = np.max(filtered_image)
        image_min = np.min(filtered_image)
        image_normalized = (image - image_min) / (image_max - image_min)
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        # ax1.imshow(image)
        # ax3.plot(histogram(filtered_image,image_min,image_max,100))
        # ax2.imshow(filtered_image)
        #
        # plt.show()
        self.image.processed_image = image_normalized

    def calculate_histogram_parameters(self) -> None:
        """
        Calculates histogram parameters of the normalized image.
        """

        histogram_values = histogram(self.image.processed_image, 0, 1, 256)
        nonzero_histogram_values = histogram_values[histogram_values > 0]
        intensity = np.linspace(0, 1, 256)
        nonzero_intensity = intensity[histogram_values > 0]
        self.image.intensity_histogram = nonzero_histogram_values
        self.image.intensity_values = nonzero_intensity

        max_index = np.argmax(self.image.intensity_histogram)
        max_value = self.image.intensity_histogram[max_index]
        low_bounds = [0, 0, 0.01, 0, 0.01, 0.01, 0, 0, 0.01]
        high_bounds = [max_value, 1, 1, max_value, 1, 1, max_value, 1, 2]
        beta0 = (max_value, 0.5, 0.1, max_value, 0.50, 0.1, 1, 0.5, 0.1)
        beta, _ = curve_fit(
            binary_image_histogram_model,
            self.image.intensity_values,
            self.image.intensity_histogram,
            p0=beta0,
            bounds=(low_bounds, high_bounds),
            maxfev=100000,
        )

        self.image.intensity_histogram_gaussian_fit_parameters = beta
        self.image.intensity_histogram_low = gaussian_profile(intensity, *beta[0:3])
        self.image.intensity_histogram_high = gaussian_profile(intensity, *beta[6:9])
        self.image.intensity_histogram_medium = gaussian_profile(intensity, *beta[3:6])
        self.image.lines_snr = np.abs(beta[1] - beta[4]) / (
            0.5 * (beta[2] + beta[5]) * 2 * np.sqrt(-2 * np.log(0.5))
        )


class EdgeDetector:
    """
    Class for detecting and analyzing edges in an image.
    """

    def __init__(self, image):
        self.image = image

    def edge_detection(self, new_edges: np.ndarray, edges_profiles: np.ndarray) -> np.ndarray:
        """
        Detects edges using polynomial, linear, threshold, or bright edge fitting.
        """

        cnt = -1
        image_size = self.image.processed_image.shape
        #print(self.image.parameters["EdgeSearchMethodRange"])
        #print(self.image.parameters["EdgeSearchMethodCDFraction"])
        if self.image.parameters["EdgeSearchMethodRange"]:
            edge_range = self.image.parameters["EdgeRange"]
        elif self.image.parameters["EdgeSearchMethodCDFraction"]:
            edge_range = np.round(self.image.parameters["CDFraction"] * self.image.pitch_estimate/2)

        #print(edge_range)

        for edge in new_edges:
            cnt = cnt + 1
            for row in range(0, image_size[1]):
                segment_start = int(np.max([0, edge - edge_range]))
                segment_end = int(np.min([edge + edge_range, image_size[0]]))
                x = np.arange(segment_start, segment_end)
                segment = self.image.processed_image[segment_start:segment_end, row]
                if self.image.parameters["Edge_fit_function"] == "polynomial":
                    p = np.polyfit(x, segment, 4)
                    p[-1] = p[-1] - np.double(self.image.parameters["Threshold"])
                    r = np.roots(p)
                    r = r[np.imag(r) == 0]
                    if len(r) > 0:
                        edge_position = r[np.argmin(np.abs(r - (segment_start + segment_end) / 2))]
                        r = np.real(edge_position)
                        if r > segment_end:
                            edge_position = segment_end
                        elif r < segment_start:
                            edge_position = segment_start
                        else:
                            edge_position = r
                        edges_profiles[cnt, row] = edge_position

                elif self.image.parameters["Edge_fit_function"] == "linear":
                    p = np.polyfit(x, segment, 1)
                    p[-1] = p[-1] - np.double(self.image.parameters["Threshold"])
                    r = np.roots(p)
                    r = r[np.imag(r) == 0]
                    if len(r) == 1:
                        if r > segment_end:
                            edge_position = segment_end
                        elif r < segment_start:
                            edge_position = segment_start
                        else:
                            edge_position = r
                        edges_profiles[cnt, row] = np.real(edge_position)
                elif self.image.parameters["Edge_fit_function"] == "threshold":
                    # # This needs to be improved
                    # a = np.argmin(np.abs(segment - np.double(self.image.parameters["Threshold"])))
                    # edge_position = x[a]
                    # edges_profiles[cnt, row] = np.real(edge_position)
                    edges_profiles[cnt, row] = edge
                elif self.image.parameters["Edge_fit_function"] == "bright_edge":
                    print("Add code for bright edge finding")
        return edges_profiles

    def filter_and_reduce_noise(self) -> tuple[float, np.ndarray]:
        """
        Filters the image to reduce noise using median filtering and Butterworth filtering.
        """

        median_filter_kernel = 5
        filtered_image = medfilt2d(self.image.processed_image, median_filter_kernel)
        image_sum = np.sum(filtered_image, 1)
        backward_filter, forward_filter = butter(8, 0.125)
        image_sum_filtered = filtfilt(backward_filter, forward_filter, image_sum, method="gust")
        return image_sum, image_sum_filtered

    def detect_peaks(self) -> tuple[np.ndarray, tuple[np.ndarray, dict]]:
        """
        Detects peaks in the filtered image sum.
        """

        image_sum, image_sum_filtered = self.filter_and_reduce_noise()
        image_sum_derivative = np.gradient(image_sum)
        image_sum_filtered_derivative = np.abs(np.gradient(image_sum_filtered))
        peaks = find_peaks(image_sum_filtered_derivative)
        return image_sum_derivative, peaks

    def classify_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Classifies edges into leading and trailing edges.
        """

        image_sum_derivative, peaks = self.detect_peaks()
        edge_locations = peaks[0]+1
        leading_edges = np.array([])
        trailing_edges = np.array([])
        if self.image.parameters["brightEdge"]:
            print("Bright edge peak detection to be added here")
        else:
            for n in edge_locations:
                if self.image.parameters["tone_positive_radiobutton"]:
                    if image_sum_derivative[n] > 0:
                        leading_edges = np.append(leading_edges, n)
                    else:
                        trailing_edges = np.append(trailing_edges, n)
                else:
                    if image_sum_derivative[n] < 0:
                        leading_edges = np.append(leading_edges, n)
                    else:
                        trailing_edges = np.append(trailing_edges, n)

        return leading_edges, trailing_edges

    def find_leading_and_trailing_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds leading and trailing edges, ensuring each trailing edge has a corresponding leading edge.
        """

        leading_edges, trailing_edges = self.classify_edges()
        new_leading_edges = np.array([])
        new_trailing_edges = np.array([])
        for n in leading_edges:
            ve = trailing_edges > n
            if len(trailing_edges[ve]) > 0:
                new_leading_edges = np.append(new_leading_edges, n)
                new_trailing_edges = np.append(new_trailing_edges, trailing_edges[ve][0])

        return new_leading_edges, new_trailing_edges

    def determine_edge_profiles(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Determines leading and trailing edge profiles.
        """

        new_leading_edges, new_trailing_edges = self.find_leading_and_trailing_edges()

        image_size = self.image.processed_image.shape
        leading_edges_profiles_param = np.nan * np.zeros([len(new_leading_edges), image_size[1]])
        trailing_edges_profiles_param = np.nan * np.zeros([len(new_trailing_edges), image_size[1]])

        leading_edges_profiles = self.edge_detection(new_leading_edges, leading_edges_profiles_param)
        trailing_edges_profiles = self.edge_detection(new_trailing_edges, trailing_edges_profiles_param)

        return leading_edges_profiles, trailing_edges_profiles

    def find_edges(self) -> None:
        """
        Consolidating and centering leading and trailing edges.
        """

        self.image.leading_edges, self.image.trailing_edges = self.determine_edge_profiles()

        method = "interpolation"

        leading_edges_consolidation = edge_consolidation(self.image.leading_edges, method)
        self.image.consolidated_leading_edges = leading_edges_consolidation[0]
        self.image.leading_edges_consolidation = leading_edges_consolidation[1]

        trailing_edges_consolidation = edge_consolidation(self.image.trailing_edges, method)
        self.image.consolidated_trailing_edges = trailing_edges_consolidation[0]
        self.image.trailing_edges_consolidation = trailing_edges_consolidation[1]



        self.image.zero_mean_leading_edge_profiles = edge_mean_subtraction(
            self.image.consolidated_leading_edges)
        self.image.zero_mean_trailing_edge_profiles = edge_mean_subtraction(
            self.image.consolidated_trailing_edges)

        self.copy_base_attributes()

        profiles_shape = self.image.leading_edges.shape
        lines_number = profiles_shape[0]

        self.image.number_of_lines = lines_number

        # self.image.critical_dimension = self.image.trailing_edges - self.image.leading_edges
        # self.image.critical_dimension_std_estimate = np.std(np.nanmedian(self.image.critical_dimension, 1))
        # self.image.critical_dimension_estimate = np.mean(np.nanmedian(self.image.critical_dimension, 1))
        #
        # self.image.pitch_estimate = (np.mean(
        #     np.nanmedian(self.image.leading_edges[1:] - self.image.leading_edges[0:-1], 1)) + np.mean(
        #     np.nanmedian(self.image.trailing_edges[1:] - self.image.trailing_edges[0:-1], 1))) / 2

        if len(self.image.leading_edges) > 1:
            pass
        else:
            self.image.pitch_estimate = np.nan

    def copy_base_attributes(self):
        self.image.basic_consolidated_leading_edges = copy(self.image.consolidated_leading_edges)
        self.image.basic_consolidated_trailing_edges = copy(self.image.consolidated_trailing_edges)
        self.image.basic_zero_mean_leading_edge_profiles = copy(self.image.zero_mean_leading_edge_profiles)
        self.image.basic_zero_mean_trailing_edge_profiles = copy(self.image.zero_mean_trailing_edge_profiles)


class PostProcessor:
    """
    Class for calculating post processed consolidated edges, the operation aim at noise reduction.
    This class is optional by checkbox "Edge distortion correction"
    """

    def __init__(self, image):
        self.image = image

    def post_processing(self, use_post_processing=False):
        """
        Perform post-processing on the image.

        This method either restores the post-processed edges from the cache if available,
        or computes and stores them in the cache. It restores base attributes if post-processing
        is not applied.

        Args:
            use_post_processing (bool): Flag to determine if post-processing should be applied.
                                        Defaults to False.
        """

        if use_post_processing:
            if self.image.post_processing_cache:
                self.restore_cache()
            else:
                self.remove_spikes()
                self.calculate_new_post_processed_zero_mean_edges()
                self.store_cache()
        else:
            self.restore_base_attributes()

    def remove_spikes(self):
        """
                Remove the spikes from the consolidated edges and create a spike map

        """

        leading_edges = self.image.consolidated_leading_edges
        trailing_edges = self.image.consolidated_trailing_edges
        spike_threshold = self.image.parameters["Spike_Threshold"]

        spike_filtered_leading_edges = np.zeros(np.shape(leading_edges))
        spike_filtered_trailing_edges = np.zeros(np.shape(leading_edges))
        leading_edges_spikes = np.zeros(np.shape(leading_edges)) * np.nan
        trailing_edges_spikes = np.zeros(np.shape(leading_edges)) * np.nan

        print("I am removing the spikes from "+ self.image.file_name)
        print(f"spike threshold set to {spike_threshold}")

        # Leading edges
        edge_number = -1
        for edge in leading_edges:
            edge_length = len(edge)
            new_edge = deepcopy(edge)
            spike_map = np.zeros(np.shape(edge)) * np.nan
            edge_number += 1
            for edge_index in range(0, edge_length):
                edge_gradient = np.gradient(new_edge)
                if edge_index == 0:
                    if abs(edge_gradient[edge_index] - edge_gradient[edge_index + 2]) > spike_threshold:
                        new_edge[edge_index] = 0.5 * (edge[edge_index + 1] + edge[edge_index + 2])
                        spike_map[edge_index] = 0.5 * (edge[edge_index + 1] + edge[edge_index + 2])
                elif edge_index == edge_length-1:
                    if abs(edge_gradient[edge_index] - edge_gradient[edge_index - 2]) > spike_threshold:
                        new_edge[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index - 2])
                        spike_map[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index - 2])
                else:
                    if abs(edge_gradient[edge_index+1]-edge_gradient[edge_index-1]) > spike_threshold:
                        new_edge[edge_index] = 0.5*(edge[edge_index-1] + edge[edge_index+1])

                        spike_map[edge_index-1] = edge[edge_index-1]
                        spike_map[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index + 1])
                        spike_map[edge_index+1] = edge[edge_index+1]
                spike_filtered_leading_edges[edge_number, :] = new_edge
                leading_edges_spikes[edge_number, :] = spike_map

        # Trailing edges
        edge_number = -1
        for edge in trailing_edges:
            edge_length = len(edge)
            new_edge = deepcopy(edge)
            spike_map = np.zeros(np.shape(edge)) * np.nan
            edge_number += 1
            for edge_index in range(0, edge_length):
                edge_gradient = np.gradient(new_edge)
                if edge_index == 0:
                    if abs(edge_gradient[edge_index] - edge_gradient[edge_index + 2]) > spike_threshold:
                        new_edge[edge_index] = 0.5 * (edge[edge_index + 1] + edge[edge_index + 2])
                        spike_map[edge_index] = 0.5 * (edge[edge_index + 1] + edge[edge_index + 2])
                elif edge_index == edge_length-1:
                    if abs(edge_gradient[edge_index] - edge_gradient[edge_index - 2]) > spike_threshold:
                        new_edge[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index - 2])
                        spike_map[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index - 2])
                else:
                    if abs(edge_gradient[edge_index+1]-edge_gradient[edge_index-1]) > spike_threshold:
                        new_edge[edge_index] = 0.5*(edge[edge_index-1] + edge[edge_index+1])

                        spike_map[edge_index - 1] = edge[edge_index - 1]
                        spike_map[edge_index] = 0.5 * (edge[edge_index - 1] + edge[edge_index + 1])
                        spike_map[edge_index + 1] = edge[edge_index + 1]

                spike_filtered_trailing_edges[edge_number, :] = new_edge
                trailing_edges_spikes[edge_number, :] = spike_map

        self.image.consolidated_leading_edges = spike_filtered_leading_edges
        self.image.leading_edges_spikes = leading_edges_spikes

        self.image.consolidated_trailing_edges = spike_filtered_trailing_edges
        self.image.trailing_edges_spikes = trailing_edges_spikes


    def calculate_new_post_processed_zero_mean_edges(self):
        """
        Calculate new post-processed zero-mean edges, based on post-processed consolidated edges.
        """

        self.image.zero_mean_leading_edge_profiles = edge_mean_subtraction(self.image.consolidated_leading_edges)
        self.image.zero_mean_trailing_edge_profiles = edge_mean_subtraction(self.image.consolidated_trailing_edges)

    def restore_base_attributes(self):
        """
        This method restores the leading and trailing edge profiles to their basic, pre-processed state.
        """

        self.image.consolidated_leading_edges = self.image.basic_consolidated_leading_edges
        self.image.consolidated_trailing_edges = self.image.basic_consolidated_trailing_edges
        self.image.zero_mean_leading_edge_profiles = self.image.basic_zero_mean_leading_edge_profiles
        self.image.zero_mean_trailing_edge_profiles = self.image.basic_zero_mean_trailing_edge_profiles

    def store_cache(self):
        """
        This method saves the current post-processed leading and trailing edge profiles
        to the image's post-processing cache.
        """

        self.image.post_processing_cache = (
            copy(self.image.consolidated_leading_edges),
            copy(self.image.consolidated_trailing_edges),
            copy(self.image.zero_mean_leading_edge_profiles),
            copy(self.image.zero_mean_trailing_edge_profiles),
        )

    def restore_cache(self):
        """
        This method restores the post-processed leading and trailing edge profiles
        from the image's post-processing cache.
        """

        (self.image.consolidated_leading_edges,
         self.image.consolidated_trailing_edges,
         self.image.zero_mean_leading_edge_profiles,
         self.image.zero_mean_trailing_edge_profiles) = self.image.post_processing_cache


class MultiTaper:
    def __init__(self, image):
        self.image = image

    def multi_taper(self, use_multi_taper=False):
        """
        Apply multi-taper processing to the image.

        This method either restores the multi-taper processed edges from the cache if available,
        or computes and stores them in the cache. It restores base attributes if multi-taper processing
        is not applied.

        Args:
            use_multi_taper (bool): Flag to determine if multi-taper processing should be applied.
                                    Defaults to False.
        """

        if use_multi_taper:
            if self.image.multi_taper_cache:
                self.restore_cache()
            else:
                self.calculate_new_multi_taper_consolidated_edges()
                self.calculate_new_multi_taper_zero_mean_edges()
                self.store_cache()
        else:
            self.restore_base_attributes()

    def calculate_new_multi_taper_consolidated_edges(self):
        pass  # TODO: Implement the function to apply multi-taper and save the results in the image container

    def calculate_new_multi_taper_zero_mean_edges(self):
        """
        Calculate new zero-mean edges after multi-taper process, based on consolidated edges after multi-taper processing.
        """

        self.image.zero_mean_leading_edge_profiles = edge_mean_subtraction(self.image.consolidated_leading_edges)
        self.image.zero_mean_trailing_edge_profiles = edge_mean_subtraction(self.image.consolidated_trailing_edges)

    def restore_base_attributes(self):
        """
        This method restores the leading and trailing edge profiles to their basic, state without multi-taper processing.
        """

        self.image.consolidated_leading_edges = self.image.basic_consolidated_leading_edges
        self.image.consolidated_trailing_edges = self.image.basic_consolidated_trailing_edges
        self.image.zero_mean_leading_edge_profiles = self.image.basic_zero_mean_leading_edge_profiles
        self.image.zero_mean_trailing_edge_profiles = self.image.basic_zero_mean_trailing_edge_profiles

    def store_cache(self):
        """
        This method saves the current multi-taper processed leading and trailing edge profiles
        to the image's multi-taper processing cache.
        """

        self.image.multi_taper_cache = (
            copy(self.image.consolidated_leading_edges),
            copy(self.image.consolidated_trailing_edges),
            copy(self.image.zero_mean_leading_edge_profiles),
            copy(self.image.zero_mean_trailing_edge_profiles),
        )

    def restore_cache(self):
        """
        This method restores the multi-taper processed leading and trailing edge profiles
        from the image's multi-taper processing cache.
        """

        (self.image.consolidated_leading_edges,
         self.image.consolidated_trailing_edges,
         self.image.zero_mean_leading_edge_profiles,
         self.image.zero_mean_trailing_edge_profiles) = self.image.multi_taper_cache


class MetricCalculator:
    """
    Class for calculating metrics related to edge profiles and image precision
    that will later serve for plotting on metric tab.
    """

    def __init__(self, image):
        self.image = image

    def calculate_and_fit_hhcf(self, input_data: np.ndarray) -> tuple[
        ndarray, Any, np.ndarray]:
        """
                Calculates and fits the Height-Height Correlation Function (HHCF) of input data.

                Args:
                - input_data (np.ndarray): The lines edges.

                Returns:
                - tuple: A tuple containing the HHCF, the HHCF fit parameters and the fitted HHCF.
                """
        hhcf_length = int(np.size(input_data,1)/2)
        hhcf = np.zeros((np.size(input_data, 0), hhcf_length))
        for n in range(0, np.size(input_data, 0)):
            profile = input_data[n, :]
            for m in range(1, hhcf_length):
                hhcf[n, m] = np.mean(abs(profile[0:-m]-profile[m:]))
                #hhcf[n, m] = np.mean((profile[0:-m] - profile[m:]) ** 2)
        height_height_correlation_function = np.mean(hhcf,0)

        background = 2*np.mean(height_height_correlation_function[0:10])
        sigma2 = np.abs(np.mean(height_height_correlation_function[-100:])-background)
        x = np.arange(0, np.size(height_height_correlation_function))
        beta0 = np.array([1, 20, 1, 0.645])
        beta_min = [0.1, 2, 0.5, 0]
        beta_max = [7, 500, 2, 5]
        beta, _ = curve_fit(
            hhcf_,
            x,
            height_height_correlation_function,
            #sigma=(1+np.arange(0,len(x))),
            p0=beta0,
            bounds=(beta_min, beta_max),
            maxfev=100000,
            method="dogbox"
        )
        #fitted_hhcf = hhcf_(x, *beta)
        fitted_hhcf = hhcf_(x, *beta0)

        return fitted_hhcf, height_height_correlation_function, beta

    def setup_frequency(self) -> None:
        """
        Sets up the frequency domain parameters based on image pixel size and profile length. It also stores the profile
        length coordinate in the property called distance.

        """

        self.image.pixel_size = self.image.parameters["PixelSize"]
        Fs = 1 / self.image.pixel_size
        s = np.shape(self.image.consolidated_leading_edges)
        profiles_length = s[1]
        num_freq_bins = profiles_length // 2 + 1
        self.image.frequency = 1000 * np.linspace(0, Fs / 2, num_freq_bins)
        self.image.distance = self.image.pixel_size * np.arange(0, profiles_length)

        # self.image.frequency = 1000 * np.arange(0, Fs / 2 + Fs / profiles_length, Fs / profiles_length)

    def select_psd_model(self) -> tuple[Union[Callable, None], Union[Callable, None], Union[Callable, None]]:
        """
        Selects the appropriate Power Spectral Density (PSD) model based on user parameters.

        Returns:
        - tuple: A tuple containing three callable functions representing the selected PSD model,
                 its beta parameters optimizer, and an alternative model function.
        """

        selected_model = self.image.parameters["PSD_model"]
        valid_models = {"Palasantzas 2", "Palasantzas 1", "Integral", "Gaussian", "Floating alpha", "No white noise"}
        if selected_model in valid_models:
            model = Palasantzas_2_minimize
            model_beta = Palasantzas_2_beta
            model_2 = Palasantzas_2b
        else:
            raise ValueError('Please select valid model')

        return model, model_beta, model_2

    def calculate_and_fit_psd(self, input_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates and fits Power Spectral Density (PSD) of input data.

        Args:
        - input_data (np.ndarray): Input data for which PSD is to be calculated.

        Returns:
        - tuple: A tuple containing PSD, PSD fit parameters, fitted PSD, unbiased PSD,
                 and fitted unbiased PSD.
        """

        # Calculate standard deviation to check
        sigma = np.mean(np.nanstd(input_data, 0))
        model, model_beta, model_2 = self.select_psd_model()
        PSD = np.nanmean(np.abs(np.fft.rfft(input_data)) ** 2, 0)
        PSD /= len(PSD) ** 2
        PSD[0] = PSD[1]

        beta0, beta_min, beta_max = model_beta(self, PSD)
        bounds = Bounds(lb=beta_min, ub=beta_max)
        optimized_parameters = minimize(
            model,
            beta0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10},
            args=(self.image.frequency, PSD),
            bounds=bounds
        )

        PSD_fit_parameters = optimized_parameters['x']
        PSD_fit = model_2(self.image.frequency, optimized_parameters['x'])
        beta = PSD_fit_parameters
        PSD_unbiased = PSD - beta[2]
        beta[2] = 0
        PSD_fit_unbiased = model_2(self.image.frequency, beta)

        sigma_standard = 3 * sigma
        biased_sigma = 3 * np.sqrt(np.sum(PSD))
        unbiased_sigma = 3 * np.sqrt(np.sum(PSD_unbiased))
        unbiased_sigma_fit = 3 * np.sqrt(np.sum(PSD_fit_unbiased))


        return PSD, PSD_fit_parameters, PSD_fit, PSD_unbiased, PSD_fit_unbiased, unbiased_sigma, biased_sigma, sigma_standard, unbiased_sigma_fit

    def calculate_metrics(self) -> None:
        """
        Calculates various metrics related to edge profiles and PSD (Power Spectral Density).
        """
        # CD and pitch
        self.image.critical_dimension = self.image.consolidated_trailing_edges - self.image.consolidated_leading_edges
        self.image.critical_dimension_std_estimate = np.std(np.nanmedian(self.image.critical_dimension, 1))
        self.image.critical_dimension_estimate = np.mean(np.nanmedian(self.image.critical_dimension, 1))

        self.image.pitch_estimate = np.median((
            np.nanmedian(self.image.consolidated_leading_edges[1:] - self.image.consolidated_leading_edges[0:-1], 1),
            np.nanmedian(self.image.consolidated_trailing_edges[1:] - self.image.consolidated_trailing_edges[0:-1],
                         1)))
        # LWR PSD
        line_width = np.abs(
            self.image.consolidated_leading_edges - self.image.consolidated_trailing_edges) * self.image.pixel_size

        (self.image.LWR_PSD,
         self.image.LWR_PSD_fit_parameters,
         self.image.LWR_PSD_fit,
         self.image.LWR_PSD_unbiased,
         self.image.LWR_PSD_fit_unbiased,
         self.image.unbiased_LWR,
         self.image.biased_LWR,
         self.image.standard_LWR,
         self.image.unbiased_LWR_fit) = self.calculate_and_fit_psd(line_width)

        # LER PSD
        all_edges = np.vstack((self.image.zero_mean_leading_edge_profiles * self.image.pixel_size,
                               self.image.zero_mean_trailing_edge_profiles * self.image.pixel_size))

        (self.image.LER_PSD,
         self.image.LER_PSD_fit_parameters,
         self.image.LER_PSD_fit,
         self.image.LER_PSD_unbiased,
         self.image.LER_PSD_fit_unbiased,
         self.image.unbiased_LER,
         self.image.biased_LER,
         self.image.standard_LER,
         self.image.unbiased_LER_fit) = self.calculate_and_fit_psd(all_edges)

        # Leading edges LER
        leading_edges = self.image.zero_mean_leading_edge_profiles * self.image.pixel_size

        (self.image.LER_Leading_PSD,
         self.image.LER_Leading_PSD_fit_parameters,
         self.image.LER_Leading_PSD_fit,
         self.image.LER_Leading_PSD_unbiased,
         self.image.LER_Leading_PSD_fit_unbiased,
         self.image.unbiased_LER_Leading,
         self.image.biased_LER_Leading,
         self.image.standard_LER_Leading,
         self.image.unbiased_LER_Leading_fit) = self.calculate_and_fit_psd(leading_edges)

        # Trailing edges LER
        trailing_edges = self.image.zero_mean_trailing_edge_profiles * self.image.pixel_size

        (self.image.LER_Trailing_PSD,
         self.image.LER_Trailing_PSD_fit_parameters,
         self.image.LER_Trailing_PSD_fit,
         self.image.LER_Trailing_PSD_unbiased,
         self.image.LER_Trailing_PSD_fit_unbiased,
         self.image.unbiased_LER_Trailing,
         self.image.biased_LER_Trailing,
         self.image.standard_LER_Trailing,
         self.image.unbiased_LER_Trailing_fit) = self.calculate_and_fit_psd(trailing_edges)

        # Line width HHCF
        (self.image.LW_HHCF_fit,
         self.image.LW_HHCF,
         self.image.LW_HHCF_parameters) = self.calculate_and_fit_hhcf(line_width)

        # Line edge HHCF
        (self.image.Lines_edge_HHCF_fit,
         self.image.Lines_edge_HHCF,
         self.image.Lines_edge_HHCF_parameters) = self.calculate_and_fit_hhcf(all_edges)

        # Line leading edge HHCF
        (self.image.Lines_leading_edges_HHCF_fit,
         self.image.Lines_leading_edges_HHCF,
         self.image.Lines_leading_edges_HHCF_parameters) = self.calculate_and_fit_hhcf(leading_edges)

        # Line trailing edge HHCF
        (self.image.Lines_trailing_edges_HHCF_fit,
         self.image.Lines_trailing_edges_HHCF,
         self.image.Lines_trailing_edges_HHCF_parameters) = self.calculate_and_fit_hhcf(trailing_edges)

        #print(self.image.LW_HHCF_parameters)