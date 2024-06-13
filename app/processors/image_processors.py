from typing import Union, Callable

import numpy as np
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit, minimize, Bounds
from scipy.ndimage import histogram
from scipy.signal import medfilt2d, filtfilt, butter, find_peaks

from app.utils.poly import (_poly11, _poly22, _poly33, poly11, poly22,
                                                poly33, binary_image_histogram_model, gaussian_profile)
from app.utils.psd import Palasantzas_2_minimize, Palasantzas_2_beta, Palasantzas_2b

from typing import Union
import numpy as np
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit
from scipy.ndimage import histogram
from app.utils.poly import _poly11, poly11, binary_image_histogram_model, gaussian_profile


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
        brightness_map = image > np.mean(image)
        x, y = np.meshgrid(
            np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[0])
        )
        image_array = image[brightness_map]
        xdata = (x[brightness_map], y[brightness_map])
        parameters_initial_guess = [0, 0, np.mean(image_array)]

        optimized_parameters, _ = curve_fit(
            _poly11, xdata, image_array, parameters_initial_guess
        )

        brightness = poly11((x, y), optimized_parameters)
        image_flattened = image - brightness

        return image_flattened

    def normalize_image(self) -> None:
        """
        Normalizes the image.
        """
        image = self.remove_brightness_gradient()
        image_max = np.max(image)
        image_min = np.min(image)
        image_normalized = (image - image_min) / (image_max - image_min)
        self.image.processed_image = image_normalized

    def calculate_histogram_parameters(self) -> None:
        """
        Calculates histogram parameters of the normalized image.
        """
        self.image.intensity_histogram = histogram(self.image.processed_image, 0, 1, 256)
        intensity = np.linspace(0, 1, 256)
        max_index = np.argmax(self.image.intensity_histogram)
        max_value = self.image.intensity_histogram[max_index]
        low_bounds = [max_value / 4, 0, 0.01, max_value / 4, 0.1, 0.01, 0, 0, 0.01]
        high_bounds = [max_value, 1, 0.5, max_value, 1, 0.5, max_value / 4, 1, 2]
        beta0 = (max_value, 0.25, 0.1, max_value, 0.75, 0.1, 1, 0.5, 0.1)
        beta, _ = curve_fit(
            binary_image_histogram_model,
            intensity,
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
        edge_range = self.image.parameters["EdgeRange"]
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
                        edges_profiles[cnt, row] = np.real(edge_position)
                elif self.image.parameters["Edge_fit_function"] == "linear":
                    print("Add code for linear edge finding")
                elif self.image.parameters["Edge_fit_function"] == "threshold":
                    print("Add code for threshold edge finding")
                elif self.image.parameters["Edge_fit_function"] == "bright_edge":
                    print("Add code for bright edge finding")
        return edges_profiles

    def edge_consolidation(self, raw_edge_profiles: np.ndarray) -> np.ndarray:
        """
        Consolidates raw edge profiles.
        """

        consolidated_edge_profiles = raw_edge_profiles.copy()
        for edge in consolidated_edge_profiles:
            mean_value = np.nanmean(edge)
            edge[edge is np.nan] = mean_value
        return consolidated_edge_profiles

    def edge_mean_subtraction(self, absolute_edge_profiles: np.ndarray) -> np.ndarray:
        """
        Subtracts the mean value from edge profiles to center them around zero.
        """

        zero_mean_edge_profiles = absolute_edge_profiles.copy()
        for edge in zero_mean_edge_profiles:
            mean_value = np.nanmean(edge)
            edge[:] = edge - mean_value
        return zero_mean_edge_profiles

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
        image_sum_derivative = np.diff(image_sum)
        image_sum_filtered_derivative = np.abs(np.diff(image_sum_filtered))
        peaks = find_peaks(image_sum_filtered_derivative)
        return image_sum_derivative, peaks

    def classify_edges(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Classifies edges into leading and trailing edges.
        """

        image_sum_derivative, peaks = self.detect_peaks()
        edge_locations = peaks[0]
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

        self.image.consolidated_leading_edges = self.edge_consolidation(self.image.leading_edges)
        self.image.consolidated_trailing_edges = self.edge_consolidation(self.image.trailing_edges)
        self.image.zero_mean_leading_edge_profiles = self.edge_mean_subtraction(
            self.image.consolidated_leading_edges)
        self.image.zero_mean_trailing_edge_profiles = self.edge_mean_subtraction(
            self.image.consolidated_trailing_edges)

        profiles_shape = self.image.leading_edges.shape
        lines_number = profiles_shape[0]

        self.image.number_of_lines = lines_number

        self.image.critical_dimension = self.image.trailing_edges - self.image.leading_edges
        self.image.critical_dimension_std_estimate = np.std(np.nanmedian(self.image.critical_dimension, 1))
        self.image.critical_dimension_estimate = np.mean(np.nanmedian(self.image.critical_dimension, 1))

        self.image.pitch_estimate = (np.mean(
            np.nanmedian(self.image.leading_edges[1:] - self.image.leading_edges[0:-1], 1)) + np.mean(
            np.nanmedian(self.image.trailing_edges[1:] - self.image.trailing_edges[0:-1], 1))) / 2

        if len(self.image.leading_edges) > 1:
            pass
        else:
            self.image.pitch_estimate = np.nan


class MetricCalculator:
    """
    Class for calculating metrics related to edge profiles and image precision
    that will later serve for plotting on metric tab.
    """

    def __init__(self, image):
        self.image = image

    def setup_frequency(self) -> None:
        """
        Sets up the frequency domain parameters based on image pixel size and profile length.
        """

        self.image.pixel_size = self.image.parameters["PixelSize"]
        Fs = 1 / self.image.pixel_size
        s = np.shape(self.image.consolidated_leading_edges)
        profiles_length = s[1]
        self.image.frequency = 1000 * np.arange(0, Fs / 2 + Fs / profiles_length, Fs / profiles_length)

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
            model, model_beta, model_2 = None, None, None

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

        return PSD, PSD_fit_parameters, PSD_fit, PSD_unbiased, PSD_fit_unbiased

    def calculate_metrics(self) -> None:
        """
        Calculates various metrics related to edge profiles and PSD (Power Spectral Density).
        """

        # LWR PSD
        line_width = np.abs(
            self.image.consolidated_leading_edges - self.image.consolidated_trailing_edges) * self.image.pixel_size

        (self.image.LWR_PSD,
         self.image.LWR_PSD_fit_parameters,
         self.image.LWR_PSD_fit,
         self.image.LWR_PSD_unbiased,
         self.image.LWR_PSD_fit_unbiased) = self.calculate_and_fit_psd(line_width)

        # LER PSD
        all_edges = np.vstack((self.image.zero_mean_leading_edge_profiles * self.image.pixel_size,
                               self.image.zero_mean_trailing_edge_profiles * self.image.pixel_size))

        (self.image.LER_PSD,
         self.image.LER_PSD_fit_parameters,
         self.image.LER_PSD_fit,
         self.image.LER_PSD_unbiased,
         self.image.LER_PSD_fit_unbiased) = self.calculate_and_fit_psd(all_edges)

        # Leading edges LER
        input_data = self.image.zero_mean_leading_edge_profiles * self.image.pixel_size

        (self.image.LER_Leading_PSD,
         self.image.LER_Leading_PSD_fit_parameters,
         self.image.LER_Leading_PSD_fit,
         self.image.LER_Leading_PSD_unbiased,
         self.image.LER_Leading_PSD_fit_unbiased) = self.calculate_and_fit_psd(input_data)

        # Trailing edges LER
        input_data = self.image.zero_mean_trailing_edge_profiles * self.image.pixel_size

        (self.image.LER_Trailing_PSD,
         self.image.LER_Trailing_PSD_fit_parameters,
         self.image.LER_Trailing_PSD_fit,
         self.image.LER_Trailing_PSD_unbiased,
         self.image.LER_Trailing_PSD_fit_unbiased) = self.calculate_and_fit_psd(input_data)
