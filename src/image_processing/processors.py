from dataclasses import dataclass
from typing import List

import numpy as np
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit, minimize, Bounds
from scipy.ndimage import histogram
from scipy.signal import medfilt2d, filtfilt, butter, find_peaks

from src.utilities.pre_processing_utils import (_poly11, _poly22, _poly33, poly11, poly22,
                                                poly33, binary_image_histogram_model, gaussian_profile)


@dataclass
class PreProcessorResults:
    processed_image: List[int]
    lines_snr: int
    intensity_histogram_medium: int
    intensity_histogram_high: int
    intensity_histogram_low: int
    intensity_histogram_gaussian_fit_parameters: int
    intensity_histogram: tuple


class PreProcessor:

    # normalization, brightness removal (poly11()), rotation to find the lines
    # histogram analysis (gaussian_profile(), binary_image_histogram_model())

    def crop_and_rotate_image(self, image, parameters):
        # Crop images to the specified ROI
        x1 = int(parameters["X1"])
        x2 = int(parameters["X2"])
        y1 = int(parameters["Y1"])
        y2 = int(parameters["Y2"])

        image_cropped = image[x1:x2, y1:y2]
        theta = np.linspace(85.0, 95.0, 50, endpoint=False)
        radius = np.min(np.array(image_cropped.shape)) / 2 - 2
        dim1 = image_cropped.shape[0]
        dim2 = image_cropped.shape[1]
        x, y = np.meshgrid(
            np.linspace(-dim2 / 2 + 1, dim2 / 2 - 1, dim2),
            np.linspace(-dim1 / 2 + 1, dim1 / 2 - 1, dim1),
        )
        circle = np.array((x * x + y * y) < (radius * radius))
        sinogram = radon(image_cropped * circle.astype(int), theta=theta)
        R = np.sum(np.power(sinogram, 2), 0)
        max_id = np.argmax(R)

        rotated_image = rotate(image, -float(theta[max_id]) + 90, order=0)
        image_rotated_cropped = rotated_image[x1:x2, y1:y2]
        return image_rotated_cropped

    def remove_brightness_gradient(self):
        image = self.crop_and_rotate_image()
        brightness_map = image > np.mean(image)
        x, y = np.meshgrid(
            np.linspace(-1, 1, image.shape[1]), np.linspace(-1, 1, image.shape[0])
        )
        image_array = image[brightness_map]
        xdata = (x[brightness_map], y[brightness_map])
        parameters_initial_guess = [0, 0, np.mean(image_array)]

        optimized_parameters, covariance = curve_fit(
            _poly11, xdata, image_array, parameters_initial_guess
        )

        brightness = poly11((x, y), optimized_parameters)
        image_flattened = image - brightness
        return image_flattened

    def normalize_image(self):
        image = self.remove_brightness_gradient()
        image_max = np.max(image)
        image_min = np.min(image)
        image_normalized = (image - image_min) / (image_max - image_min)
        return image_normalized

    def calculate_histogram_parameters(self):
        image_normalized = self.normalize_image()
        intensity_histogram = np.histogram(image_normalized, 0, 1, 256)
        intensity = np.linspace(0, 1, 256)
        image_histogram = intensity_histogram
        max_index = np.argmax(image_histogram)
        max_value = image_histogram[max_index]
        low_bounds = [max_value / 4, 0, 0.01, max_value / 4, 0.1, 0.01, 0, 0, 0.01]
        high_bounds = [max_value, 1, 0.5, max_value, 1, 0.5, max_value / 4, 1, 2]
        beta0 = (max_value, 0.25, 0.1, max_value, 0.75, 0.1, 1, 0.5, 0.1)
        beta, covariance = curve_fit(
            binary_image_histogram_model,
            intensity,
            self.intensity_histogram,
            p0=beta0,
            bounds=(low_bounds, high_bounds),
            maxfev=100000,
        )

        intensity_histogram_gaussian_fit_parameters = beta
        intensity_histogram_low = gaussian_profile(intensity, *beta[0:3])
        intensity_histogram_high = gaussian_profile(intensity, *beta[6:9])
        intensity_histogram_medium = gaussian_profile(intensity, *beta[3:6])

        lines_snr = np.abs(beta[1] - beta[4]) / (
                0.5 * (beta[2] + beta[5]) * 2 * np.sqrt(-2 * np.log(0.5))
        )

        return PreProcessorResults(image_normalized, lines_snr, intensity_histogram_medium, intensity_histogram_high,
                                   intensity_histogram_low, intensity_histogram_gaussian_fit_parameters,
                                   intensity_histogram)


class EdgeDetector:
    # find the pick in image
    # consolidate the edges and try to center that around zero
    # calculate critical dimensions and pitch estimates based on the detected edges
    # basically makes necessary calculations
    def find_edges(self):
        def edge_detection(new_edges, edges_profiles):
            cnt = -1
            for edge in new_edges:
                cnt = cnt + 1
                for row in range(0, image_size[1]):
                    segment_start = int(np.max([0, edge - edge_range]))
                    segment_end = int(np.min([edge + edge_range, image_size[0]]))
                    x = np.arange(segment_start, segment_end)
                    segment = processed_image[segment_start:segment_end, row]
                    if (self.parameters["Edge_fit_function"] == "polynomial"):
                        p = np.polyfit(x, segment, 4)
                        p[-1] = p[-1] - np.double(self.parameters["Threshold"])
                        r = np.roots(p)
                        r = r[np.imag(r) == 0]
                        if len(r) > 0:
                            edge_position = r[np.argmin(np.abs(r - (segment_start + segment_end) / 2))]
                            edges_profiles[cnt, row] = np.real(edge_position)
                    elif (self.parameters["Edge_fit_function"] == "linear"):
                        print("Add code for linear edge finding")
                    elif (self.parameters["Edge_fit_function"] == "threshold"):
                        print("Add code for threshold edge finding")
                    elif (self.parameters["Edge_fit_function"] == "bright_edge"):
                        print("Add code for bright edge finding")
            return edges_profiles

        def edge_consolidation(raw_edge_profiles):
            consolidated_edge_profiles = raw_edge_profiles.copy()
            for edge in consolidated_edge_profiles:
                mean_value = np.nanmean(edge)
                edge[edge is np.nan] = mean_value
            return consolidated_edge_profiles

        def edge_mean_subtraction(absolute_edge_profiles):
            zero_mean_edge_profiles = absolute_edge_profiles.copy()
            for edge in zero_mean_edge_profiles:
                mean_value = np.nanmean(edge)
                edge[:] = edge - mean_value
            return zero_mean_edge_profiles

        processed_image = self.processed_image
        # Filter image with a 2d median filter to remove eventual outliers (bright pixels for instance)
        median_filter_kernel = 5
        imageF = medfilt2d(processed_image, median_filter_kernel)

        # Sum all the columns of the image to calculate the average lines profile
        S = np.sum(imageF, 1)
        # Filter the lines profile to reduce the noise
        b, a = butter(8, 0.125)
        Sf = filtfilt(b, a, S, method="gust")

        dS = np.diff(S)
        dSf = np.abs(np.diff(Sf))
        peaks = find_peaks(dSf)
        edge_locations = peaks[0]
        leading_edges = np.array([])
        trailing_edges = np.array([])
        if self.parameters["brightEdge"]:
            print("Bright edge peak detection to be added here")
        else:
            for n in edge_locations:
                if self.parameters["tone_positive_radiobutton"]:
                    if dS[n] > 0:
                        leading_edges = np.append(leading_edges, n)
                    else:
                        trailing_edges = np.append(trailing_edges, n)
                else:
                    if dS[n] < 0:
                        leading_edges = np.append(leading_edges, n)
                    else:
                        trailing_edges = np.append(trailing_edges, n)

        # Consider only complete lines: for each trailing edge there must be 1 leading edge
        new_leading_edges = np.array([])
        new_trailing_edges = np.array([])
        for n in leading_edges:
            ve = trailing_edges > n
            if len(trailing_edges[ve]) > 0:
                new_leading_edges = np.append(new_leading_edges, n)
                new_trailing_edges = np.append(new_trailing_edges, trailing_edges[ve][0])
            # Rough Estimate of pitch and Critical Dimension (CD)
            # critical_dimension = np.mean(new_trailing_edges - new_leading_edges)
            # pitch = 0.5 * (
            #         np.mean(np.diff(new_trailing_edges)) + np.mean(np.diff(new_leading_edges))
            # )

            cd_fraction = np.double(self.parameters["CDFraction"])
            edge_range = np.int16(self.parameters["EdgeRange"])

        # Determine leading edge profiles
        image_size = processed_image.shape
        leading_edges_profiles = np.nan * np.zeros([len(new_leading_edges), image_size[1]])
        trailing_edges_profiles = np.nan * np.zeros([len(new_trailing_edges), image_size[1]])

        leading_edges_profiles = edge_detection(new_leading_edges, leading_edges_profiles)
        trailing_edges_profiles = edge_detection(new_trailing_edges, trailing_edges_profiles)

        self.leading_edges = leading_edges_profiles
        self.trailing_edges = trailing_edges_profiles
        self.consolidated_leading_edges = edge_consolidation(leading_edges_profiles)
        self.consolidated_trailing_edges = edge_consolidation(trailing_edges_profiles)
        self.zero_mean_leading_edge_profiles = edge_mean_subtraction(self.consolidated_leading_edges)
        self.zero_mean_trailing_edge_profiles = edge_mean_subtraction(self.consolidated_trailing_edges)

        profiles_shape = self.leading_edges.shape

        lines_number = profiles_shape[0]
        self.number_of_lines = lines_number

        self.critical_dimension = trailing_edges_profiles - leading_edges_profiles
        self.critical_dimension_std_estimate = np.std(
            np.nanmedian(self.critical_dimension, 1)
        )
        self.critical_dimension_estimate = np.mean(
            np.nanmedian(self.critical_dimension, 1)
        )
        self.pitch_estimate = (np.mean(
            np.nanmedian(leading_edges_profiles[1:] - leading_edges_profiles[0:-1], 1)) + np.mean(
            np.nanmedian(trailing_edges_profiles[1:] - trailing_edges_profiles[0:-1], 1))) / 2
        if len(self.leading_edges) > 1:
            pass
        else:
            self.pitch_estimate = np.nan

        # leading_edges_profiles = self.data.leading_edges[0]
        # trailing_edges_profiles = self.data.trailing_edges[0]
        # for n in np.arange(0, np.size(xlines, 0)):
        #     plt.plot(leading_edges_profiles[n, :], xlines[n, :], "r")
        #     plt.plot(trailing_edges_profiles[n, :], xlines[n, :], "b")
        # plt.show()


def post_processing(self):
    print('Postprocessing')


class MetricCalculator():
    # Make fourier transformation and calculates and optimize parameters
    def calculate_metrics(self):
        pixel_size = self.parameters["PixelSize"]
        Fs = 1 / pixel_size
        s = np.shape(self.consolidated_leading_edges)
        profiles_length = s[1]
        self.frequency = 1000 * np.arange(0, Fs / 2 + Fs / profiles_length, Fs / profiles_length)

        # Assign chosen PSD model
        selected_model = self.parameters["PSD_model"]
        if selected_model == "Palasantzas 2":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b
        elif selected_model == "Palasantzas 1":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b
        elif selected_model == "Integral":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b
        elif selected_model == "Gaussian":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b
        elif selected_model == "Floating alpha":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b
        elif selected_model == "No white noise":
            model = PSD_models.Palasantzas_2_minimize
            model_beta = PSD_models.Palasantzas_2_beta
            model_2 = PSD_models.Palasantzas_2b

        # LWR
        line_width = np.abs(self.consolidated_leading_edges - self.consolidated_trailing_edges) * pixel_size
        self.LWR_PSD = np.nanmean((np.fft.rfft(line_width)) ** 2, 0)
        self.LWR_PSD = self.LWR_PSD / len(self.LWR_PSD) ** 2
        self.LWR_PSD[0] = self.LWR_PSD[1]

        # Calculate Unbiased LWR
        beta0, beta_min, beta_max = model_beta(self, self.LWR_PSD)
        bounds = Bounds(lb=beta_min, ub=beta_max)

        # Alternative fit using scipy.optimize.curve_fit
        # optimized_parameters, covariance = curve_fit(
        #     model,
        #     self.frequency,
        #     self.LWR_PSD,
        #     p0=beta0,
        #     bounds=bounds,
        #     maxfev=100000,
        # )

        optimized_parameters = minimize(
            model,
            beta0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10},
            args=(self.frequency, self.LWR_PSD),
            bounds=bounds
        )

        self.LWR_PSD_fit_parameters = optimized_parameters['x']
        self.LWR_PSD_fit = model_2(self.frequency, optimized_parameters['x'])
        beta = self.LWR_PSD_fit_parameters
        self.LWR_PSD_unbiased = self.LWR_PSD - beta[2]
        beta[2] = 0
        self.LWR_PSD_fit_unbiased = model_2(self.frequency, beta)

        all_edges = np.vstack((
            self.zero_mean_leading_edge_profiles * pixel_size, self.zero_mean_trailing_edge_profiles * pixel_size))
        self.LER_PSD = np.nanmean(np.abs(np.fft.rfft(all_edges)) ** 2, 0)

        # LER
        self.LER_PSD = np.nanmean(np.abs(np.fft.rfft(all_edges)) ** 2, 0)
        self.LER_PSD = self.LER_PSD / len(self.LER_PSD) ** 2
        # Calculate Unbiased LER
        beta0, beta_min, beta_max = model_beta(self, self.LER_PSD)
        bounds = Bounds(lb=beta_min, ub=beta_max)

        optimized_parameters = minimize(
            model,
            beta0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10},
            args=(self.frequency, self.LER_PSD),
            bounds=bounds
        )

        self.LER_PSD_fit_parameters = optimized_parameters['x']
        self.LER_PSD_fit = model_2(self.frequency, optimized_parameters['x'])
        beta = self.LER_PSD_fit_parameters
        self.LER_PSD_unbiased = self.LER_PSD - beta[2]
        beta[2] = 0
        self.LER_PSD_fit_unbiased = model_2(self.frequency, beta)

        # Leading edges LER
        self.LER_Leading_PSD = np.nanmean(np.abs(np.fft.rfft(self.zero_mean_leading_edge_profiles * pixel_size)) ** 2,
                                          0)
        self.LER_Leading_PSD = self.LER_Leading_PSD / len(self.LER_Leading_PSD) ** 2
        # Calculate Unbiased Leading edges LER
        beta0, beta_min, beta_max = model_beta(self, self.LER_Leading_PSD)
        bounds = Bounds(lb=beta_min, ub=beta_max)

        optimized_parameters = minimize(
            model,
            beta0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10},
            args=(self.frequency, self.LER_Leading_PSD),
            bounds=bounds
        )

        self.LER_Leading_PSD_fit_parameters = optimized_parameters['x']
        self.LER_Leading_PSD_fit = model_2(self.frequency, optimized_parameters['x'])
        beta = self.LER_Leading_PSD_fit_parameters
        self.LER_Leading_PSD_unbiased = self.LER_Leading_PSD - beta[2]
        beta[2] = 0
        self.LER_Leading_PSD_fit_unbiased = model_2(self.frequency, beta)

        # Trailing edges LER
        self.LER_Trailing_PSD = np.nanmean(np.abs(np.fft.rfft(self.zero_mean_trailing_edge_profiles * pixel_size)) ** 2,
                                           0)
        self.LER_Trailing_PSD = self.LER_Trailing_PSD / len(self.LER_Trailing_PSD) ** 2
        # Calculate Unbiased Leading edges LER
        beta0, beta_min, beta_max = model_beta(self, self.LER_Trailing_PSD)
        bounds = Bounds(lb=beta_min, ub=beta_max)

        optimized_parameters = minimize(
            model,
            beta0,
            method='Nelder-Mead',
            options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10},
            args=(self.frequency, self.LER_Trailing_PSD),
            bounds=bounds
        )

        self.LER_Trailing_PSD_fit_parameters = optimized_parameters['x']
        self.LER_Trailing_PSD_fit = model_2(self.frequency, optimized_parameters['x'])
        beta = self.LER_Trailing_PSD_fit_parameters
        self.LER_Trailing_PSD_unbiased = self.LER_Trailing_PSD - beta[2]
        beta[2] = 0
        self.LER_Trailing_PSD_fit_unbiased = model_2(self.frequency, beta)
