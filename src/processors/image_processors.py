import numpy as np
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit, minimize, Bounds
from scipy.ndimage import histogram
from scipy.signal import medfilt2d, filtfilt, butter, find_peaks

from src.utils.poly import (_poly11, poly11, binary_image_histogram_model, gaussian_profile)
from src.utils.psd import Palasantzas_2_minimize, Palasantzas_2_beta, Palasantzas_2b


class PreProcessor:

    # normalization, brightness removal (poly11()), rotation to find the lines
    # histogram analysis (gaussian_profile(), binary_image_histogram_model())

    def __init__(self, smile_lines_image):
        self.smile_image = smile_lines_image

    def crop_and_rotate_image(self):
        # Crop images to the specified ROI
        x1 = int(self.smile_image.parameters["X1"])
        x2 = int(self.smile_image.parameters["X2"])
        y1 = int(self.smile_image.parameters["Y1"])
        y2 = int(self.smile_image.parameters["Y2"])

        image_cropped = self.smile_image.image[x1:x2, y1:y2]
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

        rotated_image = rotate(self.smile_image.image, -float(theta[max_id]) + 90, order=0)
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
        self.smile_image.processed_image = image_normalized

    def calculate_histogram_parameters(self):
        self.smile_image.intensity_histogram = histogram(self.smile_image.processed_image, 0, 1, 256)
        intensity = np.linspace(0, 1, 256)
        max_index = np.argmax(self.smile_image.intensity_histogram)
        max_value = self.smile_image.intensity_histogram[max_index]
        low_bounds = [max_value / 4, 0, 0.01, max_value / 4, 0.1, 0.01, 0, 0, 0.01]
        high_bounds = [max_value, 1, 0.5, max_value, 1, 0.5, max_value / 4, 1, 2]
        beta0 = (max_value, 0.25, 0.1, max_value, 0.75, 0.1, 1, 0.5, 0.1)
        beta, covariance = curve_fit(
            binary_image_histogram_model,
            intensity,
            self.smile_image.intensity_histogram,
            p0=beta0,
            bounds=(low_bounds, high_bounds),
            maxfev=100000,
        )

        self.smile_image.intensity_histogram_gaussian_fit_parameters = beta
        self.smile_image.intensity_histogram_low = gaussian_profile(intensity, *beta[0:3])
        self.smile_image.intensity_histogram_high = gaussian_profile(intensity, *beta[6:9])
        self.smile_image.intensity_histogram_medium = gaussian_profile(intensity, *beta[3:6])

        self.smile_image.lines_snr = np.abs(beta[1] - beta[4]) / (
                0.5 * (beta[2] + beta[5]) * 2 * np.sqrt(-2 * np.log(0.5))
        )


class EdgeDetector:

    def __init__(self, smile_lines_image):
        self.smile_image = smile_lines_image

    # find the pick in image
    # consolidate the edges and try to center that around zero
    # calculate critical dimensions and pitch estimates based on the detected edges
    # basically makes necessary calculations
    # call in determine_edge_profiles

    def edge_detection(self, new_edges, edges_profiles):
        cnt = -1
        image_size = self.smile_image.processed_image.shape
        edge_range = self.smile_image.parameters["EdgeRange"]
        for edge in new_edges:
            cnt = cnt + 1
            for row in range(0, image_size[1]):
                segment_start = int(np.max([0, edge - edge_range]))
                segment_end = int(np.min([edge + edge_range, image_size[0]]))
                x = np.arange(segment_start, segment_end)
                segment = self.smile_image.processed_image[segment_start:segment_end, row]
                if self.smile_image.parameters["Edge_fit_function"] == "polynomial":
                    p = np.polyfit(x, segment, 4)
                    p[-1] = p[-1] - np.double(self.smile_image.parameters["Threshold"])
                    r = np.roots(p)
                    r = r[np.imag(r) == 0]
                    if len(r) > 0:
                        edge_position = r[np.argmin(np.abs(r - (segment_start + segment_end) / 2))]
                        edges_profiles[cnt, row] = np.real(edge_position)
                elif self.smile_image.parameters["Edge_fit_function"] == "linear":
                    print("Add code for linear edge finding")
                elif self.smile_image.parameters["Edge_fit_function"] == "threshold":
                    print("Add code for threshold edge finding")
                elif self.smile_image.parameters["Edge_fit_function"] == "bright_edge":
                    print("Add code for bright edge finding")
        return edges_profiles

    def edge_consolidation(self, raw_edge_profiles):
        consolidated_edge_profiles = raw_edge_profiles.copy()
        for edge in consolidated_edge_profiles:
            mean_value = np.nanmean(edge)
            edge[edge is np.nan] = mean_value
        return consolidated_edge_profiles

    def edge_mean_subtraction(self, absolute_edge_profiles):
        zero_mean_edge_profiles = absolute_edge_profiles.copy()
        for edge in zero_mean_edge_profiles:
            mean_value = np.nanmean(edge)
            edge[:] = edge - mean_value
        return zero_mean_edge_profiles

    def filter_and_reduce_noise(self):
        # Filter image with a 2d median filter to remove eventual outliers (bright pixels for instance)
        median_filter_kernel = 5
        filtered_image = medfilt2d(self.smile_image.processed_image, median_filter_kernel)

        # Sum all the columns of the image to calculate the average lines profile
        image_sum = np.sum(filtered_image, 1)
        # Filter the lines profile to reduce the noise
        backward_filter, forward_filter = butter(8, 0.125)
        image_sum_filtered = filtfilt(backward_filter, forward_filter, image_sum, method="gust")
        return image_sum, image_sum_filtered

    def detect_peaks(self):
        image_sum, image_sum_filtered = self.filter_and_reduce_noise()
        image_sum_derivative = np.diff(image_sum)
        image_sum_filtered_derivative = np.abs(np.diff(image_sum_filtered))
        peaks = find_peaks(image_sum_filtered_derivative)
        return image_sum_derivative, peaks

    def classify_edges(self):
        image_sum_derivative, peaks = self.detect_peaks()
        edge_locations = peaks[0]
        leading_edges = np.array([])
        trailing_edges = np.array([])
        if self.smile_image.parameters["brightEdge"]:
            print("Bright edge peak detection to be added here")
        else:
            for n in edge_locations:
                if self.smile_image.parameters["tone_positive_radiobutton"]:
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

    def find_leading_and_trailing_edges(self):
        leading_edges, trailing_edges = self.classify_edges()

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
            # pitch = 0.5 * (np.mean(np.diff(new_trailing_edges)) + np.mean(np.diff(new_leading_edges)))

        return new_leading_edges, new_trailing_edges

    def determine_edge_profiles(self, ):
        # Determine leading edge profiles

        new_leading_edges, new_trailing_edges = self.find_leading_and_trailing_edges()

        image_size = self.smile_image.processed_image.shape
        leading_edges_profiles_param = np.nan * np.zeros([len(new_leading_edges), image_size[1]])
        trailing_edges_profiles_param = np.nan * np.zeros([len(new_trailing_edges), image_size[1]])

        leading_edges_profiles = self.edge_detection(new_leading_edges, leading_edges_profiles_param)
        trailing_edges_profiles = self.edge_detection(new_trailing_edges, trailing_edges_profiles_param)

        return leading_edges_profiles, trailing_edges_profiles

    def find_edges(self):
        self.smile_image.leading_edges, self.smile_image.trailing_edges = self.determine_edge_profiles()

        self.smile_image.consolidated_leading_edges = self.edge_consolidation(self.smile_image.leading_edges)
        self.smile_image.consolidated_trailing_edges = self.edge_consolidation(self.smile_image.trailing_edges)
        self.smile_image.zero_mean_leading_edge_profiles = self.edge_mean_subtraction(
            self.smile_image.consolidated_leading_edges)
        self.smile_image.zero_mean_trailing_edge_profiles = self.edge_mean_subtraction(
            self.smile_image.consolidated_trailing_edges)

        profiles_shape = self.smile_image.leading_edges.shape
        lines_number = profiles_shape[0]

        self.smile_image.number_of_lines = lines_number

        self.smile_image.critical_dimension = self.smile_image.trailing_edges - self.smile_image.leading_edges
        self.smile_image.critical_dimension_std_estimate = np.std(np.nanmedian(self.smile_image.critical_dimension, 1))
        self.smile_image.critical_dimension_estimate = np.mean(np.nanmedian(self.smile_image.critical_dimension, 1))

        self.smile_image.pitch_estimate = (np.mean(
            np.nanmedian(self.smile_image.leading_edges[1:] - self.smile_image.leading_edges[0:-1], 1)) + np.mean(
            np.nanmedian(self.smile_image.trailing_edges[1:] - self.smile_image.trailing_edges[0:-1], 1))) / 2

        if len(self.smile_image.leading_edges) > 1:
            pass
        else:
            self.smile_image.pitch_estimate = np.nan

        # leading_edges_profiles = self.data.leading_edges[0]
        # trailing_edges_profiles = self.data.trailing_edges[0]
        # for n in np.arange(0, np.size(xlines, 0)):
        #     plt.plot(leading_edges_profiles[n, :], xlines[n, :], "r")
        #     plt.plot(trailing_edges_profiles[n, :], xlines[n, :], "b")
        # plt.show()


class MetricCalculator:
    # Make fourier transformation and calculates and optimize parameters

    def __init__(self, smile_lines_image):
        self.smile_image = smile_lines_image

    def setup_frequency(self):
        self.smile_image.pixel_size = self.smile_image.parameters["PixelSize"]
        Fs = 1 / self.smile_image.pixel_size
        s = np.shape(self.smile_image.consolidated_leading_edges)
        profiles_length = s[1]
        self.smile_image.frequency = 1000 * np.arange(0, Fs / 2 + Fs / profiles_length, Fs / profiles_length)

    def select_psd_model(self):
        # Assign chosen PSD model

        selected_model = self.smile_image.parameters["PSD_model"]
        valid_models = {"Palasantzas 2", "Palasantzas 1", "Integral", "Gaussian", "Floating alpha", "No white noise"}
        if selected_model in valid_models:
            model = Palasantzas_2_minimize
            model_beta = Palasantzas_2_beta
            model_2 = Palasantzas_2b
        else:
            model, model_beta, model_2 = None, None, None

        return model, model_beta, model_2

    def calculate_and_fit_psd(self, input_data):
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
            args=(self.smile_image.frequency, PSD),
            bounds=bounds
        )

        PSD_fit_parameters = optimized_parameters['x']
        PSD_fit = model_2(self.smile_image.frequency, optimized_parameters['x'])
        beta = PSD_fit_parameters
        PSD_unbiased = PSD - beta[2]
        beta[2] = 0
        PSD_fit_unbiased = model_2(self.smile_image.frequency, beta)

        return PSD, PSD_fit_parameters, PSD_fit, PSD_unbiased, PSD_fit_unbiased

    def calculate_metrics(self):

        # LWR PSD
        line_width = np.abs(
            self.smile_image.consolidated_leading_edges - self.smile_image.consolidated_trailing_edges) * self.smile_image.pixel_size

        (self.smile_image.LWR_PSD,
         self.smile_image.LWR_PSD_fit_parameters,
         self.smile_image.LWR_PSD_fit,
         self.smile_image.LWR_PSD_unbiased,
         self.smile_image.LWR_PSD_fit_unbiased) = self.calculate_and_fit_psd(line_width)

        # LER PSD
        all_edges = np.vstack((self.smile_image.zero_mean_leading_edge_profiles * self.smile_image.pixel_size,
                               self.smile_image.zero_mean_trailing_edge_profiles * self.smile_image.pixel_size))

        (self.smile_image.LER_PSD,
         self.smile_image.LER_PSD_fit_parameters,
         self.smile_image.LER_PSD_fit,
         self.smile_image.LER_PSD_unbiased,
         self.smile_image.LER_PSD_fit_unbiased) = self.calculate_and_fit_psd(all_edges)

        # Leading edges LER
        input_data = self.smile_image.zero_mean_leading_edge_profiles * self.smile_image.pixel_size

        (self.smile_image.LER_Leading_PSD,
         self.smile_image.LER_Leading_PSD_fit_parameters,
         self.smile_image.LER_Leading_PSD_fit,
         self.smile_image.LER_Leading_PSD_unbiased,
         self.smile_image.LER_Leading_PSD_fit_unbiased) = self.calculate_and_fit_psd(input_data)

        # Trailing edges LER
        input_data = self.smile_image.zero_mean_trailing_edge_profiles * self.smile_image.pixel_size

        (self.smile_image.LER_Trailing_PSD,
         self.smile_image.LER_Trailing_PSD_fit_parameters,
         self.smile_image.LER_Trailing_PSD_fit,
         self.smile_image.LER_Trailing_PSD_unbiased,
         self.smile_image.LER_Trailing_PSD_fit_unbiased) = self.calculate_and_fit_psd(input_data)
