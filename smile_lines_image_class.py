import numpy as np
import os
from skimage.transform import radon, rotate
from scipy.optimize import curve_fit
from scipy.ndimage import histogram
from scipy.signal import medfilt2d, filtfilt, butter, find_peaks
from PIL import Image
import pyqtgraph as pg


class SmileLinesImage:
    def __init__(self, id, file_name, path, feature):
        self.pitch_estimate = None
        self.parameters = None
        self.leading_edges = None
        self.trailing_edges = None
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
        s = os.path.join(path, file_name)
        img = Image.open(s)
        # img = np.transpose(np.array(img))
        img = np.rot90(img, 3)
        self.image = img
        self.feature = feature
        self.id = id
        self.selected = True
        self.processed = False

    def pre_processing(self):
        def _poly11(M, *args):
            x, y = M
            return args[0] * x + args[1] * y + args[2]

        def _poly22(M, *args):
            x, y = M
            return (
                    args[0] * x
                    + args[1] * y
                    + args[2]
                    + args[3] * x ** 2
                    + args[4] * x * y
                    + args[5] * y ** 2
            )

        def _poly33(M, *args):
            x, y = M
            return (
                    args[0] * x
                    + args[1] * y
                    + args[2]
                    + args[3] * x ** 2
                    + args[4] * x * y
                    + args[5] * y ** 2
                    + args[6] * x ** 3
                    + args[7] * y * x ** 2
                    + args[8] * x * y ** 2
                    + args[7] * y ** 3
            )

        def poly11(M, args):
            x, y = M
            return args[0] * x + args[1] * y + args[2]

        def poly22(M, args):
            x, y = M
            return (
                    args[0] * x
                    + args[1] * y
                    + args[2]
                    + args[3] * x ** 2
                    + args[4] * x * y
                    + args[5] * y ** 2
            )

        def poly33(M, args):
            x, y = M
            return (
                    args[0] * x
                    + args[1] * y
                    + args[2]
                    + args[3] * x ** 2
                    + args[4] * x * y
                    + args[5] * y ** 2
                    + args[6] * x ** 3
                    + args[7] * y * x ** 2
                    + args[8] * x * y ** 2
                    + args[7] * y ** 3
            )

        def binary_image_histogram_model(x, *beta):
            return (
                    beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
                    + beta[3] * np.exp(-(((x - beta[4]) / beta[5]) ** 2))
                    + beta[6] * np.exp(-(((x - beta[7]) / beta[8]) ** 2))
            )

        def gaussian_profile(x, *beta):
            return (
                    beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
            )

        # Crop images to the specified ROI
        x1 = int(self.parameters["X1"])
        x2 = int(self.parameters["X2"])
        y1 = int(self.parameters["Y1"])
        y2 = int(self.parameters["Y2"])

        image_cropped = self.image[x1:x2, y1:y2]
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

        rotated_image = rotate(self.image, -float(theta[max_id]) + 90, order=0)
        image_rotated_cropped = rotated_image[x1:x2, y1:y2]

        # Remove brightness gradient

        image = image_rotated_cropped
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

        # self.data.processed_images[n] = normal(
        #    filters.median(image - brightness, np.ones((3, 3)))
        # )
        image_flattened = image - brightness
        image_flattened_max = np.max(image_flattened)
        image_flattened_min = np.min(image_flattened)
        image_normalized = (image_flattened - image_flattened_min) / (image_flattened_max - image_flattened_min)
        self.processed_image = image_normalized
        # print(optimized_parameters)
        # self.data_table.setCurrentIndex(self.current_image)
        # display_data(self)

        # Store the pixel size for the image
        # self.pixel_size = np.float64(self.pixel_size.text())

        # Store the processed image histogram and estimate the image line scan error
        self.intensity_histogram = histogram(
            self.processed_image, 0, 1, 256
        )
        intensity = np.linspace(0, 1, 256)
        image_histogram = self.intensity_histogram
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

        self.intensity_histogram_gaussian_fit_parameters = beta
        self.intensity_histogram_low = gaussian_profile(intensity, *beta[0:3])
        self.intensity_histogram_high = gaussian_profile(intensity, *beta[6:9])
        self.intensity_histogram_medium = gaussian_profile(intensity, *beta[3:6])

        self.lines_snr = np.abs(beta[1] - beta[4]) / (
                0.5 * (beta[2] + beta[5]) * 2 * np.sqrt(-2 * np.log(0.5))
        )

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
            critical_dimension = np.mean(new_trailing_edges - new_leading_edges)
            pitch = 0.5 * (
                    np.mean(np.diff(new_trailing_edges)) + np.mean(np.diff(new_leading_edges))
            )

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

        self.critical_dimension = trailing_edges_profiles - leading_edges_profiles
        self.critical_dimension_std_estimate = np.std(
            np.nanmedian(self.critical_dimension, 1)
        )
        self.critical_dimension_estimate = np.mean(
            np.nanmedian(self.critical_dimension, 1)
        )
        self.pitch_estimate = (np.mean(np.nanmedian(leading_edges_profiles[1:] - leading_edges_profiles[0:-1], 1))+ np.mean(np.nanmedian(trailing_edges_profiles[1:] - trailing_edges_profiles[0:-1], 1))) / 2
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

    def calculate_metrics(self):
        print('calculate_metrics')
