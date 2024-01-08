import numpy as np
import os
from skimage.transform import radon, rotate
from PIL import Image


class SmileLinesImage:
  def __init__(self, id, file_name, path, feature):
    self.parameters = None
    self.edges = None
    self.metrics = None
    self.processed_image = None
    self.file_name = file_name
    self.folder = path
    s = os.path.join(path ,file_name)
    self.image = Image.open(s)
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

      # Crop images to the specified ROI
      x1 = int(self.parameters.roi_x1)
      x2 = int(self.parameters.roi_x2)
      y1 = int(self.parameters.roi_y1)
      y2 = int(self.parameters.roi_y2)

      image = self.data.raw_images[self.current_image]
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
      self.data.processed_images[self.current_image] = normal(image - brightness)
      # print(optimized_parameters)
      # self.data_table.setCurrentIndex(self.current_image)
      # display_data(self)

      # Store the pixel size for the image
      self.data.pixel_size[self.current_image] = np.float64(self.pixel_size.text())

      # Store the processed image histogram and estimate the image line scan error
      self.data.intensity_histogram[self.current_image] = scipy.ndimage.histogram(
          self.data.processed_images[self.current_image], 0, 1, 256
      )
      intensity = np.linspace(0, 1, 256)
      image_histogram = self.data.intensity_histogram[self.current_image]
      max_index = np.argmax(image_histogram)
      max_value = image_histogram[max_index]
      low_bounds = [max_value / 4, 0, 0.01, max_value / 4, 0.1, 0.01, 0, 0, 0.01]
      high_bounds = [max_value, 1, 0.5, max_value, 1, 0.5, max_value / 4, 1, 2]
      beta0 = (max_value, 0.25, 0.1, max_value, 0.75, 0.1, 1, 0.5, 0.1)
      beta, covariance = curve_fit(
          binary_image_histogram_model,
          intensity,
          self.data.intensity_histogram[self.current_image],
          p0=beta0,
          bounds=(low_bounds, high_bounds),
          maxfev=100000,
      )

      self.data.intensity_histogram_gaussian_fit_parameters[self.current_image] = beta

      self.data.lines_snr[self.current_image] = np.abs(beta[1] - beta[4]) / (
              0.5 * (beta[2] + beta[5]) * 2 * np.sqrt(-2 * np.log(0.5))
      )

  def find_edges(self):
      print('Find edges')

  def post_processing(self):
      print('Postprocessing')

  def calculate_metrics(self):
      print('calculate_metrics')