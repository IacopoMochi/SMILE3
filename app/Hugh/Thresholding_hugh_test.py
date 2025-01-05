import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.signal import medfilt2d
from scipy.optimize import curve_fit
from app.utils.poly import binary_image_histogram_model
from skimage.transform import hough_ellipse
from skimage.feature import canny
image = cv2.imread('Ctest1.tif')
image = cv2.imread('/Users/iacopomochi/Documents/GitHub/SMILE3/app/DNN/9500.tif')

# BGR -> RGB
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = np.mean(img[0:800, 0:1500],2)
img = medfilt2d(img,[3, 3])
img = medfilt2d(img,[31, 31])
h = np.histogram(img/255, bins=np.linspace(0, 1, 256))
x = h[1][1:]
y = h[0]
x = x[:-1]
y = y[:-1]
xx = np.linspace(0, 255, 254)

intensity = np.linspace(0, 1, 254)
max_index = np.argmax(y)
max_value = y[max_index]
low_bounds = [max_value / 10, 0, 0.01, max_value / 10, 0.1, 0.01, 0, 0, 0.01]
high_bounds = [max_value, 1, 0.5, max_value, 1, 0.5, max_value / 4, 1, 2]
beta0 = (max_value, 0.4, 0.1, max_value, 0.45, 0.1, 1, 0.5, 0.1)
beta, _ = curve_fit(
    binary_image_histogram_model,
    intensity,
    y,
    p0=beta0,
    bounds=(low_bounds, high_bounds),
    maxfev=100000,
)
print(beta)
ax0 = plt.subplot(121)
ax1 = plt.subplot(122)

ax0.plot(x, y)

g1 = beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
g2 = beta[3] * np.exp(-(((x - beta[4]) / beta[5]) ** 2))
g3 = beta[6] * np.exp(-(((x - beta[7]) / beta[8]) ** 2))
ax0.plot(xx, g1)
ax0.plot(xx, g2)
ax0.plot(xx, g3)
ax0.plot(xx, g1+g2+g3)
threshold = 104
img[img < threshold] = 0
img[img > 0] = 1
img = medfilt2d(img,[3, 3])
img = medfilt2d(img,[31, 31])
ax1.imshow(img)
edges = canny(img, sigma=1.0, low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of the histogram for minor axis lengths.
# A higher `accuracy` value will lead to more ellipses being found, at the
# cost of a lower precision on the minor axis length estimation.
# A higher `threshold` will lead to less ellipses being found, filtering out those
# with fewer edge points (as found above by the Canny detector) on their perimeter.
#result = hough_ellipse(edges, accuracy=20, threshold=250, min_size=20, max_size=120)
#result.sort(order='accumulator')

#print(result.tolist())
plt.show()