import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

def contact_image(CD: float, pitch: float, arrangement: float, pixel_size: float, image_width: float, image_height: float, noise: float, inner: float, outer: float, taper: float, spline_values: np.ndarray, radius_error: float):

    x, y = np.meshgrid(np.linspace(-image_width*pixel_size/2, image_width*pixel_size/2, image_width),
                       np.linspace(-image_height*pixel_size/2, image_height*pixel_size/2, image_height))
    image = np.zeros(np.shape(x))

    centers = []

    origin = 0
    edge = CD
    end = 1

    origin_value = spline_values[0]
    edge_value = spline_values[2]
    inner_edge_value = spline_values[1]
    outer_edge_value = spline_values[3]
    taper_value = spline_values[4]
    end_value = spline_values[5]

    radii = []
    if arrangement == "orthogonal":
        for cx in np.arange(-image_width*pixel_size/2, image_width*pixel_size/2, pitch):
            for cy in np.arange(-image_height * pixel_size / 2, image_height * pixel_size / 2, pitch):

                radius = edge+np.random.random(1)*radius_error
                inner_edge = radius[0] * inner
                outer_edge = radius[0]* outer
                taper = outer_edge + (1 - outer_edge) * taper

                radii.append(radius)
                xs = np.array([origin, inner_edge, radius, outer_edge, taper, end])
                ys = np.array([origin_value, inner_edge_value, edge_value, outer_edge_value, taper_value, end_value])
                spline = interpolate.PchipInterpolator(xs, ys)

                centers.append((cx, cy))
                local_map = np.logical_and((np.abs(x-cx) <= pitch/2), (np.abs(y-cy) <= pitch/2))
                rx = x[local_map]-cx
                ry = y[local_map]-cy
                maximum = np.max([rx, ry])
                rx = rx / maximum
                ry = ry / maximum
                rho = np.sqrt(rx**2 + ry**2)
                rho_noise = np.random.poisson(rho*noise)/noise
                image[local_map] = spline(rho_noise)

    elif arrangement == "triangular":
        cnt = 0
        for cx in np.arange(-image_width*pixel_size/2, image_width*pixel_size/2, pitch):
            cnt += 1
            for cy in np.arange(-image_height * pixel_size / 2, image_height * pixel_size / 2, pitch):

                radius = edge + np.random.random(1) * radius_error
                xinner_edge = radius[0] * inner
                xouter_edge = radius[0] * outer
                xtaper = xouter_edge + (1 - xouter_edge) * taper
                radii.append(radius)
                xs = np.array([0, xinner_edge, radius[0], xouter_edge, xtaper, 1])
                print(xs)
                ys = np.array([origin_value, inner_edge_value, edge_value, outer_edge_value, taper_value, end_value])
                spline = interpolate.PchipInterpolator(xs, ys)

                if np.mod(cnt, 2) == 0:
                    centers.append((cx, cy+pitch/2))
                else:
                    centers.append((cx, cy))
                    center = centers[-1]
                    local_map = np.logical_and((np.abs(x - center[0]) <= pitch / 2), (np.abs(y - center[1]) <= pitch / 2))
                    rx = x[local_map] - cx
                    ry = y[local_map] - cy
                    maximum = np.max([rx, ry])
                    rx = rx / maximum
                    ry = ry / maximum
                    rho = np.sqrt(rx ** 2 + ry ** 2)
                    rho_noise = np.random.poisson(rho * noise) / noise
                    image[local_map] = spline(rho_noise)


    return image, centers, radii

image, centers, radii = contact_image(16, 32, 'triangular', 0.2, 2000, 1200, 5, 0.2, 0.2, 0.1, np.array([0.7,0.9,1.0,0.9,0.1,1.0]), 1)
plt.imshow(image)
plt.show()