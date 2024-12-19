import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from PIL import Image

def contact_image(CD: float, pitch: float, arrangement: str, pixel_size: float, image_width: float, image_height: float, noise: float, inner: float, outer: float, taper: float, spline_values: np.ndarray, radius_error: float):

    x, y = np.meshgrid(np.linspace(-image_width*pixel_size/2, image_width*pixel_size/2, image_width),
                       np.linspace(-image_height*pixel_size/2, image_height*pixel_size/2, image_height))

    centers = []

    edge = CD

    origin_value = spline_values[0]
    inner_edge_value = spline_values[1]
    edge_value = spline_values[2]
    outer_edge_value = spline_values[3]
    taper_value = spline_values[4]
    end_value = spline_values[5]

    image = np.zeros(np.shape(x)) + end_value

    radii = []
    if arrangement == "orthogonal":
        for cx in np.arange(-image_width*pixel_size/2, image_width*pixel_size/2, pitch):
            for cy in np.arange(-image_height * pixel_size / 2, image_height * pixel_size / 2, pitch):

                radius = edge / pitch + np.random.random(1) * radius_error
                x_inner_edge = radius[0] * inner
                x_outer_edge = radius[0] * outer
                x_taper = x_outer_edge + (1 - x_outer_edge) * taper
                #radii.append(radius*pitch)
                xs = np.array([0, x_inner_edge, radius[0], x_outer_edge, x_taper, np.sqrt(2)])
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
                x_profile = np.linspace(0, np.sqrt(2),100)
                profile = spline(x_profile)
                radius_position = np.argmin(np.diff(profile)) * pitch/100
                radii.append(radius_position)

    elif arrangement == "triangular":
        cnt = 0
        for cx in np.arange(-image_width*pixel_size/2+CD, image_width*pixel_size/2-CD, pitch):
            cnt += 1
            for cy in np.arange(-image_height * pixel_size / 2+CD, image_height * pixel_size / 2-CD, pitch):

                radius = edge/pitch + np.random.random(1) * radius_error
                x_inner_edge = radius[0] * inner
                x_outer_edge = radius[0] * outer
                x_taper = x_outer_edge + (1 - x_outer_edge) * taper
                #radii.append(radius*pitch)
                xs = np.array([0, x_inner_edge, radius[0], x_outer_edge, x_taper, 1])
                ys = np.array([origin_value, inner_edge_value, edge_value, outer_edge_value, taper_value, end_value])
                spline = interpolate.PchipInterpolator(xs, ys)

                if np.mod(cnt, 2) == 0:
                    centers.append((cx, cy+pitch/2))
                else:
                    centers.append((cx, cy))
                center = centers[-1]
                local_map = np.logical_and((np.abs(x - center[0]) <= pitch / 2), (np.abs(y - center[1]) <= pitch / 2))
                rx = x[local_map] - center[0]
                ry = y[local_map] - center[1]
                maximum = np.max([rx, ry])
                rx = rx / maximum
                ry = ry / maximum
                rho = np.sqrt(rx ** 2 + ry ** 2)
                rho_noise = np.random.poisson(rho * noise) / noise
                image[local_map] = spline(rho_noise)
                x_profile = np.linspace(0, np.sqrt(2), 100)
                profile = spline(x_profile)
                radius_position = np.argmin(np.diff(profile)) * pitch / 100
                radii.append(radius_position)

    image[image < 0] = 0
    return image, centers, radii


for img_idx in range(0, 100):
    CD = random.randint(20,100)
    pitch = CD * random.randint(6,10)/4
    arrangement_flag = random.randint(0,1)
    if arrangement_flag == 0:
        arrangement = "orthogonal"
    else:
        arrangement = "triangular"
    image_height = random.randint(600,1000)
    image_width = random.randint(800, 2000)
    noise_idx = random.randint(0,9)
    noise = np.round(np.linspace(100,10000, 10))
    noise_value = noise[noise_idx]
    inner = random.randint(1,10) / 100
    outer = random.randint(1, 10) / 100
    taper = random.randint(1, 20) / 100

    center_value = random.randint(2,10)/10
    inner_value = random.randint(2, 10) / 10
    outer_value = inner_value
    edge_value = random.randint(2, 10) / 10
    taper_value = random.randint(6, 7) / 10
    background_value = random.randint(0, 2) / 10
    radius_error = random.randint(0,15)/100
    image, centers, radii = contact_image(32, 64, 'triangular', 0.8, 1000,
                                      600, 100, 0.05, 1.05, 0.1,
                                      np.array([center_value, inner_value, edge_value, outer_value,
                                                taper_value,background_value]), 0.15)

    im = Image.fromarray(np.uint8(image*255))
    s = "synt_circles_"+f'{img_idx:03}'+".jpg"
    print(s)
    im.save(s)

# fig, ax = plt.subplots(1,1)
# cs = ax.imshow(image)
# #
# t = np.linspace(0, 2*np.pi,100)
# for n in range(0, np.size(radii)):
#     c = centers[n]
#     r = radii[n]
#     ax.plot(c[0]/0.8+500, c[1]/0.8+300,'ro')
#     ax.plot(c[0]/0.8+500 + r*np.cos(t), c[1]/0.8+300 + r*np.sin(t),'-b')
# fig.colorbar(cs)
# plt.show()