import random

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from PIL import Image
import pickle

def contact_image(CD: float, pitch: float, arrangement: str, pixel_size: float, image_width: float, image_height: float, noise: float, inner: float, outer: float, taper: float, spline_values: np.ndarray, radius_error: float, missc: int):

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
                miss = random.randint(1, missc) - 2
                if miss>=0:
                    radius = edge / pitch + (np.random.randint(1, 10) / 10 - 0.5) * radius_error
                    x_inner_edge = radius * inner
                    x_outer_edge = radius * outer
                    x_taper = x_outer_edge + (np.sqrt(2) - x_outer_edge) * taper                    #radii.append(radius*pitch)
                    xs = np.array([0, x_inner_edge, radius, x_outer_edge, x_taper, np.sqrt(2)])
                    ys = np.array([origin_value, inner_edge_value, edge_value, outer_edge_value, taper_value, end_value])
                    spline = interpolate.PchipInterpolator(xs, ys)

                    centers.append((cx-1, cy-1))
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
                    radius_position = np.argmax(np.abs(np.diff(profile))) * pitch/(np.sqrt(2)*100 * pixel_size)
                    radii.append(radius_position)
    elif arrangement == "triangular":
        cnt = 0
        for cx in np.arange(-image_width*pixel_size/2+CD, image_width*pixel_size/2-CD, pitch):
            cnt += 1
            for cy in np.arange(-image_height * pixel_size / 2+CD, image_height * pixel_size / 2-CD, pitch):
                miss = random.randint(1, missc) - 2
                if miss >= 0:
                    radius = edge/pitch + (np.random.randint(1,10)/10-0.5) * radius_error
                    x_inner_edge = radius * inner
                    x_outer_edge = radius * outer
                    x_taper = x_outer_edge + (np.sqrt(2)-x_outer_edge) * taper
                    #radii.append(radius*pitch)
                    xs = np.array([0, x_inner_edge, radius, x_outer_edge, x_taper, np.sqrt(2)])
                    ys = np.array([origin_value, inner_edge_value, edge_value, outer_edge_value, taper_value, end_value])
                    spline = interpolate.PchipInterpolator(xs, ys)

                    if np.mod(cnt, 2) == 0:
                        centers.append((cx-1, cy-1+pitch/2))
                    else:
                        centers.append((cx-1, cy-1))
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
                    radius_position = np.argmax(abs(np.diff(profile))) * pitch / (np.sqrt(2) * 100 * pixel_size)
                    radii.append(radius_position)
    image[image < 0] = 0
    return image, centers, radii

image_list = []
annotations = []
for img_idx in range(0, 100):
    CD = random.randint(20,100)
    pitch = CD * (1 + random.randint(1,20)/10)
    arrangement_flag = random.randint(0,1)
    if arrangement_flag == 0:
        arrangement = "orthogonal"
    else:
        arrangement = "triangular"
    image_height = random.randint(600,1000)
    image_width = random.randint(800, 2000)
    noise_idx = random.randint(0,99)
    noise = np.round(np.linspace(100,15000, 100))
    noise_value = noise[noise_idx]
    inner = 1 - random.randint(1,10) / 100
    outer = 1 + random.randint(1, 10) / 100
    taper = random.randint(2, 8) / 10

    center_value = random.randint(3,10)/10
    inner_value = random.randint(5, 10) / 10
    edge_value = inner_value + random.randint(1, 10) / 10 -0.5
    outer_value = edge_value * (1-random.randint(1, 90) / 100)
    taper_value = random.randint(1, 5) / 10
    background_value = taper_value*(1-random.randint(0,10)/100)
    radius_error = random.randint(0,20)/100
    missc = random.randint(10,100)
    # image, centers, radii = contact_image(32, 64, 'triangular', 0.8, 1000,
    #                                       600, 100, 0.05, 1.05, 0.1,
    #                                       np.array([center_value, inner_value, edge_value, outer_value,
    #                                                 taper_value, background_value]), 0.15, 100)
    image, centers, radii = contact_image(CD, pitch, arrangement, 0.8, image_width,
                                          image_height, noise_value, inner, outer, taper,
                                          np.array([center_value, inner_value, edge_value, outer_value,
                                                    taper_value, background_value]), radius_error, missc)

    im = Image.fromarray(np.uint8(image*255/np.max(image)))
    s = "synt_circles_"+f'{img_idx:03}'+".jpg"
    print(s)
    im.save(s)
    image_list.append(s)  # Paths to your images

    boxes = []
    labels = []
    for contact in range(0,np.size(radii)):
        center_box = centers[contact]
        xmin = center_box[0] - radii[contact]
        xmax = center_box[0] + radii[contact]
        ymin = center_box[1] - radii[contact]
        ymax = center_box[1] + radii[contact]

        boxes.append([xmin,ymin,xmax,ymax])
        annotations.append({"boxes": boxes, "labels": np.ones((1,np.size(radii)))})

with open('annotations.pkl', 'wb+') as f:
    pickle.dump(annotations, f)
with open('image_list.pkl', 'wb+') as f:
    pickle.dump(image_list, f)
    """
    fig, ax = plt.subplots(1,1)
    cs = ax.imshow(image)

    t = np.linspace(0, 2*np.pi,100)
    for npl in range(0, np.size(radii)):
      cpl = centers[npl]
      rpl = radii[npl]
      ax.plot(cpl[0]/0.8+image_width/2, cpl[1]/0.8+image_height/2,'ro')
      ax.plot(cpl[0]/0.8+image_width/2 + rpl*np.cos(t), cpl[1]/0.8+image_height/2 + rpl*np.sin(t),'-b')
    fig.colorbar(cs)
    plt.show()
    """