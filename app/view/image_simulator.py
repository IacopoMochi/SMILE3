import numpy as np
from matplotlib import pyplot as plt


# VISUAL OUTPUT


CD = 16
pitch = 32
number_of_lines = 32
profile_length = 4000
pixel_size = 0.2
image_width = int((number_of_lines+2) * pitch/pixel_size)
x,y = np.meshgrid(np.linspace(-image_width*pixel_size/2,image_width*pixel_size/2,image_width), np.arange(0,profile_length))
image = np.zeros((profile_length,image_width))
print(np.shape(image))
print(np.shape(x))
coherence_length = 20
sigma = 0.6
alpha = 2
uncorrelated_noise_level = 0

profile_position = np.linspace(pixel_size, profile_length*pixel_size, int(profile_length))
Fs = 1/pixel_size
frequency = np.arange(0, Fs, Fs / profile_length)

#PSD palasantzas model
#y0 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[0]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#y1 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[1]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#y2 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[2]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#PSD gaussian model
y0 = sigma ** 2 * np.exp(-((frequency[0:1+int(profile_length/2)] * coherence_length)) ** alpha) + np.abs(uncorrelated_noise_level)

for n in range(0, number_of_lines-1):
    #Create a random profile with a normal distribution (sigma = 1)
    profile0 = np.random.randn(profile_length)
    profile1 = np.random.randn(profile_length)

    #Filter the profile with the chosen PSD
    profile0_fft = np.fft.rfft(profile0)
    profile1_fft = np.fft.rfft(profile1)
    PSD0 = profile0_fft*y0
    PSD1 = profile0_fft * y0
    # Normalization to the integral of the PSD model (Parseval theorem) and rescaling to the desired sigma
    filtered_profile0 = (n+1-(number_of_lines/2))*pitch+sigma*(np.fft.irfft(PSD0)) / (np.sqrt(sum(y0**2)/(0.5*profile_length)))
    filtered_profile1 = (n+1-(number_of_lines/2))*pitch+CD+sigma * (np.fft.irfft(PSD1)) / (np.sqrt(sum(y0 ** 2) / (0.5 * profile_length)))
    for m in range(0, profile_length):
        image[m, np.logical_and(x[m,:]>=filtered_profile0[m], x[m,:]<filtered_profile1[m])] = 1

s = np.shape(image)
image = image+np.abs(np.random.randn(s[0], s[1]))
# fig, axs = plt.subplots(3)
# axs[0].loglog(frequency[0:1+int(profile_length/2)], y0)
# axs[1].plot(profile_position, filtered_profile0)


plt.imshow(image)

plt.show()

