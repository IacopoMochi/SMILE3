import numpy as np
from matplotlib import pyplot as plt


profile_length = 2000
pixel_size = 1
coherence_length = [10, 35, 60]
sigma = 4
alpha = 2
uncorrelated_noise_level = 0.1

profile_position = np.linspace(pixel_size, profile_length*pixel_size, int(profile_length))
Fs = 1/pixel_size
frequency = np.arange(0, Fs, Fs / profile_length)

#PSD palasantzas model
#y0 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[0]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#y1 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[1]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#y2 = (sigma**2 / (1 + (frequency[0:1+int(profile_length/2)] * coherence_length[2]) ** 2) ** (0.5 + alpha)) + np.abs(uncorrelated_noise_level)
#PSD gaussian model
y0 = sigma**2 *np.exp(-((frequency[0:1+int(profile_length/2)] * coherence_length[0])) ** alpha)  + np.abs(uncorrelated_noise_level)
y1 = sigma**2 *np.exp(-((frequency[0:1+int(profile_length/2)] * coherence_length[1])) ** alpha)  + np.abs(uncorrelated_noise_level)
y2 = sigma**2 *np.exp(-((frequency[0:1+int(profile_length/2)] * coherence_length[2])) ** alpha)  + np.abs(uncorrelated_noise_level)

#Create a random profile with a normal distribution (sigma = 1)
profile = np.random.randn(profile_length)

#Filter the profile with the chosen PSD
profile_fft = np.fft.rfft(profile)
PSD0 = profile_fft*y0
PSD1 = profile_fft*y1
PSD2 = profile_fft*y2

filtered_profile0 = (np.fft.irfft(PSD0))
filtered_profile1 = (np.fft.irfft(PSD1))
filtered_profile2 = (np.fft.irfft(PSD2))

fig, axs = plt.subplots(2)
axs[0].loglog(frequency[0:1+int(profile_length/2)], y0)
axs[0].loglog(frequency[0:1+int(profile_length/2)], y1)
axs[0].loglog(frequency[0:1+int(profile_length/2)], y2)
axs[1].plot(profile_position, filtered_profile0)
axs[1].plot(profile_position, filtered_profile1)
axs[1].plot(profile_position, filtered_profile2)
plt.show()