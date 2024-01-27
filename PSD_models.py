# PSD models
import numpy as np

def Palasantzas_2_beta(image):
    beta = [0, 0, 0, 0]
    LWR_PSD = image.LWR_PSD
    LWR_PSD_length = np.size(image.LWR_PSD)
    parameters = image.parameters
    High_frequency_max = parameters["High_frequency_cut"]
    High_frequency_average = parameters["High_frequency_average"]
    High_frequency_min = -High_frequency_max - High_frequency_average
    Low_frequency_min = parameters["Low_frequency_cut"]
    Low_frequency_average = parameters["Low_frequency_average"]
    Low_frequency_max = Low_frequency_min + Low_frequency_average
    correlation_length = parameters['Correlation_length']
    alpha = parameters['Alpha']

    beta[0] = np.nanmean(LWR_PSD[Low_frequency_min:Low_frequency_max])*correlation_length
    beta[1] = correlation_length
    beta[2] = np.nanmean(LWR_PSD[High_frequency_min:-High_frequency_max])
    beta[3] = alpha


    beta_min = [beta[0]*0.5, beta[1]*0.5, 0, 0]
    beta_max = [beta[0] * 2, beta[1] * 2, beta[2]*2, beta[3]*2]

    return beta, beta_min, beta_max

def Palasantzas_2(beta, freq, PSD):
    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    S = np.nanmean(np.abs(PSD - y))
    return S


def Palasantzas_2b(freq, beta):
    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    return y


def Palasantzas_1(freq, *beta):
    sig2 = beta[0]
    Lc = 1.0 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    a = beta[4]
    y = (Lc * sig2 / (1 + a * (freq * Lc) ** 2) ** (1 + alpha)) + np.abs(Nl)
    return y


def Gaussian(freq, *beta):
    A = beta[0]
    B = beta[1]
    C = beta[2]
    y = A * np.exp(-((freq * B) ** 2)) + np.abs(C)
    return y


def NoWhiteNoise(freq, *beta):
    sig2 = beta[0]
    Lc = 1.0 / beta[1]
    alpha = beta[2]
    y = Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)
    return y
