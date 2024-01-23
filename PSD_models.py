# PSD models
import numpy as np

def Palasantzas_2_beta(image):
    beta = np.zeros((1,4))
    parameters = image.parmeters
    minH = parameters["minimumHigh"]
    maxH = parameters["maximumHigh"]
    LWR_PSD = image.LWR_PSD
    beta[0] = np.nanmean(LWR_PSD[minH:maxH])
    beta[1] = np.nanmean(LWR_PSD[minH:maxH])
    beta[2] = np.nanmean(LWR_PSD[minH:maxH])
    beta[3] = np.nanmean(LWR_PSD[minH:maxH])

    return beta

def Palasantzas_2(freq, *beta):
    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    return y


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
