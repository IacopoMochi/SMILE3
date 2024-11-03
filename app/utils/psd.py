from typing import List, Tuple
import numpy as np


def Palasantzas_2_beta(image, PSD: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates the beta parameters and their min and max ranges for the Palasantzas 2 model.

    Args:
        image: The image object containing the parameters.
        PSD (np.ndarray): The power spectral density data.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing the beta parameters,
                                                      their minimum values, and their maximum values.
    """

    beta = [0, 0, 0, 0]
    parameters = image.image.parameters
    High_frequency_max = parameters["High_frequency_cut"]
    High_frequency_average = parameters["High_frequency_average"]
    High_frequency_min = -High_frequency_max - High_frequency_average
    Low_frequency_min = parameters["Low_frequency_cut"]
    Low_frequency_average = parameters["Low_frequency_average"]
    Low_frequency_max = Low_frequency_min + Low_frequency_average
    correlation_length = parameters['Correlation_length']
    alpha = parameters['Alpha']

    beta[0] = np.nanmean(PSD[Low_frequency_min:Low_frequency_max]) * correlation_length
    beta[1] = correlation_length
    beta[2] = np.nanmean(PSD[High_frequency_min:-High_frequency_max])
    beta[3] = alpha

    beta_min = [beta[0] * 0.5, beta[1] * 0.5, 0, 0]
    beta_max = [beta[0] * 2, beta[1] * 2, beta[2] * 2, beta[3] * 2]

    return beta, beta_min, beta_max


def Palasantzas_2_minimize(beta: List[float], freq: np.ndarray, PSD: np.ndarray) -> float:
    """
    Model for use with the scipy.optimize.minimize function.

    Args:
        beta (List[float]): List of beta parameters.
        freq (np.ndarray): Frequency data.
        PSD (np.ndarray): Power spectral density data.

    Returns:
        float: The minimized value.
    """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    S = np.nanmean(np.abs(PSD - y))
    return S


def Palasantzas_2(freq: np.ndarray, *beta: float) -> np.ndarray:
    """
    Model for use with the scipy.optimize.curve_fit function.

    Args:
        freq (np.ndarray): Frequency data.
        *beta (float): Beta parameters.

    Returns:
        np.ndarray: The computed model values.
    """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    return y


def Palasantzas_2b(freq: np.ndarray, beta: List[float]) -> np.ndarray:
    """
    Computes the Palasantzas 2b model.

    Args:
        freq (np.ndarray): Frequency data.
        beta (List[float]): List of beta parameters.

    Returns:
        np.ndarray: The computed model values.
    """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    y = (Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    return y


def Palasantzas_1_beta(image, PSD):
    """
    Calculates the beta parameters and their min and max ranges for the Palasantzas 1 model.

    Args:
        image: The image object containing the parameters required for the calculation.
        PSD (np.ndarray): The power spectral density data used for estimating the parameters.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing:
            - A list of beta parameters estimated from the image and PSD data.
            - A list of minimum values for each beta parameter.
            - A list of maximum values for each beta parameter.
    """

    beta = [0, 0, 0, 0, 0]
    parameters = image.parameters
    High_frequency_max = parameters["High_frequency_cut"]
    High_frequency_average = parameters["High_frequency_average"]
    High_frequency_min = -High_frequency_max - High_frequency_average
    Low_frequency_min = parameters["Low_frequency_cut"]
    Low_frequency_average = parameters["Low_frequency_average"]
    Low_frequency_max = Low_frequency_min + Low_frequency_average
    correlation_length = parameters['Correlation_length']
    alpha = parameters['Alpha']
    a = 1.0

    beta[0] = np.nanmean(PSD[Low_frequency_min:Low_frequency_max]) * correlation_length
    beta[1] = correlation_length
    beta[2] = np.nanmean(PSD[High_frequency_min:-High_frequency_max])
    beta[3] = alpha
    beta[4] = a

    beta_min = [beta[0] * 0.5, beta[1] * 0.5, 0, 0, 0]
    beta_max = [beta[0] * 2, beta[1] * 2, beta[2] * 2, beta[3] * 2, 2]

    return beta, beta_min, beta_max


def Palasantzas_1_minimize(beta, freq, PSD):
    """
    Objective function for minimizing the difference between the observed PSD and the model's predicted PSD.

    Args:
        beta (List[float]): List of beta parameters for the Palasantzas 1 model.
        freq (np.ndarray): Frequency data used to compute the model's predicted PSD.
        PSD (np.ndarray): Observed power spectral density data.

    Returns:
        float: The minimized value, which is the mean absolute difference between the observed and predicted PSD.
    """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    a = beta[4]
    y = (Lc * sig2 / (1 + a * (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    S = np.nanmean(np.abs(PSD - y))
    return S


def Palasantzas_1(freq: np.ndarray, *beta: float) -> np.ndarray:
    """
    Computes the Palasantzas 1 model.

    Args:
        freq (np.ndarray): Frequency data.
        *beta (float): Beta parameters.

    Returns:
        np.ndarray: The computed model values.
    """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    a = beta[4]
    y = (Lc * sig2 / (1 + a * (freq * Lc) ** 2) ** (1 + alpha)) + np.abs(Nl)
    return y


def Palasantzas_1b(freq, beta):
    """
        Calculates the beta parameters and their min and max ranges for the Palasantzas 1 model.

        Args:
            image: The image object containing the parameters.
            PSD (np.ndarray): The power spectral density data.

        Returns:
            Tuple[List[float], List[float], List[float]]: A tuple containing the beta parameters,
                                                          their minimum values, and their maximum values.
        """

    sig2 = beta[0]
    Lc = 1 / beta[1]
    Nl = beta[2]
    alpha = beta[3]
    a = beta[4]
    y = (Lc * sig2 / (1 + a * (freq * Lc) ** 2) ** (0.5 + alpha)) + np.abs(Nl)
    return y


def Gaussian(freq: np.ndarray, *beta: float) -> np.ndarray:
    """
    Computes the Gaussian model.

    Args:
        freq (np.ndarray): Frequency data.
        *beta (float): Beta parameters.

    Returns:
        np.ndarray: The computed Gaussian model values.
    """

    A = beta[0]
    B = beta[1]
    C = beta[2]
    y = A * np.exp(-((freq * B) ** 2)) + np.abs(C)
    return y


def NoWhiteNoise(freq: np.ndarray, *beta: float) -> np.ndarray:
    """
    Computes the NoWhiteNoise model.

    Args:
        freq (np.ndarray): Frequency data.
        *beta (float): Beta parameters.

    Returns:
        np.ndarray: The computed NoWhiteNoise model values.
    """

    sig2 = beta[0]
    Lc = 1.0 / beta[1]
    alpha = beta[2]
    y = Lc * sig2 / (1 + (freq * Lc) ** 2) ** (0.5 + alpha)
    return y

    # y = (Lc * sig2 / (1 + a * (freq * Lc) ** 2) ** (1 + alpha)) + np.abs(Nl)
    # return y