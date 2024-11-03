from typing import List, Tuple
import numpy as np

def hhcf_minimize(beta: List[float], x: np.ndarray, HHCF: np.ndarray) -> float:
    """
        Model for use with the scipy.optimize.minimize function.

        Args:
            beta (List[float]): List of beta parameters.
            x (np.ndarray): length coordinate of the profile.
            HHCF (np.ndarray): measured height-height correlation function.

        Returns:
            float: The minimized value.
        """
    sigma2 = beta[0]
    correlation_length = beta[1]
    hurst_coefficient = beta[2]

    hhcf = 2*sigma2*(np.exp(-(x/correlation_length)**(2*hurst_coefficient)))
    s = np.nanmean((hhcf - HHCF)**2)
    return s


def hhcf_(x: np.ndarray, *beta: float) -> np.ndarray:
    """
    Model for use with the scipy.optimize.curve_fit function.

    Args:
        beta (List[float]): List of beta parameters.
        x (np.ndarray): length coordinate of the profile.

    Returns:
        np.ndarray: The computed height-height correlation function values.
    """

    sigma2 = beta[0]
    correlation_length = beta[1]
    hurst_coefficient = beta[2]

    y = 2 * sigma2 * (np.exp(-(x / correlation_length) ** (2 * hurst_coefficient)))

    return y
