import numpy as np
from typing import Tuple, Any

def _poly11(M: Tuple[float, float], *args: float) -> np.ndarray:
    """
    Computes a polynomial of degree 1 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        *args (float): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """

    x, y = M
    return np.array(args[0] * x + args[1] * y + args[2])


def _poly22(M: Tuple[float, float], *args: float) -> np.ndarray:
    """
    Computes a polynomial of degree 2 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        *args (float): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """

    x, y = M
    return np.array(
        args[0] * x
        + args[1] * y
        + args[2]
        + args[3] * x ** 2
        + args[4] * x * y
        + args[5] * y ** 2
    )


def _poly33(M: Tuple[float, float], *args: float) -> np.ndarray:
    """
    Computes a polynomial of degree 3 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        *args (float): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """
    x, y = M
    return np.array(
        args[0] * x
        + args[1] * y
        + args[2]
        + args[3] * x ** 2
        + args[4] * x * y
        + args[5] * y ** 2
        + args[6] * x ** 3
        + args[7] * y * x ** 2
        + args[8] * x * y ** 2
        + args[9] * y ** 3
    )


def poly11(M: Tuple[float, float], args: Tuple[float, float, float]) -> np.ndarray:
    """
    Computes a polynomial of degree 1 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        args (Tuple[float, float, float]): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """
    x, y = M
    return np.array(args[0] * x + args[1] * y + args[2])


def poly22(M: Tuple[float, float], args: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Computes a polynomial of degree 2 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        args (Tuple[float, float, float, float, float, float]): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """
    x, y = M
    return np.array(
        args[0] * x
        + args[1] * y
        + args[2]
        + args[3] * x ** 2
        + args[4] * x * y
        + args[5] * y ** 2
    )


def poly33(M: Tuple[float, float], args: Tuple[float, float, float, float, float, float, float, float, float, float]) -> np.ndarray:
    """
    Computes a polynomial of degree 3 in two variables (x and y).

    Args:
        M (Tuple[float, float]): A tuple containing the x and y coordinates.
        args (Tuple[float, float, float, float, float, float, float, float, float, float]): Coefficients for the polynomial.

    Returns:
        np.ndarray: The computed polynomial value.
    """
    x, y = M
    return np.array(
        args[0] * x
        + args[1] * y
        + args[2]
        + args[3] * x ** 2
        + args[4] * x * y
        + args[5] * y ** 2
        + args[6] * x ** 3
        + args[7] * y * x ** 2
        + args[8] * x * y ** 2
        + args[9] * y ** 3
    )


def binary_image_histogram_model(x: float, *beta: float) -> np.ndarray:
    """
    Computes a binary image histogram model using multiple Gaussian profiles.

    Args:
        x (float): The variable for the histogram model.
        *beta (float): Coefficients for the Gaussian profiles.

    Returns:
        np.ndarray: The computed histogram model value.
    """
    return np.array(
        beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
        + beta[3] * np.exp(-(((x - beta[4]) / beta[5]) ** 2))
        + beta[6] * np.exp(-(((x - beta[7]) / beta[8]) ** 2))
    )


def gaussian_profile(x: float, *beta: float) -> np.ndarray:
    """
    Computes a Gaussian profile.

    Args:
        x (float): The variable for the Gaussian profile.
        *beta (float): Coefficients for the Gaussian profile.

    Returns:
        np.ndarray: The computed Gaussian profile value.
    """
    return np.array(
        beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
    )
