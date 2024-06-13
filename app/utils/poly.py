import numpy as np
from typing import Tuple, Any

def _poly11(M: Tuple[float, float], *args: float) -> np.ndarray:
    x, y = M
    return np.array(args[0] * x + args[1] * y + args[2])


def _poly22(M: Tuple[float, float], *args: float) -> np.ndarray:
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
    x, y = M
    return np.array(args[0] * x + args[1] * y + args[2])


def poly22(M: Tuple[float, float], args: Tuple[float, float, float, float, float, float]) -> np.ndarray:
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
    return np.array(
        beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
        + beta[3] * np.exp(-(((x - beta[4]) / beta[5]) ** 2))
        + beta[6] * np.exp(-(((x - beta[7]) / beta[8]) ** 2))
    )


def gaussian_profile(x: float, *beta: float) -> np.ndarray:
    return np.array(
        beta[0] * np.exp(-(((x - beta[1]) / beta[2]) ** 2))
    )
