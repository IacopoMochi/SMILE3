import numpy as np
from app.models.image_container import Image


def gather_parameters(window, image: Image) -> None:
    """
    Gathers parameters from the GUI and assigns them to the image object.

    Args:
        window: The GUI window object containing input fields.
        image (Image): The image object to which the parameters will be assigned.
    """

    edge_fit_function = get_fit_function(window)

    parameters = {'Threshold': np.double(window.threshold_line_edit.text()),
                  'MinPeakDistance': np.double(window.minPeakDistance_line_edit.text()),
                  'MinPeakProminence': np.double(window.minPeakProminence_line_edit.text()),
                  'PixelSize': np.double(window.pixelSize_line_edit.text()),
                  'X1': np.double(window.X1.text()),
                  'X2': np.double(window.X2.text()),
                  'Y1': np.double(window.Y1.text()),
                  'Y2': np.double(window.Y2.text()),
                  'tone_positive_radiobutton': window.tone_positive_radiobutton.isChecked(),
                  'brightEdge': window.brightEdge.isChecked(),
                  'Edge_fit_function': edge_fit_function,
                  'CDFraction': np.double(window.CDFraction.text()),
                  'EdgeRange': np.double(window.EdgeRange.text()),
                  'EdgeSearchMethodRange': window.edge_search_range.isChecked(),
                  'EdgeSearchMethodCDFraction': window.edge_search_CD_fraction.isChecked(),
                  'High_frequency_cut': int(window.high_freq_cut.text()),
                  'Low_frequency_cut': int(window.low_freq_cut.text()),
                  'Low_frequency_average': int(window.low_freq_average.text()),
                  'High_frequency_average': int(window.high_freq_average.text()),
                  'Correlation_length': np.double(window.correlation_length.text()),
                  'Alpha': np.double(window.alpha.text()),
                  'PSD_model': window.PSD_model.currentText(),
                  }
    image.parameters = parameters


def get_fit_function(window) -> str:
    """
    Determines the edge fit function based on the GUI settings.

    Args:
        window: The GUI window object containing input fields.

    Returns:
        str: The selected edge fit function.
    """

    if window.Polynomial.isChecked():
        edge_fit_function = 'polynomial'
    elif window.Linear.isChecked():
        edge_fit_function = 'linear'
    elif window.ThresholdEdge.isChecked():
        edge_fit_function = 'threshold'
    else:
        edge_fit_function = 'bright_edge'
    return edge_fit_function
