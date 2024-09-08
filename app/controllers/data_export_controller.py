import copy
import os
from PyQt6 import QtWidgets
from app.models.images_list import ImagesList
from PyQt6.QtCore import QSettings
import pandas as pd
import numpy as np

class DataExporter:
    """
    A class to export the precessed data to a file in EXCEL or CSV format
    """
    def __init__(self, images_list: ImagesList, window):
        self.images_list = images_list
        self.window = window
        self.settings = QSettings("PSI", "SMILE3")

    def select_target_file(self):
        filename, filters = QtWidgets.QFileDialog.getSaveFileName()
        return filename

    def gather_parameter_for_exporting(self):
        parameters_of_the_first_image = self.images_list.images_list[0].parameters
        parameters = copy.copy(parameters_of_the_first_image)
        for key in parameters_of_the_first_image:
            parameters[key] = [parameters_of_the_first_image[key]]
        for processed_image in self.images_list.images_list:
            image_parameters = processed_image.parameters
            for key in image_parameters:
                if key in parameters:
                    parameters[key].append(image_parameters[key])
        return parameters

    def gather_data_for_exporting(self):
        selected = []
        image_name = []
        processed = []
        number_of_lines = []
        cd_average = []
        cd_std = []
        unbiased_LWR_fit = []
        biased_LWR = []
        unbiased_LWR = []
        standard_LWR = []
        unbiased_LER_fit = []
        biased_LER = []
        unbiased_LER = []
        standard_LER = []
        unbiased_LER_leading_fit = []
        biased_LER_leading = []
        unbiased_LER_leading = []
        standard_LER_leading = []
        unbiased_LER_trailing_fit = []
        biased_LER_trailing = []
        unbiased_LER_trailing = []
        standard_LER_trailing = []
        for processed_image in self.images_list.images_list:
            selected.append(processed_image.selected)
            processed.append(processed_image.processed)
            number_of_lines.append(processed_image.number_of_lines)
            image_name.append(processed_image.file_name)
            cd_average.append(processed_image.critical_dimension_estimate)
            cd_std.append(processed_image.critical_dimension_std_estimate)
            unbiased_LWR_fit.append(processed_image.unbiased_LWR_fit)
            unbiased_LWR.append(processed_image.unbiased_LWR)
            biased_LWR.append(processed_image.biased_LWR)
            standard_LWR.append(processed_image.standard_LWR)
            unbiased_LER_fit.append(processed_image.unbiased_LER_fit)
            unbiased_LER.append(processed_image.unbiased_LER)
            biased_LER.append(processed_image.biased_LER)
            standard_LER.append(processed_image.standard_LER)
            unbiased_LER_leading_fit.append(processed_image.unbiased_LER_Leading_fit)
            unbiased_LER_leading.append(processed_image.unbiased_LER_Leading)
            biased_LER_leading.append(processed_image.biased_LER_Leading)
            standard_LER_leading.append(processed_image.standard_LER_Leading)
            unbiased_LER_trailing_fit.append(processed_image.unbiased_LER_Trailing_fit)
            unbiased_LER_trailing.append(processed_image.unbiased_LER_Trailing)
            biased_LER_trailing.append(processed_image.biased_LER_Trailing)
            standard_LER_trailing.append(processed_image.standard_LER_Trailing)

        data = {
            "Selected": selected,
            "Processed": processed,
            "Name": image_name,
            "Number of lines": number_of_lines,
            "CD Average ": cd_average,
            "CD std ": cd_std,
            "Unbiased LWR fit": unbiased_LWR_fit,
            "Unbiased LWR": biased_LWR,
            "Biased LWR": biased_LWR,
            "Standard LWR": standard_LWR,

            "Unbiased LER fit": unbiased_LER_fit,
            "Unbiased LER": biased_LER,
            "Biased LER": biased_LER,
            "Standard LER": standard_LER,

            "Standard leading-edge LER": standard_LER_leading,
            "Unbiased leading-edge LER fit": unbiased_LER_leading_fit,
            "Unbiased leading-edge LER": biased_LER_leading,
            "Biased leading-edge LER": biased_LER_leading,

            "Standard trailing-edge LER": standard_LER_trailing,
            "Unbiased trailing-edge LER fit": unbiased_LER_trailing_fit,
            "Unbiased trailing-edge LER": biased_LER_trailing,
            "Biased trailing-edge LER": biased_LER_trailing

        }

        # load data into a DataFrame object:
        return data

    def export_data(self):

        line_metrics = pd.DataFrame(self.gather_data_for_exporting())
        parameters = pd.DataFrame(self.gather_parameter_for_exporting())

        filename = self.select_target_file()

        with pd.ExcelWriter(filename) as writer:
            line_metrics.to_excel(writer, sheet_name='Line_Metrics')
            parameters.to_excel(writer, sheet_name='Parameters')
            for processed_image in self.images_list.images_list:
                leading_edges = np.transpose(processed_image.consolidated_leading_edges)
                trailing_edges = np.transpose(processed_image.consolidated_leading_edges)
                edges = np.empty((leading_edges.shape[0], leading_edges.shape[1] + trailing_edges.shape[1]), dtype=leading_edges.dtype)
                edges[:, ::2] = leading_edges
                edges[:, 1::2] = trailing_edges
                pd.DataFrame(edges).to_excel(writer, sheet_name=processed_image.file_name + ' - Edges', startrow=1, header=False)

