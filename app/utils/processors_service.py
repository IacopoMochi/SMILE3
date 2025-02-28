import numpy as np
from scipy import interpolate


def edge_consolidation(raw_edge_profiles: np.ndarray, method: str) -> [np.ndarray, np.ndarray]:
    """
    Consolidates raw edge profiles.
    """

    consolidated_edge_profiles = raw_edge_profiles.copy()
    consolidation = np.zeros(np.shape(consolidated_edge_profiles)) * np.nan

    if method == "average":
        for i, edge in enumerate(consolidated_edge_profiles):
            mean_value = np.nanmean(edge)
            new_edge = edge.copy()
            consolidated_edge_profiles[i] = np.where(np.isnan(new_edge), mean_value, new_edge)
            consolidation[i] = np.where(np.isnan(new_edge), mean_value, np.nan)
    elif method == 'interpolation':
        for i, edge in enumerate(consolidated_edge_profiles):
            new_edge = edge.copy()
            x = np.array(range(len(new_edge)), dtype=float)
            # Find indices where values are not NaN
            valid = ~np.isnan(new_edge)
            x_valid = x[valid]
            y_valid = new_edge[valid]

            interp_func = interpolate.interp1d(x_valid, y_valid, kind='linear', fill_value="extrapolate")
            consolidated_edge_profiles[i] = np.where(np.isnan(new_edge), interp_func(x), new_edge)
            consolidation[i] = np.where(np.isnan(new_edge), interp_func(x), np.nan)


    return consolidated_edge_profiles, consolidation


def edge_mean_subtraction(absolute_edge_profiles: np.ndarray) -> np.ndarray:
    """
    Subtracts the mean value from edge profiles to center them around zero.
    """

    zero_mean_edge_profiles = absolute_edge_profiles.copy()
    for edge in zero_mean_edge_profiles:
        mean_value = np.nanmean(edge)
        edge[:] = edge - mean_value
    return zero_mean_edge_profiles
