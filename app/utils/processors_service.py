import numpy as np

def edge_consolidation(raw_edge_profiles: np.ndarray) -> np.ndarray:
    """
    Consolidates raw edge profiles.
    """

    consolidated_edge_profiles = raw_edge_profiles.copy()
    for i, edge in enumerate(consolidated_edge_profiles):
        mean_value = np.nanmean(edge)
        consolidated_edge_profiles[i] = np.where(np.isnan(edge), mean_value, edge)

    return consolidated_edge_profiles


def edge_mean_subtraction(absolute_edge_profiles: np.ndarray) -> np.ndarray:
    """
    Subtracts the mean value from edge profiles to center them around zero.
    """

    zero_mean_edge_profiles = absolute_edge_profiles.copy()
    for edge in zero_mean_edge_profiles:
        mean_value = np.nanmean(edge)
        edge[:] = edge - mean_value
    return zero_mean_edge_profiles
