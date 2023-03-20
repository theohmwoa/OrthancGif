import numpy as np


def set_contrast(array: np.ndarray, min_percentile: int = 1, max_percentile: int = 99) -> np.ndarray:
    """Set dynamic range of value of the numpy array, by default filter extreme value 1% lower and upper value

    Args:
        array (np.ndarray): numpy arry image
        min_percentile (int, optional): min percentile value as minimum value. Defaults to 1.
        max_percentile (int, optional): max percentile value as maximum value. Defaults to 99.

    Returns:
        np.ndarray: _description_
    """
    max = np.percentile(array, max_percentile)
    min = np.percentile(array, min_percentile)
    normalized = ((array - min) / (max - min)) * 255
    return normalized
