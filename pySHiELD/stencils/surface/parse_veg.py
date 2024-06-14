import numpy as np

def parse_veg_data(
    veg_data: np.ndarray,
    nroot_data: np.ndarray,
):
    nroot = np.zeros_like(veg_data)
    nroot[:, :] = nroot_data[veg_data[:, :]]  # and similar for other arrays

    return nroot
