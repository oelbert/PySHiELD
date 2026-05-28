from .ozinterp import read_o3_data, ozinterpolate, setindexoz
from .ozphys import ozphys, ozphys_2015

"""
read_o3_data: Reads ozone data from file
ozinterpolate: interpolates ozone data onto model levels
setindexoz: sets indexes for ozone interpolation
ozphys: Models ozone production and destruction based on input data
ozphys_2015:  Models ozone production and destruction based on input data,
    updated for more coefficients
"""

__all__ = ["read_o3_data", "ozinterpolate", "setindexoz", "ozphys", "ozphys_2015"]
