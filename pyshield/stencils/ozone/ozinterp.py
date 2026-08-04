import numpy as np
import xarray as xr
from ndsl.logging import ndsl_log
import datetime as dt

def read_o3_data(
    ntoz: int, o3_data: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads ozone data from a NetCDF file and returns it as an xarray DataArray.

    Args:
        ntoz (int): The number of ozone tracer species
        o3_data (string): The path to the NetCDF file containing ozone data.
    """
    if ntoz <= 0:
        raise NotImplementedError("Diagnostic ozone (ntoz <= 0) is not supported")

    ds = xr.open_dataset(o3_data)
    time = ds['time'].values
    oz_coeff = ds['ozcoeff'].values
    oz_pres = np.log(100 * ds['lev'].values)
    levozp = len(ds['lev'].values)
    ozlat = ds['lat'].values
    ozplin = ds['ozplin'].values
    ds.close()

    ndsl_log.info(f"Read ozone data from {o3_data}")
    ndsl_log.info(f"oz_coeff: {(oz_coeff)}")
    ndsl_log.info(f"latsozp: {len(ozlat)}")
    ndsl_log.info(f"levozp: {levozp}")
    ndsl_log.info(f"timeoz: {len(time)}")

    return levozp, oz_coeff, ozlat, oz_pres, time, ozplin

def setindexoz(dlat: np.ndarray, oz_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jindx1 = np.zeros(dlat.shape, dtype=int)
    jindx2 = np.zeros(dlat.shape, dtype=int)
    ddy = np.zeros(dlat.shape, dtype=float)
    for j, k in np.ndindex(dlat.shape):
        jindx2[j, k] = len(oz_lat)
        for i in range(len(oz_lat)):
            if dlat[j, k] < oz_lat[i]:
                jindx2[j, k] = i
                break
        jindx1[j, k] = max(jindx2[j, k]-1, 0)
        jindx2[j, k] = min(jindx2[j, k], len(oz_lat))
        if jindx2[j, k] != jindx1[j, k]:
            ddy[j, k] = (dlat[j, k] - oz_lat[jindx1[j, k]]) / (
                oz_lat[jindx2[j, k]] - oz_lat[jindx1[j, k]]
            )
        else:
            ddy[j, k] = 1.0
    return jindx1, jindx2, ddy


def ozinterpolate(
    model_time: dt.datetime,
    jindx1: np.ndarray,
    jindx2: np.ndarray,
    ddy: np.ndarray,
    oz_time: np.ndarray,
    oz_coeff: int,
    levozp: int,
    ozplin: np.ndarray
) -> np.ndarray:
    """
    Interpolate ozone onto model levels at a given time in-place onto ozplout
    """
    jdoy = int(model_time.strftime("%j"))  # day of year 1-365
    rjday = jdoy + (model_time.hour / 24.)
    if rjday < oz_time[0]:
        rjday += 365.

    n2 = len(oz_time) - 1
    for j in range(1, len(oz_time) - 1):
        if rjday < oz_time[j]:
            n2 = j
            break
    n1 = n2 - 1
    tx1 = (oz_time[n2] - rjday) / (oz_time[n2] - oz_time[n1])
    tx2 = 1.0 - tx1
    ozplout = np.zeros((ddy.shape[0], ddy.shape[1], levozp, oz_coeff))
    for nc in range(oz_coeff):
        for ll in range(levozp):
            for j, k in np.ndindex(ddy.shape):
                j1 = jindx1[j, k]
                j2 = jindx2[j, k]
                tem = 1.0 - ddy[j, k]
                ozplout[j, k, ll, nc] = tx1 * (
                    tem * ozplin[n1, nc, ll, j1] + ddy[j, k] * ozplin[n1, nc, ll, j2]
                ) + tx2 * (
                    tem * ozplin[n2, nc, ll, j1] + ddy[j, k] * ozplin[n2, nc, ll, j2]
                )
    return ozplout
