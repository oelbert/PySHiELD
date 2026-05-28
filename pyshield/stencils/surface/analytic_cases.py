import numpy as np
from ndsl import constants
from ndsl import QuantityFactory
from ndsl.grid import GridData
from ndsl.typing import Communicator
from .sfc_state import SurfaceState


NHALO = constants.N_HALO_DEFAULT

def set_aquaplanet_surface(
    tsea: np.ndarray,
    islmsk: np.ndarray,
    sfcemis: np.ndarray,
    gridlat: np.ndarray,
    sst_profile: int,
    tmax: float,
    tmin: float,
):
    """
    Sets sea surface temperature according the selected profile:
        0: constant sst
        1: cosine profile in SHiELD
        2: from equation (1) of Neale and Hoskins
    """
    islmsk[:] = 0
    sfcemis[:] = 0.97
    if sst_profile == 0:
        tsea[:] = tmax
    elif sst_profile == 1:
        tsea[:] = tmin + (tmax - tmin) * np.cos(gridlat[:])
    elif sst_profile == 2:
        tsea[:] = tmin
        select = gridlat >= (-constants.PI / 3.0) and gridlat <= (constants.PI / 3.0)
        tsea[select] = tmax - ((tmax - tmin) * np.sin(3 * gridlat[select] / 2) ** 2)
    else:
        raise NotImplementedError(f"sst profile {sst_profile} not implemented")

def init_aquaplanet(
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    sst_profile: int,
    tmax: float,
    tmin: float,
    comm: Communicator,
) -> SurfaceState:
    state = SurfaceState.init_zeros(quantity_factory)
    set_aquaplanet_surface(
        state.tsfc[:],
        state.islmsk[:],
        state.sfcemis[:],
        grid_data.lat[:],
        sst_profile,
        tmax,
        tmin,
    )

    comm.halo_update(state.tsfc, n_points=NHALO)
    comm.halo_update(state.islmsk, n_points=NHALO)
    comm.halo_update(state.sfcemis, n_points=NHALO)

    return state
