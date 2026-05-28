from enum import Enum

from ndsl import CubedSphereCommunicator, MetaEnumStr, QuantityFactory
from ndsl.grid import GridData
from ndsl.typing import Communicator
from .sfc_state import SurfaceState
from .analytic_cases import init_aquaplanet
from pyfv3.initialization.analytic_init import AnalyticCase


def init_analytic_state(
    analytic_init_case: AnalyticCase,
    grid_data: GridData,
    quantity_factory: QuantityFactory,
    sst_profile: int,
    tsea_max: float,
    tsea_min: float,
    comm: Communicator,
) -> SurfaceState:
    """
    This method initializes the surface state for a given analytic state
    Args:
        analytic_init_case:     test case specifier
        grid_data:              current selected grid data values
        quantity_factory:       inclusion of QuantityFactory class
        sst_profile:            Which profile to use for sea surface temperatures
        tsea_max:               Maximum sea surface temperature
        tsea_min:               Minimum sea surface temperature
        comm:                   inclusion of CubedSphereCommunicator class

    Returns:
        an instance of SurfaceState class
    """
    spherical_cases = [
        AnalyticCase.baroclinic_instability,
        AnalyticCase.baroclinic_steady,
        AnalyticCase.aquaplanet,
    ]

    cartesian_cases = []

    if analytic_init_case in spherical_cases:
        # TODO: Consider CubedSphereCommunicator check within individual init_*() calls
        if not isinstance(comm, CubedSphereCommunicator):
            raise TypeError(
                f"Expected CubedSphereCommunicator instance for 'comm', "
                f"got {type(comm).__name__} instead."
            )

        if analytic_init_case in [
            AnalyticCase.baroclinic_instability,
            AnalyticCase.baroclinic_steady,
            AnalyticCase.aquaplanet
        ]:
            return init_aquaplanet(
                grid_data=grid_data,
                quantity_factory=quantity_factory,
                sst_profile=sst_profile,
                tmax=tsea_max,
                tmin=tsea_min,
                comm=comm,
            )
        else:
            raise ValueError(f"Case {analytic_init_case} not implemented")
    elif analytic_init_case in cartesian_cases:
        raise ValueError(f"Case {analytic_init_case} not implemented")
    else:
        raise ValueError(f"Case {analytic_init_case} not recognized")
