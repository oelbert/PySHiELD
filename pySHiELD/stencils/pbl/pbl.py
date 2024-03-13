import numpy as np
import physical_functions as physfun
from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    interval,
    sqrt,
)

import ndsl.constants as constants
import pyFV3.stencils.basic_operations as basic
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import GridIndexing, StencilFactory
from ndsl.dsl.typing import Float, FloatField, FloatFieldIJ
from ndsl.grid import GridData
from ndsl.initialization.allocator import QuantityFactory
from ndsl.performance.timer import Timer

from ..._config import PBLConfig


class PBL:
    """
    Planetary Boundary Layer scheme to calculate vertical diffusion
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_data: GridData,
        config: PBLConfig,
        dt_atmos: float,
    ):
        # Allocate variables, compile stencils
        pass

    def __call__(self, args):
        # Execute stencils
        pass
