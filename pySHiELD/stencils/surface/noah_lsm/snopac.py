from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval, log

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    FloatFieldK,
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from pySHiELD._config import SurfaceConfig
from pySHiELD.functions.physics_functions import fpvs


class SNOPAC:
    def __init__(self):
        pass

    def __call__(self):
        pass
