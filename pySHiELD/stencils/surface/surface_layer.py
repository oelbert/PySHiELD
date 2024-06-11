from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    sqrt,
)

import ndsl.constants as constants
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.grid import GridData
from ndsl.initialization.allocator import QuantityFactory
from pySHiELD._config import COND_DIM, SFC_CONFIG, TRACER_DIM
from pySHiELD.functions.physics_functions import fpvs
from pySHiELD.stencils.surface.sfc_diff import SurfaceExchange


class SurfaceLayers:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SFC_CONFIG,
    ):

        self._exchange = SurfaceExchange(
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
        )
        pass

    def __call__(self):
        for iter in range(2):
            self._exchange()
        pass
