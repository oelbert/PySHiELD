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
from ndsl.quantity import Quantity
from pySHiELD._config import COND_DIM, TRACER_DIM, SurfaceConfig
from pySHiELD.functions.physics_functions import fpvs
from pySHiELD.stencils.surface.sfc_diff import SurfaceExchange
from pySHiELD.stencils.surface.sfc_state import SurfaceState


def init_step_vars(
    tsfc: FloatFieldIJ,
    phil: FloatField,
    tsurf: FloatFieldIJ,
    flag_guess: BoolFieldIJ,
    flag_iter: BoolFieldIJ,
    drain: FloatFieldIJ,
    ep1d: FloatFieldIJ,
    runof: FloatFieldIJ,
    hflx: FloatFieldIJ,
    evap: FloatFieldIJ,
    evbs: FloatFieldIJ,
    evcw: FloatFieldIJ,
    trans: FloatFieldIJ,
    sbsno: FloatFieldIJ,
    snowc: FloatFieldIJ,
    snohf: FloatFieldIJ,
    qss: FloatFieldIJ,
    gflx: FloatFieldIJ,
    zlvl: FloatFieldIJ,
    smcwlt2: FloatFieldIJ,
    smcref2: FloatFieldIJ,
):
    with computation(PARALLEL), interval(-1, None):
        tsurf = tsfc
        flag_guess = False
        flag_iter = True
        drain = 0.0
        ep1d = 0.0
        runof = 0.0
        hflx = 0.0
        evap = 0.0
        evbs = 0.0
        evcw = 0.0
        trans = 0.0
        sbsno = 0.0
        snowc = 0.0
        snohf = 0.0
        qss = 0.0
        gflx = 0.0
        zlvl = phil * constants.RGRAV
        smcwlt2 = 0.0
        smcref2 = 0.0


def update_guess(
    wind: FloatFieldIJ,
    iter: Int,
    flag_guess: BoolFieldIJ,
):
    with computation(PARALLEL), interval(...):
        if (iter == 1) and (wind < 2.0):
            flag_guess[0, 0] = True


class SurfaceLayer:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SurfaceConfig,
    ):
        grid_indexing = stencil_factory.grid_indexing

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Float,
            )

        self._cdq = make_quantity_2d()

        self._flag_guess = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Bool,
        )

        self._exchange = SurfaceExchange(stencil_factory=stencil_factory, config=config)
        self._update_guess = stencil_factory.from_origin_domain(
            update_guess,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass

    def __call__(self, state: SurfaceState):
        for iter in range(2):
            self._exchange()
            self._update_guess(state.wind, iter, self._flag_guess)
        pass
