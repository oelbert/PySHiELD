from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, interval

from ndsl.constants import X_DIM, Y_DIM, Z_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    FloatFieldK,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.stencils.tridiag import tridiag_solve
import pySHiELD.constants as physcons

def init_sstep(
    rhstt: FloatField,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
    surface_mask: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(PARALLEL), interval(...):
        if surface_mask:
            rhstt *= dt
            ai *= dt
            bi = 1. + (bi * dt)
            ci *= dt

def finish_sstep(
    sh2o: FloatField,
    rhsct: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    zsoil: FloatFieldK,
    sice: FloatFieldIJ,
    cmc: FloatFieldIJ,
    p: FloatField,
    runoff3: FloatFieldIJ,
    smc: FloatField,
    surface_mask: BoolFieldIJ,
):
    from __externals__ import dt
    # sum the previous smc value and the matrix solution
    with computation(FORWARD):
        with interval(0, 1):
            if surface_mask:
                runoff3 = 0.0
                runoff3 = 0.0
                ddz = -zsoil
        with interval(1, None):
            if surface_mask:
                ddz = zsoil[0, 0, -1] - zsoil

    with computation(FORWARD), interval(...):
        if surface_mask:
            sh2o = sh2o + p + runoff3 / ddz
            stot = sh2o + sice
            if stot > smcmax:
                runoff3 = (stot - smcmax) * ddz
            else:
                runoff3 = 0.0

            smc = max(min(stot, smcmax), 0.02)
            sh2o = max(smc - sice, 0.0)
            runoff3 = 0.0

            # update canopy water content/interception
            cmc += dt * rhsct
            if cmc < 1.0e-20:
                cmc = 0.0
            cmc = min(cmc, physcons.CMCMAX)


class SoilCanopyMoisture:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt: Float,
    ):
        """
        Fortran name is sstep
        """
        grid_indexing = stencil_factory.grid_indexing

        self._delta = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )

        self._p = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )

        self._init_sstep = stencil_factory.from_origin_domain(
            func=init_sstep,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._tridiag = stencil_factory.from_origin_domain(
            func=tridiag_solve,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._finish_sstep = stencil_factory.from_origin_domain(
            func=finish_sstep,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        sh2o: FloatField,
        rhsct: FloatFieldIJ,
        smcmax: FloatFieldIJ,
        zsoil: FloatFieldK,
        sice: FloatField,
        cmc: FloatFieldIJ,
        rhstt: FloatField,
        ai: FloatField,
        bi: FloatField,
        ci: FloatField,
        runoff3: FloatFieldIJ,
        smc: FloatField,
        surface_mask: BoolFieldIJ
    ):
        """
        Original Fortran description:
        ! ===================================================================== !
        !  description:                                                         !
        !    subroutine sstep calculates/updates soil moisture content values   !
        !    and canopy moisture content values.                                !
        !                                                                       !
        !  subprogram called:  rosr12                                           !
        !                                                                       !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs:                                                       size   !
        !     nsoil    - integer, number of soil layers                    1    !
        !     sh2oin   - real, unfrozen soil moisture                    nsoil  !
        !     rhsct    - real,                                             1    !
        !     dt       - real, time step                                   1    !
        !     smcmax   - real, porosity                                    1    !
        !     cmcmax   - real, maximum canopy water parameters             1    !
        !     zsoil    - real, soil layer depth below ground             nsoil  !
        !     sice     - real, ice content at each soil layer            nsoil  !
        !                                                                       !
        !  input/outputs:                                                       !
        !     cmc      - real, canopy moisture content                     1    !
        !     rhstt    - real, soil water time tendency                  nsoil  !
        !     ai       - real, matrix coefficients                       nsold  !
        !     bi       - real, matrix coefficients                       nsold  !
        !     ci       - real, matrix coefficients                       nsold  !
        !                                                                       !
        !  outputs:                                                             !
        !     sh2oout  - real, updated soil moisture content             nsoil  !
        !     runoff3  - real, excess of porosity                          1    !
        !     smc      - real, total soil moisture                       nsoil  !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """
        self._init_sstep(
            rhstt,
            ai,
            bi,
            ci,
        )

        self._tridiag(
            ai,
            bi,
            ci,
            rhstt,
            self._p,
            self._delta,
        )

        self._finish_sstep(
            sh2o,
            rhsct,
            smcmax,
            zsoil,
            sice,
            cmc,
            self._p,
            runoff3,
            smc,
        )
