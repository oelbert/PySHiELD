from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval

import pySHiELD.constants as physcons
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    FloatFieldK,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from ndsl.stencils.basic_operations import average_in
from pySHiELD.stencils.surface.noah_lsm.sstep import SoilCanopyMoisture


@gtscript.function
def wdfcnd_fn(smc, smcmax, bexp, dksat, dwsat, sicemax):
    # calc the ratio of the actual to the max psbl soil h2o content of each layer
    factr = min(1.0, max(0.0, 0.2 / smcmax))
    factr0 = min(1.0, max(0.0, smc / smcmax))

    # prep an expntl coef and calc the soil water diffusivity
    expon = bexp + 2.0
    wdf = dwsat * factr0 ** expon

    # frozen soil hydraulic diffusivity.
    if sicemax > 0.0:
        vkwgt = 1.0 / (1.0 + (500.0 * sicemax) ** 3.0)
        wdf = vkwgt * wdf + (1.0 - vkwgt) * dwsat * factr ** expon

    # reset the expntl coef and calc the hydraulic conductivity
    expon = (2.0 * bexp) + 3.0
    wcnd = dksat * factr0 ** expon

    return wdf, wcnd


def start_smflx(
    smcmax: FloatFieldIJ,
    prcp1: FloatFieldIJ,
    zsoil: FloatFieldK,
    shdfac: FloatFieldIJ,
    ec1: FloatFieldIJ,
    cmc: FloatFieldIJ,
    sh2o: FloatField,
    smc: FloatField,
    sh2o_in: FloatField,
    sice: FloatField,
    surface_mask: BoolFieldIJ,
    frozen_ground: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(FORWARD), interval(0, 1):
        if surface_mask:
            # compute the right hand side of the canopy eqn term
            rhsct = shdfac * prcp1 - ec1

            drip = 0.0
            trhsct = dt * rhsct
            excess = cmc + trhsct

            if excess > physcons.CMCMAX:
                drip = excess - physcons.CMCMAX

            # pcpdrp is the combined prcp1 and drip (from cmc) that goes into the soil
            pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt

            frozen_ground = (pcpdrp * dt) > (0.001 * 1000.0 * (-zsoil) * smcmax)

    with computation(PARALLEL), interval(...):
        if surface_mask:
            # store ice content at each soil layer before calling srt and sstep
            sice = smc - sh2o
            sh2o_in = sh2o


def srt(
    edir: FloatFieldIJ,
    et: FloatField,
    sh2o: FloatField,
    pcpdrp: FloatFieldIJ,
    zsoil: FloatFieldK,
    dwsat: FloatFieldIJ,
    dksat: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    bexp: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    slope: FloatFieldIJ,
    kdt: FloatFieldIJ,
    frzx: FloatFieldIJ,
    sice: FloatField,
    rhstt: FloatField,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
    surface_mask: BoolFieldIJ,
):
    """
    ! ===================================================================== !
    !  description:                                                         !
    !    subroutine srt calculates the right hand side of the time tendency !
    !    term of the soil water diffusion equation.  also to compute        !
    !    ( prepare ) the matrix coefficients for the tri-diagonal matrix    !
    !    of the implicit time scheme.                                       !
    !                                                                       !
    !  subprogram called:  wdfcnd                                           !
    !                                                                       !
    !                                                                       !
    !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs:                                                       size   !
    !     nsoil    - integer, number of soil layers                    1    !
    !     edir     - real, direct soil evaporation                     1    !
    !     et       - real, plant transpiration                       nsoil  !
    !     sh2o     - real, unfrozen soil moisture                    nsoil  !
    !     sh2oa    - real,                                           nsoil  !
    !     pcpdrp   - real, combined prcp and drip                      1    !
    !     zsoil    - real, soil layer depth below ground             nsoil  !
    !     dwsat    - real, saturated soil diffusivity                  1    !
    !     dksat    - real, saturated soil hydraulic conductivity       1    !
    !     smcmax   - real, porosity                                    1    !
    !     bexp     - real, soil type "b" parameter                     1    !
    !     dt       - real, time step                                   1    !
    !     smcwlt   - real, wilting point                               1    !
    !     slope    - real, linear reservoir coefficient                1    !
    !     kdt      - real,                                             1    !
    !     frzx     - real, frozen ground parameter                     1    !
    !     sice     - real, ice content at each soil layer            nsoil  !
    !                                                                       !
    !  outputs:                                                             !
    !     rhstt    - real, soil water time tendency                  nsoil  !
    !     runoff1  - real, surface runoff not infiltrating sfc         1    !
    !     runoff2  - real, sub surface runoff (baseflow)               1    !
    !     ai       - real, matrix coefficients                       nsold  !
    !     bi       - real, matrix coefficients                       nsold  !
    !     ci       - real, matrix coefficients                       nsold  !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """
    from __externals__ import dt

    with computation(FORWARD), interval(0, 1):
        if surface_mask:
            # determine rainfall infiltration rate and runoff
            cvfrz = 3
            pddum = pcpdrp
            runoff1 = 0.0
            sicemax = 0.0

    with computation(FORWARD), interval(...):
        if surface_mask:
            sicemax = max(sice, sicemax)

    with computation(FORWARD):
        with interval(0, 1):
            if surface_mask:
                if pcpdrp != 0:
                    # frozen ground version
                    dt1 = dt / 86400.0
                    smcav = smcmax - smcwlt
                    dd = -zsoil * (smcav - (sh2o + sice - smcwlt))
                    dice = -zsoil * sice
        with interval(1, None):
            if surface_mask:
                if pcpdrp != 0:
                    dd += (zsoil[-1] - zsoil) * (smcav - (sh2o + sice - smcwlt))

                    dice += (zsoil[-1] - zsoil) * sice

    with computation(FORWARD):
        with interval(0, 1):
            if surface_mask:
                if pcpdrp != 0:
                    val = 1.0 - exp(-kdt * dt1)
                    ddt = dd * val

                    px = pcpdrp * dt

                    if px < 0.0:
                        px = 0.0

                    infmax = (px * (ddt / (px + ddt))) / dt

                    # reduction of infiltration based on frozen ground parameters
                    fcr = 1.0

                    if dice > 1.0e-2:
                        acrt = cvfrz * frzx / dice
                        ialp1 = cvfrz - 1  # = 2

                        # Hardcode for ialp1 = 2
                        sum = 1.0 + acrt ** 2.0 / 2.0 + acrt

                        fcr = 1.0 - exp(-acrt) * sum

                    infmax *= fcr

                    wdf0, wcnd0 = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

                    infmax = max(infmax, wcnd0)
                    infmax = min(infmax, px)

                    if pcpdrp > infmax:
                        runoff1 = pcpdrp - infmax
                        pddum = infmax

                wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)

                # calc the matrix coefficients ai, bi, and ci for the top layer
                ddz = 1.0 / (-0.5 * zsoil[1])
                ai0 = 0.0
                bi0 = wdf0 * ddz / (-zsoil)
                ci0 = -bi0

                # calc rhstt for the top layer
                dsmdz = (sh2o - sh2o[0, 0, 1]) / (-0.5 * zsoil[1])
                rhstt = (wdf0 * dsmdz + wcnd0 - pddum + edir + et) / zsoil

    with computation(FORWARD), interval(1, -1):
        if surface_mask:
            # 2. Interior Layers
            wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)
            denom2 = zsoil[-1] - zsoil
            denom = zsoil[-1] - zsoil[1]
            dsmdz = (sh2o - sh2o[0, 0, 1]) / (denom * 0.5)
            ddz = 2.0 / denom
            ci = -wdf * ddz / denom2
            slopx = 1.0
            numer = (
                (wdf * dsmdz)
                + slopx * wcnd
                - wdf[0, 0, -1] * dsmdz[0, 0, -1]
                - wcnd[0, 0, -1]
                + et
            )
            rhstt = -numer / denom2

            # calc matrix coefs
            ai = -wdf * ddz[0, 0, -1] / denom2
            bi = -(ai + ci)

    with computation(FORWARD), interval(-1, None):
        if surface_mask:
            # 3. Bottom Layer
            denom2 = zsoil[-1] - zsoil
            slopx = slope
            wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)
            dsmdz = 0.0
            ci = 0.0
            numer = (
                (wdf * dsmdz)
                + slopx * wcnd
                - wdf[0, 0, -1] * dsmdz[0, 0, -1]
                - wcnd[0, 0, -1]
                + et
            )
            rhstt = -numer / denom2

            # calc matrix coefs
            ai = -wdf * ddz / denom2
            bi = -(ai + ci)

            runoff2 = slope * wcnd


class SoilMoistureFlux:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt: Float,
    ):
        """
        Fortran name is smflx
        """

        grid_indexing = stencil_factory.grid_indexing

        def make_quantity() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Float,
            )

        self._frozen_ground = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._mid_sh20 = make_quantity()
        self._ai = make_quantity()
        self._bi = make_quantity()
        self._ci = make_quantity()
        self._rhstt = make_quantity()
        self._sice = make_quantity()
        self._pcpdrp = make_quantity_2d()
        self._rhsct = make_quantity_2d()
        self._trhsct = make_quantity_2d()

        self._start_smflx = stencil_factory.from_origin_domain(
            func=start_smflx,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._average_in = stencil_factory.from_origin_domain(
            func=average_in,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._srt = stencil_factory.from_origin_domain(
            func=srt,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._sstep = SoilCanopyMoisture(stencil_factory, quantity_factory, dt)

    def __call__(
        self,
        kdt: FloatFieldIJ,
        smcmax: FloatFieldIJ,
        smcwlt: FloatFieldIJ,
        prcp1: FloatFieldIJ,
        zsoil: FloatFieldK,
        slope: FloatFieldIJ,
        frzx: FloatFieldIJ,
        bexp: FloatFieldIJ,
        dksat: FloatFieldIJ,
        dwsat: FloatFieldIJ,
        shdfac: FloatFieldIJ,
        edir1: FloatFieldIJ,
        ec1: FloatFieldIJ,
        et1: FloatField,
        cmc: FloatFieldIJ,
        sh2o: FloatField,
        smc: FloatField,
        runoff1: FloatFieldIJ,
        runoff2: FloatFieldIJ,
        runoff3: FloatFieldIJ,
        drip: FloatFieldIJ,
        surface_mask: BoolFieldIJ,
    ):
        """
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  subroutine smflx calculates soil moisture flux.  the soil moisture   !
        !  content (smc - a per unit volume measurement) is a dependent variable!
        !  that is updated with prognostic eqns. the canopy moisture content    !
        !  (cmc) is also updated. frozen ground version:  new states added: sh2o!
        !  and frozen ground correction factor, frzx and parameter slope.       !
        !                                                                       !
        !                                                                       !
        !  subprogram called:  srt, sstep                                       !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs:                                                       size   !
        !     nsoil    - integer, number of soil layers                    1    !
        !     dt       - real, time step                                   1    !
        !     kdt      - real,                                             1    !
        !     smcmax   - real, porosity                                    1    !
        !     smcwlt   - real, wilting point                               1    !
        !     cmcmax   - real, maximum canopy water parameters             1    !
        !     prcp1    - real, effective precip                            1    !
        !     zsoil    - real, soil layer depth below ground (negative)  nsoil  !
        !     slope    - real, linear reservoir coefficient                1    !
        !     frzx     - real, frozen ground parameter                     1    !
        !     bexp     - real, soil type "b" parameter                     1    !
        !     dksat    - real, saturated soil hydraulic conductivity       1    !
        !     dwsat    - real, saturated soil diffusivity                  1    !
        !     shdfac   - real, aeral coverage of green veg                 1    !
        !     edir1    - real, direct soil evaporation                     1    !
        !     ec1      - real, canopy water evaporation                    1    !
        !     et1      - real, plant transpiration                       nsoil  !
        !                                                                       !
        !  input/outputs:                                                       !
        !     cmc      - real, canopy moisture content                     1    !
        !     sh2o     - real, unfrozen soil moisture                    nsoil  !
        !                                                                       !
        !  outputs:                                                             !
        !     smc      - real, total soil moisture                       nsoil  !
        !     runoff1  - real, surface runoff not infiltrating sfc         1    !
        !     runoff2  - real, sub surface runoff (baseflow)               1    !
        !     runoff3  - real, excess of porosity                          1    !
        !     drip     - real, through-fall of precip and/or dew           1    !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """
        self._start_smflx(
            smcmax,
            prcp1,
            zsoil,
            shdfac,
            ec1,
            cmc,
            sh2o,
            smc,
            self._mid_sh20,
            self._sice,
            surface_mask,
            self._frozen_ground,
        )

        self._srt(
            edir1,
            et1,
            sh2o,
            self._pcpdrp,
            zsoil,
            dwsat,
            dksat,
            smcmax,
            bexp,
            smcwlt,
            slope,
            kdt,
            frzx,
            self._sice,
            self._rhstt,
            runoff1,
            runoff2,
            self._ai,
            self._bi,
            self._ci,
            self._frozen_ground,
        )

        self._sstep(
            sh2o,
            self._rhsct,
            smcmax,
            zsoil,
            self._sice,
            cmc,
            self._rhstt,
            self._ai,
            self._bi,
            self._ci,
            runoff3,
            smc,
            self._frozen_ground,
        )

        self._average_in(sh2o, self._mid_sh20)

        self._srt(
            edir1,
            et1,
            sh2o,
            self._pcpdrp,
            zsoil,
            dwsat,
            dksat,
            smcmax,
            bexp,
            smcwlt,
            slope,
            kdt,
            frzx,
            self._sice,
            self._rhstt,
            runoff1,
            runoff2,
            self._ai,
            self._bi,
            self._ci,
            surface_mask,
        )

        self._sstep(
            sh2o,
            self._rhsct,
            smcmax,
            zsoil,
            self._sice,
            cmc,
            self._rhstt,
            self._ai,
            self._bi,
            self._ci,
            runoff3,
            smc,
            surface_mask,
        )
