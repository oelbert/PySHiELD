from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval, log

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
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
    kdt,
    smcmax,
    smcwlt,
    cmcmax,
    prcp1,
    zsoil,
    slope,
    frzx,
    bexp,
    dksat,
    dwsat,
    shdfac,
    edir1,
    ec1,
    et1,
    cmc,
    sh2o,
    smc,
):
    from __externals__ import dt
    with computation(FORWARD), interval(0, 1):
        # compute the right hand side of the canopy eqn term
        rhsct = shdfac * prcp1 - ec1

        drip = 0.0
        trhsct = dt * rhsct
        excess = cmc + trhsct

        if excess > cmcmax:
            drip = excess - cmcmax

        # pcpdrp is the combined prcp1 and drip (from cmc) that goes into the soil
        pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt

    with computation(PARALLEL), interval(...):
        # store ice content at each soil layer before calling srt and sstep
        sice = smc - sh2o

        (
            rhstt0,
            rhstt1,
            rhstt2,
            rhstt3,
            runoff1,
            runoff2,
            ai0,
            ai1,
            ai2,
            ai3,
            bi0,
            bi1,
            bi2,
            bi3,
            ci0,
            ci1,
            ci2,
            ci3,
        ) = srt_fn(
            edir1,
            et1_0,
            et1_1,
            et1_2,
            et1_3,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            pcpdrp,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            dwsat,
            dksat,
            smcmax,
            bexp,
            dt,
            smcwlt,
            slope,
            kdt,
            frzx,
            sice0,
            sice1,
            sice2,
            sice3,
        )

        sh2o0, sh2o1, sh2o2, sh2o3, runoff3, smc0, smc1, smc2, smc3, cmc = sstep_fn(
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            rhsct,
            dt,
            smcmax,
            cmcmax,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            sice0,
            sice1,
            sice2,
            sice3,
            cmc,
            rhstt0,
            rhstt1,
            rhstt2,
            rhstt3,
            ai0,
            ai1,
            ai2,
            ai3,
            bi0,
            bi1,
            bi2,
            bi3,
            ci0,
            ci1,
            ci2,
            ci3,
        )

        # return (
        #     cmc,
        #     sh2o
        #     smc,
        #     runoff1,
        #     runoff2,
        #     runoff3,
        #     drip,
        # )


def srt(
    edir: FloatFieldIJ,
    et: FloatField,
    sh2o: FloatField,
    pcpdrp: FloatFieldIJ,
    zsoil: FloatField,
    dwsat: FloatFieldIJ,
    dksat: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    bexp: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    slope: FloatFieldIJ,
    kdt: FloatFieldIJ,
    frzx: FloatFieldIJ,
    sice : FloatField,
    rhstt: FloatField,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
    dd: FloatFieldIJ,
    dice: FloatFieldIJ,
):
    from __externals__ import dt
    with computation(FORWARD), interval(0, 1):
        # determine rainfall infiltration rate and runoff
        cvfrz = 3
        pddum = pcpdrp
        runoff1 = 0.0
        sicemax = 0.0

    with computation(FORWARD), interval(...):
        sicemax = max(sice, sicemax)

    with computation(FORWARD):
        with interval(0, 1):
            if pcpdrp != 0:
                # frozen ground version
                dt1 = dt / 86400.0
                smcav = smcmax - smcwlt
                dd = -zsoil * (smcav - (sh2o + sice - smcwlt))
                dice = -zsoil * sice
        with interval(1, None):
            if pcpdrp != 0:
                dd += (zsoil[-1] - zsoil) * (smcav - (sh2o + sice - smcwlt))

                dice += (zsoil[-1] - zsoil) * sice

    with computation(FORWARD):
        with interval(0, 1):
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
            ddz = 1.0 / (-0.5 * zsoil[0, 0, 1])
            ai0 = 0.0
            bi0 = wdf0 * ddz / (-zsoil)
            ci0 = -bi0

            # calc rhstt for the top layer
            dsmdz = (sh2o - sh2o[0, 0, 1]) / (-0.5 * zsoil[0, 0, 1])
            rhstt = (wdf0 * dsmdz + wcnd0 - pddum + edir + et) / zsoil

    with computation(FORWARD), interval(1, -1):
        # 2. Interior Layers
        wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)
        denom2 = zsoil[0, 0, -1] - zsoil
        denom = zsoil[0, 0, -1] - zsoil[0, 0, 1]
        dsmdz = (sh2o - sh2o[0, 0, 1]) / (denom * 0.5)
        ddz = 2.0 / denom
        ci = -wdf * ddz / denom2
        slopx = 1.0
        numer = (
            wdf * dsmdz + slopx * wcnd - wdf[0, 0, -1] * dsmdz[0, 0, -1] - wcnd[0, 0, -1] + et
        )
        rhstt = -numer / denom2

        # calc matrix coefs
        ai = -wdf * ddz[0, 0, -1] / denom2
        bi = -(ai + ci)

    with computation(FORWARD), interval(-1, None):
        # 3. Bottom Layer
        denom2 = zsoil[0, 0, -1] - zsoil
        slopx = slope
        wdf, wcnd = wdfcnd_fn(sh2o, smcmax, bexp, dksat, dwsat, sicemax)
        dsmdz = 0.0
        ci = 0.0
        numer = (
            wdf * dsmdz + slopx * wcnd - wdf[0, 0, -1] * dsmdz[-1] - wcnd[0, 0, -1] + et
        )
        rhstt = -numer / denom2

        # calc matrix coefs
        ai = -wdf * ddz / denom2
        bi = -(ai + ci)

        runoff2 = slope * wcnd


class SoilMoistureFlux:
    def __init__(self):
        """
        Fortran name is smflx
        """
        pass

    def __call__(self):
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
        pass
