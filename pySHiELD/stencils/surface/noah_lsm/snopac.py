from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.stencils.surface.noah_lsm.evapo import EvapoTranspiration
from pySHiELD.stencils.surface.noah_lsm.smflx import SoilMoistureFlux
from pySHiELD.stencils.surface.noah_lsm.shflx import SoilHeatFlux, tdfcnd
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
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity


@gtscript.function
def snowpack_fn(esd, dtsec, tsnow, tsoil, snowh, sndens):
    # --- ... subprograms called: none

    # calculates compaction of snowpack under conditions of
    # increasing snow density.

    c1 = 0.01
    c2 = 21.0

    # conversion into simulation units
    snowhc = snowh * 100.0
    esdc = esd * 100.0
    dthr = dtsec / 3600.0
    tsnowc = tsnow - constants.TICE0
    tsoilc = tsoil - constants.TICE0

    # calculating of average temperature of snow pack
    tavgc = 0.5 * (tsnowc + tsoilc)

    # calculating of snow depth and density as a result of compaction
    if esdc > 1.0e-2:
        esdcx = esdc
    else:
        esdcx = 1.0e-2

    bfac = dthr * c1 * exp(0.08 * tavgc - c2 * sndens)

    # number of terms of polynomial expansion and its accuracy is governed by iteration limit "ipol".
    ipol = 4
    # hardcode loop for ipol = 4
    pexp = 0.0
    pexp = (1.0 + pexp) * bfac * esdcx / 5.0
    pexp = (1.0 + pexp) * bfac * esdcx / 4.0
    pexp = (1.0 + pexp) * bfac * esdcx / 3.0
    pexp = (1.0 + pexp) * bfac * esdcx / 2.0
    pexp += 1.0

    dsx = sndens * pexp
    # set upper/lower limit on snow density
    dsx = max(min(dsx, 0.40), 0.05)
    sndens = dsx

    # update of snow depth and density depending on liquid water during snowmelt.
    if tsnowc >= 0.0:
        dw = 0.13 * dthr / 24.0
        sndens = min(sndens * (1.0 - dw) + dw, 0.40)

    # calculate snow depth (cm) from snow water equivalent and snow density.
    snowhc = esdc / sndens
    snowh = snowhc * 0.01

    return snowh, sndens


def snopac_fn(
    nroot,
    etp,
    prcp,
    smcmax,
    smcwlt,
    smcref,
    smcdry,
    cmcmax,
    df1,
    sfcems,
    sfctmp,
    t24,
    th2,
    fdown,
    epsca,
    bexp,
    pc,
    rch,
    rr,
    cfactr,
    slope,
    kdt,
    frzx,
    psisat,
    zsoil0,
    zsoil1,
    zsoil2,
    zsoil3,
    dwsat,
    dksat,
    zbot,
    shdfac,
    ice,
    rtdis0,
    rtdis1,
    rtdis2,
    rtdis3,
    quartz,
    fxexp,
    csoil,
    flx2,
    snowng,
    ffrozp,
    ivegsrc,
    vegtype,
    # in/outs
    prcp1,
    cmc,
    t1,
    stc0,
    stc1,
    stc2,
    stc3,
    sncovr,
    sneqv,
    sndens,
    snowh,
    sh2o0,
    sh2o1,
    sh2o2,
    sh2o3,
    tbot,
    smc0,
    smc1,
    smc2,
    smc3,
):
    # calculates soil moisture and heat flux values and
    # update soil moisture content and soil heat content values for the
    # case when a snow pack is present.

    from __externals__ import dt

    snoexp = 2.0
    esdmin = 1.0e-6

    prcp1 = prcp1 * 0.001
    edir = 0.0
    edir1 = 0.0

    ec = 0.0
    ec1 = 0.0

    runoff1 = 0.0
    runoff2 = 0.0
    runoff3 = 0.0

    drip = 0.0

    ett = 0.0
    ett1 = 0.0
    etns = 0.0
    etns1 = 0.0
    esnow = 0.0
    esnow1 = 0.0
    esnow2 = 0.0

    et1_0, et1_1, et1_2, et1_3, = (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    et_0, et_1, et_2, et_3, = (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    dew = 0.0
    etp1 = etp * 0.001

    if etp < 0.0:
        # dewfall (=frostfall in this case).
        dew = -etp1
        esnow2 = etp1 * dt
        etanrg = etp * ((1.0 - sncovr) * physcons.LSUBC + sncovr * physcons.LSUBS)

    else:
        # upward moisture flux
        if ice != 0:
            # for sea-ice and glacial-ice case
            esnow = etp
            esnow1 = esnow * 0.001
            esnow2 = esnow1 * dt
            etanrg = esnow * physcons.LSUBS

        else:
            # for non-glacial land case
            if sncovr < 1.0:
                etns1, edir1, ec1, et1_0, et1_1, et1_2, et1_3, ett1 = evapo_fn(
                    nroot,
                    cmc,
                    cmcmax,
                    etp1,
                    dt,
                    sh2o0,
                    sh2o1,
                    sh2o2,
                    sh2o3,
                    smcmax,
                    smcwlt,
                    smcref,
                    smcdry,
                    pc,
                    shdfac,
                    cfactr,
                    rtdis0,
                    rtdis1,
                    rtdis2,
                    rtdis3,
                    fxexp,
                )

                edir1 *= 1.0 - sncovr
                ec1 *= 1.0 - sncovr
                et1_0 *= 1.0 - sncovr
                et1_1 *= 1.0 - sncovr
                et1_2 *= 1.0 - sncovr
                et1_3 *= 1.0 - sncovr
                ett1 *= 1.0 - sncovr
                etns1 *= 1.0 - sncovr

                edir = edir1 * 1000.0
                ec = ec1 * 1000.0
                et_0 = et1_0 * 1000.0
                et_1 = et1_1 * 1000.0
                et_2 = et1_2 * 1000.0
                et_3 = et1_3 * 1000.0
                ett = ett1 * 1000.0
                etns = etns1 * 1000.0

            esnow = etp * sncovr
            esnow1 = esnow * 0.001
            esnow2 = esnow1 * dt
            etanrg = esnow * physcons.LSUBS + etns * physcons.LSUBC

    # if precip is falling, calculate heat flux from snow sfc to newly accumulating precip
    flx1 = 0.0
    if snowng:
        # fractional snowfall/rainfall
        flx1 = (physcons.CPICE * ffrozp + physcons.CPH2O1 * (1.0 - ffrozp)) * prcp * (t1 - sfctmp)

    elif prcp > 0.0:
        flx1 = physcons.CPH2O1 * prcp * (t1 - sfctmp)

    # calculate an 'effective snow-grnd sfc temp' based on heat fluxes between
    # the snow pack and the soil and on net radiation.
    dsoil = -0.5 * zsoil0
    dtot = snowh + dsoil
    denom = 1.0 + df1 / (dtot * rr * rch)
    t12a = (
        (fdown - flx1 - flx2 - sfcems * physcons.SIGMA1 * t24) / rch
        + th2
        - sfctmp
        - etanrg / rch
    ) / rr
    t12b = df1 * stc0 / (dtot * rr * rch)
    t12 = (sfctmp + t12a + t12b) / denom

    if t12 <= constants.TICE0:  # no snow melt will occur.

        # set the skin temp to this effective temp
        t1 = t12
        # update soil heat flux
        ssoil = df1 * (t1 - stc0) / dtot
        # update depth of snowpack
        sneqv = max(0.0, sneqv - esnow2)
        flx3 = 0.0
        ex = 0.0
        snomlt = 0.0

    else:  # snow melt will occur.
        t1 = constants.TICE0 * max(0.01, sncovr ** snoexp) + t12 * (
            1.0 - max(0.01, sncovr ** snoexp)
        )
        ssoil = df1 * (t1 - stc0) / dtot

        if sneqv - esnow2 <= esdmin:
            # snowpack has sublimated away, set depth to zero.
            sneqv = 0.0
            ex = 0.0
            snomlt = 0.0
            flx3 = 0.0

        else:
            # potential evap (sublimation) less than depth of snowpack
            sneqv -= esnow2
            seh = rch * (t1 - th2)

            t14 = t1 * t1
            t14 = t14 * t14

            flx3 = fdown - flx1 - flx2 - sfcems * physcons.SIGMA1 * t14 - ssoil - seh - etanrg
            if flx3 <= 0.0:
                flx3 = 0.0

            ex = flx3 * 0.001 / physcons.LSUBF

            # snowmelt reduction
            snomlt = ex * dt

            if sneqv - snomlt >= esdmin:
                # retain snowpack
                sneqv -= snomlt
            else:
                # snowmelt exceeds snow depth
                ex = sneqv / dt
                flx3 = ex * 1000.0 * physcons.LSUBF
                snomlt = sneqv
                sneqv = 0.0

        if ice == 0:
            prcp1 += ex

    if ice == 0:
        # smflx returns updated soil moisture values for non-glacial land.
        (
            cmc,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            smc0,
            smc1,
            smc2,
            smc3,
            runoff1,
            runoff2,
            runoff3,
            drip,
        ) = smflx_fn(
            dt,
            kdt,
            smcmax,
            smcwlt,
            cmcmax,
            prcp1,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            slope,
            frzx,
            bexp,
            dksat,
            dwsat,
            shdfac,
            edir1,
            ec1,
            et1_0,
            et1_1,
            et1_2,
            et1_3,
            cmc,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            smc0,
            smc1,
            smc2,
            smc3,
        )
    zz1 = 1.0
    yy = stc0 - 0.5 * ssoil * zsoil0 * zz1 / df1
    t11 = t1

    # shflx will calc/update the soil temps.
    ssoil1, stc0, stc1, stc2, stc3, t11, tbot, sh2o0, sh2o1, sh2o2, sh2o3 = shflx_fn(
        smc0,
        smc1,
        smc2,
        smc3,
        smcmax,
        dt,
        yy,
        zz1,
        zsoil0,
        zsoil1,
        zsoil2,
        zsoil3,
        zbot,
        psisat,
        bexp,
        df1,
        ice,
        quartz,
        csoil,
        ivegsrc,
        vegtype,
        shdfac,
        stc0,
        stc1,
        stc2,
        stc3,
        t11,
        tbot,
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
    )

    # snow depth and density adjustment based on snow compaction.
    if ice == 0:
        if sneqv > 0.0:
            snowh, sndens = snowpack_fn(sneqv, dt, t1, yy, snowh, sndens)

        else:
            sneqv = 0.0
            snowh = 0.0
            sndens = 0.0
            sncovr = 0.0

    elif ice == 1:
        if sneqv >= 0.01:
            snowh, sndens = snowpack_fn(sneqv, dt, t1, yy, snowh, sndens)
        else:
            sneqv = 0.01
            snowh = 0.05
            sncovr = 1.0
    else:
        if sneqv >= 0.10:
            snowh, sndens = snowpack_fn(sneqv, dt, t1, yy, snowh, sndens)
        else:
            sneqv = 0.10
            snowh = 0.50
            sncovr = 1.0

    return (
        prcp1,
        cmc,
        t1,
        stc0,
        stc1,
        stc2,
        stc3,
        sncovr,
        sneqv,
        sndens,
        snowh,
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
        tbot,
        smc0,
        smc1,
        smc2,
        smc3,
        ssoil,
        runoff1,
        runoff2,
        runoff3,
        edir,
        ec,
        et_0,
        et_1,
        et_2,
        et_3,
        ett,
        snomlt,
        drip,
        dew,
        flx1,
        flx3,
        esnow,
    )


class SNOPAC:
    def __init__(self):

        self._evapo = EvapoTranspiration()

        self._smflx = SoilMoistureFlux()

        self._shflx = SoilHeatFlux()

        pass

    def __call__(self):
        pass
