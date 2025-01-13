from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, computation, exp, interval, log
from numpy import ndarray

import ndsl.constants as constants
import pySHiELD.constants as physcons
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from pySHiELD._config import LSMConfig, FloatFieldTracer
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg, BARE
from pySHiELD.functions.physics_functions import fpvs


@gtscript.function
def tdfcnd_fn(smc, qz, smcmax, sh2o):
    # --- ... subprograms called: none

    # calculates thermal diffusivity and conductivity of the soil
    # for a given point and time

    # saturation ratio
    satratio = smc / smcmax

    thkice = 2.2
    thkw = 0.57
    thko = 2.0
    thkqtz = 7.7

    # solids` conductivity
    thks = (thkqtz ** qz) * (thko ** (1.0 - qz))

    # unfrozen fraction
    xunfroz = (sh2o + 1.0e-9) / (smc + 1.0e-9)

    # unfrozen volume for saturation (porosity*xunfroz)
    xu = xunfroz * smcmax

    # saturated thermal conductivity
    thksat = thks ** (1.0 - smcmax) * thkice ** (smcmax - xu) * thkw ** (xu)

    # dry density in kg/m3
    gammd = (1.0 - smcmax) * 2700.0

    # dry thermal conductivity in w.m-1.k-1
    thkdry = (0.135 * gammd + 64.7) / (2700.0 - 0.947 * gammd)

    if sh2o + 0.0005 < smc:  # frozen
        ake = satratio
    elif satratio > 0.1:
        # kersten number
        ake = log(satratio) / log(10) + 1.0  # log10 from ln
    else:
        ake = 0.0

    # thermal conductivity
    df = ake * (thksat - thkdry) + thkdry

    return df


@gtscript.function
def alcalc_fn(alb, snoalb, sncovr):
    # --- ... subprograms called: none

    # calculates albedo using snow effect
    # snoalb: max albedo over deep snow
    albedo = alb + sncovr * (snoalb - alb)
    if albedo > snoalb:
        albedo = snoalb

    return albedo


@gtscript.function
def canres_fn(
    nroot,
    swdn,
    ch,
    q2,
    q2sat,
    dqsdt2,
    sfctmp,
    cpx1,
    sfcprs,
    sfcems,
    sh2o0,
    sh2o1,
    sh2o2,
    sh2o3,
    smcwlt,
    smcref,
    zsoil0,
    zsoil1,
    zsoil2,
    zsoil3,
    rsmin,
    rsmax,
    topt,
    rgl,
    hs,
    xlai,
):
    # --- ... subprograms called: none

    # calculates canopy resistance

    # contribution due to incoming solar radiation
    ff = 0.55 * 2.0 * swdn / (rgl * xlai)
    rcs = (ff + rsmin / rsmax) / (1.0 + ff)
    rcs = max(rcs, 0.0001)

    # contribution due to air temperature at first model level above ground
    rct = 1.0 - 0.0016 * (topt - sfctmp) ** 2.0
    rct = max(rct, 0.0001)

    # contribution due to vapor pressure deficit at first model level.
    rcq = 1.0 / (1.0 + hs * (q2sat - q2))
    rcq = max(rcq, 0.01)

    # contribution due to soil moisture availability.
    if nroot > 0:
        gx0 = max(0.0, min(1.0, (sh2o0 - smcwlt) / (smcref - smcwlt)))
        zsoil = zsoil0
    else:
        gx0 = 0.0
    if nroot > 1:
        gx1 = max(0.0, min(1.0, (sh2o1 - smcwlt) / (smcref - smcwlt)))
        zsoil = zsoil1
    else:
        gx1 = 0.0
    if nroot > 2:
        gx2 = max(0.0, min(1.0, (sh2o2 - smcwlt) / (smcref - smcwlt)))
        zsoil = zsoil2
    else:
        gx2 = 0.0
    if nroot > 3:
        gx3 = max(0.0, min(1.0, (sh2o3 - smcwlt) / (smcref - smcwlt)))
        zsoil = zsoil3
    else:
        gx3 = 0.0

    # use soil depth as weighting factor
    sum = (
        zsoil0 / zsoil * gx0
        + (zsoil1 - zsoil0) / zsoil * gx1
        + (zsoil2 - zsoil1) / zsoil * gx2
        + (zsoil3 - zsoil2) / zsoil * gx3
    )

    rcsoil = max(sum, 0.0001)

    # determine canopy resistance due to all factors

    rc = rsmin / (xlai * rcs * rct * rcq * rcsoil)
    rr = (
        4.0 * sfcems * physcons.SIGMA1 * physcons.RD1 / cpx1
    ) * (sfctmp ** 4.0) / (sfcprs * ch) + 1.0
    delta = (physcons.LSUBC / cpx1) * dqsdt2

    pc = (rr + delta) / (rr * (1.0 + rc * ch) + delta)

    return rc, pc, rcs, rct, rcq, rcsoil


@gtscript.function
def csnow_fn(sndens):
    # --- ... subprograms called: none

    unit = 0.11631

    c = 0.328 * 10 ** (2.25 * sndens)
    sncond = unit * c

    return sncond


@gtscript.function
def penman_fn(
    sfctmp,
    sfcprs,
    sfcems,
    ch,
    t2v,
    th2,
    prcp,
    fdown,
    cpx,
    cpfac,
    ssoil,
    q2,
    q2sat,
    dqsdt2,
    snowng,
    frzgra,
    ffrozp,
):
    # --- ... subprograms called: none

    flx2 = 0.0
    # # prepare partial quantities for penman equation.
    delta = physcons.ELCP * cpfac * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (physcons.RD1 * t2v)
    rch = rho * cpx * ch

    # adjust the partial sums / products with the latent heat
    # effects caused by falling precipitation.
    if not snowng:
        if prcp > 0.0:
            rr += physcons.CPH2O1 * prcp / rch
    else:
        # fractional snowfall/rainfall
        rr += (physcons.CPICE * ffrozp + physcons.CPH2O1 * (1.0 - ffrozp)) * prcp / rch

    # ssoil = 13.753581783277639
    fnet = fdown - sfcems * physcons.SIGMA1 * t24 - ssoil

    # include the latent heat effects of frzng rain converting to ice
    # on impact in the calculation of flx2 and fnet.
    if frzgra:
        flx2 = -physcons.LSUBF * prcp
        fnet = fnet - flx2

    # finish penman equation calculations.

    rad = fnet / rch + th2 - sfctmp
    a = physcons.ELCP * cpfac * (q2sat - q2)

    epsca = (a * rr + rad * delta) / (delta + rr)
    etp = epsca * rch / physcons.LSUBC

    return t24, etp, rch, epsca, rr, flx2


@gtscript.function
def redprm_fn(
    vegtyp,
    dksat,
    smcmax,
    smcref,
    nroot,
    sldpth0,
    sldpth1,
    sldpth2,
    sldpth3,
    zsoil0,
    zsoil1,
    zsoil2,
    zsoil3,
    shdfac,
):
    # --- ... subprograms called: none

    kdt = physcons.REFKDT * dksat / physcons.REFDK

    frzfact = (smcmax / smcref) * (0.412 / 0.468)

    # to adjust frzk parameter to actual soil type: frzk * frzfact
    frzx = physcons.FRZK * frzfact

    if vegtyp + 1 == BARE:
        shdfac = 0.0

    # calculate root distribution.  present version assumes uniform
    # distribution based on soil layer depths.

    zsoil = zsoil3
    if nroot <= 3:
        rtdis3 = 0.0
        zsoil = zsoil2
    else:
        rtdis3 = -sldpth3 / zsoil

    if nroot <= 2:
        rtdis2 = 0.0
        zsoil = zsoil1
    else:
        rtdis2 = -sldpth2 / zsoil

    if nroot <= 1:
        rtdis1 = 0.0
        zsoil = zsoil0
    else:
        rtdis1 = -sldpth1 / zsoil

    if nroot <= 0:
        rtdis0 = 0.0
    else:
        rtdis0 = -sldpth0 / zsoil

    return kdt, shdfac, frzx, rtdis0, rtdis1, rtdis2, rtdis3


@gtscript.function
def snfrac_fn(sneqv, snup, salp):
    # --- ... subprograms called: none

    # determine snow fraction cover.
    if sneqv < snup:
        rsnow = sneqv / snup
        sncovr = 1.0 - (exp(-salp * rsnow) - rsnow * exp(-salp))
    else:
        sncovr = 1.0

    return sncovr


@gtscript.function
def snow_new_fn(sfctmp, sn_new, snowh, sndens):
    # --- ... subprograms called: none

    # conversion into simulation units
    snowhc = snowh * 100.0
    newsnc = sn_new * 100.0
    tempc = sfctmp - constants.TICE0

    # calculating new snowfall density
    if tempc <= -15.0:
        dsnew = 0.05
    else:
        dsnew = 0.05 + 0.0017 * (tempc + 15.0) ** 1.5

    # adjustment of snow density depending on new snowfall
    hnewc = newsnc / dsnew
    sndens = (snowhc * sndens + hnewc * dsnew) / (snowhc + hnewc)
    snowhc = snowhc + hnewc
    snowh = snowhc * 0.01

    return snowh, sndens


@gtscript.function
def devap_fn(etp1, smc, shdfac, smcmax, smcdry, fxexp):
    # --- ... subprograms called: none

    # calculates direct soil evaporation
    sratio = (smc - smcdry) / (smcmax - smcdry)

    if sratio > 0.0:
        fx = sratio ** fxexp
        fx = max(min(fx, 1.0), 0.0)
    else:
        fx = 0.0

    # allow for the direct-evap-reducing effect of shade
    edir1 = fx * (1.0 - shdfac) * etp1
    return edir1


@gtscript.function
def transp_fn(
    nroot,
    etp1,
    smc0,
    smc1,
    smc2,
    smc3,
    smcwlt,
    smcref,
    cmc,
    cmcmax,
    shdfac,
    pc,
    cfactr,
    rtdis0,
    rtdis1,
    rtdis2,
    rtdis3,
):
    # initialize plant transp to zero for all soil layers.
    et1_0 = 0.0
    et1_1 = 0.0
    et1_2 = 0.0
    et1_3 = 0.0

    if cmc != 0.0:
        etp1a = shdfac * pc * etp1 * (1.0 - (cmc / cmcmax) ** cfactr)
    else:
        etp1a = shdfac * pc * etp1

    if nroot > 0:
        gx0 = max(0.0, min(1.0, (smc0 - smcwlt) / (smcref - smcwlt)))
    else:
        gx0 = 0.0
    if nroot > 1:
        gx1 = max(0.0, min(1.0, (smc1 - smcwlt) / (smcref - smcwlt)))
    else:
        gx1 = 0.0
    if nroot > 2:
        gx2 = max(0.0, min(1.0, (smc2 - smcwlt) / (smcref - smcwlt)))
    else:
        gx2 = 0.0
    if nroot > 3:
        gx3 = max(0.0, min(1.0, (smc3 - smcwlt) / (smcref - smcwlt)))
    else:
        gx3 = 0.0

    sgx = (gx0 + gx1 + gx2 + gx3) / nroot

    rtx0 = rtdis0 + gx0 - sgx
    rtx1 = rtdis1 + gx1 - sgx
    rtx2 = rtdis2 + gx2 - sgx
    rtx3 = rtdis3 + gx3 - sgx

    gx0 *= max(rtx0, 0.0)
    gx1 *= max(rtx1, 0.0)
    gx2 *= max(rtx2, 0.0)
    gx3 *= max(rtx3, 0.0)

    denom = gx0 + gx1 + gx2 + gx3

    if denom <= 0.0:
        denom = 1.0

    et1_0 = etp1a * gx0 / denom
    et1_1 = etp1a * gx1 / denom
    et1_2 = etp1a * gx2 / denom
    et1_3 = etp1a * gx3 / denom

    return et1_0, et1_1, et1_2, et1_3


@gtscript.function
def evapo_fn(
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
):
    # --- ... subprograms called: devap, transp

    ec1 = 0.0
    ett1 = 0.0
    edir1 = 0.0

    et1_0, et1_1, et1_2, et1_3 = 0.0, 0.0, 0.0, 0.0

    if etp1 > 0.0:
        # retrieve direct evaporation from soil surface.
        if shdfac < 1.0:
            edir1 = devap_fn(etp1, sh2o0, shdfac, smcmax, smcdry, fxexp)
            # edir1 = 4.250472271407341e-10

        # initialize plant total transpiration, retrieve plant transpiration,
        # and accumulate it for all soil layers.
        if shdfac > 0.0:
            # calculates transpiration for the veg class

            et1_0, et1_1, et1_2, et1_3 = transp_fn(
                nroot,
                etp1,
                sh2o0,
                sh2o1,
                sh2o2,
                sh2o3,
                smcwlt,
                smcref,
                cmc,
                cmcmax,
                shdfac,
                pc,
                cfactr,
                rtdis0,
                rtdis1,
                rtdis2,
                rtdis3,
            )

            ett1 = et1_0 + et1_1 + et1_2 + et1_3

            # calculate canopy evaporation.
            if cmc > 0.0:
                ec1 = shdfac * ((cmc / cmcmax) ** cfactr) * etp1
            else:
                ec1 = 0.0

            # ec should be limited by the total amount of available water on the canopy
            cmc2ms = cmc / dt
            ec1 = min(cmc2ms, ec1)

    eta1 = edir1 + ett1 + ec1

    return eta1, edir1, ec1, et1_0, et1_1, et1_2, et1_3, ett1


@gtscript.function
def tmpavg_fn(tup, tm, tdn, dz):

    dzh = dz * 0.5

    if tup < constants.TICE0:
        if tm < constants.TICE0:
            if tdn < constants.TICE0:
                tavg = (tup + 2.0 * tm + tdn) / 4.0
            else:
                x0 = (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    0.5 * (
                        tup * dzh + tm * (dzh + x0) + constants.TICE0 * (2.0 * dzh - x0)
                    ) / dz
                )
        else:
            if tdn < constants.TICE0:
                xup = (constants.TICE0 - tup) * dzh / (tm - tup)
                xdn = dzh - (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    0.5 * (
                        tup * xup + constants.TICE0 * (2.0 * dz - xup - xdn) + tdn * xdn
                    ) / dz
                )
            else:
                xup = (constants.TICE0 - tup) * dzh / (tm - tup)
                tavg = 0.5 * (tup * xup + constants.TICE0 * (2.0 * dz - xup)) / dz
    else:
        if tm < constants.TICE0:
            if tdn < constants.TICE0:
                xup = dzh - (constants.TICE0 - tup) * dzh / (tm - tup)
                tavg = 0.5 * (constants.TICE0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh) / dz
            else:
                xup = dzh - (constants.TICE0 - tup) * dzh / (tm - tup)
                xdn = (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = 0.5 * (constants.TICE0 * (2.0 * dz - xup - xdn) + tm * (xup + xdn)) / dz
        else:
            if tdn < constants.TICE0:
                xdn = dzh - (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (constants.TICE0 * (dz - xdn) + 0.5 * (constants.TICE0 + tdn) * xdn) / dz
            else:
                tavg = (tup + 2.0 * tm + tdn) / 4.0
    return tavg


@gtscript.function
def frh2o_loop_fn(psisat, ck, swl, smcmax, smc, bx, tavg, error):
    df = log(
        (psisat * physcons.GS2 / physcons.LSUBF)
        * ((1.0 + ck * swl) ** 2.0)
        * (smcmax / (smc - swl)) ** bx
    ) - log(-(tavg - constants.TICE0) / tavg)

    denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
    swlk = swl - df / denom

    # bounds useful for mathematical solution.
    swlk = max(min(swlk, smc - 0.02), 0.0)

    # mathematical solution bounds applied.
    dswl = abs(swlk - swl)
    swl = swlk

    if dswl <= error:
        kcount = False

    free = smc - swl

    return kcount, free, swl


@gtscript.function
def frh2o_fn(psis, bexp, tavg, smc, sh2o, smcmax):
    ### ************ frh2o *********** ###
    # constant parameters
    ck = 8.0
    blim = 5.5
    error = 0.005
    bx = min(bexp, blim)

    kcount = True

    if tavg <= (constants.TICE0 - 1.0e-3):
        swl = smc - sh2o
        swl = max(min(swl, smc - 0.02), 0.0)

        kcount, free, swl = frh2o_loop_fn(psis, ck, swl, smcmax, smc, bx, tavg, error)
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )
        if kcount:
            kcount, free, swl = frh2o_loop_fn(
                psis, ck, swl, smcmax, smc, bx, tavg, error
            )

    else:
        free = smc

    return free


@gtscript.function
def snksrc_fn(psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dt, dz):
    free = frh2o_fn(psisat, bexp, tavg, smc, sh2o, smcmax)

    # estimate the new amount of liquid water
    dh2o = 1.0000e3
    xh2o = sh2o + qtot * dt / (dh2o * physcons.LSUBF * dz)

    if xh2o < sh2o and xh2o < free:
        if free > sh2o:
            xh2o = sh2o
        else:
            xh2o = free
    if xh2o > sh2o and xh2o > free:
        if free < sh2o:
            xh2o = sh2o
        else:
            xh2o = free

    xh2o = max(min(xh2o, smc), 0.0)
    tsnsr = -dh2o * physcons.LSUBF * dz * (xh2o - sh2o) / dt
    sh2o = xh2o

    return tsnsr, sh2o


@gtscript.function
def rosr12_fn(ai1, ai2, ai3, bi0, bi1, bi2, bi3, ci0, ci1, ci2, ci3, d0, d1, d2, d3):
    # solve the tri-diagonal matrix

    ci3 = 0.0

    # solve the coefs for the 1st soil layer

    p0 = -ci0 / bi0
    delta0 = d0 / bi0

    p1 = -ci1 / (bi1 + ai1 * p0)
    delta1 = (d1 - ai1 * delta0) / (bi1 + ai1 * p0)
    p2 = -ci2 / (bi2 + ai2 * p1)
    delta2 = (d2 - ai2 * delta1) / (bi2 + ai2 * p1)
    p3 = -ci3 / (bi3 + ai3 * p2)
    delta3 = (d3 - ai3 * delta2) / (bi3 + ai3 * p2)

    p3 = delta3
    p2 = p2 * p3 + delta2
    p1 = p1 * p2 + delta1
    p0 = p0 * p1 + delta0
    return p0, p1, p2, p3, delta0, delta1, delta2, delta3


@gtscript.function
def hrtice_fn(
    stc0, stc1, stc2, stc3, zsoil0, zsoil1, zsoil2, zsoil3, yy, zz1, df1, ice, tbot
):
    # calculates the right hand side of the time tendency
    # term of the soil thermal diffusion equation for sea-ice or glacial-ice

    # set a nominal universal value of specific heat capacity
    if ice == 1:  # sea-ice
        hcpct = 1.72396e6
        tbot = 271.16
    else:  # glacial-ice
        hcpct = 1.89000e6

    # set ice pack depth
    if ice == 1:
        zbot = zsoil3
    else:
        zbot = -25.0

    # 1. Layer
    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5 * zsoil1)
    ai0 = 0.0
    ci0 = (df1 * ddz) / (zsoil0 * hcpct)
    bi0 = -ci0 + df1 / (0.5 * zsoil0 * zsoil0 * hcpct * zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc0 - stc1) / (-0.5 * zsoil1)
    ssoil = df1 * (stc0 - yy) / (0.5 * zsoil0 * zz1)
    rhsts0 = (df1 * dtsdz - ssoil) / (zsoil0 * hcpct)

    # 2. Layer
    denom = 0.5 * (zsoil0 - zsoil2)
    dtsdz2 = (stc1 - stc2) / denom
    ddz2 = 2.0 / (zsoil0 - zsoil2)
    ci1 = -df1 * ddz2 / ((zsoil0 - zsoil1) * hcpct)

    denom = (zsoil1 - zsoil0) * hcpct
    rhsts1 = (df1 * dtsdz2 - df1 * dtsdz) / denom

    ai1 = -df1 * ddz / ((zsoil0 - zsoil1) * hcpct)
    bi1 = -(ai1 + ci1)

    dtsdz = dtsdz2
    ddz = ddz2

    # 3. Layer
    denom = 0.5 * (zsoil1 - zsoil3)
    dtsdz2 = (stc2 - stc3) / denom
    ddz2 = 2.0 / (zsoil1 - zsoil3)
    ci2 = -df1 * ddz2 / ((zsoil1 - zsoil2) * hcpct)

    denom = (zsoil2 - zsoil1) * hcpct
    rhsts2 = (df1 * dtsdz2 - df1 * dtsdz) / denom

    ai2 = -df1 * ddz / ((zsoil1 - zsoil2) * hcpct)
    bi2 = -(ai2 + ci2)

    dtsdz = dtsdz2
    ddz = ddz2

    # 4. Layer
    dtsdz2 = (stc3 - tbot) / (0.5 * (zsoil2 - zsoil3) - zbot)
    ci3 = 0.0

    denom = (zsoil3 - zsoil2) * hcpct
    rhsts3 = (df1 * dtsdz2 - df1 * dtsdz) / denom

    ai3 = -df1 * ddz / ((zsoil2 - zsoil3) * hcpct)
    bi3 = -(ai3 + ci3)

    return (
        tbot,
        rhsts0,
        rhsts1,
        rhsts2,
        rhsts3,
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


@gtscript.function
def hrt_fn(
    stc0,
    stc1,
    stc2,
    stc3,
    smc0,
    smc1,
    smc2,
    smc3,
    smcmax,
    zsoil0,
    zsoil1,
    zsoil2,
    zsoil3,
    yy,
    zz1,
    tbot,
    zbot,
    psisat,
    dt,
    bexp,
    df1,
    quartz,
    csoil,
    ivegsrc,
    vegtyp,
    shdfac,
    sh2o0,
    sh2o1,
    sh2o2,
    sh2o3,
):

    csoil_loc = csoil

    if ivegsrc == 1 and vegtyp == 12:
        csoil_loc = 3.0e6 * (1.0 - shdfac) + csoil * shdfac

    # calc the heat capacity of the top soil layer
    hcpct = (
        sh2o0 * physcons.CPH2O2
        + (1.0 - smcmax) * csoil_loc
        + (smcmax - smc0) * physcons.CP2
        + (smc0 - sh2o0) * physcons.CPICE1
    )

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5 * zsoil1)
    ai0 = 0.0
    ci0 = (df1 * ddz) / (zsoil0 * hcpct)
    bi0 = -ci0 + df1 / (0.5 * zsoil0 * zsoil0 * hcpct * zz1)

    # calc the vertical soil temp gradient btwn the top and 2nd soil
    dtsdz = (stc0 - stc1) / (-0.5 * zsoil1)
    ssoil = df1 * (stc0 - yy) / (0.5 * zsoil0 * zz1)
    rhsts0 = (df1 * dtsdz - ssoil) / (zsoil0 * hcpct)

    # capture the vertical difference of the heat flux at top and
    # bottom of first soil layer
    qtot = ssoil - df1 * dtsdz

    tsurf = (yy + (zz1 - 1) * stc0) / zz1

    # linear interpolation between the average layer temperatures
    tbk = stc0 + (stc1 - stc0) * zsoil0 / zsoil1
    # calculate frozen water content in 1st soil layer.
    sice = smc0 - sh2o0

    df1k = df1

    if sice > 0 or tsurf < constants.TICE0 or stc0 < constants.TICE0 or tbk < constants.TICE0:
        ### ************ tmpavg *********** ###
        dz = -zsoil0
        tavg = tmpavg_fn(tsurf, stc0, tbk, dz)
        ### ************ snksrc *********** ###
        tsnsr, sh2o0 = snksrc_fn(psisat, bexp, tavg, smc0, sh2o0, smcmax, qtot, dt, dz)
        ### ************ END snksrc *********** ###

        rhsts0 -= tsnsr / (zsoil0 * hcpct)

    # 2. Layer
    hcpct = (
        sh2o1 * physcons.CPH2O2
        + (1.0 - smcmax) * csoil_loc
        + (smcmax - smc1) * physcons.CP2
        + (smc1 - sh2o1) * physcons.CPICE1
    )

    # calculate thermal diffusivity for each layer
    df1n = tdfcnd_fn(smc1, quartz, smcmax, sh2o1)

    if ivegsrc == 1 and vegtyp == 12:
        df1n = 3.24 * (1.0 - shdfac) + shdfac * df1n

    tbk1 = stc1 + (stc2 - stc1) * (zsoil0 - zsoil1) / (zsoil0 - zsoil2)
    # calc the vertical soil temp gradient thru each layer
    denom = 0.5 * (zsoil0 - zsoil2)
    dtsdz2 = (stc1 - stc2) / denom
    ddz2 = 2.0 / (zsoil0 - zsoil2)

    ci1 = -df1n * ddz2 / ((zsoil0 - zsoil1) * hcpct)

    # calculate rhsts
    denom = (zsoil1 - zsoil0) * hcpct
    rhsts1 = (df1n * dtsdz2 - df1k * dtsdz) / denom

    qtot = -1.0 * denom * rhsts1
    sice = smc1 - sh2o1

    if sice > 0 or tbk < constants.TICE0 or stc1 < constants.TICE0 or tbk1 < constants.TICE0:
        ### ************ tmpavg *********** ###
        dz = zsoil0 - zsoil1
        tavg = tmpavg_fn(tbk, stc1, tbk1, dz)
        ### ************ snksrc *********** ###
        tsnsr, sh2o1 = snksrc_fn(psisat, bexp, tavg, smc1, sh2o1, smcmax, qtot, dt, dz)
        ### ************ END snksrc *********** ###
        rhsts1 -= tsnsr / denom

    # calc matrix coefs, ai, and bi for this layer.
    ai1 = -df1 * ddz / ((zsoil0 - zsoil1) * hcpct)
    bi1 = -(ai1 + ci1)

    tbk = tbk1
    df1k = df1n
    dtsdz = dtsdz2
    ddz = ddz2

    # 3. Layer
    hcpct = (
        sh2o2 * physcons.CPH2O2
        + (1.0 - smcmax) * csoil_loc
        + (smcmax - smc2) * physcons.CP2
        + (smc2 - sh2o2) * physcons.CPICE1
    )

    # calculate thermal diffusivity for each layer
    df1n = tdfcnd_fn(smc2, quartz, smcmax, sh2o2)

    if ivegsrc == 1 and vegtyp == 12:
        df1n = 3.24 * (1.0 - shdfac) + shdfac * df1n

    tbk1 = stc2 + (stc3 - stc2) * (zsoil1 - zsoil2) / (zsoil1 - zsoil3)
    # calc the vertical soil temp gradient thru each layer
    denom = 0.5 * (zsoil1 - zsoil3)
    dtsdz2 = (stc2 - stc3) / denom
    ddz2 = 2.0 / (zsoil1 - zsoil3)

    ci2 = -df1n * ddz2 / ((zsoil1 - zsoil2) * hcpct)

    # calculate rhsts
    denom = (zsoil2 - zsoil1) * hcpct
    rhsts2 = (df1n * dtsdz2 - df1k * dtsdz) / denom

    qtot = -1.0 * denom * rhsts2
    sice = smc2 - sh2o2

    if sice > 0 or tbk < constants.TICE0 or stc2 < constants.TICE0 or tbk1 < constants.TICE0:
        ### ************ tmpavg *********** ###
        dz = zsoil1 - zsoil2
        tavg = tmpavg_fn(tbk, stc2, tbk1, dz)

        tsnsr, sh2o2 = snksrc_fn(psisat, bexp, tavg, smc2, sh2o2, smcmax, qtot, dt, dz)

        rhsts2 -= tsnsr / denom

    # calc matrix coefs, ai, and bi for this layer.
    ai2 = -df1 * ddz / ((zsoil1 - zsoil2) * hcpct)
    bi2 = -(ai2 + ci2)

    tbk = tbk1
    df1k = df1n
    dtsdz = dtsdz2
    ddz = ddz2

    # 4. Layer
    hcpct = (
        sh2o3 * physcons.CPH2O2
        + (1.0 - smcmax) * csoil_loc
        + (smcmax - smc3) * physcons.CP2
        + (smc3 - sh2o3) * physcons.CPICE1
    )

    # calculate thermal diffusivity for each layer
    df1n = tdfcnd_fn(smc3, quartz, smcmax, sh2o3)

    if ivegsrc == 1 and vegtyp == 12:
        df1n = 3.24 * (1.0 - shdfac) + shdfac * df1n

    tbk1 = stc3 + (tbot - stc3) * (zsoil2 - zsoil3) / (zsoil2 + zsoil3 - 2.0 * zbot)

    denom = 0.5 * (zsoil2 + zsoil3) - zbot
    dtsdz2 = (stc3 - tbot) / denom
    ci3 = 0.0

    # calculate rhsts
    denom = (zsoil3 - zsoil2) * hcpct
    rhsts3 = (df1n * dtsdz2 - df1k * dtsdz) / denom

    qtot = -1.0 * denom * rhsts3
    sice = smc3 - sh2o3

    if sice > 0 or tbk < constants.TICE0 or stc3 < constants.TICE0 or tbk1 < constants.TICE0:
        ### ************ tmpavg *********** ###
        dz = zsoil2 - zsoil3
        tavg = tmpavg_fn(tbk, stc3, tbk1, dz)
        tsnsr, sh2o3 = snksrc_fn(psisat, bexp, tavg, smc3, sh2o3, smcmax, qtot, dt, dz)

        rhsts3 -= tsnsr / denom
    # calc matrix coefs, ai, and bi for this layer.
    ai3 = -df1 * ddz / ((zsoil2 - zsoil3) * hcpct)
    bi3 = -(ai3 + ci3)

    return (
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
        rhsts0,
        rhsts1,
        rhsts2,
        rhsts3,
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


@gtscript.function
def hstep_fn(
    stc0,
    stc1,
    stc2,
    stc3,
    dt,
    rhsts0,
    rhsts1,
    rhsts2,
    rhsts3,
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
):

    ci0, ci1, ci2, ci3, rhsts0, rhsts1, rhsts2, rhsts3 = rosr12_fn(
        ai1 * dt,
        ai2 * dt,
        ai3 * dt,
        1.0 + dt * bi0,
        1.0 + dt * bi1,
        1.0 + dt * bi2,
        1.0 + dt * bi3,
        ci0 * dt,
        ci1 * dt,
        ci2 * dt,
        ci3 * dt,
        rhsts0 * dt,
        rhsts1 * dt,
        rhsts2 * dt,
        rhsts3 * dt,
    )

    stc0 += ci0
    stc1 += ci1
    stc2 += ci2
    stc3 += ci3

    return stc0, stc1, stc2, stc3


@gtscript.function
def shflx_fn(
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
    vegtyp,
    shdfac,
    stc0,
    stc1,
    stc2,
    stc3,
    t1,
    tbot,
    sh2o0,
    sh2o1,
    sh2o2,
    sh2o3,
):
    # --- ... subprograms called: hstep, hrtice, hrt

    # updates the temperature state of the soil column

    ctfil1 = 0.5
    ctfil2 = 1.0 - ctfil1

    oldt1 = t1

    stsoil0 = stc0
    stsoil1 = stc1
    stsoil2 = stc2
    stsoil3 = stc3

    if ice != 0:  # sea-ice or glacial ice case
        (
            tbot,
            rhsts0,
            rhsts1,
            rhsts2,
            rhsts3,
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
        ) = hrtice_fn(
            stc0,
            stc1,
            stc2,
            stc3,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            yy,
            zz1,
            df1,
            ice,
            tbot,
        )

        stc0, stc1, stc2, stc3 = hstep_fn(
            stc0,
            stc1,
            stc2,
            stc3,
            dt,
            rhsts0,
            rhsts1,
            rhsts2,
            rhsts3,
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

    else:
        (
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            rhsts0,
            rhsts1,
            rhsts2,
            rhsts3,
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
        ) = hrt_fn(
            stc0,
            stc1,
            stc2,
            stc3,
            smc0,
            smc1,
            smc2,
            smc3,
            smcmax,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            yy,
            zz1,
            tbot,
            zbot,
            psisat,
            dt,
            bexp,
            df1,
            quartz,
            csoil,
            ivegsrc,
            vegtyp,
            shdfac,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
        )
        stc0, stc1, stc2, stc3 = hstep_fn(
            stc0,
            stc1,
            stc2,
            stc3,
            dt,
            rhsts0,
            rhsts1,
            rhsts2,
            rhsts3,
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

    # update the grnd (skin) temperature in the no snowpack case
    t1 = (yy + (zz1 - 1.0) * stc0) / zz1
    t1 = ctfil1 * t1 + ctfil2 * oldt1
    stc0 = ctfil1 * stc0 + ctfil2 * stsoil0
    stc1 = ctfil1 * stc1 + ctfil2 * stsoil1
    stc2 = ctfil1 * stc2 + ctfil2 * stsoil2
    stc3 = ctfil1 * stc3 + ctfil2 * stsoil3

    # calculate surface soil heat flux
    ssoil = df1 * (stc0 - t1) / (0.5 * zsoil0)

    return ssoil, stc0, stc1, stc2, stc3, t1, tbot, sh2o0, sh2o1, sh2o2, sh2o3


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


@gtscript.function
def srt_fn(
    edir,
    et_0,
    et_1,
    et_2,
    et_3,
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
):
    # determine rainfall infiltration rate and runoff
    cvfrz = 3

    sicemax = max(max(max(max(sice0, sice1), sice2), sice3), 0.0)

    pddum = pcpdrp
    runoff1 = 0.0

    if pcpdrp != 0:
        # frozen ground version
        dt1 = dt / 86400.0
        smcav = smcmax - smcwlt
        dd = (
            -zsoil0 * (smcav - (sh2o0 + sice0 - smcwlt))
            + (zsoil0 - zsoil1) * (smcav - (sh2o1 + sice1 - smcwlt))
            + (zsoil1 - zsoil2) * (smcav - (sh2o2 + sice2 - smcwlt))
            + (zsoil2 - zsoil3) * (smcav - (sh2o3 + sice3 - smcwlt))
        )

        dice = (
            -zsoil0 * sice0
            + (zsoil0 - zsoil1) * sice1
            + (zsoil1 - zsoil2) * sice2
            + (zsoil2 - zsoil3) * sice3
        )

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

        wdf0, wcnd0 = wdfcnd_fn(sh2o0, smcmax, bexp, dksat, dwsat, sicemax)

        infmax = max(infmax, wcnd0)
        infmax = min(infmax, px)

        if pcpdrp > infmax:
            runoff1 = pcpdrp - infmax
            pddum = infmax

    wdf0, wcnd0 = wdfcnd_fn(sh2o0, smcmax, bexp, dksat, dwsat, sicemax)
    wdf1, wcnd1 = wdfcnd_fn(sh2o1, smcmax, bexp, dksat, dwsat, sicemax)
    wdf2, wcnd2 = wdfcnd_fn(sh2o2, smcmax, bexp, dksat, dwsat, sicemax)
    wdf3, wcnd3 = wdfcnd_fn(sh2o3, smcmax, bexp, dksat, dwsat, sicemax)

    # calc the matrix coefficients ai, bi, and ci for the top layer
    ddz = 1.0 / (-0.5 * zsoil1)
    ai0 = 0.0
    bi0 = wdf0 * ddz / (-zsoil0)
    ci0 = -bi0

    # calc rhstt for the top layer
    dsmdz = (sh2o0 - sh2o1) / (-0.5 * zsoil1)
    rhstt0 = (wdf0 * dsmdz + wcnd0 - pddum + edir + et_0) / zsoil0

    # 2. Layer
    denom2 = zsoil0 - zsoil1
    denom = zsoil0 - zsoil2
    dsmdz2 = (sh2o1 - sh2o2) / (denom * 0.5)
    ddz2 = 2.0 / denom
    ci1 = -wdf1 * ddz2 / denom2
    slopx = 1.0
    numer = wdf1 * dsmdz2 + slopx * wcnd1 - wdf0 * dsmdz - wcnd0 + et_1
    rhstt1 = -numer / denom2

    # calc matrix coefs
    ai1 = -wdf0 * ddz / denom2
    bi1 = -(ai1 + ci1)

    dsmdz = dsmdz2
    ddz = ddz2

    # 3. Layer
    denom2 = zsoil1 - zsoil2
    denom = zsoil1 - zsoil3
    dsmdz2 = (sh2o2 - sh2o3) / (denom * 0.5)
    ddz2 = 2.0 / denom
    ci2 = -wdf2 * ddz2 / denom2
    slopx = 1.0
    numer = wdf2 * dsmdz2 + slopx * wcnd2 - wdf1 * dsmdz - wcnd1 + et_2
    rhstt2 = -numer / denom2

    # calc matrix coefs
    ai2 = -wdf1 * ddz / denom2
    bi2 = -(ai2 + ci2)

    dsmdz = dsmdz2
    ddz = ddz2

    # 4. Layer
    denom2 = zsoil2 - zsoil3
    dsmdz2 = 0.0
    ci3 = 0.0
    slopx = slope
    numer = wdf3 * dsmdz2 + slopx * wcnd3 - wdf2 * dsmdz - wcnd2 + et_3
    rhstt3 = -numer / denom2

    # calc matrix coefs
    ai3 = -wdf2 * ddz / denom2
    bi3 = -(ai3 + ci3)

    runoff2 = slope * wcnd3

    return (
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
    )


@gtscript.function
def sstep_fn(
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
):
    # calculates/updates soil moisture content values and
    # canopy moisture content values.

    ci0, ci1, ci2, ci3, rhstt0, rhstt1, rhstt2, rhstt3 = rosr12_fn(
        ai1 * dt,
        ai2 * dt,
        ai3 * dt,
        1.0 + dt * bi0,
        1.0 + dt * bi1,
        1.0 + dt * bi2,
        1.0 + dt * bi3,
        ci0 * dt,
        ci1 * dt,
        ci2 * dt,
        ci3 * dt,
        rhstt0 * dt,
        rhstt1 * dt,
        rhstt2 * dt,
        rhstt3 * dt,
    )

    # sum the previous smc value and the matrix solution
    ddz0 = -zsoil0
    ddz1 = zsoil0 - zsoil1
    ddz2 = zsoil1 - zsoil2
    ddz3 = zsoil2 - zsoil3

    wplus = 0.0

    # 1. Layer
    sh2o0 = sh2o0 + ci0 + wplus / ddz0
    stot = sh2o0 + sice0

    if stot > smcmax:
        wplus = (stot - smcmax) * ddz0
    else:
        wplus = 0.0

    smc0 = max(min(stot, smcmax), 0.02)
    sh2o0 = max(smc0 - sice0, 0.0)

    # 2. Layer
    sh2o1 = sh2o1 + ci1 + wplus / ddz1
    stot = sh2o1 + sice1

    if stot > smcmax:
        wplus = (stot - smcmax) * ddz1
    else:
        wplus = 0.0

    smc1 = max(min(stot, smcmax), 0.02)
    sh2o1 = max(smc1 - sice1, 0.0)

    # 3. Layer
    sh2o2 = sh2o2 + ci2 + wplus / ddz2
    stot = sh2o2 + sice2

    if stot > smcmax:
        wplus = (stot - smcmax) * ddz2
    else:
        wplus = 0.0

    smc2 = max(min(stot, smcmax), 0.02)
    sh2o2 = max(smc2 - sice2, 0.0)

    # 4. Layer
    sh2o3 = sh2o3 + ci3 + wplus / ddz3
    stot = sh2o3 + sice3

    if stot > smcmax:
        wplus = (stot - smcmax) * ddz3
    else:
        wplus = 0.0

    smc3 = max(min(stot, smcmax), 0.02)
    sh2o3 = max(smc3 - sice3, 0.0)

    runoff3 = wplus

    # update canopy water content/interception
    cmc += dt * rhsct
    if cmc < 1.0e-20:
        cmc = 0.0
    cmc = min(cmc, cmcmax)

    return sh2o0, sh2o1, sh2o2, sh2o3, runoff3, smc0, smc1, smc2, smc3, cmc


@gtscript.function
def smflx_fn(
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
):
    # compute the right hand side of the canopy eqn term
    rhsct = shdfac * prcp1 - ec1

    drip = 0.0
    trhsct = dt * rhsct
    excess = cmc + trhsct

    if excess > cmcmax:
        drip = excess - cmcmax

    # pcpdrp is the combined prcp1 and drip (from cmc) that goes into the soil
    pcpdrp = (1.0 - shdfac) * prcp1 + drip / dt

    # store ice content at each soil layer before calling srt and sstep
    sice0 = smc0 - sh2o0
    sice1 = smc1 - sh2o1
    sice2 = smc2 - sh2o2
    sice3 = smc3 - sh2o3

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

    return (
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
    )


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

    # number of terms of polynomial expansion and its accuracy
    # is governed by iteration limit "ipol".
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


@gtscript.function
def nopac_fn(
    nroot,
    etp,
    prcp,
    smcmax,
    smcwlt,
    smcref,
    smcdry,
    cmcmax,
    dt,
    shdfac,
    sbeta,
    sfctmp,
    sfcems,
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
    dksat,
    dwsat,
    zbot,
    ice,
    rtdis0,
    rtdis1,
    rtdis2,
    rtdis3,
    quartz,
    fxexp,
    csoil,
    ivegsrc,
    vegtyp,
    cmc,
    t1,
    stc0,
    stc1,
    stc2,
    stc3,
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
    # convert etp from kg m-2 s-1 to ms-1 and initialize dew.
    prcp1 = prcp * 0.001
    etp1 = etp * 0.001
    dew = 0.0
    edir = 0.0
    edir1 = 0.0
    ec = 0.0
    ec1 = 0.0

    ett = 0.0
    ett1 = 0.0
    eta = 0.0
    eta1 = 0.0

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

    if etp > 0.0:
        eta1, edir1, ec1, et1_0, et1_1, et1_2, et1_3, ett1 = evapo_fn(
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

    else:
        # if etp < 0, assume dew forms
        eta1 = 0.0
        dew = -etp1
        prcp1 += dew

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

    # convert modeled evapotranspiration fm  m s-1  to  kg m-2 s-1
    eta = eta1 * 1000.0
    edir = edir1 * 1000.0
    ec = ec1 * 1000.0
    et_0 = et1_0 * 1000.0
    et_1 = et1_1 * 1000.0
    et_2 = et1_2 * 1000.0
    et_3 = et1_3 * 1000.0
    ett = ett1 * 1000.0

    # based on etp and e values, determine beta
    if etp < 0.0:
        beta = 1.0
    elif etp == 0.0:
        beta = 0.0
    else:
        beta = eta / etp

    # get soil thermal diffuxivity/conductivity for top soil lyr, calc.
    df1 = tdfcnd_fn(smc0, quartz, smcmax, sh2o0)

    if (ivegsrc == 1) and (vegtyp == 12):
        df1 = 3.24 * (1.0 - shdfac) + shdfac * df1 * exp(sbeta * shdfac)
    else:
        df1 *= exp(sbeta * shdfac)

    # compute intermediate terms passed to routine hrt
    yynum = fdown - sfcems * physcons.SIGMA1 * t24
    yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca) / rr
    zz1 = df1 / (-0.5 * zsoil0 * rch * rr) + 1.0

    ssoil, stc0, stc1, stc2, stc3, t1, tbot, sh2o0, sh2o1, sh2o2, sh2o3 = shflx_fn(
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
        vegtyp,
        shdfac,
        stc0,
        stc1,
        stc2,
        stc3,
        t1,
        tbot,
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
    )

    flx1 = 0.0
    flx3 = 0.0

    return (
        cmc,
        t1,
        stc0,
        stc1,
        stc2,
        stc3,
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
        tbot,
        eta,
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
        beta,
        drip,
        dew,
        flx1,
        flx3,
    )


@gtscript.function
def snopac_fn(
    nroot,
    etp,
    prcp,
    smcmax,
    smcwlt,
    smcref,
    smcdry,
    cmcmax,
    dt,
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
    vegtyp,
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

    # if precip is falling, calculate heat flux from snow sfc
    # to newly accumulating precip
    flx1 = 0.0
    if snowng:
        # fractional snowfall/rainfall
        flx1 = (physcons.CPICE * ffrozp + physcons.CPH2O1 * (
            1.0 - ffrozp
        )) * prcp * (t1 - sfctmp)

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
        vegtyp,
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


@gtscript.function
def sflx(
    couple,
    ice,
    ffrozp,
    dt,
    sldpth0,
    sldpth1,
    sldpth2,
    sldpth3,
    dksat,
    smcmax,
    smcref,
    smcwlt,
    smcdry,
    bexp,
    xlai,
    snup,
    quartz,
    rgl,
    hs,
    slope,
    psisat,
    dwsat,
    nroot,
    rsmin,
    swdn,
    swnet,
    lwdn,
    sfcems,
    sfcprs,
    sfctmp,
    prcp,
    q2,
    q2sat,
    dqsdt2,
    th2,
    ivegsrc,
    vegtyp,
    soiltyp,
    slopetyp,
    alb,
    snoalb,
    bexpp,
    xlaip,
    lheatstrg,
    tbot,
    cmc,
    t1,
    stc0,
    stc1,
    stc2,
    stc3,
    smc0,
    smc1,
    smc2,
    smc3,
    sh2o0,
    sh2o1,
    sh2o2,
    sh2o3,
    sneqv,
    ch,
    cm,
    z0,
    shdfac,
    snowh,
):
    # --- ... subprograms called: redprm, snow_new, csnow, snfrac,
    # alcalc, tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.
    # initialization

    shdfac0 = shdfac

    # is not called
    if ivegsrc == 2 and vegtyp == 12:
        ice = -1
        shdfac = 0.0

    if ivegsrc == 1 and vegtyp == 14:
        ice = -1
        shdfac = 0.0

    zsoil0 = -sldpth0
    zsoil1 = zsoil0 - sldpth1
    zsoil2 = zsoil1 - sldpth2
    zsoil3 = zsoil2 - sldpth3

    kdt, shdfac, frzx, rtdis0, rtdis1, rtdis2, rtdis3 = redprm_fn(
        vegtyp,
        dksat,
        smcmax,
        smcref,
        nroot,
        sldpth0,
        sldpth1,
        sldpth2,
        sldpth3,
        zsoil0,
        zsoil1,
        zsoil2,
        zsoil3,
        shdfac,
    )

    if ivegsrc == 1 and vegtyp == 12:
        rsmin = 400.0 * (1 - shdfac0) + 40.0 * shdfac0
        shdfac = shdfac0
        smcmax = 0.45 * (1 - shdfac0) + smcmax * shdfac0
        smcref = 0.42 * (1 - shdfac0) + smcref * shdfac0
        smcwlt = 0.40 * (1 - shdfac0) + smcwlt * shdfac0
        smcdry = 0.40 * (1 - shdfac0) + smcdry * shdfac0

    bexp = bexp * min(1.0 + bexpp, 2.0)

    xlai = xlai * (1.0 + xlaip)
    xlai = max(xlai, 0.75)

    # over sea-ice or glacial-ice, if s.w.e. (sneqv) below threshold
    # lower bound (0.01 m for sea-ice, 0.10 m for glacial-ice), then
    # set at lower bound and store the source increment in subsurface
    # runoff/baseflow (runoff2).
    if (ice == 1) and (sneqv < 0.10):
        sneqv = 0.10
        snowh = 1.00
    elif (ice == -1) and (sneqv < 0.10):
        # TODO: check if it is called
        sneqv = 0.10
        snowh = 1.00

    # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
    # as a flag for non-soil medium
    if ice != 0:
        smc0 = 1.0
        smc1 = 1.0
        smc2 = 1.0
        smc3 = 1.0
        sh2o0 = 1.0
        sh2o1 = 1.0
        sh2o2 = 1.0
        sh2o3 = 1.0

    # if input snowpack is nonzero, then compute snow density "sndens"
    # and snow thermal conductivity "sncond" (note that csnow is a function subroutine)
    if sneqv == 0.0:
        sndens = 0.0
        snowh = 0.0
        sncond = 1.0
    else:
        sndens = sneqv / snowh
        sndens = max(0.0, min(1.0, sndens))
        # TODO: sncond is that necessary? is it later overwritten without using before?
        sncond = csnow_fn(sndens)

    # determine if it's precipitating and what kind of precip it is.
    # if it's prcping and the air temp is colder than 0 c, it's snowing!
    # if it's prcping and the air temp is warmer than 0 c, but the grnd
    # temp is colder than 0 c, freezing rain is presumed to be falling.
    snowng = (prcp > 0.0) and (ffrozp > 0.0)
    frzgra = (prcp > 0.0) and (ffrozp <= 0.0) and (t1 <= constants.TICE0)

    # if either prcp flag is set, determine new snowfall (converting
    # prcp rate from kg m-2 s-1 to a liquid equiv snow depth in meters)
    # and add it to the existing snowpack.

    # snowfall
    if snowng:
        sn_new = ffrozp * prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcp1 = (1.0 - ffrozp) * prcp

    # freezing rain
    if frzgra:
        sn_new = prcp * dt * 0.001
        sneqv = sneqv + sn_new
        prcp1 = 0.0

    if snowng or frzgra:

        # update snow density based on new snowfall, using old and new
        # snow.  update snow thermal conductivity
        snowh, sndens = snow_new_fn(sfctmp, sn_new, snowh, sndens)
        sncond = csnow_fn(sndens)

    else:
        # precip is liquid (rain), hence save in the precip variable
        # that later can wholely or partially infiltrate the soil (along
        # with any canopy "drip" added to this later)
        prcp1 = prcp

    # determine snowcover fraction and albedo fraction over land.
    if ice != 0:
        sncovr = 1.0
        albedo = 0.65  # albedo over sea-ice, glacial- ice

    else:
        # non-glacial land
        # if snow depth=0, set snowcover fraction=0, albedo=snow free albedo.
        if sneqv == 0.0:
            sncovr = 0.0
            albedo = alb

        else:
            # determine snow fraction cover.
            # determine surface albedo modification due to snowdepth state.
            sncovr = snfrac_fn(sneqv, snup, physcons.SALP)
            albedo = alcalc_fn(alb, snoalb, sncovr)

    # thermal conductivity for sea-ice case, glacial-ice case
    if ice != 0:
        df1 = 2.2

    else:
        # calculate the subsurface heat flux, which first requires calculation
        # of the thermal diffusivity.
        df1 = tdfcnd_fn(smc0, quartz, smcmax, sh2o0)
        if ivegsrc == 1 and vegtyp == 12:
            df1 = 3.24 * (1.0 - shdfac) + shdfac * df1 * exp(physcons.SBETA * shdfac)
        else:
            df1 = df1 * exp(physcons.SBETA * shdfac)

    dsoil = -0.5 * zsoil0

    # df1 = 0.41081215623353906
    if sneqv == 0.0:
        ssoil = df1 * (t1 - stc0) / dsoil
    else:
        dtot = snowh + dsoil
        frcsno = snowh / dtot
        frcsoi = dsoil / dtot

        # arithmetic mean (parallel flow)
        df1a = frcsno * sncond + frcsoi * df1

        # geometric mean (intermediate between harmonic and arithmetic mean)
        df1 = df1a * sncovr + df1 * (1.0 - sncovr)

        # calculate subsurface heat flux
        ssoil = df1 * (t1 - stc0) / dtot

    # calc virtual temps and virtual potential temps needed by
    # subroutines sfcdif and penman.
    t2v = sfctmp * (1.0 + 0.61 * q2)

    # surface exchange coefficients computed externally and passed in,
    # hence subroutine sfcdif not called.
    fdown = swnet + lwdn

    # enhance cp as a function of z0 to mimic heat storage
    cpx = constants.CP_AIR
    cpx1 = physcons.CP1
    cpfac = 1.0

    # call penman subroutine to calculate potential evaporation (etp),
    # and other partial products and sums save in common/rite for later
    # calculations.

    t24, etp, rch, epsca, rr, flx2 = penman_fn(
        sfctmp,
        sfcprs,
        sfcems,
        ch,
        t2v,
        th2,
        prcp,
        fdown,
        cpx,
        cpfac,
        ssoil,
        q2,
        q2sat,
        dqsdt2,
        snowng,
        frzgra,
        ffrozp,
    )

    # etp = 1.712958945801106e-06
    # call canres to calculate the canopy resistance and convert it
    # into pc if nonzero greenness fraction

    rc = 0.0
    rcs = 0.0
    rct = 0.0
    rcq = 0.0
    rcsoil = 0.0
    runoff1 = 0.0
    runoff2 = 0.0
    runoff3 = 0.0
    snomlt = 0.0

    pc = 0.0

    if shdfac > 0.0:

        # frozen ground extension: total soil water "smc" was replaced
        # by unfrozen soil water "sh2o" in call to canres below
        rc, pc, rcs, rct, rcq, rcsoil = canres_fn(
            nroot,
            swdn,
            ch,
            q2,
            q2sat,
            dqsdt2,
            sfctmp,
            cpx1,
            sfcprs,
            sfcems,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            smcwlt,
            smcref,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            rsmin,
            physcons.RSMAX,
            physcons.TOPT,
            rgl,
            hs,
            xlai,
        )

    # now decide major pathway branch to take depending on whether
    # snowpack exists or not:
    esnow = 0.0

    if sneqv == 0.0:
        (
            cmc,
            t1,
            stc0,
            stc1,
            stc2,
            stc3,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            tbot,
            eta,
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
            beta,
            drip,
            dew,
            flx1,
            flx3,
        ) = nopac_fn(
            nroot,
            etp,
            prcp,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            physcons.CMCMAX,
            dt,
            shdfac,
            physcons.SBETA,
            sfctmp,
            sfcems,
            t24,
            th2,
            fdown,
            epsca,
            bexp,
            pc,
            rch,
            rr,
            physcons.CFACTR,
            slope,
            kdt,
            frzx,
            psisat,
            zsoil0,
            zsoil1,
            zsoil2,
            zsoil3,
            dksat,
            dwsat,
            physcons.ZBOT,
            ice,
            rtdis0,
            rtdis1,
            rtdis2,
            rtdis3,
            quartz,
            physcons.FXEXP,
            physcons.CSOIL,
            ivegsrc,
            vegtyp,
            cmc,
            t1,
            stc0,
            stc1,
            stc2,
            stc3,
            sh2o0,
            sh2o1,
            sh2o2,
            sh2o3,
            tbot,
            smc0,
            smc1,
            smc2,
            smc3,
        )

    else:
        (
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
        ) = snopac_fn(
            nroot,
            etp,
            prcp,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            physcons.CMCMAX,
            dt,
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
            physcons.CFACTR,
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
            physcons.ZBOT,
            shdfac,
            ice,
            rtdis0,
            rtdis1,
            rtdis2,
            rtdis3,
            quartz,
            physcons.FXEXP,
            physcons.CSOIL,
            flx2,
            snowng,
            ffrozp,
            ivegsrc,
            vegtyp,
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
        )

    # prepare sensible heat (h) for return to parent model
    sheat = -(ch * physcons.CP1 * sfcprs) / (physcons.RD1 * t2v) * (th2 - t1)

    # convert units and/or sign of total evap (eta), potential evap (etp),
    # subsurface heat flux (s), and runoffs for what parent model expects
    # convert eta from kg m-2 s-1 to w m-2
    edir = edir * physcons.LSUBC
    ec = ec * physcons.LSUBC
    et_0 = et_0 * physcons.LSUBC
    et_1 = et_1 * physcons.LSUBC
    et_2 = et_2 * physcons.LSUBC
    et_3 = et_3 * physcons.LSUBC

    ett = ett * physcons.LSUBC
    esnow = esnow * physcons.LSUBS
    etp = etp * ((1.0 - sncovr) * physcons.LSUBC + sncovr * physcons.LSUBS)

    # esnow = 0.0
    if etp > 0.0:
        eta = edir + ec + ett + esnow
    else:
        eta = etp

    beta = eta / etp

    # convert the sign of soil heat flux so that:
    # ssoil>0: warm the surface  (night time)
    # ssoil<0: cool the surface  (day time)
    ssoil = -1.0 * ssoil

    if ice == 0:
        # for the case of land (but not glacial-ice):
        # convert runoff3 (internal layer runoff from supersat) from m
        # to m s-1 and add to subsurface runoff/baseflow (runoff2).
        # runoff2 is already a rate at this point.
        runoff3 = runoff3 / dt
        runoff2 = runoff2 + runoff3

    else:
        # for the case of sea-ice (ice=1) or glacial-ice (ice=-1), add any
        # snowmelt directly to surface runoff (runoff1) since there is no
        # soil medium, and thus no call to subroutine smflx (for soil
        # moisture tendency).
        runoff1 = snomlt / dt

    # total column soil moisture in meters (soilm) and root-zone
    # soil moisture availability (fraction) relative to porosity/saturation
    soilm = (
        -smc0 * zsoil0
        + smc1 * (zsoil0 - zsoil1)
        + smc2 * (zsoil1 - zsoil2)
        + smc3 * (zsoil2 - zsoil3)
    )
    soilwm = -(smcmax - smcwlt) * zsoil3
    soilww = (
        -(smc0 - smcwlt) * zsoil0
        + (smc1 - smcwlt) * (zsoil0 - zsoil1)
        + (smc2 - smcwlt) * (zsoil1 - zsoil2)
        + (smc3 - smcwlt) * (zsoil2 - zsoil3)
    )
    soilw = soilww / soilwm

    return (
        tbot,
        cmc,
        t1,
        stc0,
        stc1,
        stc2,
        stc3,
        smc0,
        smc1,
        smc2,
        smc3,
        sh2o0,
        sh2o1,
        sh2o2,
        sh2o3,
        sneqv,
        ch,
        cm,
        z0,
        shdfac,
        snowh,
        nroot,
        albedo,
        eta,
        sheat,
        ec,
        edir,
        et_0,
        et_1,
        et_2,
        et_3,
        ett,
        esnow,
        drip,
        dew,
        beta,
        etp,
        ssoil,
        flx1,
        flx2,
        flx3,
        runoff1,
        runoff2,
        runoff3,
        snomlt,
        sncovr,
        rc,
        pc,
        rsmin,
        xlai,
        rcs,
        rct,
        rcq,
        rcsoil,
        soilw,
        soilm,
        smcwlt,
        smcdry,
        smcref,
        smcmax,
    )


def sfc_drv(
    ps: FloatFieldIJ,
    t1: FloatField,
    q1: FloatFieldTracer,
    soiltyp: IntFieldIJ,
    vegtype: IntFieldIJ,
    sigmaf: FloatFieldIJ,
    sfcemis: FloatFieldIJ,
    dlwflx: FloatFieldIJ,
    dswsfc: FloatFieldIJ,
    snet: FloatFieldIJ,
    tg3: FloatFieldIJ,
    cm: FloatFieldIJ,
    ch: FloatFieldIJ,
    prsl1: FloatFieldIJ,
    prslki: FloatFieldIJ,
    zf: FloatFieldIJ,
    land: BoolFieldIJ,
    wind: FloatFieldIJ,
    slopetyp: IntFieldIJ,
    snoalb: FloatFieldIJ,
    sfalb: FloatFieldIJ,
    flag_iter: BoolFieldIJ,
    flag_guess: BoolFieldIJ,
    bexppert: FloatFieldIJ,
    xlaipert: FloatFieldIJ,
    weasd: FloatFieldIJ,
    snwdph: FloatFieldIJ,
    tskin: FloatFieldIJ,
    tprcp: FloatFieldIJ,
    srflag: FloatFieldIJ,
    smc0: FloatFieldIJ,
    smc1: FloatFieldIJ,
    smc2: FloatFieldIJ,
    smc3: FloatFieldIJ,
    stc0: FloatFieldIJ,
    stc1: FloatFieldIJ,
    stc2: FloatFieldIJ,
    stc3: FloatFieldIJ,
    slc0: FloatFieldIJ,
    slc1: FloatFieldIJ,
    slc2: FloatFieldIJ,
    slc3: FloatFieldIJ,
    canopy: FloatFieldIJ,
    trans: FloatFieldIJ,
    tsurf: FloatFieldIJ,
    zorl: FloatFieldIJ,
    sncovr1: FloatFieldIJ,
    qsurf: FloatFieldIJ,
    gflux: FloatFieldIJ,
    drain: FloatFieldIJ,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    ep: FloatFieldIJ,
    runoff: FloatFieldIJ,
    cmm: FloatFieldIJ,
    chh: FloatFieldIJ,
    evbs: FloatFieldIJ,
    evcw: FloatFieldIJ,
    sbsno: FloatFieldIJ,
    snowc: FloatFieldIJ,
    stm: FloatFieldIJ,
    snohf: FloatFieldIJ,
    smcwlt2: FloatFieldIJ,
    smcref2: FloatFieldIJ,
    wet1: FloatFieldIJ,
    delt: float,
    lheatstrg: int,
    ivegsrc: int,
    bexp: FloatFieldIJ,
    dksat: FloatFieldIJ,
    dwsat: FloatFieldIJ,
    f1: FloatFieldIJ,
    psisat: FloatFieldIJ,
    quartz: FloatFieldIJ,
    smcdry: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    smcref: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    nroot: IntFieldIJ,
    snup: FloatFieldIJ,
    rsmin: FloatFieldIJ,
    rgl: FloatFieldIJ,
    hs: FloatFieldIJ,
    xlai: FloatFieldIJ,
    slope: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):

        # set constant parameters
        cpinv = 1.0 / constants.CP_AIR
        hvapi = 1.0 / constants.HLV
        elocp = physcons.HOCP
        rhoh2o = 1000.0
        a2 = 17.2693882
        a3 = 273.16
        a4 = 35.86
        a23m4 = a2 * (a3 - a4)
        zsoil_noah0 = -0.1
        zsoil_noah1 = -0.4
        zsoil_noah2 = -1.0
        zsoil_noah3 = -2.0

        # save land-related prognostic fields for guess run
        if land and flag_guess:
            weasd_old = weasd
            snwdph_old = snwdph
            tskin_old = tskin
            canopy_old = canopy
            tprcp_old = tprcp
            srflag_old = srflag
            smc_old0 = smc0
            smc_old1 = smc1
            smc_old2 = smc2
            smc_old3 = smc3
            stc_old0 = stc0
            stc_old1 = stc1
            stc_old2 = stc2
            stc_old3 = stc3
            slc_old0 = slc0
            slc_old1 = slc1
            slc_old2 = slc2
            slc_old3 = slc3

        if flag_iter and land:
            # initialization block
            ep = 0.0
            evap = 0.0
            hflx = 0.0
            gflux = 0.0
            drain = 0.0
            canopy = max(canopy, 0.0)

            evbs = 0.0
            evcw = 0.0
            trans = 0.0
            sbsno = 0.0
            snowc = 0.0
            snohf = 0.0

            q0 = max(q1[0, 0, 0][0], 1.0e-8)
            theta1 = t1 * prslki
            rho = prsl1 / (constants.RDGAS * t1 * (1.0 + constants.ZVIR * q0))
            qs1 = fpvs(t1)
            qs1 = max(constants.EPS * qs1 / (prsl1 + (constants.EPS - 1) * qs1), 1.0e-8)

            q0 = min(qs1, q0)

            # noah: prepare variables to run noah lsm
            # configuration information
            couple = 1
            ice = 0
            sldpth0 = -zsoil_noah0
            sldpth1 = zsoil_noah0 - zsoil_noah1
            sldpth2 = zsoil_noah1 - zsoil_noah2
            sldpth3 = zsoil_noah2 - zsoil_noah3

            # forcing data
            prcp = rhoh2o * tprcp / delt
            dqsdt2 = qs1 * a23m4 / (t1 - a4) ** 2

            # history variables
            cmc = canopy * 0.001
            snowh = snwdph * 0.001
            sneqv = weasd * 0.001

            if (sneqv != 0.0) and (snowh == 0.0):
                snowh = 10.0 * sneqv

            chx = ch * wind
            cmx = cm * wind
            chh = chx * rho
            cmm = cmx

            z0 = zorl / 100.0

            (
                tg3,
                cmc,
                tsurf,
                stc0,
                stc1,
                stc2,
                stc3,
                smc0,
                smc1,
                smc2,
                smc3,
                slc0,
                slc1,
                slc2,
                slc3,
                sneqv,
                chx,
                cmx,
                z0,
                shdfac,
                snowh,
                nroot,
                albedo,
                eta,
                sheat,
                ec,
                edir,
                et_0,
                et_1,
                et_2,
                et_3,
                ett,
                esnow,
                drip,
                dew,
                beta,
                etp,
                ssoil,
                flx1,
                flx2,
                flx3,
                runoff1,
                runoff2,
                runoff3,
                snomlt,
                sncovr,
                rc,
                pc,
                rsmin,
                xlai,
                rcs,
                rct,
                rcq,
                rcsoil,
                soilw,
                soilm,
                smcwlt,
                smcdry,
                smcref,
                smcmax,
            ) = sflx(  # inputs
                couple,
                ice,
                srflag,
                delt,
                sldpth0,
                sldpth1,
                sldpth2,
                sldpth3,
                dksat,
                smcmax,
                smcref,
                smcwlt,
                smcdry,
                bexp,
                xlai,
                snup,
                quartz,
                rgl,
                hs,
                slope,
                psisat,
                dwsat,
                nroot,
                rsmin,
                dswsfc,
                snet,
                dlwflx,
                sfcemis,
                prsl1,
                t1,
                prcp,
                q0,
                qs1,
                dqsdt2,
                theta1,
                ivegsrc,
                vegtype,
                soiltyp,
                slopetyp,
                sfalb,
                snoalb,
                bexppert,
                xlaipert,
                lheatstrg,
                tg3,
                cmc,
                tsurf,
                stc0,
                stc1,
                stc2,
                stc3,
                smc0,
                smc1,
                smc2,
                smc3,
                slc0,
                slc1,
                slc2,
                slc3,
                sneqv,
                chx,
                cmx,
                z0,
                sigmaf,
                snowh,
            )

            # output
            evap = eta
            hflx = sheat
            gflux = ssoil

            evbs = edir
            evcw = ec
            trans = ett
            sbsno = esnow
            snowc = sncovr
            stm = soilm * 1000.0
            snohf = flx1 + flx2 + flx3

            smcwlt2 = smcwlt
            smcref2 = smcref

            ep = etp
            wet1 = smc0 / smcmax

            runoff = runoff1 * 1000.0
            drain = runoff2 * 1000.0

            canopy = cmc * 1000.0
            snwdph = snowh * 1000.0
            weasd = sneqv * 1000.0
            sncovr1 = sncovr
            zorl = z0 * 100.0

            # compute qsurf
            rch = rho * constants.CP_AIR * ch * wind
            qsurf = q1[0, 0, 0][0] + evap / (elocp * rch)
            tem = 1.0 / rho
            hflx = hflx * tem * cpinv
            evap = evap * tem * hvapi

        # restore land-related prognostic fields for guess run
        if land and flag_guess:
            weasd = weasd_old
            snwdph = snwdph_old
            tskin = tskin_old
            canopy = canopy_old
            tprcp = tprcp_old
            srflag = srflag_old
            smc0 = smc_old0
            smc1 = smc_old1
            smc2 = smc_old2
            smc3 = smc_old3
            stc0 = stc_old0
            stc1 = stc_old1
            stc2 = stc_old2
            stc3 = stc_old3
            slc0 = slc_old0
            slc1 = slc_old1
            slc2 = slc_old2
            slc3 = slc_old3
        elif land:
            tskin = tsurf

def set_2d_fields(
    smc: FloatField,
    stc: FloatField,
    slc: FloatField,
    smc0: FloatFieldIJ,
    smc1: FloatFieldIJ,
    smc2: FloatFieldIJ,
    smc3: FloatFieldIJ,
    stc0: FloatFieldIJ,
    stc1: FloatFieldIJ,
    stc2: FloatFieldIJ,
    stc3: FloatFieldIJ,
    slc0: FloatFieldIJ,
    slc1: FloatFieldIJ,
    slc2: FloatFieldIJ,
    slc3: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        smc0 = smc[0, 0, 0]
        smc1 = smc[0, 0, 1]
        smc2 = smc[0, 0, 2]
        smc3 = smc[0, 0, 3]
        stc0 = stc[0, 0, 0]
        stc1 = stc[0, 0, 1]
        stc2 = stc[0, 0, 2]
        stc3 = stc[0, 0, 3]
        slc0 = slc[0, 0, 0]
        slc1 = slc[0, 0, 1]
        slc2 = slc[0, 0, 2]
        slc3 = slc[0, 0, 3]

def set_3d_fields(
    smc: FloatField,
    stc: FloatField,
    slc: FloatField,
    smc0: FloatFieldIJ,
    smc1: FloatFieldIJ,
    smc2: FloatFieldIJ,
    smc3: FloatFieldIJ,
    stc0: FloatFieldIJ,
    stc1: FloatFieldIJ,
    stc2: FloatFieldIJ,
    stc3: FloatFieldIJ,
    slc0: FloatFieldIJ,
    slc1: FloatFieldIJ,
    slc2: FloatFieldIJ,
    slc3: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        smc[0, 0, 0] = smc0
        stc[0, 0, 0] = stc0
        slc[0, 0, 0] = slc0
    with computation(FORWARD), interval(1, 2):
        smc[0, 0, 0] = smc1
        stc[0, 0, 0] = stc1
        slc[0, 0, 0] = slc1
    with computation(FORWARD), interval(2, 3):
        smc[0, 0, 0] = smc2
        stc[0, 0, 0] = stc2
        slc[0, 0, 0] = slc2
    with computation(FORWARD), interval(3, 4):
        smc[0, 0, 0] = smc3
        stc[0, 0, 0] = stc3
        slc[0, 0, 0] = slc3

class NoahLSM_2D:
    """
    Called sfc_drv in SHiELD
    The Noah LSM (Chen et al., 1996; Koren et al., 1999; Ek et al., 2003) is targeted
    for moderate complexity and good computational efficiency for numerical weather
    prediction and climate models. Thus, it omits subgrid surface tiling and uses a
    single-layer snowpack. The surface energy balance is solved via a Penman-based
    approximation for latent heat flux. The Noah model includes packages to simulate
    soil moisture, soil ice, soil temperature, skin temperature, snow depth, snow water
    equivalent, energy fluxes such as latent heat, sensible heat and ground heat, and
    water fluxes such as evaporation and total runoff. The Noah surface infiltration
    scheme follows that of Schaake et al. (1996) for its treatment of the subgrid
    variability of precipitation and soil moisture.
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: LSMConfig,
        land_data: ndarray,
        veg_data: ndarray,
        soil_data: ndarray,
        slope_data: ndarray,
        vegfrac_data: ndarray,
        dt: Float,
    ):
        assert (
            config.pertvegf[0] < 0
        ), f"pertvegf[0] > 0 not implemented, got {config.pertvegf[0]}"
        assert config.ivegsrc == 1, f"ivegsrc !=1 not implemented, got {config.ivegsrc}"
        assert config.isot == 1, f"isot != 1 not implemented, got {config.isot}"
        self._delt = dt
        self._ivegsrc = config.ivegsrc
        self._isot = config.isot
        self._lheatstrg = config.lheatstrg

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

        grid_indexing = stencil_factory.grid_indexing

        domain = grid_indexing.domain
        domain2D = (domain[0], domain[1], 1)

        (
            nroot,
            zroot,
            sldpth,
            zsoil,
            slope,
            snup,
            rsmin,
            rgl,
            hs,
            xlai,
            bexp,
            dksat,
            dwsat,
            f1,
            kdt,
            psisat,
            quartz,
            smcdry,
            smcmax,
            smcref,
            smcwlt,
            shdfac,
            frzx,
            rtdis,
            land,
            ice,
        ) = set_soil_veg(land_data, veg_data, soil_data, vegfrac_data, slope_data)

        self._vegtype = quantity_factory.from_array(
            veg_data, dims=[X_DIM, Y_DIM], units="",
        )
        self._soiltype = quantity_factory.from_array(
            soil_data, dims=[X_DIM, Y_DIM], units="",
        )
        self._slopetype = quantity_factory.from_array(
            slope_data, dims=[X_DIM, Y_DIM], units="",
        )
        self._land = quantity_factory.from_array(
            land, dims=[X_DIM, Y_DIM], units="",
        )
        self._ice = quantity_factory.from_array(
            ice, dims=[X_DIM, Y_DIM], units="",
        )
        self._nroot = quantity_factory.from_array(nroot, dims=[X_DIM, Y_DIM], units="")
        self._zroot = quantity_factory.from_array(zroot, dims=[X_DIM, Y_DIM], units="")
        self._sldpth = quantity_factory.from_array(
            sldpth, dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM], units="m"
        )
        self._snup = quantity_factory.from_array(snup, dims=[X_DIM, Y_DIM], units="")
        self._rsmin = quantity_factory.from_array(rsmin, dims=[X_DIM, Y_DIM], units="")
        self._rgl = quantity_factory.from_array(rgl, dims=[X_DIM, Y_DIM], units="")
        self._hs = quantity_factory.from_array(hs, dims=[X_DIM, Y_DIM], units="")
        self._xlai = quantity_factory.from_array(xlai, dims=[X_DIM, Y_DIM], units="")
        self._bexp = quantity_factory.from_array(bexp, dims=[X_DIM, Y_DIM], units="")
        self._dksat = quantity_factory.from_array(dksat, dims=[X_DIM, Y_DIM], units="")
        self._dwsat = quantity_factory.from_array(dwsat, dims=[X_DIM, Y_DIM], units="")
        self._f1 = quantity_factory.from_array(f1, dims=[X_DIM, Y_DIM], units="")
        self._kdt = quantity_factory.from_array(kdt, dims=[X_DIM, Y_DIM], units="")
        self._psisat = quantity_factory.from_array(
            psisat, dims=[X_DIM, Y_DIM], units=""
        )
        self._quartz = quantity_factory.from_array(
            quartz, dims=[X_DIM, Y_DIM], units=""
        )
        self._smcdry = quantity_factory.from_array(
            smcdry, dims=[X_DIM, Y_DIM], units=""
        )
        self._smcmax = quantity_factory.from_array(
            smcmax, dims=[X_DIM, Y_DIM], units=""
        )
        self._smcref = quantity_factory.from_array(
            smcref, dims=[X_DIM, Y_DIM], units=""
        )
        self._smcwlt = quantity_factory.from_array(
            smcwlt, dims=[X_DIM, Y_DIM], units=""
        )
        self._shdfac = quantity_factory.from_array(
            shdfac, dims=[X_DIM, Y_DIM], units=""
        )
        self._frzx = quantity_factory.from_array(frzx, dims=[X_DIM, Y_DIM], units="")
        self._rtdis = quantity_factory.from_array(
            rtdis, dims=[X_DIM, Y_DIM, Z_DIM], units=""
        )
        self._zsoil = quantity_factory.from_array(
            zsoil,
            dims=[Z_DIM],
            units="",
        )
        self._slope = quantity_factory.from_array(slope, dims=[X_DIM, Y_DIM], units="")

        self._smc0 = make_quantity_2d()
        self._smc1 = make_quantity_2d()
        self._smc2 = make_quantity_2d()
        self._smc3 = make_quantity_2d()
        self._stc0 = make_quantity_2d()
        self._stc1 = make_quantity_2d()
        self._stc2 = make_quantity_2d()
        self._stc3 = make_quantity_2d()
        self._slc0 = make_quantity_2d()
        self._slc1 = make_quantity_2d()
        self._slc2 = make_quantity_2d()
        self._slc3 = make_quantity_2d()

        self._sfc_drv = stencil_factory.from_origin_domain(
            func=sfc_drv,
            origin=grid_indexing.origin_compute(),
            domain=domain2D,
        )

        self._set_2d_fields = stencil_factory.from_origin_domain(
            func=set_2d_fields,
            origin=grid_indexing.origin_compute(),
            domain=domain2D,
        )

        self._set_3d_fields = stencil_factory.from_origin_domain(
            func=set_3d_fields,
            origin=grid_indexing.origin_compute(),
            domain=domain,
        )

    def __call__(
        self,
        ps: FloatFieldIJ,
        t1: FloatField,
        q1: FloatFieldTracer,
        sfcemis: FloatFieldIJ,
        dlwflx: FloatFieldIJ,
        dswsfc: FloatFieldIJ,
        snet: FloatFieldIJ,
        tg3: FloatFieldIJ,
        cm: FloatFieldIJ,
        ch: FloatFieldIJ,
        prsl1: FloatFieldIJ,
        prslki: FloatFieldIJ,
        zf: FloatFieldIJ,
        wind: FloatFieldIJ,
        snoalb: FloatFieldIJ,
        sfalb: FloatFieldIJ,
        flag_iter: BoolFieldIJ,
        flag_guess: BoolFieldIJ,
        bexppert: FloatFieldIJ,
        xlaipert: FloatFieldIJ,
        weasd: FloatFieldIJ,
        snwdph: FloatFieldIJ,
        tskin: FloatFieldIJ,
        tprcp: FloatFieldIJ,
        srflag: FloatFieldIJ,
        smc: FloatField,
        stc: FloatField,
        slc: FloatField,
        canopy: FloatFieldIJ,
        trans: FloatFieldIJ,
        tsurf: FloatFieldIJ,
        z0rl: FloatFieldIJ,
        sncovr1: FloatFieldIJ,
        qsurf: FloatFieldIJ,
        gflux: FloatFieldIJ,
        drain: FloatFieldIJ,
        evap: FloatFieldIJ,
        hflx: FloatFieldIJ,
        ep: FloatFieldIJ,
        runoff: FloatFieldIJ,
        cmm: FloatFieldIJ,
        chh: FloatFieldIJ,
        evbs: FloatFieldIJ,
        evcw: FloatFieldIJ,
        sbsno: FloatFieldIJ,
        snowc: FloatFieldIJ,
        stm: FloatFieldIJ,
        snohf: FloatFieldIJ,
        smcwlt2: FloatFieldIJ,
        smcref2: FloatFieldIJ,
        wet1: FloatFieldIJ,
    ):
        """
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs:                                                       size   !
        !     ps       - real, surface pressure (pa)                       im   !
        !     t1       - real, surface layer mean temperature (k)          im   !
        !     q1       - real, surface layer mean specific humidity        im   !
        !     sfcemis  - real, sfc lw emissivity ( fraction )              im   !
        !     dlwflx   - real, total sky sfc downward lw flux ( w/m**2 )   im   !
        !     dswflx   - real, total sky sfc downward sw flux ( w/m**2 )   im   !
        !     snet     - real, total sky sfc netsw flx into ground(w/m**2) im   !
        !     delt     - real, time interval (second)                      1    !
        !     tg3      - real, deep soil temperature (k)                   im   !
        !     cm       - real, surface exchange coeff for momentum (m/s)   im   !
        !     ch       - real, surface exchange coeff heat & moisture(m/s) im   !
        !     prsl1    - real, sfc layer 1 mean pressure (pa)              im   !
        !     prslki   - real,                                             im   !
        !     zf       - real, height of bottom layer (m)                  im   !
        !     wind     - real, wind speed (m/s)                            im   !
        !     shdmin   - real, min fractional coverage of green veg        im   !
        !     shdmax   - real, max fractnl cover of green veg (not used)   im   !
        !     snoalb   - real, upper bound on max albedo over deep snow    im   !
        !     sfalb    - real, mean sfc diffused sw albedo (fractional)    im   !
        !     flag_iter- logical,                                          im   !
        !     flag_guess-logical,                                          im   !
        !                                                                       !
        !  input/outputs:                                                       !
        !     weasd    - real, water equivalent accumulated snow depth (mm) im  !
        !     snwdph   - real, snow depth (water equiv) over land          im   !
        !     tskin    - real, ground surface skin temperature ( k )       im   !
        !     tprcp    - real, total precipitation                         im   !
        !     srflag   - real, snow/rain flag for precipitation            im   !
        !     smc      - real, total soil moisture content (fractional)   im,km !
        !     stc      - real, soil temp (k)                              im,km !
        !     slc      - real, liquid soil moisture                       im,km !
        !     canopy   - real, canopy moisture content (m)                 im   !
        !     trans    - real, total plant transpiration (m/s)             im   !
        !     tsurf    - real, surface skin temperature (after iteration)  im   !
        !     z0rl     - real, surface roughness                           im   !
        !                                                                       !
        !  outputs:                                                             !
        !     sncovr1  - real, snow cover over land (fractional)           im   !
        !     qsurf    - real, specific humidity at sfc                    im   !
        !     gflux    - real, soil heat flux (w/m**2)                     im   !
        !     drain    - real, subsurface runoff (mm/s)                    im   !
        !     evap     - real, evaperation from latent heat flux           im   !
        !     hflx     - real, sensible heat flux                          im   !
        !     ep       - real, potential evaporation                       im   !
        !     runoff   - real, surface runoff (m/s)                        im   !
        !     cmm      - real,                                             im   !
        !     chh      - real,                                             im   !
        !     evbs     - real, direct soil evaporation (m/s)               im   !
        !     evcw     - real, canopy water evaporation (m/s)              im   !
        !     sbsno    - real, sublimation/deposit from snopack (m/s)      im   !
        !     snowc    - real, fractional snow cover                       im   !
        !     stm      - real, total soil column moisture content (m)      im   !
        !     snohf    - real, snow/freezing-rain latent heat flux (w/m**2)im   !
        !     smcwlt2  - real, dry soil moisture threshold                 im   !
        !     smcref2  - real, soil moisture threshold                     im   !
        !     wet1     - real, normalized soil wetness                     im   !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """

        self._set_2d_fields(
            smc,
            stc,
            slc,
            self._smc0,
            self._smc1,
            self._smc2,
            self._smc3,
            self._stc0,
            self._stc1,
            self._stc2,
            self._stc3,
            self._slc0,
            self._slc1,
            self._slc2,
            self._slc3,
        )

        self._sfc_drv(
            ps,
            t1,
            q1,
            self._soiltype,
            self._vegtype,
            self._shdfac,
            sfcemis,
            dlwflx,
            dswsfc,
            snet,
            tg3,
            cm,
            ch,
            prsl1,
            prslki,
            zf,
            self._land,
            wind,
            self._slopetype,
            snoalb,
            sfalb,
            flag_iter,
            flag_guess,
            bexppert,
            xlaipert,
            weasd,
            snwdph,
            tskin,
            tprcp,
            srflag,
            self._smc0,
            self._smc1,
            self._smc2,
            self._smc3,
            self._stc0,
            self._stc1,
            self._stc2,
            self._stc3,
            self._slc0,
            self._slc1,
            self._slc2,
            self._slc3,
            canopy,
            trans,
            tsurf,
            z0rl,
            sncovr1,
            qsurf,
            gflux,
            drain,
            evap,
            hflx,
            ep,
            runoff,
            cmm,
            chh,
            evbs,
            evcw,
            sbsno,
            snowc,
            stm,
            snohf,
            smcwlt2,
            smcref2,
            wet1,
            self._delt,
            self._lheatstrg,
            self._ivegsrc,
            self._bexp,
            self._dksat,
            self._dwsat,
            self._f1,
            self._psisat,
            self._quartz,
            self._smcdry,
            self._smcmax,
            self._smcref,
            self._smcwlt,
            self._nroot,
            self._snup,
            self._rsmin,
            self._rgl,
            self._hs,
            self._xlai,
            self._slope,
        )

        self._set_3d_fields(
            smc,
            stc,
            slc,
            self._smc0,
            self._smc1,
            self._smc2,
            self._smc3,
            self._stc0,
            self._stc1,
            self._stc2,
            self._stc3,
            self._slc0,
            self._slc1,
            self._slc2,
            self._slc3,
        )
