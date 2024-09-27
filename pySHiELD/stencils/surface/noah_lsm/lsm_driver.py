from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval
from numpy import ndarray, zeros

import ndsl.constants as constants
import pySHiELD.constants as physcons
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
    IntFieldIJ,
    IntFieldK,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from pySHiELD._config import LSMConfig, FloatFieldTracer
from pySHiELD.functions.physics_functions import fpvs
from pySHiELD.stencils.surface.noah_lsm.nopac import NOPAC
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg
from pySHiELD.stencils.surface.noah_lsm.shflx import tdfcnd
from pySHiELD.stencils.surface.noah_lsm.snopac import SNOPAC


@gtscript.function
def snowz0(sncovr, z0):
    """
    calculates total roughness length over snow
    """
    # z0s = 0.001
    z0s = z0
    z0 = (1.0 - sncovr) * z0 + sncovr * z0s
    return z0


@gtscript.function
def alcalc(alb, snoalb, sncovr):
    # --- ... subprograms called: none

    # calculates albedo using snow effect
    # snoalb: max albedo over deep snow
    albedo = alb + sncovr * (snoalb - alb)
    if albedo > snoalb:
        albedo = snoalb

    return albedo


def canres(
    nroot: IntFieldIJ,
    swdn: FloatFieldIJ,
    ch: FloatFieldIJ,
    q2: FloatFieldIJ,
    q2sat: FloatFieldIJ,
    dqsdt2: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    sfcprs: FloatFieldIJ,
    sfcems: FloatFieldIJ,
    sh2o: FloatField,
    smcwlt: FloatFieldIJ,
    smcref: FloatFieldIJ,
    zsoil: FloatFieldK,
    zroot: FloatFieldIJ,
    rsmin: FloatFieldIJ,
    rgl: FloatFieldIJ,
    hs: FloatFieldIJ,
    xlai: FloatFieldIJ,
    k_mask: IntFieldK,
    shdfac: FloatFieldIJ,
    rc: FloatFieldIJ,
    pc: FloatFieldIJ,
    rcs: FloatFieldIJ,
    rct: FloatFieldIJ,
    rcq: FloatFieldIJ,
    rcsoil: FloatFieldIJ,
    lsm_mask: BoolFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            if lsm_mask:
                if shdfac > 0.0:
                    # calculates canopy resistance

                    # contribution due to incoming solar radiation
                    ff = 0.55 * 2.0 * swdn / (rgl * xlai)
                    rcs = (ff + rsmin / physcons.RSMAX) / (1.0 + ff)
                    rcs = max(rcs, 0.0001)

                    # contribution due to air temperature
                    # at first model level above ground
                    rct = 1.0 - 0.0016 * (physcons.TOPT - sfctmp) ** 2.0
                    rct = max(rct, 0.0001)

                    # contribution due to vapor pressure deficit at first model level.
                    rcq = 1.0 / (1.0 + hs * (q2sat - q2))
                    rcq = max(rcq, 0.01)

                    # contribution due to soil moisture availability.
                    rcsoil = 0.0

                    # use soil depth as weighting factor
                    gx = 0.0
                    if k_mask < nroot:
                        gx = max(0.0, min(1.0, (sh2o - smcwlt) / (smcref - smcwlt)))
                        rcsoil = rcsoil + (zsoil / zroot * gx)
        with interval(1, None):
            if lsm_mask:
                if shdfac > 0.0:
                    if k_mask < nroot:
                        gx = max(0.0, min(1.0, (sh2o - smcwlt) / (smcref - smcwlt)))
                        rcsoil = rcsoil + ((zsoil - zsoil[-1]) / zroot * gx)
    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            if shdfac > 0.0:
                rcsoil = max(rcsoil, 0.0001)

                # determine canopy resistance due to all factors

                rc = rsmin / (xlai * rcs * rct * rcq * rcsoil)
                rr = (4.0 * sfcems * physcons.SIGMA1 * physcons.RD1 / physcons.CP1) * (
                    sfctmp ** 4.0
                ) / (sfcprs * ch) + 1.0
                delta = (physcons.LSUBC / physcons.CP1) * dqsdt2

                pc = (rr + delta) / (rr * (1.0 + rc * ch) + delta)


@gtscript.function
def csnow(sndens):
    # --- ... subprograms called: none

    unit = 0.11631

    c = 0.328 * 10 ** (2.25 * sndens)
    sncond = unit * c

    return sncond


@gtscript.function
def penman(
    sfctmp,
    sfcprs,
    sfcems,
    ch,
    t2v,
    th2,
    prcp,
    fdown,
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
    delta = physcons.ELCP * dqsdt2
    t24 = sfctmp * sfctmp * sfctmp * sfctmp
    rr = t24 * 6.48e-8 / (sfcprs * ch) + 1.0
    rho = sfcprs / (physcons.RD1 * t2v)
    rch = rho * constants.CP_AIR * ch

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
    a = physcons.ELCP * (q2sat - q2)

    epsca = (a * rr + rad * delta) / (delta + rr)
    etp = epsca * rch / physcons.LSUBC

    return t24, etp, rch, epsca, rr, flx2


@gtscript.function
def snfrac(sneqv, snup, salp):
    # --- ... subprograms called: none

    # determine snow fraction cover.
    if sneqv < snup:
        rsnow = sneqv / snup
        sncovr = 1.0 - (exp(-salp * rsnow) - rsnow * exp(-salp))
    else:
        sncovr = 1.0

    return sncovr


@gtscript.function
def snow_new(sfctmp, sn_new, snowh, sndens):
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


def init_lsm(
    smc: FloatField,
    stc: FloatField,
    slc: FloatField,
    weasd: FloatFieldIJ,
    snwdph: FloatFieldIJ,
    tskin: FloatFieldIJ,
    canopy: FloatFieldIJ,
    tprcp: FloatFieldIJ,
    srflag: FloatFieldIJ,
    q1: FloatFieldTracer,
    t1: FloatField,
    prslki: FloatFieldIJ,
    prsl1: FloatFieldIJ,
    ch: FloatFieldIJ,
    cm: FloatFieldIJ,
    wind: FloatFieldIJ,
    zorl: FloatFieldIJ,
    smc_old: FloatField,
    stc_old: FloatField,
    slc_old: FloatField,
    weasd_old: FloatFieldIJ,
    snwdph_old: FloatFieldIJ,
    tskin_old: FloatFieldIJ,
    canopy_old: FloatFieldIJ,
    tprcp_old: FloatFieldIJ,
    srflag_old: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    ep: FloatFieldIJ,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    gflux: FloatFieldIJ,
    drain: FloatFieldIJ,
    evbs: FloatFieldIJ,
    evcw: FloatFieldIJ,
    trans: FloatFieldIJ,
    sbsno: FloatFieldIJ,
    snowc: FloatFieldIJ,
    snohf: FloatFieldIJ,
    q0: FloatFieldIJ,
    qsi: FloatFieldIJ,
    dqsdt2: FloatFieldIJ,
    theta1: FloatFieldIJ,
    cmc: FloatFieldIJ,
    chh: FloatFieldIJ,
    cmm: FloatFieldIJ,
    z0: FloatFieldIJ,
    prcp: FloatFieldIJ,
    snowh: FloatFieldIJ,
    sneqv: FloatFieldIJ,
    rho: FloatFieldIJ,
    ice: IntFieldIJ,
    land: BoolFieldIJ,
    flag_guess: BoolFieldIJ,
    flag_iter: BoolFieldIJ,
    lsm_mask: BoolFieldIJ,
):
    from __externals__ import dt

    # save land-related prognostic fields for guess run
    with computation(PARALLEL), interval(...):
        if land and flag_guess:
            smc_old = smc
            stc_old = stc
            slc_old = slc

    with computation(FORWARD), interval(0, 1):
        if land and flag_guess:
            weasd_old = weasd
            snwdph_old = snwdph
            tskin_old = tskin
            canopy_old = canopy
            tprcp_old = tprcp
            srflag_old = srflag

        if flag_iter and land:
            lsm_mask = True

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

            # forcing data
            prcp = constants.RHO_H2O * tprcp / dt
            dqsdt2 = qs1 * physcons.A23M4 / (t1 - physcons.A4) ** 2

            # history variables
            cmc = canopy * 0.001
            snowh = snwdph * 0.001
            sneqv = weasd * 0.001
            sfctmp = t1

            if (sneqv != 0.0) and (snowh == 0.0):
                snowh = 10.0 * sneqv

            chx = ch * wind
            cmx = cm * wind
            chh = chx * rho
            cmm = cmx

            z0 = zorl / 100.0


def sflx_1(
    ice: IntFieldIJ,
    ffrozp: FloatFieldIJ,
    zsoil: FloatFieldK,
    swdn: FloatFieldIJ,
    swnet: FloatFieldIJ,
    lwdn: FloatFieldIJ,
    sfcems: FloatFieldIJ,
    sfcprs: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    prcp: FloatFieldIJ,
    q2: FloatFieldIJ,
    q2sat: FloatFieldIJ,
    dqsdt2: FloatFieldIJ,
    th2: FloatFieldIJ,
    vegtype: IntFieldIJ,
    soiltype: IntFieldIJ,
    slopetype: IntFieldIJ,
    alb: FloatFieldIJ,
    snoalb: FloatFieldIJ,
    bexpp: FloatFieldIJ,
    xlaip: FloatFieldIJ,
    cmc: FloatFieldIJ,
    t1: FloatFieldIJ,
    stc: FloatField,
    smc: FloatField,
    sh2o: FloatField,
    sneqv: FloatFieldIJ,
    sndens: FloatFieldIJ,
    ch: FloatFieldIJ,
    cm: FloatFieldIJ,
    z0: FloatFieldIJ,
    snowh: FloatFieldIJ,
    nroot: FloatFieldIJ,
    zroot: FloatFieldIJ,
    sldpth: FloatField,
    snup: FloatFieldIJ,
    rsmin: FloatFieldIJ,
    rgl: FloatFieldIJ,
    hs: FloatFieldIJ,
    xlai: FloatFieldIJ,
    bexp: FloatFieldIJ,
    dksat: FloatFieldIJ,
    dwsat: FloatFieldIJ,
    f1: FloatFieldIJ,
    kdt: FloatFieldIJ,
    psisat: FloatFieldIJ,
    quartz: FloatFieldIJ,
    smcdry: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    smcref: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    shdfac: FloatFieldIJ,
    frzx: FloatFieldIJ,
    rtdis: FloatFieldIJ,
    df1: FloatFieldIJ,
    t24: FloatFieldIJ,
    fdown: FloatFieldIJ,
    epsca: FloatFieldIJ,
    etp: FloatFieldIJ,
    rr: FloatFieldIJ,
    rch: FloatFieldIJ,
    flx2: FloatFieldIJ,
    prcp1: FloatFieldIJ,
    rc: FloatFieldIJ,
    rcs: FloatFieldIJ,
    rct: FloatFieldIJ,
    rcq: FloatFieldIJ,
    rcsoil: FloatFieldIJ,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    runoff3: FloatFieldIJ,
    snomlt: FloatFieldIJ,
    pc: FloatFieldIJ,
    esnow: FloatFieldIJ,
    t2v: FloatFieldIJ,
    snowng: BoolFieldIJ,
    lsm_mask: BoolFieldIJ,
    snopac_mask: BoolFieldIJ,
    nopac_mask: BoolFieldIJ,
):
    """
    Fortran docs:
    ! ===================================================================== !
    !  description:                                                         !
    !                                                                       !
    !  subroutine sflx - version 2.7:                                       !
    !  sub-driver for "noah/osu lsm" family of physics subroutines for a    !
    !  soil/veg/snowpack land-surface model to update soil moisture, soil   !
    !  ice, soil temperature, skin temperature, snowpack water content,     !
    !  snowdepth, and all terms of the surface energy balance and surface   !
    !  water balance (excluding input atmospheric forcings of downward      !
    !  radiation and precip)                                                !
    !                                                                       !
    !  usage:                                                               !
    !                                                                       !
    !      call sflx                                                        !
    !  ---  inputs:                                                         !
    !          ( nsoil, couple, icein, ffrozp, dt, zlvl, sldpth,            !
    !            swdn, swnet, lwdn, sfcems, sfcprs, sfctmp,                 !
    !            sfcspd, prcp, q2, q2sat, dqsdt2, th2,ivegsrc,              !
    !            vegtyp, soiltyp, slopetyp, shdmin, alb, snoalb,            !
    !  ---  input/outputs:                                                  !
    !            tbot, cmc, t1, stc, smc, sh2o, sneqv, ch, cm,              !
    !  ---  outputs:                                                        !
    !            nroot, shdfac, snowh, albedo, eta, sheat, ec,              !
    !            edir, et, ett, esnow, drip, dew, beta, etp, ssoil,         !
    !            flx1, flx2, flx3, runoff1, runoff2, runoff3,               !
    !            snomlt, sncovr, rc, pc, rsmin, xlai, rcs, rct, rcq,        !
    !            rcsoil, soilw, soilm, smcwlt, smcdry, smcref, smcmax )     !
    !                                                                       !
    !                                                                       !
    !  subprograms called:  redprm, snow_new, csnow, snfrac, alcalc,        !
    !            tdfcnd, snowz0, sfcdif, penman, canres, nopac, snopac.     !
    !                                                                       !
    !                                                                       !
    !  program history log:                                                 !
    !    jun  2003  -- k. mitchell et. al -- created version 2.7            !
    !         200x  -- sarah lu    modified the code including:             !
    !                       added passing argument, couple; replaced soldn  !
    !                       and solnet by radflx; call sfcdif if couple=0;  !
    !                       apply time filter to stc and tskin; and the     !
    !                       way of namelist inport.                         !
    !    feb  2004 -- m. ek noah v2.7.1 non-linear weighting of snow vs     !
    !                       non-snow covered portions of gridbox            !
    !    apr  2009  -- y.-t. hou   added lw surface emissivity effect,      !
    !                       streamlined and reformatted the code, and       !
    !                       consolidated constents/parameters by using      !
    !                       module physcons, and added program documentation!
    !    sep  2009 -- s. moorthi minor fixes                                !
    !                                                                       !
    !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs:                                                       size   !
    !     nsoil    - integer, number of soil layers (>=2 but <=nsold)  1    !
    !     couple   - integer, =0:uncoupled (land model only)           1    !
    !                         =1:coupled with parent atmos model            !
    !     icein    - integer, sea-ice flag (=1: sea-ice, =0: land)     1    !
    !     ffrozp   - real, fractional snow/rain                        1    !
    !     dt       - real, time step (<3600 sec)                       1    !
    !     zlvl     - real, height abv atmos ground forcing vars (m)    1    !
    !     sldpth   - real, thickness of each soil layer (m)          nsoil  !
    !     swdn     - real, downward sw radiation flux (w/m**2)         1    !
    !     swnet    - real, downward sw net (dn-up) flux (w/m**2)       1    !
    !     lwdn     - real, downward lw radiation flux (w/m**2)         1    !
    !     sfcems   - real, sfc lw emissivity (fractional)              1    !
    !     sfcprs   - real, pressure at height zlvl abv ground(pascals) 1    !
    !     sfctmp   - real, air temp at height zlvl abv ground (k)      1    !
    !     sfcspd   - real, wind speed at height zlvl abv ground (m/s)  1    !
    !     prcp     - real, precip rate (kg m-2 s-1)                    1    !
    !     q2       - real, mixing ratio at hght zlvl abv grnd (kg/kg)  1    !
    !     q2sat    - real, sat mixing ratio at zlvl abv grnd (kg/kg)   1    !
    !     dqsdt2   - real, slope of sat specific humidity curve at     1    !
    !                      t=sfctmp (kg kg-1 k-1)                           !
    !     th2      - real, air potential temp at zlvl abv grnd (k)     1    !
    !     ivegsrc  - integer, sfc veg type data source umd or igbp          !
    !     vegtyp   - integer, vegetation type (integer index)          1    !
    !     soiltyp  - integer, soil type (integer index)                1    !
    !     slopetyp - integer, class of sfc slope (integer index)       1    !
    !     shdmin   - real, min areal coverage of green veg (fraction)  1    !
    !     alb      - real, bkground snow-free sfc albedo (fraction)    1    !
    !     snoalb   - real, max albedo over deep snow     (fraction)    1    !
    !     lheatstrg- logical, flag for canopy heat storage             1    !
    !                         parameterization                              !
    !                                                                       !
    !  input/outputs:                                                       !
    !     tbot     - real, bottom soil temp (k)                        1    !
    !                      (local yearly-mean sfc air temp)                 !
    !     cmc      - real, canopy moisture content (m)                 1    !
    !     t1       - real, ground/canopy/snowpack eff skin temp (k)    1    !
    !     stc      - real, soil temp (k)                             nsoil  !
    !     smc      - real, total soil moisture (vol fraction)        nsoil  !
    !     sh2o     - real, unfrozen soil moisture (vol fraction)     nsoil  !
    !                      note: frozen part = smc-sh2o                     !
    !     sneqv    - real, water-equivalent snow depth (m)             1    !
    !                      note: snow density = snwqv/snowh                 !
    !     ch       - real, sfc exchange coeff for heat & moisture (m/s)1    !
    !                      note: conductance since it's been mult by wind   !
    !     cm       - real, sfc exchange coeff for momentum (m/s)       1    !
    !                      note: conductance since it's been mult by wind   !
    !                                                                       !
    !  outputs:                                                             !
    !     nroot    - integer, number of root layers                    1    !
    !     shdfac   - real, aeral coverage of green veg (fraction)      1    !
    !     snowh    - real, snow depth (m)                              1    !
    !     albedo   - real, sfc albedo incl snow effect (fraction)      1    !
    !     eta      - real, downward latent heat flux (w/m2)            1    !
    !     sheat    - real, downward sensible heat flux (w/m2)          1    !
    !     ec       - real, canopy water evaporation (w/m2)             1    !
    !     edir     - real, direct soil evaporation (w/m2)              1    !
    !     et       - real, plant transpiration     (w/m2)            nsoil  !
    !     ett      - real, total plant transpiration (w/m2)            1    !
    !     esnow    - real, sublimation from snowpack (w/m2)            1    !
    !     drip     - real, through-fall of precip and/or dew in excess 1    !
    !                      of canopy water-holding capacity (m)             !
    !     dew      - real, dewfall (or frostfall for t<273.15) (m)     1    !
    !     beta     - real, ratio of actual/potential evap              1    !
    !     etp      - real, potential evaporation (w/m2)                1    !
    !     ssoil    - real, upward soil heat flux (w/m2)                1    !
    !     flx1     - real, precip-snow sfc flux  (w/m2)                1    !
    !     flx2     - real, freezing rain latent heat flux (w/m2)       1    !
    !     flx3     - real, phase-change heat flux from snowmelt (w/m2) 1    !
    !     snomlt   - real, snow melt (m) (water equivalent)            1    !
    !     sncovr   - real, fractional snow cover                       1    !
    !     runoff1  - real, surface runoff (m/s) not infiltrating sfc   1    !
    !     runoff2  - real, sub sfc runoff (m/s) (baseflow)             1    !
    !     runoff3  - real, excess of porosity for a given soil layer   1    !
    !     rc       - real, canopy resistance (s/m)                     1    !
    !     pc       - real, plant coeff (fraction) where pc*etp=transpi 1    !
    !     rsmin    - real, minimum canopy resistance (s/m)             1    !
    !     xlai     - real, leaf area index  (dimensionless)            1    !
    !     rcs      - real, incoming solar rc factor  (dimensionless)   1    !
    !     rct      - real, air temp rc factor        (dimensionless)   1    !
    !     rcq      - real, atoms vapor press deficit rc factor         1    !
    !     rcsoil   - real, soil moisture rc factor   (dimensionless)   1    !
    !     soilw    - real, available soil mois in root zone            1    !
    !     soilm    - real, total soil column mois (frozen+unfrozen) (m)1    !
    !     smcwlt   - real, wilting point (volumetric)                  1    !
    !     smcdry   - real, dry soil mois threshold (volumetric)        1    !
    !     smcref   - real, soil mois threshold     (volumetric)        1    !
    !     smcmax   - real, porosity (sat val of soil mois)             1    !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """
    from __externals__ import dt, ivegsrc, lheatstrg

    with computation(FORWARD), interval(0, 1):
        if lsm_mask:

            # initialization
            shdfac0 = shdfac

            if ivegsrc == 2 and vegtype == 13:
                ice = -1
                shdfac = 0.0

            if ivegsrc == 1 and vegtype == 15:
                ice = -1
                shdfac = 0.0

            if ivegsrc == 1 and vegtype == 13:
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
            if (ice == 1) and (sneqv < 0.01):
                sneqv = 0.01
                snowh = 0.10

            if (ice == -1) and (sneqv < 0.10):
                # TODO: check if it is called
                sneqv = 0.10
                snowh = 1.00

    with computation(PARALLEL), interval(...):
        if lsm_mask:
            # for sea-ice and glacial-ice cases, set smc and sh2o values = 1
            # as a flag for non-soil medium
            if ice != 0:
                smc = 1.0
                sh2o = 1.0

    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            # if input snowpack is nonzero, then compute snow density "sndens"
            # and snow thermal conductivity "sncond"
            if sneqv == 0.0:
                sndens = 0.0
                snowh = 0.0
                sncond = 1.0
            else:
                sndens = sneqv / snowh
                sndens = max(0.0, min(1.0, sndens))
                # TODO: sncond is that necessary?
                # is it later overwritten without using before?
                sncond = csnow(sndens)

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
                snowh, sndens = snow_new(sfctmp, sn_new, snowh, sndens)
                sncond = csnow(sndens)

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
                    sncovr = snfrac(sneqv, snup, physcons.SALP)
                    albedo = alcalc(alb, snoalb, sncovr)

            # thermal conductivity for sea-ice case, glacial-ice case
            if ice != 0:
                df1 = 2.2

            else:
                # calculate the subsurface heat flux, which first requires calculation
                # of the thermal diffusivity. treatment of the
                # latter follows that on pages 148-149 from "heat transfer in
                # cold climates", by v. j. lunardini (published in 1981
                # by van nostrand reinhold co.) i.e. treatment of two contiguous
                # "plane parallel" mediums (namely here the first soil layer
                # and the snowpack layer, if any). this diffusivity treatment
                # behaves well for both zero and nonzero snowpack, including the
                # limit of very thin snowpack.  this treatment also eliminates
                # the need to impose an arbitrary upper bound on subsurface
                # heat flux when the snowpack becomes extremely thin.

                # first calculate thermal diffusivity of top soil layer, using
                # both the frozen and liquid soil moisture, following the
                # soil thermal diffusivity function of peters-lidard et al.
                # (1998,jas, vol 55, 1209-1224), which requires the specifying
                # the quartz content of the given soil class (see routine redprm)
                df1 = tdfcnd(smc, quartz, smcmax, sh2o)
                if (not lheatstrg) and (ivegsrc == 1) and (vegtype == 13):
                    df1 = 3.24 * (1.0 - shdfac) + shdfac * df1 * exp(
                        physcons.SBETA * shdfac
                    )
                else:
                    df1 = df1 * exp(physcons.SBETA * shdfac)

            # finally "plane parallel" snowpack effect following
            # v.j. linardini reference cited above. note that dtot is
            # combined depth of snowdepth and thickness of first soil layer

            dsoil = -0.5 * zsoil

            # df1 = 0.41081215623353906
            if sneqv == 0.0:
                ssoil = df1 * (t1 - stc) / dsoil
            else:
                dtot = snowh + dsoil
                frcsno = snowh / dtot
                frcsoi = dsoil / dtot

                # arithmetic mean (parallel flow)
                df1a = frcsno * sncond + frcsoi * df1

                # geometric mean (intermediate between harmonic and arithmetic mean)
                df1 = df1a * sncovr + df1 * (1.0 - sncovr)

                # calculate subsurface heat flux
                ssoil = df1 * (t1 - stc) / dtot

            # determine surface roughness over snowpack using snow condition
            # from the previous timestep.
            if sncovr > 0.0:
                z0 = snowz0(sncovr, z0)

            # calc virtual temps and virtual potential temps needed by
            # subroutines sfcdif and penman.
            t2v = sfctmp * (1.0 + 0.61 * q2)

            # surface exchange coefficients computed externally and passed in,
            # hence subroutine sfcdif not called.
            fdown = swnet + lwdn

            # call penman subroutine to calculate potential evaporation (etp),
            # and other partial products and sums save in common/rite for later
            # calculations.

            t24, etp, rch, epsca, rr, flx2 = penman(
                sfctmp,
                sfcprs,
                sfcems,
                ch,
                t2v,
                th2,
                prcp,
                fdown,
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
            # TODO: rename these runoff variables
            runoff1 = 0.0
            runoff2 = 0.0
            runoff3 = 0.0
            snomlt = 0.0

            pc = 0.0

            esnow = 0.0

        # TODO: Split these out completely
        if lsm_mask:
            esnow = 0.0

            # now decide major pathway branch to take depending on whether
            # snowpack exists or not:
            if sneqv == 0.0:
                nopac_mask = True
                snopac_mask = False
            else:
                nopac_mask = False
                snopac_mask = True


def sflx_2(
    nroot: IntFieldIJ,
    zsoil: FloatFieldK,
    ice: IntFieldIJ,
    t2v: FloatFieldIJ,
    th2: FloatFieldIJ,
    sfcprs: FloatFieldIJ,
    t1: FloatFieldIJ,
    smc: FloatField,
    ch: FloatFieldIJ,
    eta: FloatFieldIJ,
    sheat: FloatFieldIJ,
    ec: FloatFieldIJ,
    edir: FloatFieldIJ,
    et: FloatField,
    ett: FloatFieldIJ,
    esnow: FloatFieldIJ,
    beta: FloatFieldIJ,
    etp: FloatFieldIJ,
    ssoil: FloatFieldIJ,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    runoff3: FloatFieldIJ,
    snomlt: FloatFieldIJ,
    sncovr: FloatFieldIJ,
    soilw: FloatFieldIJ,
    soilm: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    lsm_mask: BoolFieldIJ,
    k_mask: IntFieldK,
):
    with computation(PARALLEL), interval(...):
        from __externals__ import dt

        if lsm_mask:
            et = et * physcons.LSUBC

    with computation(FORWARD):
        with interval(0, 1):
            if lsm_mask:

                # prepare sensible heat (h) for return to parent model
                sheat = (
                    -(ch * physcons.CP1 * sfcprs) / (physcons.RD1 * t2v) * (th2 - t1)
                )

                # convert units and/or sign of total evap (eta), potential evap (etp),
                # subsurface heat flux (s), and runoffs for what parent model expects
                # convert eta from kg m-2 s-1 to w m-2
                edir = edir * physcons.LSUBC
                ec = ec * physcons.LSUBC

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
                soilm = -smc * zsoil
                soilww = -(smc - smcwlt) * zsoil
                soilwm = -(smcmax - smcwlt) * zsoil
        with interval(1, -1):
            if lsm_mask:
                soilm = soilm + smc * (zsoil[-1] - zsoil)
                if k_mask < nroot:
                    soilww = soilww + (smc - smcwlt) * (zsoil[-1] - zsoil)
        with interval(-1, None):
            if lsm_mask:
                soilm = soilm + smc * (zsoil[-1] - zsoil)
                if k_mask < nroot:
                    soilww = soilww + (smc - smcwlt) * (zsoil[-1] - zsoil)
    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            soilw = soilww / soilwm


def finalize_outputs(
    soilm: FloatFieldIJ,
    flx1: FloatFieldIJ,
    flx2: FloatFieldIJ,
    flx3: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    smcref: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    cmc: FloatFieldIJ,
    snowh: FloatFieldIJ,
    sneqv: FloatFieldIJ,
    z0: FloatFieldIJ,
    rho: FloatFieldIJ,
    ch: FloatFieldIJ,
    wind: FloatFieldIJ,
    q1: FloatFieldTracer,
    t1: FloatField,
    smc_old: FloatField,
    stc_old: FloatField,
    slc_old: FloatField,
    weasd_old: FloatFieldIJ,
    snwdph_old: FloatFieldIJ,
    tskin_old: FloatFieldIJ,
    canopy_old: FloatFieldIJ,
    tprcp_old: FloatFieldIJ,
    srflag_old: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    tsurf: FloatFieldIJ,
    smc: FloatField,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    gflux: FloatFieldIJ,
    snowc: FloatFieldIJ,
    stm: FloatFieldIJ,
    snohf: FloatFieldIJ,
    smcwlt2: FloatFieldIJ,
    smcref2: FloatFieldIJ,
    wet1: FloatFieldIJ,
    runoff: FloatFieldIJ,
    drain: FloatFieldIJ,
    canopy: FloatFieldIJ,
    snwdph: FloatFieldIJ,
    weasd: FloatFieldIJ,
    sncovr1: FloatFieldIJ,
    zorl: FloatFieldIJ,
    rch: FloatFieldIJ,
    qsurf: FloatFieldIJ,
    stc: FloatField,
    slc: FloatField,
    tskin: FloatFieldIJ,
    tprcp: FloatFieldIJ,
    srflag: FloatFieldIJ,
    lsm_mask: BoolFieldIJ,
    land: BoolFieldIJ,
    flag_guess: BoolFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            stm *= 1000.0
            snohf = flx1 + flx2 + flx3

            smcwlt2 = smcwlt
            smcref2 = smcref

            wet1 = smc / smcmax

            runoff = runoff1 * 1000.0
            drain = runoff2 * 1000.0

            canopy = cmc * 1000.0
            snwdph = snowh * 1000.0
            weasd = sneqv * 1000.0
            sncovr1 = snowc
            zorl = z0 * 100.0
            t1 = sfctmp

            # compute qsurf
            rch = rho * constants.CP_AIR * ch * wind
            qsurf = q1[0, 0, 0][0] + evap / (physcons.HOCP * rch)
            tem = 1.0 / rho
            hflx = hflx * tem / constants.CP_AIR
            evap = evap * tem / constants.HLV

        # restore land-related prognostic fields for guess run
    with computation(PARALLEL), interval(...):
        if land and flag_guess:
            smc = smc_old
            stc = stc_old
            slc = slc_old
    with computation(FORWARD), interval(0, 1):
        if land and flag_guess:
            weasd = weasd_old
            snwdph = snwdph_old
            tskin = tskin_old
            canopy = canopy_old
            tprcp = tprcp_old
            srflag = srflag_old
        elif land:
            tskin = tsurf


class NoahLSM:
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
        kmask = zeros(domain[2], dtype=Int)
        for k in range(domain[2]):
            kmask[k] = k
        self._k_mask = quantity_factory.from_compute_array(
            kmask, dims=[Z_DIM], units=""
        )

        (
            nroot,
            zroot,
            sldpth,
            zsoil,
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
        ) = set_soil_veg(land_data, veg_data, soil_data, vegfrac_data)

        self._vegtype = quantity_factory.from_array(
            veg_data, dims=[X_DIM, Y_DIM], units="",
        )
        self._soiltype = quantity_factory.from_array(
            soil_data, dims=[X_DIM, Y_DIM], units="",
        )
        self._slope = quantity_factory.from_array(
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

        self._lsm_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._snopac_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._nopac_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._snowng = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._smc_old = make_quantity()
        self._stc_old = make_quantity()
        self._slc_old = make_quantity()
        self._et = make_quantity()

        self._rho = make_quantity_2d()
        self._q0 = make_quantity_2d()
        self._theta1 = make_quantity_2d()
        self._z0 = make_quantity_2d()
        self._weasd_old = make_quantity_2d()
        self._snwdph_old = make_quantity_2d()
        self._tskin_old = make_quantity_2d()
        self._canopy_old = make_quantity_2d()
        self._tprcp_old = make_quantity_2d()
        self._srflag_old = make_quantity_2d()
        self._prcp = make_quantity_2d()
        self._qsi = make_quantity_2d()
        self._dqsdt2 = make_quantity_2d()
        self._cmc = make_quantity_2d()
        self._snowh = make_quantity_2d()
        self._sneqv = make_quantity_2d()
        self._sndens = make_quantity_2d()
        self._rc = make_quantity_2d()
        self._pc = make_quantity_2d()
        self._rcs = make_quantity_2d()
        self._rct = make_quantity_2d()
        self._rcq = make_quantity_2d()
        self._rcsoil = make_quantity_2d()
        self._df1 = make_quantity_2d()
        self._t24 = make_quantity_2d()
        self._fdown = make_quantity_2d()
        self._epsca = make_quantity_2d()
        self._rr = make_quantity_2d()
        self._rch = make_quantity_2d()
        self._flx2 = make_quantity_2d()
        self._prcp1 = make_quantity_2d()
        self._runoff1 = make_quantity_2d()
        self._runoff2 = make_quantity_2d()
        self._runoff3 = make_quantity_2d()
        self._snomlt = make_quantity_2d()
        self._esnow = make_quantity_2d()
        self._drip = make_quantity_2d()
        self._dew = make_quantity_2d()
        self._flx1 = make_quantity_2d()
        self._flx3 = make_quantity_2d()
        self._beta = make_quantity_2d()
        self._t2v = make_quantity_2d()
        self._soilw = make_quantity_2d()
        self._sfctmp = make_quantity_2d()

        self._init_lsm = stencil_factory.from_origin_domain(
            func=init_lsm,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._sflx_1 = stencil_factory.from_origin_domain(
            func=sflx_1,
            externals={
                "dt": dt,
                "ivegsrc": config.ivegsrc,
                "lheatstrg": config.lheatstrg,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._canres = stencil_factory.from_origin_domain(
            func=canres,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._snopac = SNOPAC(
            stencil_factory,
            quantity_factory,
            config.ivegsrc,
            config.lheatstrg,
            dt,
        )

        self._nopac = NOPAC(
            stencil_factory,
            quantity_factory,
            config.ivegsrc,
            config.lheatstrg,
            dt,
        )

        self._sflx_2 = stencil_factory.from_origin_domain(
            func=sflx_2,
            externals={
                "dt": dt,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._finalize_outputs = stencil_factory.from_origin_domain(
            func=finalize_outputs,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass

    def __call__(
        self,
        ps: FloatFieldIJ,
        t1: FloatField,
        q1: FloatFieldTracer,
        sfcemis: FloatFieldIJ,
        dlwflx: FloatFieldIJ,
        dswflx: FloatFieldIJ,
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
        !     zorl     - real, surface roughness                           im   !
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
        self._init_lsm(
            smc,
            stc,
            slc,
            weasd,
            snwdph,
            tskin,
            canopy,
            tprcp,
            srflag,
            q1,
            t1,
            prslki,
            prsl1,
            ch,
            cm,
            wind,
            zorl,
            self._smc_old,
            self._stc_old,
            self._slc_old,
            self._weasd_old,
            self._snwdph_old,
            self._tskin_old,
            self._canopy_old,
            self._tprcp_old,
            self._srflag_old,
            self._sfctmp,
            ep,
            evap,
            hflx,
            gflux,
            drain,
            evbs,
            evcw,
            trans,
            sbsno,
            snowc,
            snohf,
            self._q0,
            self._qsi,
            self._dqsdt2,
            self._theta1,
            self._cmc,
            chh,
            cmm,
            self._z0,
            self._prcp,
            self._snowh,
            self._sneqv,
            self._rho,
            self._ice,
            self._land,
            flag_guess,
            flag_iter,
            self._lsm_mask,
        )

        self._sflx_1(
            self._ice,
            srflag,
            self._zsoil,
            dswflx,
            snet,
            dlwflx,
            sfcemis,
            prsl1,
            self._sfctmp,
            self._prcp,
            self._q0,
            self._qsi,
            self._dqsdt2,
            self._theta1,
            self._vegtype,
            self._soiltype,
            self._slope,
            sfalb,
            snoalb,
            bexppert,
            xlaipert,
            self._cmc,
            tsurf,
            stc,
            smc,
            slc,
            self._sneqv,
            self._sndens,
            ch,
            cm,
            self._z0,
            self._snowh,
            self._nroot,
            self._zroot,
            self._sldpth,
            self._snup,
            self._rsmin,
            self._rgl,
            self._hs,
            self._xlai,
            self._bexp,
            self._dksat,
            self._dwsat,
            self._f1,
            self._kdt,
            self._psisat,
            self._quartz,
            self._smcdry,
            self._smcmax,
            self._smcref,
            self._smcwlt,
            self._shdfac,
            self._frzx,
            self._rtdis,
            self._df1,
            self._t24,
            self._fdown,
            self._epsca,
            ep,
            self._rr,
            self._rch,
            self._flx2,
            self._prcp1,
            self._rc,
            self._rcs,
            self._rct,
            self._rcq,
            self._rcsoil,
            self._runoff1,
            self._runoff2,
            self._runoff3,
            self._snomlt,
            self._pc,
            self._esnow,
            self._t2v,
            self._snowng,
            self._k_mask,
            self._lsm_mask,
            self._snopac_mask,
            self._nopac_mask,
        )

        self._canres(
            self._nroot,
            dswflx,
            ch,
            self._q0,
            self._qsi,
            self._dqsdt2,
            self._sfctmp,
            prsl1,
            sfcemis,
            slc,
            self._smcwlt,
            self._smcref,
            self._zsoil,
            self._zroot,
            self._rsmin,
            self._rgl,
            self._hs,
            self._xlai,
            self._k_mask,
            self._shdfac,
            self._rc,
            self._pc,
            self._rcs,
            self._rct,
            self._rcq,
            self._rcsoil,
            self._lsm_mask,
        )

        self._snopac(
            self._nroot,
            ep,
            self._prcp,
            self._smcmax,
            self._smcwlt,
            self._smcref,
            self._smcdry,
            self._df1,
            sfcemis,
            self._sfctmp,
            self._t24,
            self._theta1,
            self._fdown,
            self._epsca,
            self._bexp,
            self._pc,
            self._rch,
            self._rr,
            self._slope,
            self._kdt,
            self._frzx,
            self._psisat,
            self._zsoil,
            self._dwsat,
            self._dksat,
            self._shdfac,
            self._ice,
            self._rtdis,
            self._quartz,
            self._flx2,
            self._snowng,
            srflag,
            self._vegtype,
            self._prcp1,
            self._cmc,
            tsurf,
            stc,
            snowc,
            self._sneqv,
            self._sndens,
            self._snowh,
            slc,
            tg3,
            smc,
            gflux,
            self._runoff1,
            self._runoff2,
            self._runoff3,
            evbs,
            evcw,
            self._et,
            trans,
            self._snomlt,
            self._drip,
            self._dew,
            self._flx1,
            self._flx3,
            sbsno,
            self._snopac_mask,
            self._k_mask,
        )

        self._nopac(
            self._nroot,
            ep,
            self._prcp,
            self._smcmax,
            self._smcwlt,
            self._smcref,
            self._smcdry,
            self._shdfac,
            self._sfctmp,
            sfcemis,
            self._t24,
            self._theta1,
            self._fdown,
            self._epsca,
            self._bexp,
            self._pc,
            self._rch,
            self._rr,
            self._slope,
            self._kdt,
            self._frzx,
            self._psisat,
            self._zsoil,
            self._dksat,
            self._dwsat,
            self._ice,
            self._rtdis,
            self._quartz,
            self._vegtype,
            self._cmc,
            tsurf,
            stc,
            slc,
            tg3,
            smc,
            evap,
            gflux,
            self._runoff1,
            self._runoff2,
            self._runoff3,
            evbs,
            evcw,
            self._et,
            trans,
            self._beta,
            self._drip,
            self._dew,
            self._flx1,
            self._flx3,
            self._k_mask,
            self._nopac_mask,
        )

        self._sflx_2(
            self._nroot,
            self._zsoil,
            self._ice,
            self._t2v,
            self._theta1,
            prsl1,
            tsurf,
            smc,
            ch,
            evap,
            hflx,
            evcw,
            evbs,
            self._et,
            trans,
            sbsno,
            self._beta,
            ep,
            gflux,
            self._runoff1,
            self._runoff2,
            self._runoff3,
            self._snomlt,
            snowc,
            self._soilw,
            stm,
            self._smcwlt,
            self._smcmax,
            self._lsm_mask,
            self._k_mask,
        )

        self._finalize_outputs(
            self._flx1,
            self._flx2,
            self._flx3,
            self._smcwlt,
            self._smcref,
            self._smcmax,
            self._runoff1,
            self._runoff2,
            self._cmc,
            self._snowh,
            self._sneqv,
            self._z0,
            self._rho,
            ch,
            wind,
            q1,
            t1,
            self._smc_old,
            self._stc_old,
            self._slc_old,
            self._weasd_old,
            self._snwdph_old,
            self._tskin_old,
            self._canopy_old,
            self._tprcp_old,
            self._srflag_old,
            self._sfctmp,
            tsurf,
            smc,
            evap,
            hflx,
            gflux,
            snowc,
            stm,
            snohf,
            smcwlt2,
            smcref2,
            wet1,
            runoff,
            drain,
            canopy,
            snwdph,
            weasd,
            sncovr1,
            zorl,
            self._rch,
            qsurf,
            stc,
            slc,
            tskin,
            tprcp,
            srflag,
            self._lsm_mask,
            self._land,
            flag_guess,
        )

        pass
