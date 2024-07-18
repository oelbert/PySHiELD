from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval, log

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg
from pySHiELD.stencils.surface.noah_lsm.nopac import NOPAC
from pySHiELD.stencils.surface.noah_lsm.snopac import SNOPAC
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
    zsoil: FloatField,
    zroot: FloatFieldIJ,
    rsmin: FloatFieldIJ,
    rsmax: FloatFieldIJ,
    topt: FloatFieldIJ,
    rgl: FloatFieldIJ,
    hs: FloatFieldIJ,
    xlai: FloatFieldIJ,
    kmask: IntFieldIJ,
    flag_iter: BoolFieldIJ,
    land: BoolFieldIJ,
    shdfac: FloatFieldIJ,
    rc: FloatFieldIJ,
    pc: FloatFieldIJ,
    rcs: FloatFieldIJ,
    rct: FloatFieldIJ,
    rcq: FloatFieldIJ,
    rcsoil: FloatFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            if flag_iter and land:
                if shdfac > 0.0:
                    # calculates canopy resistance

                    # contribution due to incoming solar radiation
                    ff = 0.55 * 2.0 * swdn / (rgl * xlai)
                    rcs = (ff + rsmin / rsmax) / (1.0 + ff)
                    rcs = max(rcs, 0.0001)

                    # contribution due to air temperature
                    # at first model level above ground
                    rct = 1.0 - 0.0016 * (topt - sfctmp) ** 2.0
                    rct = max(rct, 0.0001)

                    # contribution due to vapor pressure deficit at first model level.
                    rcq = 1.0 / (1.0 + hs * (q2sat - q2))
                    rcq = max(rcq, 0.01)

                    # contribution due to soil moisture availability.
                    rcsoil = 0.

                    # use soil depth as weighting factor
                    gx = 0.
                    if (kmask < nroot):
                        gx = max(0.0, min(1.0, (sh2o - smcwlt) / (smcref - smcwlt)))
                        rcsoil = rcsoil + (zsoil / zroot * gx)
        with interval(1, None):
            if flag_iter and land:
                if shdfac > 0.0:
                    if (kmask < nroot):
                        gx = max(0.0, min(1.0, (sh2o - smcwlt) / (smcref - smcwlt)))
                        rcsoil = rcsoil + ((zsoil - zsoil[0, 0, -1]) / zroot * gx)
    with computation(FORWARD), interval(0, 1):
        if flag_iter and land:
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
    ps: FloatFieldIJ,
    t1: FloatFieldIJ,
    q1: FloatFieldIJ,
    soiltype: IntFieldIJ,
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
    slopetype: IntFieldIJ,
    shdmin: FloatFieldIJ,
    shdmax: FloatFieldIJ,
    snoalb: FloatFieldIJ,
    sfalb: FloatFieldIJ,
    flag_iter: BoolFieldIJ,
    flag_guess: BoolFieldIJ,
    bexppert: FloatFieldIJ,
    xlaipert: FloatFieldIJ,
    vegfpert: FloatFieldIJ,
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
    zsoil: FloatFieldK,
    sldpth: FloatFieldK,
    lsm_mask: BoolFieldIJ,
    # bexp: DT_F,
    # dksat: DT_F,
    # dwsat: DT_F,
    # psisat: DT_F,
    # quartz: DT_F,
    # smcdry: DT_F,
    # smcmax: DT_F,
    # smcref: DT_F,
    # smcwlt: DT_F,
    # nroot: DT_I,
    # snup: DT_F,
    # rsmin: DT_F,
    # rgl: DT_F,
    # hs: DT_F,
    # xlai: DT_F,
    # slope: DT_F,
):
    from __externals__ import delt
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

            q0 = max(q1, 1.0e-8)
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
            prcp = physcons.RHOH2O * tprcp / delt
            dqsdt2 = qs1 * physcons.A23M4 / (t1 - physcons.A4) ** 2

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


def sflx_1(
    ice: IntFieldIJ,
    ffrozp: FloatFieldIJ,
    zsoil: FloatField,
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
    tbot: FloatFieldIJ,
    cmc: FloatFieldIJ,
    t1: FloatFieldIJ,
    stc: FloatField,
    smc: FloatField,
    sh2o: FloatField,
    sneqv: FloatFieldIJ,
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
    kmask: IntField,
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
    from __externals__ import lheatstrg, ivegsrc, dt
    with computation(PARALLEL), interval(0, 1):
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

    with computation(PARALLEL), interval(0, 1):
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

            esnow = 0.

    # TODO: Split these out completely
    with computation(PARALLEL), interval(...):
        if lsm_mask:
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
                    sfcprs,
                    sfcems,
                    sh2o,
                    smcwlt,
                    smcref,
                    zsoil,
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
                nopac_mask = True
                snopac_mask = False
            else:
                nopac_mask = False
                snopac_mask = True
                
                
                
                (
                    cmc,
                    t1,
                    stc,
                    sh2o,
                    tbot,
                    eta,
                    smc,
                    ssoil,
                    runoff1,
                    runoff2,
                    runoff3,
                    edir,
                    ec,
                    et,
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
                    cmcmax,
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
                    cfactr,
                    slopetype,
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
                    vegtype,
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
                    stc,
                    sncovr,
                    sneqv,
                    sndens,
                    snowh,
                    sh2o,
                    tbot,
                    smc,
                    ssoil,
                    runoff1,
                    runoff2,
                    runoff3,
                    edir,
                    ec,
                    et,
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
                    slopetype,
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


def sflx_2():
    with computation(PARALLEL), interval(...):
        from __externals__ import dt
        if lsm_mask:
            et = et * physcons.LSUBC

    with computation(FORWARD):
        with interval(0, 1):
            if lsm_mask:

                # prepare sensible heat (h) for return to parent model
                sheat = -(ch * physcons.CP1 * sfcprs) / (physcons.RD1 * t2v) * (
                    th2 - t1
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
                soilm = soilm + smc * (zsoil[0, 0, -1] - zsoil)
                if kmask < nroot:
                    soilww = soilww + (smc - smcwlt) * (zsoil[0, 0, -1] - zsoil)
        with interval(-1, None):
            if lsm_mask:
                soilm = soilm + smc * (zsoil[0, 0, -1] - zsoil)
                if kmask < nroot:
                    soilww = soilww + (smc - smcwlt) * (zsoil[0, 0, -1] - zsoil)
    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            soilw = soilww / soilwm

            # return (
            #     tbot,
            #     cmc,
            #     t1,
            #     stc,
            #     smc,
            #     sh2o,
            #     sneqv,
            #     ch,
            #     cm,
            #     z0,
            #     shdfac,
            #     snowh,
            #     nroot,
            #     albedo,
            #     eta,
            #     sheat,
            #     ec,
            #     edir,
            #     et,
            #     ett,
            #     esnow,
            #     drip,
            #     dew,
            #     beta,
            #     etp,
            #     ssoil,
            #     flx1,
            #     flx2,
            #     flx3,
            #     runoff1,
            #     runoff2,
            #     runoff3,
            #     snomlt,
            #     sncovr,
            #     rc,
            #     pc,
            #     rsmin,
            #     xlai,
            #     rcs,
            #     rct,
            #     rcq,
            #     rcsoil,
            #     soilw,
            #     soilm,
            #     smcwlt,
            #     smcdry,
            #     smcref,
            #     smcmax,
            # )


def finalize_outputs(
    ps: FloatFieldIJ,
    t1: FloatFieldIJ,
    q1: FloatFieldIJ,
    soiltype: IntFieldIJ,
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
    wind: FloatFieldIJ,
    slopetype: IntFieldIJ,
    shdmin: FloatFieldIJ,
    shdmax: FloatFieldIJ,
    snoalb: FloatFieldIJ,
    sfalb: FloatFieldIJ,
    flag_guess: BoolFieldIJ,
    bexppert: FloatFieldIJ,
    xlaipert: FloatFieldIJ,
    vegfpert: FloatFieldIJ,
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
    zsoil: FloatFieldK,
    sldpth: FloatFieldK,
    lsm_mask: BoolFieldIJ,
    land: BoolFieldIJ,
):
    from __externals__ import delt, lheatstrg, ivegsrc

    with computation(PARALLEL), interval(...):
        if lsm_mask:
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
            qsurf = q1 + evap / (physcons.HOCP * rch)
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
    """
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: SurfaceConfig,
        land_mask,
        veg_data,
        soil_data,
        vegfrac_data,
        dt: Float,
        lheatstrg: Bool,
        ivegsrc: Bool,
    ):

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
        ) = set_soil_veg(land_mask, veg_data, soil_data, vegfrac_data)

        self._nroot = quantity_factory.from_array(
            nroot, dims=[X_DIM, Y_DIM], units=""
        )
        self._zroot = quantity_factory.from_array(
            zroot, dims=[X_DIM, Y_DIM], units=""
        )
        self._sldpth = quantity_factory.from_array(
            sldpth, dims=[X_DIM, Y_DIM, Z_INTERFACE_DIM], units="m"
        )
        self._zsoil = quantity_factory.from_array(
            sldpth, dims=[X_DIM, Y_DIM, Z_DIM], units="m"
        )
        self._snup = quantity_factory.from_array(
            snup, dims=[X_DIM, Y_DIM], units=""
        )
        self._rsmin = quantity_factory.from_array(
            rsmin, dims=[X_DIM, Y_DIM], units=""
        )
        self._rgl = quantity_factory.from_array(
            rgl, dims=[X_DIM, Y_DIM], units=""
        )
        self._hs = quantity_factory.from_array(
            hs, dims=[X_DIM, Y_DIM], units=""
        )
        self._xlai = quantity_factory.from_array(
            xlai, dims=[X_DIM, Y_DIM], units=""
        )
        self._bexp = quantity_factory.from_array(
            bexp, dims=[X_DIM, Y_DIM], units=""
        )
        self._dksat = quantity_factory.from_array(
            dksat, dims=[X_DIM, Y_DIM], units=""
        )
        self._dwsat = quantity_factory.from_array(
            dwsat, dims=[X_DIM, Y_DIM], units=""
        )
        self._f1 = quantity_factory.from_array(
            f1, dims=[X_DIM, Y_DIM], units=""
        )
        self._kdt = quantity_factory.from_array(
            kdt, dims=[X_DIM, Y_DIM], units=""
        )
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
        self._frzx = quantity_factory.from_array(
            frzx, dims=[X_DIM, Y_DIM], units=""
        )
        self._rtdis = quantity_factory.from_array(
            rtdis, dims=[X_DIM, Y_DIM, Z_DIM], units=""
        )

        self._init_lsm = stencil_factory.from_origin_domain(
            func=init_lsm,
            externals={"delt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._sflx_1 = stencil_factory.from_origin_domain(
            func=sflx_1,
            externals={
                "dt": dt,
                "ivegsrc": ivegsrc,
                "lheatstrg": lheatstrg,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._canres = stencil_factory.from_origin_domain(
            func=canres,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._snopac = stencil_factory.from_origin_domain(
            func=snopac,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._nopac = stencil_factory.from_origin_domain(
            func=nopac,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._sflx_2 = stencil_factory.from_origin_domain(
            func=sflx_2,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._finalize_outputs = stencil_factory.from_origin_domain(
            func=finalize_outputs,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass

    def __call__(self):
        self._init_lsm()
        self._sflx_1()
        self._canres()

        # now decide major pathway branch to take depending on whether
        # snowpack exists or not:
        # Each column will either call one or the other...
        self._snopac()
        self._nopac()

        self._sflx_2()
        self._finalize_outputs()
        pass
