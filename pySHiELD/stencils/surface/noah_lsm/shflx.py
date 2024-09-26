from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, interval, log

import ndsl.constants as constants
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
    Int,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from ndsl.stencils.tridiag import tridiag_solve


@gtscript.function
def frh2o_fn(psis, bexp, tavg, smc, sh2o, smcmax):
    # constant parameters
    ck = 8.0
    blim = 5.5
    error = 0.005
    bx = min(bexp, blim)

    kcount = True
    nlog = 0

    if tavg <= (constants.TICE0 - 1.0e-3):
        swl = smc - sh2o
        swl = max(min(swl, smc - 0.02), 0.0)

        while (nlog < 10) and kcount:
            nlog += 1
            df = log(
                (psis * physcons.GS2 / physcons.LSUBF)
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
        if not kcount:
            fk = (
                (
                    (physcons.LSUBF / (physcons.GS2 * (-psis)))
                    * ((tavg - constants.TICE0) / tavg)
                )
                ** (-1 / bx)
            ) * smcmax

            fk = max(fk, 0.02)

            free = min(fk, smc)

    else:
        free = smc

    return free

@gtscript.function
def snksrc_fn(psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dz):
    from __externals__ import dt

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
def tmpavg_fn(tup, tm, tdn, dz):

    dzh = dz * 0.5

    if tup < constants.TICE0:
        if tm < constants.TICE0:
            if tdn < constants.TICE0:
                tavg = (tup + 2.0 * tm + tdn) / 4.0
            else:
                x0 = (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    0.5
                    * (tup * dzh + tm * (dzh + x0) + constants.TICE0 * (2.0 * dzh - x0))
                    / dz
                )
        else:
            if tdn < constants.TICE0:
                xup = (constants.TICE0 - tup) * dzh / (tm - tup)
                xdn = dzh - (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    0.5
                    * (tup * xup + constants.TICE0 * (2.0 * dz - xup - xdn) + tdn * xdn)
                    / dz
                )
            else:
                xup = (constants.TICE0 - tup) * dzh / (tm - tup)
                tavg = 0.5 * (tup * xup + constants.TICE0 * (2.0 * dz - xup)) / dz
    else:
        if tm < constants.TICE0:
            if tdn < constants.TICE0:
                xup = dzh - (constants.TICE0 - tup) * dzh / (tm - tup)
                tavg = (
                    0.5
                    * (constants.TICE0 * (dz - xup) + tm * (dzh + xup) + tdn * dzh)
                    / dz
                )
            else:
                xup = dzh - (constants.TICE0 - tup) * dzh / (tm - tup)
                xdn = (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    0.5
                    * (constants.TICE0 * (2.0 * dz - xup - xdn) + tm * (xup + xdn))
                    / dz
                )
        else:
            if tdn < constants.TICE0:
                xdn = dzh - (constants.TICE0 - tm) * dzh / (tdn - tm)
                tavg = (
                    constants.TICE0 * (dz - xdn) + 0.5 * (constants.TICE0 + tdn) * xdn
                ) / dz
            else:
                tavg = (tup + 2.0 * tm + tdn) / 4.0
    return tavg


@gtscript.function
def tdfcnd(smc, qz, smcmax, sh2o):
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

def start_shflx(
    ice: IntFieldIJ,
    stc: FloatField,
    t1: FloatFieldIJ,
    ctfil1: FloatFieldIJ,
    ctfil2: FloatFieldIJ,
    oldt1: FloatFieldIJ,
    stsoil: FloatField,
    surface_mask: BoolFieldIJ,
    ice_mask: BoolFieldIJ,
    no_ice_mask: BoolFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        if surface_mask:
            # updates the temperature state of the soil column

            ctfil1 = 0.5
            ctfil2 = 1.0 - ctfil1

            oldt1 = t1

            if ice != 0:  # sea-ice or glacial ice case
                ice_mask = True
                no_ice_mask = False
            else:
                ice_mask = False
                no_ice_mask = True

    with computation(PARALLEL), interval(...):
        if surface_mask:
            stsoil = stc


def hrtice(
    stc: FloatField,
    zsoil: FloatFieldK,
    yy: FloatFieldIJ,
    zz1: FloatFieldIJ,
    df1: FloatFieldIJ,
    ice: IntFieldIJ,
    tbot: FloatFieldIJ,
    rhsts: FloatField,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
    surface_mask: BoolFieldIJ,
):
    with computation(FORWARD):
        with interval(0, 1):
            if surface_mask:
                # calculates the right hand side of the time tendency
                # term of the soil thermal diffusion equation for sea-ice or glacial-ice

                # set a nominal universal value of specific heat capacity
                if ice == 1:  # sea-ice
                    hcpct = 1.72396e6
                    tbot = 271.16
                else:  # glacial-ice
                    hcpct = 1.89000e6

                # 1. Top Layer
                # calc the matrix coefficients ai, bi, and ci for the top layer
                ddz = 1.0 / (-0.5 * zsoil[0, 0, 1])
                ai = 0.0
                ci = (df1 * ddz) / (zsoil * hcpct)
                bi = -ci + df1 / (0.5 * zsoil * zsoil * hcpct * zz1)

                # calc the vertical soil temp gradient btwn the top and 2nd soil
                dtsdz = (stc - stc[0, 0, 1]) / (-0.5 * zsoil[0, 0, 1])
                ssoil = df1 * (stc - yy) / (0.5 * zsoil * zz1)
                rhsts = (df1 * dtsdz - ssoil) / (zsoil * hcpct)

        with interval(1, -1):
            if surface_mask:
                # 2. Inner Layers
                denom = 0.5 * (zsoil[0, 0, -1] - zsoil[0, 0, 1])
                dtsdz2 = (stc - stc[0, 0, 1]) / denom
                ddz2 = 2.0 / (zsoil[0, 0, -1] - zsoil[0, 0, 1])
                ci = -df1 * ddz2 / ((zsoil[0, 0, -1] - zsoil) * hcpct)

                denom = (zsoil - zsoil[0, 0, -1]) * hcpct
                rhsts = (df1 * dtsdz2 - df1 * dtsdz) / denom

                ai = -df1 * ddz / ((zsoil[0, 0, -1] - zsoil) * hcpct)
                bi = -(ai + ci)

                dtsdz = dtsdz2
                ddz = ddz2

        with interval(-1, None):
            if surface_mask:
                # 3. Bottom Layer
                # set ice pack depth
                if ice == 1:
                    zbot = zsoil
                else:
                    zbot = -25.0
                dtsdz2 = (stc - tbot) / (0.5 * (zsoil[0, 0, -1] - zsoil) - zbot)
                ci = 0.0

                denom = (zsoil - zsoil[0, 0, -1]) * hcpct
                rhsts = (df1 * dtsdz2 - df1 * dtsdz) / denom

                ai = -df1 * ddz / ((zsoil[0, 0, -1] - zsoil) * hcpct)
                bi = -(ai + ci)


def hrt(
    stc: FloatField,
    smc: FloatField,
    smcmax: FloatFieldIJ,
    zsoil: FloatFieldK,
    yy: FloatFieldIJ,
    zz1: FloatFieldIJ,
    tbot: FloatFieldIJ,
    psisat: FloatFieldIJ,
    bexp: FloatFieldIJ,
    df1: FloatFieldIJ,
    quartz: FloatFieldIJ,
    vegtype: IntFieldIJ,
    shdfac: FloatFieldIJ,
    sh2o: FloatField,
    rhsts: FloatField,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
    surface_mask: FloatFieldIJ,
):
    from __externals__ import ivegsrc, lheatstrg

    with computation(FORWARD):
        with interval(0, 1):
            if surface_mask:
                csoil_loc = physcons.CSOIL

                if (not lheatstrg) and (ivegsrc == 1) and (vegtype == 13):
                    csoil_loc = 3.0e6 * (1.0 - shdfac) + physcons.CSOIL * shdfac

                # calc the heat capacity of the top soil layer
                hcpct = (
                    sh2o * physcons.CPH2O2
                    + (1.0 - smcmax) * csoil_loc
                    + (smcmax - smc) * physcons.CP2
                    + (smc - sh2o) * physcons.CPICE1
                )

                # calc the matrix coefficients ai, bi, and ci for the top layer
                ddz = 1.0 / (-0.5 * zsoil[0, 0, 1])
                ai = 0.0
                ci = (df1 * ddz) / (zsoil * hcpct)
                bi = -ci + df1 / (0.5 * zsoil * zsoil * hcpct * zz1)

                # calc the vertical soil temp gradient btwn the top and 2nd soil
                dtsdz = (stc - stc[0, 0, 1]) / (-0.5 * zsoil[0, 0, 1])
                ssoil = df1 * (stc - yy) / (0.5 * zsoil * zz1)
                rhsts = (df1 * dtsdz - ssoil) / (zsoil * hcpct)

                # capture the vertical difference of the heat flux at top and
                # bottom of first soil layer
                qtot = ssoil - df1 * dtsdz

                tsurf = (yy + (zz1 - 1) * stc) / zz1

                # linear interpolation between the average layer temperatures
                tbk = stc + (stc[0, 0, 1] - stc) * zsoil / zsoil[0, 0, 1]
                # calculate frozen water content in 1st soil layer.
                sice = smc - sh2o

                df1k = df1

                if (
                    (sice > 0)
                    or tsurf < (constants.TICE0)
                    or (stc < constants.TICE0)
                    or (tbk < constants.TICE0)
                ):
                    dz = -zsoil
                    tavg = tmpavg_fn(tsurf, stc, tbk, dz)
                    tsnsr, sh2o = snksrc_fn(
                        psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dz
                    )

                    rhsts -= tsnsr / (zsoil * hcpct)

        with interval(1, -1):
            if surface_mask:
                # 2. Inner Layers
                hcpct = (
                    sh2o * physcons.CPH2O2
                    + (1.0 - smcmax) * csoil_loc
                    + (smcmax - smc) * physcons.CP2
                    + (smc - sh2o) * physcons.CPICE1
                )

                # calculate thermal diffusivity for each layer
                df1k = tdfcnd(smc, quartz, smcmax, sh2o)

                if (not lheatstrg) and (ivegsrc == 1) and (vegtype == 13):
                    df1k = 3.24 * (1.0 - shdfac) + shdfac * df1k

                tbk = stc + (stc[0, 0, 1] - stc) * (zsoil[0, 0, -1] - zsoil) / (
                    zsoil[0, 0, -1] - zsoil[0, 0, 1]
                )
                # calc the vertical soil temp gradient thru each layer
                denom = 0.5 * (zsoil[0, 0, -1] - zsoil[0, 0, 1])
                dtsdz = (stc - stc[0, 0, 1]) / denom
                ddz = 2.0 / (zsoil[0, 0, -1] - zsoil[0, 0, 1])

                ci = -df1k * ddz / ((zsoil[0, 0, -1] - zsoil) * hcpct)

                # calculate rhsts
                denom = (zsoil - zsoil[0, 0, -1]) * hcpct
                rhsts = (df1k * dtsdz - df1k[0, 0, -1] * dtsdz[0, 0, -1]) / denom

                qtot = -1.0 * denom * rhsts
                sice = smc - sh2o

                if (
                    (sice > 0)
                    or (tbk[0, 0, -1] < constants.TICE0)
                    or (stc < constants.TICE0)
                    or (tbk < constants.TICE0)
                ):
                    dz = zsoil[0, 0, -1] - zsoil
                    tavg = tmpavg_fn(tbk[0, 0, -1], stc, tbk, dz)
                    tsnsr, sh2o = snksrc_fn(
                        psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dz
                    )
                    rhsts -= tsnsr / denom

                # calc matrix coefs, ai, and bi for this layer.
                ai = -df1 * ddz[0, 0, -1] / ((zsoil[0, 0, -1] - zsoil) * hcpct)
                bi = -(ai + ci)

        with interval(-1, None):
            if surface_mask:
                # 3. Bottom Layer
                hcpct = (
                    sh2o * physcons.CPH2O2
                    + (1.0 - smcmax) * csoil_loc
                    + (smcmax - smc) * physcons.CP2
                    + (smc - sh2o) * physcons.CPICE1
                )

                # calculate thermal diffusivity for each layer
                df1k = tdfcnd(smc, quartz, smcmax, sh2o)

                if (not lheatstrg) and (ivegsrc == 1) and (vegtype == 13):
                    df1k = 3.24 * (1.0 - shdfac) + shdfac * df1k

                tbk = stc + (tbot - stc) * (zsoil[0, 0, -1] - zsoil) / (
                    zsoil[0, 0, -1] + zsoil - 2.0 * physcons.ZBOT
                )

                denom = 0.5 * (zsoil[0, 0, -1] + zsoil) - physcons.ZBOT
                dtsdz = (stc - tbot) / denom
                ci = 0.0

                # calculate rhsts
                denom = (zsoil - zsoil[0, 0, -1]) * hcpct
                rhsts = (df1k * dtsdz - df1k[0, 0, -1] * dtsdz[0, 0, -1]) / denom

                qtot = -1.0 * denom * rhsts
                sice = smc - sh2o

                if (
                    (sice > 0)
                    or (tbk[0, 0, -1] < constants.TICE0)
                    or (stc < constants.TICE0)
                    or (tbk < constants.TICE0)
                ):
                    dz = zsoil[0, 0, -1] - zsoil
                    tavg = tmpavg_fn(tbk[0, 0, -1], stc, tbk, dz)
                    tsnsr, sh2o = snksrc_fn(
                        psisat, bexp, tavg, smc, sh2o, smcmax, qtot, dz
                    )

                    rhsts -= tsnsr / denom
                # calc matrix coefs, ai, and bi for this layer.
                ai = -df1 * ddz[0, 0, -1] / ((zsoil[0, 0, -1] - zsoil) * hcpct)
                bi = -(ai + ci)


def prep_hstep(
    stc: FloatField,
    rhsts: FloatField,
    ai: FloatField,
    bi: FloatField,
    ci: FloatField,
):
    from __externals__ import dt

    with computation(PARALLEL), interval(...):
        ai *= dt
        bi = 1.0 + dt * bi
        ci *= dt
        rhsts *= dt


def finish_hstep(
    heat_flux: FloatField,
    stc: FloatField,
    surface_mask: BoolFieldIJ,
):
    with computation(PARALLEL), interval(...):
        if surface_mask:
            stc += heat_flux


def finish_shflux(
    stc: FloatField,
    zsoil: FloatFieldK,
    yy: FloatFieldIJ,
    zz1: FloatFieldIJ,
    df1: FloatFieldIJ,
    ctfil1: FloatFieldIJ,
    ctfil2: FloatFieldIJ,
    oldt1: FloatFieldIJ,
    stsoil: FloatField,
    ssoil: FloatFieldIJ,
    surface_mask: BoolFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        if surface_mask:
            # update the grnd (skin) temperature in the no snowpack case
            t1 = (yy + (zz1 - 1.0) * stc) / zz1
            t1 = ctfil1 * t1 + ctfil2 * oldt1

            stc = ctfil1 * stc + ctfil2 * stsoil

            # calculate surface soil heat flux
            ssoil = df1 * (stc - t1) / (0.5 * zsoil)
    with computation(FORWARD), interval(1, None):
        if surface_mask:
            stc = ctfil1 * stc + ctfil2 * stsoil


class SoilHeatFlux:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        ivegsrc: Int,
        lheatstrg: Bool,
        dt: Float,
    ):
        """
        Fortran name is shflux
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

        self._ice_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._no_ice_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._ctfil1 = make_quantity_2d()
        self._ctfil2 = make_quantity_2d()
        self._oldt1 = make_quantity_2d()
        self._stsoil = make_quantity_2d()
        self._rhsts = make_quantity()
        self._ai = make_quantity()
        self._bi = make_quantity()
        self._ci = make_quantity()
        self._heat_flux = make_quantity()
        self._delta = make_quantity()

        self._start_shflx = stencil_factory.from_origin_domain(
            func=start_shflx,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._hrtice = stencil_factory.from_origin_domain(
            func=hrtice,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._hrt = stencil_factory.from_origin_domain(
            func=hrt,
            externals={
                "dt": dt,
                "lheatstrg": lheatstrg,
                "ivegsrc": ivegsrc,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._prep_hstep = stencil_factory.from_origin_domain(
            func=prep_hstep,
            externals={
                "dt": dt,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._tridiag_solve = stencil_factory.from_origin_domain(
            func=tridiag_solve,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._finish_hstep = stencil_factory.from_origin_domain(
            func=finish_hstep,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._finish_shflux = stencil_factory.from_origin_domain(
            func=finish_shflux,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        pass

    def __call__(
        self,
        smc: FloatField,
        smcmax: FloatFieldIJ,
        yy: FloatFieldIJ,
        zz1: FloatFieldIJ,
        zsoil: FloatFieldK,
        psisat: FloatFieldIJ,
        bexp: FloatFieldIJ,
        df1: FloatFieldIJ,
        ice: IntFieldIJ,
        quartz: FloatFieldIJ,
        vegtype: IntFieldIJ,
        shdfac: FloatFieldIJ,
        stc: FloatField,
        t1: FloatFieldIJ,
        tbot: FloatFieldIJ,
        sh2o: FloatField,
        ssoil: FloatFieldIJ,
        surface_mask: BoolFieldIJ,
    ):
        """
        Fortran Description:
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  subroutine shflx updates the temperature state of the soil column    !
        !  based on the thermal diffusion equation and update the frozen soil   !
        !  moisture content based on the temperature.                           !
        !                                                                       !
        !  subprogram called:  hstep, hrtice, hrt                               !
        !                                                                       !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs:                                                       size   !
        !     nsoil    - integer, number of soil layers                    1    !
        !     smc      - real, total soil moisture                       nsoil  !
        !     smcmax   - real, porosity (sat val of soil mois)             1    !
        !     dt       - real, time step                                   1    !
        !     yy       - real, soil temperature at the top of column       1    !
        !     zz1      - real,                                             1    !
        !     zsoil    - real, soil layer depth below ground (negative)  nsoil  !
        !     zbot     - real, specify depth of lower bd soil              1    !
        !     psisat   - real, saturated soil potential                    1    !
        !     bexp     - real, soil type "b" parameter                     1    !
        !     df1      - real, thermal diffusivity and conductivity        1    !
        !     ice      - integer, sea-ice flag (=1: sea-ice, =0: land)     1    !
        !     quartz   - real, soil quartz content                         1    !
        !     csoil    - real, soil heat capacity                          1    !
        !     vegtyp   - integer, vegtation type                           1    !
        !     shdfac   - real, aeral coverage of green vegetation          1    !
        !     lheatstrg- logical, flag for canopy heat storage             1    !
        !                         parameterization                              !
        !                                                                       !
        !  input/outputs:                                                       !
        !     stc      - real, soil temp                                 nsoil  !
        !     t1       - real, ground/canopy/snowpack eff skin temp        1    !
        !     tbot     - real, bottom soil temp                            1    !
        !     sh2o     - real, unfrozen soil moisture                    nsoil  !
        !                                                                       !
        !  outputs:                                                             !
        !     ssoil    - real, upward soil heat flux                       1    !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """
        self._start_shflx(
            ice,
            stc,
            t1,
            self._ctfil1,
            self._ctfil2,
            self._oldt1,
            self._stsoil,
            surface_mask,
            self._ice_mask,
            self._no_ice_mask,
        )

        self._hrtice(
            stc,
            zsoil,
            yy,
            zz1,
            df1,
            ice,
            tbot,
            self._rhsts,
            self._ai,
            self._bi,
            self._ci,
            surface_mask,
        )

        self._hrt(
            stc,
            smc,
            smcmax,
            zsoil,
            yy,
            zz1,
            tbot,
            psisat,
            bexp,
            df1,
            quartz,
            vegtype,
            shdfac,
            sh2o,
            self._rhsts,
            self._ai,
            self._bi,
            self._ci,
            surface_mask,
        )

        self._prep_hstep(
            stc,
            self._rhsts,
            self._ai,
            self._bi,
            self._ci,
        )

        self._tridiag_solve(
            self._ai,
            self._bi,
            self._ci,
            self._rhsts,
            self._heat_flux,
            self._delta,
        )

        self._finish_hstep(
            self._heat_flux,
            stc,
            surface_mask,
        )

        self._finish_shflux(
            stc,
            zsoil,
            yy,
            zz1,
            df1,
            self._ctfil1,
            self._ctfil2,
            self._oldt1,
            self._stsoil,
            ssoil,
            surface_mask,
        )
