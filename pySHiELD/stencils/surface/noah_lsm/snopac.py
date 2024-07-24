from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval

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
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.quantity import Quantity
from pySHiELD.stencils.surface.noah_lsm.evapo import EvapoTranspiration
from pySHiELD.stencils.surface.noah_lsm.shflx import SoilHeatFlux
from pySHiELD.stencils.surface.noah_lsm.smflx import SoilMoistureFlux


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

    # dsx = sndens * ((dexp(bfac*esdc)-1.0) / (bfac*esdc))

    # the function of the form (e**x-1)/x embedded in above expression
    # for dsx was causing numerical difficulties when the denominator "x"
    # (i.e. bfac*esdc) became zero or approached zero (despite the fact
    # that the analytical function (e**x-1)/x has a well defined limit
    # as "x" approaches zero), hence below we replace the (e**x-1)/x
    # expression with an equivalent, numerically well-behaved
    # polynomial expansion.
    # The number of terms of polynomial expansion and its accuracy
    # is governed by iteration limit "ipol":
    # ipol greater than 9 only makes a difference on double
    # precision (relative errors given in percent %).
    # ipol=9, for rel.error <~ 1.6 e-6 % (8 significant digits)
    # ipol=8, for rel.error <~ 1.8 e-5 % (7 significant digits)
    # ipol=7, for rel.error <~ 1.8 e-4 % ...

    ipol = 4
    ii = 0
    pexp = 0.0
    while ii < ipol:
        pexp = (1.0 + pexp) * bfac * esdcx / (ipol + 1 - ii)
        ii += 1
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


def start_snopac(
    etp: FloatFieldIJ,
    ice: IntFieldIJ,
    prcp1: FloatFieldIJ,
    sncovr: FloatFieldIJ,
    snoexp: FloatFieldIJ,
    esdmin: FloatFieldIJ,
    et1: FloatField,
    et: FloatField,
    edir: FloatFieldIJ,
    edir1: FloatFieldIJ,
    ec: FloatFieldIJ,
    ec1: FloatFieldIJ,
    runoff1: FloatFieldIJ,
    runoff2: FloatFieldIJ,
    runoff3: FloatFieldIJ,
    drip: FloatFieldIJ,
    ett: FloatFieldIJ,
    ett1: FloatFieldIJ,
    etns: FloatFieldIJ,
    etns1: FloatFieldIJ,
    esnow: FloatFieldIJ,
    esnow1: FloatFieldIJ,
    esnow2: FloatFieldIJ,
    dew: FloatFieldIJ,
    etanrg: FloatFieldIJ,
    snopac_mask: BoolFieldIJ,
    evapo_mask: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(PARALLEL), interval(...):
        if snopac_mask:
            et1 = 0.0
            et = 0.0

    with computation(FORWARD), interval(0, 1):
        if snopac_mask:
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

            dew = 0.0
            etp1 = etp * 0.001

            if etp < 0.0:
                # dewfall (=frostfall in this case).
                dew = -etp1
                esnow2 = etp1 * dt
                etanrg = etp * (
                    (1.0 - sncovr) * physcons.LSUBC + sncovr * physcons.LSUBS
                )

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
                        evapo_mask = True


def update_temp_and_melt_snow(
    etp: FloatFieldIJ,
    ice: IntFieldIJ,
    sncovr: FloatFieldIJ,
    et1: FloatField,
    et: FloatField,
    edir1: FloatFieldIJ,
    ec1: FloatFieldIJ,
    ett1: FloatFieldIJ,
    etns1: FloatFieldIJ,
    edir: FloatFieldIJ,
    ec: FloatFieldIJ,
    ett: FloatFieldIJ,
    etns: FloatFieldIJ,
    etanrg: FloatFieldIJ,
    flx1: FloatFieldIJ,
    ffrozp: FloatFieldIJ,
    prcp: FloatFieldIJ,
    t1: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    zsoil: FloatFieldK,
    snowh: FloatFieldIJ,
    sneqv: FloatFieldIJ,
    snoexp: FloatFieldIJ,
    df1: FloatFieldIJ,
    rr: FloatFieldIJ,
    rch: FloatFieldIJ,
    fdown: FloatFieldIJ,
    flx2: FloatFieldIJ,
    sfcems: FloatFieldIJ,
    t24: FloatFieldIJ,
    th2: FloatFieldIJ,
    stc: FloatFieldIJ,
    esdmin: FloatFieldIJ,
    prcp1: FloatFieldIJ,
    snowng: BoolFieldIJ,
    snopac_mask: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(PARALLEL), interval(...):
        if snopac_mask:
            if etp >= 0.0:
                if ice == 0:
                    if sncovr < 1.0:
                        et1 *= 1.0 - sncovr
                        et = et1 * 1000.0

    with computation(FORWARD), interval(0, 1):
        if snopac_mask:
            if etp >= 0.0:
                if ice == 0:
                    if sncovr < 1.0:
                        edir1 *= 1.0 - sncovr
                        ec1 *= 1.0 - sncovr
                        ett1 *= 1.0 - sncovr
                        etns1 *= 1.0 - sncovr

                        edir = edir1 * 1000.0
                        ec = ec1 * 1000.0
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
                flx1 = (
                    (physcons.CPICE * ffrozp + physcons.CPH2O1 * (1.0 - ffrozp))
                    * prcp
                    * (t1 - sfctmp)
                )

            elif prcp > 0.0:
                flx1 = physcons.CPH2O1 * prcp * (t1 - sfctmp)

            # calculate an 'effective snow-grnd sfc temp' based on heat fluxes between
            # the snow pack and the soil and on net radiation.
            dsoil = -0.5 * zsoil
            dtot = snowh + dsoil
            denom = 1.0 + df1 / (dtot * rr * rch)
            t12a = (
                (fdown - flx1 - flx2 - sfcems * physcons.SIGMA1 * t24) / rch
                + th2
                - sfctmp
                - etanrg / rch
            ) / rr
            t12b = df1 * stc / (dtot * rr * rch)
            t12 = (sfctmp + t12a + t12b) / denom

            if t12 <= constants.TICE0:  # no snow melt will occur.

                # set the skin temp to this effective temp
                t1 = t12
                # update soil heat flux
                ssoil = df1 * (t1 - stc) / dtot
                # update depth of snowpack
                sneqv = max(0.0, sneqv - esnow2)
                flx3 = 0.0
                ex = 0.0
                snomlt = 0.0

            else:  # snow melt will occur.
                t1 = constants.TICE0 * max(0.01, sncovr ** snoexp) + t12 * (
                    1.0 - max(0.01, sncovr ** snoexp)
                )
                ssoil = df1 * (t1 - stc) / dtot

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

                    flx3 = (
                        fdown
                        - flx1
                        - flx2
                        - sfcems * (physcons.SIGMA1) * t14
                        - ssoil
                        - seh
                        - etanrg
                    )
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
                smflx_mask = True

            zz1 = 1.0
            yy = stc - 0.5 * ssoil * zsoil * zz1 / df1
            t11 = t1


def adjust_for_snow_compaction(
    ice: IntFieldIJ,
    sneqv: FloatFieldIJ,
    snowh: FloatFieldIJ,
    sndens: FloatFieldIJ,
    sncovr: FloatFieldIJ,
    t1: FloatFieldIJ,
    yy: FloatFieldIJ,
    snopac_mask: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(FORWARD), interval(0, 1):
        if snopac_mask:
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


class SNOPAC:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        ivegsrc: Int,
        lheatstrg: Bool,
        dt: Float,
    ):
        grid_indexing = stencil_factory.grid_indexing

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Float,
            )

        self._et1 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Float,
        )

        self._df1 = make_quantity_2d()
        self._eta1 = make_quantity_2d()
        self._etp1 = make_quantity_2d()
        self._prcp1 = make_quantity_2d()
        self._yy = make_quantity_2d()
        self._zz1 = make_quantity_2d()
        self._ec1 = make_quantity_2d()
        self._edir1 = make_quantity_2d()
        self._ett1 = make_quantity_2d()
        self._snoexp = make_quantity_2d()
        self._esdmin = make_quantity_2d()
        self._etns = make_quantity_2d()
        self._etns1 = make_quantity_2d()
        self._esnow1 = make_quantity_2d()
        self._esnow2 = make_quantity_2d()
        self._etanrg = make_quantity_2d()
        self._evapo_mask = make_quantity_2d()

        self._evapo_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._start_snopac = stencil_factory.stencil_factory.from_origin_domain(
            func=start_snopac,
            externals={
                "dt": dt,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._evapo = EvapoTranspiration(
            stencil_factory,
            quantity_factory,
            dt,
        )

        self._update_temp_and_melt_snow = (
            stencil_factory.stencil_factory.from_origin_domain(
                func=update_temp_and_melt_snow,
                externals={
                    "dt": dt,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        )

        self._smflx = SoilMoistureFlux(
            stencil_factory,
            quantity_factory,
            dt,
        )

        self._shflx = SoilHeatFlux(
            stencil_factory,
            quantity_factory,
            ivegsrc,
            lheatstrg,
            dt,
        )

        self._adjust_for_snow_compaction = (
            stencil_factory.stencil_factory.from_origin_domain(
                func=adjust_for_snow_compaction,
                externals={
                    "dt": dt,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        )

    def __call__(
        self,
        nroot: IntFieldIJ,
        etp: FloatFieldIJ,
        prcp: FloatFieldIJ,
        smcmax: FloatFieldIJ,
        smcwlt: FloatFieldIJ,
        smcref: FloatFieldIJ,
        smcdry: FloatFieldIJ,
        df1: FloatFieldIJ,
        sfcems: FloatFieldIJ,
        sfctmp: FloatFieldIJ,
        t24: FloatFieldIJ,
        th2: FloatFieldIJ,
        fdown: FloatFieldIJ,
        epsca: FloatFieldIJ,
        bexp: FloatFieldIJ,
        pc: FloatFieldIJ,
        rch: FloatFieldIJ,
        rr: FloatFieldIJ,
        slope: FloatFieldIJ,
        kdt: FloatFieldIJ,
        frzx: FloatFieldIJ,
        psisat: FloatFieldIJ,
        zsoil: FloatFieldK,
        dwsat: FloatFieldIJ,
        dksat: FloatFieldIJ,
        shdfac: FloatFieldIJ,
        ice: IntFieldIJ,
        rtdis: FloatField,
        quartz: FloatFieldIJ,
        flx2: FloatFieldIJ,
        snowng: BoolFieldIJ,
        ffrozp: FloatFieldIJ,
        vegtype: IntFieldIJ,
        prcp1: FloatFieldIJ,
        cmc: FloatFieldIJ,
        t1: FloatFieldIJ,
        stc: FloatField,
        sncovr: FloatFieldIJ,
        sneqv: FloatFieldIJ,
        sndens: FloatFieldIJ,
        snowh: FloatFieldIJ,
        sh2o: FloatField,
        tbot: FloatFieldIJ,
        smc: FloatField,
        ssoil: FloatFieldIJ,
        runoff1: FloatFieldIJ,
        runoff2: FloatFieldIJ,
        runoff3: FloatFieldIJ,
        edir: FloatFieldIJ,
        ec: FloatFieldIJ,
        et: FloatFieldIJ,
        ett: FloatFieldIJ,
        snomlt: FloatFieldIJ,
        drip: FloatFieldIJ,
        dew: FloatFieldIJ,
        flx1: FloatFieldIJ,
        flx3: FloatFieldIJ,
        esnow: FloatFieldIJ,
        snopac_mask: BoolFieldIJ,
        k_mask: IntField,
    ):
        """
        Fortran description:
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  subroutine snopac calculates soil moisture and heat flux values and  !
        !  update soil moisture content and soil heat content values for the    !
        !  case when a snow pack is present.                                    !
        !                                                                       !
        !                                                                       !
        !  subprograms called:  evapo, smflx, shflx, snowpack
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs from the calling program:                              size   !
        !     nsoil    - integer, number of soil layers                    1    !
        !     nroot    - integer, number of root layers                    1    !
        !     etp      - real, potential evaporation                       1    !
        !     prcp     - real, precip rate                                 1    !
        !     smcmax   - real, porosity                                    1    !
        !     smcwlt   - real, wilting point                               1    !
        !     smcref   - real, soil mois threshold                         1    !
        !     smcdry   - real, dry soil mois threshold                     1    !
        !     cmcmax   - real, maximum canopy water parameters             1    !
        !     dt       - real, time step                                   1    !
        !     df1      - real, thermal diffusivity                         m    !
        !     sfcems   - real, lw surface emissivity                       1    !
        !     sfctmp   - real, sfc temperature                             1    !
        !     t24      - real, sfctmp**4                                   1    !
        !     th2      - real, sfc air potential temperature               1    !
        !     fdown    - real, net solar + downward lw flux at sfc         1    !
        !     epsca    - real,                                             1    !
        !     bexp     - real, soil type "b" parameter                     1    !
        !     pc       - real, plant coeff                                 1    !
        !     rch      - real, companion coefficient of ch                 1    !
        !     rr       - real,                                             1    !
        !     cfactr   - real, canopy water parameters                     1    !
        !     slope    - real, linear reservoir coefficient                1    !
        !     kdt      - real,                                             1    !
        !     frzx     - real, frozen ground parameter                     1    !
        !     psisat   - real, saturated soil potential                    1    !
        !     zsoil    - real, soil layer depth below ground (negative)  nsoil  !
        !     dwsat    - real, saturated soil diffusivity                  1    !
        !     dksat    - real, saturated soil hydraulic conductivity       1    !
        !     zbot     - real, specify depth of lower bd soil              1    !
        !     shdfac   - real, aeral coverage of green vegetation          1    !
        !     ice      - integer, sea-ice flag (=1: sea-ice, =0: land)     1    !
        !     rtdis    - real, root distribution                         nsoil  !
        !     quartz   - real, soil quartz content                         1    !
        !     fxexp    - real, bare soil evaporation exponent              1    !
        !     csoil    - real, soil heat capacity                          1    !
        !     flx2     - real, freezing rain latent heat flux              1    !
        !     snowng   - logical, snow flag                                1    !
        !     lheatstrg- logical, flag for canopy heat storage             1    !
        !                         parameterization                              !
        !                                                                       !
        !  input/outputs from and to the calling program:                       !
        !     prcp1    - real, effective precip                            1    !
        !     cmc      - real, canopy moisture content                     1    !
        !     t1       - real, ground/canopy/snowpack eff skin temp        1    !
        !     stc      - real, soil temperature                          nsoil  !
        !     sncovr   - real, snow cover                                  1    !
        !     sneqv    - real, water-equivalent snow depth                 1    !
        !     sndens   - real, snow density                                1    !
        !     snowh    - real, snow depth                                  1    !
        !     sh2o     - real, unfrozen soil moisture                    nsoil  !
        !     tbot     - real, bottom soil temperature                     1    !
        !     beta     - real, ratio of actual/potential evap              1    !
        !                                                                       !
        !  outputs to the calling program:                                      !
        !     smc      - real, total soil moisture                       nsoil  !
        !     ssoil    - real, upward soil heat flux                       1    !
        !     runoff1  - real, surface runoff not infiltrating sfc         1    !
        !     runoff2  - real, sub surface runoff                          1    !
        !     runoff3  - real, excess of porosity for a given soil layer   1    !
        !     edir     - real, direct soil evaporation                     1    !
        !     ec       - real, canopy water evaporation                    1    !
        !     et       - real, plant transpiration                       nsoil  !
        !     ett      - real, total plant transpiration                   1    !
        !     snomlt   - real, snow melt water equivalent                  1    !
        !     drip     - real, through-fall of precip                      1    !
        !     dew      - real, dewfall (or frostfall)                      1    !
        !     flx1     - real, precip-snow sfc flux                        1    !
        !     flx3     - real, phase-change heat flux from snowmelt        1    !
        !     esnow    - real, sublimation from snowpack                   1    !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """

        self._start_snopac(
            etp,
            ice,
            prcp1,
            sncovr,
            self._snoexp,
            self._esdmin,
            self._et1,
            et,
            edir,
            self._edir1,
            ec,
            self._ec1,
            runoff1,
            runoff2,
            runoff3,
            drip,
            ett,
            self._ett1,
            self._etns,
            self._etns1,
            esnow,
            self._esnow1,
            self._esnow2,
            dew,
            self._etanrg,
            snopac_mask,
            self._evapo_mask,
        )

        self._evapo(
            nroot,
            cmc,
            self._etp1,
            sh2o,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            pc,
            shdfac,
            rtdis,
            self._etns1,
            self._edir1,
            self._ec1,
            self._et1,
            self._ett1,
            k_mask,
            self._evapo_mask,
        )

        self._update_temp_and_melt_snow(
            etp,
            ice,
            sncovr,
            self._et1,
            et,
            self._edir1,
            self._ec1,
            self._ett1,
            self._etns1,
            edir,
            ec,
            ett,
            self._etns,
            self._etanrg,
            flx1,
            ffrozp,
            prcp,
            t1,
            sfctmp,
            zsoil,
            snowh,
            sneqv,
            self._snoexp,
            df1,
            rr,
            rch,
            fdown,
            flx2,
            sfcems,
            t24,
            th2,
            stc,
            self._esdmin,
            prcp1,
            snowng,
            snopac_mask,
        )

        self._smflx(
            kdt,
            smcmax,
            smcwlt,
            self._prcp1,
            zsoil,
            slope,
            frzx,
            bexp,
            dksat,
            dwsat,
            shdfac,
            self._edir1,
            self._ec1,
            self._et1,
            cmc,
            sh2o,
            smc,
            runoff1,
            runoff2,
            runoff3,
            drip,
            snopac_mask,
        )

        self._shflx(
            smc,
            smcmax,
            self._yy,
            self._zz1,
            zsoil,
            psisat,
            bexp,
            self._df1,
            ice,
            quartz,
            vegtype,
            shdfac,
            stc,
            t1,
            tbot,
            sh2o,
            ssoil,
            snopac_mask,
        )

        self._adjust_for_snow_compaction(
            ice,
            sneqv,
            snowh,
            sndens,
            sncovr,
            t1,
            self._yy,
            snopac_mask,
        )
