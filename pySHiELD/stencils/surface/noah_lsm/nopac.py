from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval

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


def start_nopac(
    prcp: FloatFieldIJ,
    prcp1: FloatFieldIJ,
    etp: FloatFieldIJ,
    etp1: FloatFieldIJ,
    dew: FloatFieldIJ,
    edir: FloatFieldIJ,
    edir1: FloatFieldIJ,
    ec: FloatFieldIJ,
    ec1: FloatFieldIJ,
    ett: FloatFieldIJ,
    ett1: FloatFieldIJ,
    eta: FloatFieldIJ,
    eta1: FloatFieldIJ,
    et1: FloatField,
    et: FloatField,
    nopac_mask: BoolFieldIJ,
    evapo_mask: BoolFieldIJ,
):
    """
    !  subroutine nopac calculates soil moisture and heat flux values and   !
    !  update soil moisture content and soil heat content values for the    !
    !  case when no snow pack is present.
    """

    with computation(FORWARD), interval(0, 1):
        if nopac_mask:
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

    with computation(PARALLEL), interval(...):
        if nopac_mask:
            et1 = 0.0
            et = 0.0

    with computation(FORWARD), interval(0, 1):
        if nopac_mask:
            if etp > 0.0:
                evapo_mask = True
            else:
                evapo_mask = False
                eta1 = 0.0
                dew = -etp1
                prcp1 += dew


def prep_for_flux_calc(
    etp: FloatFieldIJ,
    etp1: FloatFieldIJ,
    dew: FloatFieldIJ,
    edir: FloatFieldIJ,
    edir1: FloatFieldIJ,
    ec: FloatFieldIJ,
    ec1: FloatFieldIJ,
    ett: FloatFieldIJ,
    ett1: FloatFieldIJ,
    eta: FloatFieldIJ,
    eta1: FloatFieldIJ,
    et1: FloatField,
    et: FloatField,
    smc: FloatField,
    quartz: FloatFieldIJ,
    smcmax: FloatFieldIJ,
    sh2o: FloatField,
    zsoil: FloatField,
    df1: FloatFieldIJ,
    vegtype: IntFieldIJ,
    shdfac: FloatFieldIJ,
    sbeta: FloatFieldIJ,
    fdown: FloatFieldIJ,
    t24: FloatFieldIJ,
    th2: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    sfcems: FloatFieldIJ,
    epsca: FloatFieldIJ,
    rch: FloatFieldIJ,
    rr: FloatFieldIJ,
    yy: FloatFieldIJ,
    zz1: FloatFieldIJ,
    flx1: FloatFieldIJ,
    flx3: FloatFieldIJ,
    nopac_mask: BoolFieldIJ,
):
    from __externals__ import lheatstrg, ivegsrc
    with computation(FORWARD), interval(...):
        if nopac_mask:
            et = et1 * 1000.0
    with computation(FORWARD), interval(0, 1):
        if nopac_mask:
            # convert modeled evapotranspiration fm  m s-1  to  kg m-2 s-1
            eta = eta1 * 1000.0
            edir = edir1 * 1000.0
            ec = ec1 * 1000.0
            ett = ett1 * 1000.0

            # based on etp and e values, determine beta
            if etp < 0.0:
                beta = 1.0
            elif etp == 0.0:
                beta = 0.0
            else:
                beta = eta / etp

            # get soil thermal diffuxivity/conductivity for top soil lyr, calc.
            df1 = tdfcnd(smc, quartz, smcmax, sh2o)

            if (not lheatstrg) and (ivegsrc == 1) and (vegtype == 13):
                df1 = 3.24 * (1.0 - shdfac) + shdfac * df1 * exp(sbeta * shdfac)
            else:
                df1 *= exp(sbeta * shdfac)

            # compute intermediate terms passed to routine hrt
            yynum = fdown - sfcems * physcons.SIGMA1 * t24
            yy = sfctmp + (yynum / rch + th2 - sfctmp - beta * epsca) / rr
            zz1 = df1 / (-0.5 * zsoil * rch * rr) + 1.0

            flx1 = 0.0
            flx3 = 0.0


class NOPAC:
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
        self._yynum = make_quantity_2d()
        self._zz1 = make_quantity_2d()
        self._ec1 = make_quantity_2d()
        self._edir1 = make_quantity_2d()
        self._ett1 = make_quantity_2d()

        self._evapo_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._start_nopac = stencil_factory.stencil_factory.from_origin_domain(
            func=start_nopac,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._evapo = EvapoTranspiration(
            stencil_factory,
            quantity_factory,
            dt,
        )

        self._smflx = SoilMoistureFlux(
            stencil_factory,
            quantity_factory,
            dt,
        )

        self._prep_for_flux_calc = stencil_factory.stencil_factory.from_origin_domain(
            func=prep_for_flux_calc,
            externals={
                "dt": dt,
                "lheatstrg": lheatstrg,
                "ivegsrc": ivegsrc,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

        self._shflx = SoilHeatFlux(
            stencil_factory,
            quantity_factory,
            ivegsrc,
            lheatstrg,
            dt,
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
        cmcmax: FloatFieldIJ,
        shdfac: FloatFieldIJ,
        sbeta: FloatFieldIJ,
        sfctmp: FloatFieldIJ,
        sfcems: FloatFieldIJ,
        t24: FloatFieldIJ,
        th2: FloatFieldIJ,
        fdown: FloatFieldIJ,
        epsca: FloatFieldIJ,
        bexp: FloatFieldIJ,
        pc: FloatFieldIJ,
        rch: FloatFieldIJ,
        rr: FloatFieldIJ,
        cfactr: FloatFieldIJ,
        slope: FloatFieldIJ,
        kdt: FloatFieldIJ,
        frzx: FloatFieldIJ,
        psisat: FloatFieldIJ,
        zsoil: FloatFieldK,
        dksat: FloatFieldIJ,
        dwsat: FloatFieldIJ,
        zbot: FloatFieldIJ,
        ice: IntFieldIJ,
        rtdis: FloatField,
        quartz: FloatFieldIJ,
        fxexp: FloatFieldIJ,
        csoil: FloatFieldIJ,
        vegtype: IntFieldIJ,
        cmc: FloatFieldIJ,
        t1: FloatFieldIJ,
        stc: FloatField,
        sh2o: FloatField,
        tbot: FloatFieldIJ,
        smc: FloatField,
        eta: FloatFieldIJ,
        ssoil: FloatFieldIJ,
        runoff1: FloatFieldIJ,
        runoff2: FloatFieldIJ,
        runoff3: FloatFieldIJ,
        edir: FloatFieldIJ,
        ec: FloatFieldIJ,
        et: FloatField,
        ett: FloatFieldIJ,
        beta: FloatFieldIJ,
        drip: FloatFieldIJ,
        dew: FloatFieldIJ,
        flx1: FloatFieldIJ,
        flx3: FloatFieldIJ,
        k_mask: IntField,
        nopac_mask: BoolFieldIJ
    ):
        """
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  subroutine nopac calculates soil moisture and heat flux values and   !
        !  update soil moisture content and soil heat content values for the    !
        !  case when no snow pack is present.                                   !
        !                                                                       !
        !                                                                       !
        !  subprograms called:  evapo, smflx, tdfcnd, shflx                     !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs from calling program:                                  size   !
        !     nsoil    - integer, number of soil layers                    1    !
        !     nroot    - integer, number of root layers                    1    !
        !     etp      - real, potential evaporation                       1    !
        !     prcp     - real, precip rate                                 1    !
        !     smcmax   - real, porosity (sat val of soil mois)             1    !
        !     smcwlt   - real, wilting point                               1    !
        !     smcref   - real, soil mois threshold                         1    !
        !     smcdry   - real, dry soil mois threshold                     1    !
        !     cmcmax   - real, maximum canopy water parameters             1    !
        !     dt       - real, time step                                   1    !
        !     shdfac   - real, aeral coverage of green veg                 1    !
        !     sbeta    - real, param to cal veg effect on soil heat flux   1    !
        !     sfctmp   - real, air temp at height zlvl abv ground          1    !
        !     sfcems   - real, sfc lw emissivity                           1    !
        !     t24      - real, sfctmp**4                                   1    !
        !     th2      - real, air potential temp at zlvl abv grnd         1    !
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
        !     dksat    - real, saturated soil hydraulic conductivity       1    !
        !     dwsat    - real, saturated soil diffusivity                  1    !
        !     zbot     - real, specify depth of lower bd soil              1    !
        !     ice      - integer, sea-ice flag (=1: sea-ice, =0: land)     1    !
        !     rtdis    - real, root distribution                         nsoil  !
        !     quartz   - real, soil quartz content                         1    !
        !     fxexp    - real, bare soil evaporation exponent              1    !
        !     csoil    - real, soil heat capacity                          1    !
        !     lheatstrg- logical, flag for canopy heat storage             1    !
        !                         parameterization                              !
        !                                                                       !
        !  input/outputs from and to the calling program:                       !
        !     cmc      - real, canopy moisture content                     1    !
        !     t1       - real, ground/canopy/snowpack eff skin temp        1    !
        !     stc      - real, soil temp                                 nsoil  !
        !     sh2o     - real, unfrozen soil moisture                    nsoil  !
        !     tbot     - real, bottom soil temp                            1    !
        !                                                                       !
        !  outputs to the calling program:                                      !
        !     eta      - real, downward latent heat flux                   1    !
        !     smc      - real, total soil moisture                       nsoil  !
        !     ssoil    - real, upward soil heat flux                       1    !
        !     runoff1  - real, surface runoff not infiltrating sfc         1    !
        !     runoff2  - real, sub surface runoff (baseflow)               1    !
        !     runoff3  - real, excess of porosity                          1    !
        !     edir     - real, direct soil evaporation                     1    !
        !     ec       - real, canopy water evaporation                    1    !
        !     et       - real, plant transpiration                       nsoil  !
        !     ett      - real, total plant transpiration                   1    !
        !     beta     - real, ratio of actual/potential evap              1    !
        !     drip     - real, through-fall of precip and/or dew           1    !
        !     dew      - real, dewfall (or frostfall)                      1    !
        !     flx1     - real, precip-snow sfc flux                        1    !
        !     flx3     - real, phase-change heat flux from snowmelt        1    !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """
        self._start_nopac(
            prcp,
            self._prcp1,
            etp,
            self._etp1,
            dew,
            edir,
            self._edir1,
            ec,
            self._ec1,
            ett,
            self._ett1,
            eta,
            self._eta1,
            self._et1,
            et,
            nopac_mask,
            self._evapo_mask,
        )

        self._evapo(
            nroot,
            cmc,
            cmcmax,
            self._etp1,
            sh2o,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            pc,
            shdfac,
            cfactr,
            rtdis,
            fxexp,
            self._eta1,
            self._edir1,
            self._ec1,
            self._et1,
            self._ett1,
            k_mask,
            self._evapo_mask,
        )

        self._smflx(
            kdt,
            smcmax,
            smcwlt,
            cmcmax,
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
            nopac_mask,
        )

        self._prep_for_flux_calc(
            etp,
            self._etp1,
            dew,
            edir,
            self._edir1,
            ec,
            self._ec1,
            ett,
            self._ett1,
            eta,
            self._eta1,
            self._et1,
            et,
            smc,
            quartz,
            smcmax,
            sh2o,
            zsoil,
            self._df1,
            vegtype,
            shdfac,
            sbeta,
            fdown,
            t24,
            th2,
            sfctmp,
            sfcems,
            epsca,
            rch,
            rr,
            self._yy,
            self._zz1,
            flx1,
            flx3,
            nopac_mask,
        )

        self._shflx(
            smc,
            smcmax,
            self._yy,
            self._zz1,
            zsoil,
            zbot,
            psisat,
            bexp,
            self._df1,
            ice,
            quartz,
            csoil,
            vegtype,
            shdfac,
            stc,
            t1,
            tbot,
            sh2o,
            ssoil,
            nopac_mask,
        )
