from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, PARALLEL, computation, exp, interval, log

import ndsl.constants as constants
import pySHiELD.constants as physcons
from pySHiELD.stencils.surface.noah_lsm.evapo import EvapoTranspiration
from pySHiELD.stencils.surface.noah_lsm.smflx import SoilMoistureFlux
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
from pySHiELD._config import SurfaceConfig
from pySHiELD.functions.physics_functions import fpvs


def start_nopac(
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
    kmask: IntFieldIJ,
    snopac_mask: BoolFieldIJ,
    evapo_mask: BoolFieldIJ,
):
    """
    !  subroutine nopac calculates soil moisture and heat flux values and   !
    !  update soil moisture content and soil heat content values for the    !
    !  case when no snow pack is present.
    """
    from __externals__ import dt

    with computation(FORWARD), interval(0, 1):
        if not snopac_mask:
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
        if not snopac_mask:
            et1 = 0.0
            et = 0.0

    with computation(FORWARD), interval(0, 1):
        if not snopac_mask:
            if etp > 0.0:
                evapo_mask = True
            else:
                evapo_mask = False
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

def nopac_2():
    with computation(FORWARD), interval(0, 1):
        if not snopac_mask:
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
            df1 = tdfcnd(smc0, quartz, smcmax, sh2o0)

            if (ivegsrc == 1) and (vegtype == 12):
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
                vegtype,
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

        self._evapo_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._start_nopac = stencil_factory.stencil_factory.from_origin_domain(
            func=start_nopac,
            externals={"dt": dt},
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

        pass

    def __call__(
        self,
        snopac_mask: BoolFieldIJ
    ):
        self._start_nopac(
            snopac_mask,
            self._evapo_mask
        )
        self._evapo(
            self._evapo_mask
        )
        self._smflx(
            snopac_mask
        )

        pass
