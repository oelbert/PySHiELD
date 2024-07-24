from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, computation, interval

import pySHiELD.constants as physcons
from ndsl.constants import X_DIM, Y_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory


@gtscript.function
def devap_fn(etp1, smc, shdfac, smcmax, smcdry):
    # calculates direct soil evaporation
    sratio = (smc - smcdry) / (smcmax - smcdry)

    if sratio > 0.0:
        fx = sratio ** physcons.FXEXP
        fx = max(min(fx, 1.0), 0.0)
    else:
        fx = 0.0

    # allow for the direct-evap-reducing effect of shade
    edir1 = fx * (1.0 - shdfac) * etp1
    return edir1


def start_evaporation(
    etp1: FloatFieldIJ,
    sh2o: FloatField,
    smcmax: FloatFieldIJ,
    smcdry: FloatFieldIJ,
    shdfac: FloatFieldIJ,
    ec1: FloatFieldIJ,
    ett1: FloatFieldIJ,
    edir1: FloatFieldIJ,
    et1: FloatField,
    evapo_mask: BoolFieldIJ,
    transp_mask: BoolFieldIJ,
):
    with computation(FORWARD), interval(...):
        if evapo_mask:
            et1 = 0.0

    with computation(FORWARD), interval(0, 1):
        if evapo_mask:
            ec1 = 0.0
            ett1 = 0.0
            edir1 = 0.0

            if etp1 > 0.0:
                # retrieve direct evaporation from soil surface.
                if shdfac < 1.0:
                    edir1 = devap_fn(etp1, sh2o, shdfac, smcmax, smcdry)
                    # edir1 = 4.250472271407341e-10

                # initialize plant total transpiration, retrieve plant
                # transpiration and accumulate it for all soil layers.
                transp_mask = True if shdfac > 0.0 else False


def transpiration(
    nroot: IntFieldIJ,
    etp1: FloatFieldIJ,
    smc: FloatField,
    smcwlt: FloatFieldIJ,
    smcref: FloatFieldIJ,
    cmc: FloatFieldIJ,
    shdfac: FloatFieldIJ,
    pc: FloatFieldIJ,
    rtdis: FloatField,
    et1: FloatField,
    sgx: FloatFieldIJ,
    transp_mask: BoolFieldIJ,
    k_mask: IntField,
):
    """
    Fortran name is transp
    Calculates transpiration for the veg class
    ! ===================================================================== !
    !  description:                                                         !
    !     subroutine transp calculates transpiration for the veg class.     !
    !                                                                       !
    !  subprogram called:  none                                             !
    !                                                                       !
    !                                                                       !
    !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs:                                                       size   !
    !     nroot    - integer, number of root layers                    1    !
    !     etp1     - real, potential evaporation                       1    !
    !     smc      - real, unfrozen soil moisture                    nsoil  !
    !     smcwlt   - real, wilting point                               1    !
    !     smcref   - real, soil mois threshold                         1    !
    !     cmc      - real, canopy moisture content                     1    !
    !     cmcmax   - real, maximum canopy water parameters             1    !
    !     shdfac   - real, aeral coverage of green vegetation          1    !
    !     pc       - real, plant coeff                                 1    !
    !     cfactr   - real, canopy water parameters                     1    !
    !     rtdis    - real, root distribution                         nsoil  !
    !                                                                       !
    !  outputs:                                                             !
    !     et1      - real, plant transpiration                       nsoil  !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """
    with computation(FORWARD), interval(...):
        if transp_mask:
            # initialize plant total transpiration, retrieve plant
            # transpiration and accumulate it for all soil layers.
            # initialize plant transp to zero for all soil layers.
            et1 = 0.0

    with computation(FORWARD), interval(0, 1):
        if transp_mask:
            if cmc != 0.0:
                etp1a = (
                    shdfac
                    * pc
                    * etp1
                    * (1.0 - (cmc / physcons.CMCMAX) ** physcons.CFACTR)
                )
            else:
                etp1a = shdfac * pc * etp1
            sgx = 0.0

    with computation(FORWARD), interval(...):
        if transp_mask:
            if nroot > k_mask:
                gx = max(0.0, min(1.0, (smc - smcwlt) / (smcref - smcwlt)))
                sgx = sgx + gx
    with computation(FORWARD), interval(0, 1):
        if transp_mask:
            sgx = sgx / nroot
            denom = 0.0

    with computation(FORWARD), interval(...):
        if transp_mask:
            rtx = rtdis + gx - sgx
            gx *= max(rtx, 0.0)

            denom += gx

    with computation(FORWARD), interval(0, 1):
        if transp_mask:
            if denom <= 0.0:
                denom = 1.0

    with computation(FORWARD), interval(...):
        if transp_mask:
            et1 = etp1a * gx / denom

            # return et1


def finish_evaporation(
    cmc: FloatFieldIJ,
    eta1: FloatFieldIJ,
    edir1: FloatFieldIJ,
    ec1: FloatFieldIJ,
    et1: FloatField,
    ett1: FloatFieldIJ,
    etp1: FloatFieldIJ,
    shdfac: FloatFieldIJ,
    evapo_mask: BoolFieldIJ,
    transp_mask: BoolFieldIJ,
):
    from __externals__ import dt

    with computation(FORWARD), interval(...):
        if transp_mask:
            ett1 = ett1 + et1

    with computation(FORWARD), interval(0, 1):
        if evapo_mask:
            if etp1 > 0.0:
                if shdfac > 0.0:
                    # calculate canopy evaporation.
                    if cmc > 0.0:
                        ec1 = (
                            shdfac * ((cmc / physcons.CMCMAX) ** physcons.CFACTR) * etp1
                        )
                    else:
                        ec1 = 0.0

                    # ec should be limited by the total amount
                    # of available water on the canopy
                    cmc2ms = cmc / dt
                    ec1 = min(cmc2ms, ec1)

            eta1 = edir1 + ett1 + ec1


class EvapoTranspiration:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        dt: Float,
    ):
        """
        Fortran name is evapo
        """

        grid_indexing = stencil_factory.grid_indexing

        self._transp_mask = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Bool,
        )

        self._sgx = quantity_factory.zeros(
            dims=[X_DIM, Y_DIM],
            units="",
            dtype=Float,
        )

        self._start_evaporation = stencil_factory.stencil_factory.from_origin_domain(
            func=start_evaporation,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._transpiration = stencil_factory.stencil_factory.from_origin_domain(
            func=transpiration,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._finish_evaporation = stencil_factory.stencil_factory.from_origin_domain(
            func=finish_evaporation,
            externals={"dt": dt},
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        nroot: IntFieldIJ,
        cmc: FloatFieldIJ,
        etp1: FloatFieldIJ,
        sh2o: FloatField,
        smcmax: FloatFieldIJ,
        smcwlt: FloatFieldIJ,
        smcref: FloatFieldIJ,
        smcdry: FloatFieldIJ,
        pc: FloatFieldIJ,
        shdfac: FloatFieldIJ,
        rtdis: FloatField,
        eta1: FloatFieldIJ,
        edir1: FloatFieldIJ,
        ec1: FloatFieldIJ,
        et1: FloatField,
        ett1: FloatFieldIJ,
        k_mask: IntField,
        evapo_mask: BoolFieldIJ,
    ):
        """
        Description from Fortran:
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  subroutine evapo calculates soil moisture flux.  the soil moisture   !
        !  content (smc - a per unit volume measurement) is a dependent variable!
        !  that is updated with prognostic eqns. the canopy moisture content    !
        !  (cmc) is also updated. frozen ground version:  new states added:     !
        !  sh2o, and frozen ground correction factor, frzfact and parameter     !
        !  slope.                                                               !
        !                                                                       !
        !                                                                       !
        !  subprogram called:  devap, transp                                    !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs from calling program:                                  size   !
        !     nroot    - integer, number of root layers                    1    !
        !     cmc      - real, canopy moisture content                     1    !
        !     cmcmax   - real, maximum canopy water parameters             1    !
        !     etp1     - real, potential evaporation                       1    !
        !     dt       - real, time step                                   1    !
        !     zsoil    - real, soil layer depth below ground             nsoil  !
        !     sh2o     - real, unfrozen soil moisture                    nsoil  !
        !     smcmax   - real, porosity                                    1    !
        !     smcwlt   - real, wilting point                               1    !
        !     smcref   - real, soil mois threshold                         1    !
        !     smcdry   - real, dry soil mois threshold                     1    !
        !     pc       - real, plant coeff                                 1    !
        !     cfactr   - real, canopy water parameters                     1    !
        !     rtdis    - real, root distribution                         nsoil  !
        !     fxexp    - real, bare soil evaporation exponent              1    !
        !                                                                       !
        !  outputs to calling program:                                          !
        !     eta1     - real, latent heat flux                            1    !
        !     edir1    - real, direct soil evaporation                     1    !
        !     ec1      - real, canopy water evaporation                    1    !
        !     et1      - real, plant transpiration                       nsoil  !
        !     ett1     - real, total plant transpiration                   1    !
        !                                                                       !
        !  ====================    end of description    =====================  !
        """

        self._start_evaporation(
            etp1,
            sh2o,
            smcmax,
            smcdry,
            shdfac,
            ec1,
            ett1,
            edir1,
            et1,
            evapo_mask,
            self._transp_mask,
        )

        self._transpiration(
            nroot,
            etp1,
            sh2o,
            smcwlt,
            smcref,
            cmc,
            shdfac,
            pc,
            rtdis,
            et1,
            self._sgx,
            self._transp_mask,
            k_mask,
        )

        self._finish_evaporation(
            cmc,
            eta1,
            edir1,
            ec1,
            et1,
            ett1,
            etp1,
            shdfac,
            evapo_mask,
            self._transp_mask,
        )
