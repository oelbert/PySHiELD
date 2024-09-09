from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import FORWARD, computation, interval

import ndsl.constants as constants
import pySHiELD.constants as physcons

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatFieldIJ,
    FloatField,
    Int,
    IntFieldIJ,
)
from pySHiELD.functions.physics_functions import fpvsx
from pySHiELD._config import FloatFieldTracer


@gtscript.function
def ice3lay(
    fice,
    hfi,
    hfd,
    sneti,
    focn,
    snowd,
    hice,
    stc0,
    stc1,
    tice,
    snof,
    snowmt,
    gflux,
):
    """three-layer sea ice vertical thermodynamics
                                                                           *
    based on:  m. winton, "a reformulated three-layer sea ice model",      *
    journal of atmospheric and oceanic technology, 2000                    *
                                                                           *
                                                                           *
          -> +---------+ <- tice - diagnostic surface temperature ( <= 0c )*
         /   |         |                                                   *
     snowd   |  snow   | <- 0-heat capacity snow layer                     *
         \   |         |                                                   *
          => +---------+                                                   *
         /   |         |                                                   *
        /    |         | <- t1 - upper 1/2 ice temperature; this layer has *
       /     |         |         a variable (t/s dependent) heat capacity  *
     hice    |...ice...|                                                   *
       \     |         |                                                   *
        \    |         | <- t2 - lower 1/2 ice temp. (fixed heat capacity) *
         \   |         |                                                   *
          -> +---------+ <- base of ice fixed at seawater freezing temp.   *
                                                                           *
    =====================  definition of variables  =====================  *
                                                                           *
    inputs:                                                         size   *
       fice     - real, sea-ice concentration                         im   *
       hfi      - real, net non-solar and heat flux @ surface(w/m^2)  im   *
       hfd      - real, heat flux derivatice @ sfc (w/m^2/deg-c)      im   *
       sneti    - real, net solar incoming at top  (w/m^2)            im   *
       focn     - real, heat flux from ocean    (w/m^2)               im   *
       delt     - real, timestep                (sec)                 1    *
                                                                           *
    input/outputs:                                                         *
       snowd    - real, surface pressure                              im   *
       hice     - real, sea-ice thickness                             im   *
       stc0     - real, temp @ midpt of ice levels (deg c), 1st layer im   *
       stc1     - real, temp @ midpt of ice levels (deg c), 2nd layer im   *
       tice     - real, surface temperature     (deg c)               im   *
       snof     - real, snowfall rate           (m/sec)               im   *
                                                                           *
    outputs:                                                               *
       snowmt   - real, snow melt during delt   (m)                   im   *
       gflux    - real, conductive heat flux    (w/m^2)               im   *
                                                                           *
    locals:                                                                *
       hdi      - real, ice-water interface     (m)                        *
       hsni     - real, snow-ice                (m)                        *
                                                                           *
    ====================================================================== *
    """
    from __externals__ import delt
    # constants
    TFI = -physcons.MU * physcons.SI  # sea ice freezing temp = -MU*salinity
    TFI0 = TFI - 0.0001

    snowd = snowd * constants.RHO_H2O / physcons.RHO_SNO
    hdi = physcons.RHO_SNO / constants.RHO_H2O * snowd + (
        physcons.RHO_ICE / constants.RHO_H2O * hice
    )

    if hice < hdi:
        snowd = snowd + hice - hdi
        hice = hice + (hdi - hice) * physcons.RHO_SNO / physcons.RHO_ICE

    snof = snof * constants.RHO_H2O / physcons.RHO_SNO
    tice = tice - constants.TICE0
    stc0 = stc0 - constants.TICE0 if stc0 - constants.TICE0 < TFI0 else TFI0
    stc1 = TFI0 if TFI0 < stc1 - constants.TICE0 else stc1 - constants.TICE0  # degc

    ip = physcons.I0 * sneti  # ip +v here (in winton ip=-I0*sneti)
    if snowd > 0.0:
        tsf = 0.0
        ip = 0.0
    else:
        tsf = TFI

    tice = tsf if tsf < tice else tice

    # compute ice temperature

    bi = hfd
    ai = hfi - sneti + ip - tice * bi  # +v sol input here
    k12 = (physcons.KI * 4.0) * physcons.KS / (
        physcons.KS * hice + (physcons.KI * 4.0) * snowd
    )
    k32 = (physcons.KI + physcons.KI) / hice

    wrk = 1.0 / (6.0 * delt * k32 + physcons.RHO_ICE / physcons.CI * hice)
    a10 = physcons.RHO_ICE / physcons.CI * hice * (0.5 / delt) + k32 * (
        4.0 * delt * k32 + physcons.RHO_ICE / physcons.CI * hice
    ) * wrk
    b10 = (
        -physcons.RHO_ICE * hice * (
            physcons.CI * stc0 + physcons.LI * TFI / stc0
        ) * (0.5 / delt)
        - ip
        - k32 * ((
            4.0 * delt * k32 * physcons.TFW
        ) + physcons.RHO_ICE / physcons.CI * hice * stc1) * wrk
    )

    wrk1 = k12 / (k12 + bi)
    a1 = a10 + bi * wrk1
    b1 = b10 + ai * wrk1
    c1 = physcons.RHO_ICE / physcons.LI * TFI * (0.5 / delt) * hice

    stc0 = -((b1 * b1 - 4.0 * a1 * c1) ** 0.5 + b1) / (a1 + a1)
    tice = (k12 * stc0 - ai) / (k12 + bi)

    if tice > tsf:
        a1 = a10 + k12
        b1 = b10 - k12 * tsf
        stc0 = -((b1 * b1 - 4.0 * a1 * c1) ** 0.5 + b1) / (a1 + a1)
        tice = tsf
        tmelt = (k12 * (stc0 - tsf) - (ai + bi * tsf)) * delt
    else:
        tmelt = 0.0
        snowd = snowd + snof * delt

    stc1 = (2.0 * delt * k32 * (
        stc0 + physcons.TFW + physcons.TFW
    ) + physcons.RHO_ICE / physcons.CI * hice * stc1) * wrk
    bmelt = (focn + (physcons.KI * 4.0) * (stc1 - physcons.TFW) / hice) * delt

    # resize the ice ...

    h1 = 0.5 * hice
    h2 = 0.5 * hice

    # top ...
    if tmelt <= snowd * physcons.RHO_SNO / physcons.LI:
        snowmt = tmelt / (physcons.RHO_SNO / physcons.LI)
        snowd = snowd - snowmt
    else:
        snowmt = snowd
        h1 = h1 - (tmelt - snowd * physcons.RHO_SNO / physcons.LI) / (
            physcons.RHO_ICE * (physcons.CI - physcons.LI / stc0) * (TFI - stc0)
        )
        snowd = 0.0

    # and bottom

    if bmelt < 0.0:
        dh = -bmelt / (
            physcons.RHO_ICE / physcons.LI + physcons.RHO_ICE / physcons.CI * (
                TFI - physcons.TFW
            )
        )
        stc1 = (h2 * stc1 + dh * physcons.TFW) / (h2 + dh)
        h2 = h2 + dh
    else:
        h2 = h2 - bmelt / (
            physcons.RHO_ICE / physcons.LI + physcons.RHO_ICE / physcons.CI * (
                TFI - stc1
            )
        )

    # if ice remains, even up 2 layers, else, pass negative energy back in snow

    hice = h1 + h2

    # begin if_hice_block
    if hice > 0.0:
        if h1 > 0.5 * hice:
            f1 = 1.0 - 2.0 * h2 / hice
            stc1 = f1 * (
                stc0 + physcons.LI * TFI / (physcons.CI * stc0)
            ) + (1.0 - f1) * stc1

            if stc1 > TFI:
                hice = hice - h2 * physcons.CI * (stc1 - TFI) / (physcons.LI * delt)
                stc1 = TFI

        else:
            f1 = 2.0 * h1 / hice
            stc0 = f1 * (
                stc0 + physcons.LI * TFI / (physcons.CI * stc0)
            ) + (1.0 - f1) * stc1
            stc0 = (
                stc0 - (stc0 * stc0 - 4.0 * TFI * physcons.LI / physcons.CI) ** 0.5
            ) * 0.5

        k12 = (physcons.KI * 4.0) * physcons.KS / (
            physcons.KS * hice + (physcons.KI * 4.0) * snowd
        )
        gflux = k12 * (stc0 - tice)

    else:
        snowd = (
            snowd
            + (
                h1 * (physcons.CI * (stc0 - TFI) - physcons.LI * (1.0 - TFI / stc0))
                + h2 * (physcons.CI * (stc1 - TFI) - physcons.LI)
            )
            / physcons.LI
        )
        hice = snowd * physcons.RHO_SNO / physcons.RHO_ICE if (
            snowd * physcons.RHO_SNO / physcons.RHO_ICE < 0.
        ) else 0.0
        snowd = 0.0
        stc0 = physcons.TFW
        stc1 = physcons.TFW
        gflux = 0.0

    gflux = fice * gflux
    snowmt = snowmt * physcons.RHO_SNO / constants.RHO_H2O
    snowd = snowd * physcons.RHO_SNO / constants.RHO_H2O
    tice = tice + constants.TICE0
    stc0 = stc0 + constants.TICE0
    stc1 = stc1 + constants.TICE0

    return snowd, hice, stc0, stc1, tice, snof, snowmt, gflux


def sfc_sice(
    ps: FloatFieldIJ,
    wind: FloatFieldIJ,
    t1: FloatField,
    q1: FloatFieldTracer,
    sfcemis: FloatFieldIJ,
    dlwflx: FloatFieldIJ,
    sfcnsw: FloatFieldIJ,
    sfcdsw: FloatFieldIJ,
    srflag: FloatFieldIJ,
    cm: FloatFieldIJ,
    ch: FloatFieldIJ,
    prsl1: FloatFieldIJ,
    prslki: FloatFieldIJ,
    islimsk: IntFieldIJ,
    flag_iter: BoolFieldIJ,
    hice: FloatFieldIJ,
    fice: FloatFieldIJ,
    tice: FloatFieldIJ,
    weasd: FloatFieldIJ,
    tskin: FloatFieldIJ,
    tprcp: FloatFieldIJ,
    stc0: FloatFieldIJ,
    stc1: FloatFieldIJ,
    ep: FloatFieldIJ,
    snwdph: FloatFieldIJ,
    qsurf: FloatFieldIJ,
    cmm: FloatFieldIJ,
    chh: FloatFieldIJ,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    gflux: FloatFieldIJ,
    snowmt: FloatFieldIJ,
):
    from __externals__ import mom4ice, lsm
    with computation(FORWARD), interval(0, 1):
        # set flag for sea-ice
        flag = (islimsk == 2) and flag_iter

        if flag_iter and (islimsk < 2):
            hice = 0.0
            fice = 0.0

        if flag:
            if mom4ice:
                hi_save = hice
                hs_save = weasd * 0.001
            elif lsm > 0:
                if srflag == 1.0:
                    ep = 0.
                    weasd = weasd + 1.e3 * tprcp
                    tprcp = 0.0

            #     initialize variables. all units are supposedly m.k.s. unless specified
            #     psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
            #     temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
            #     is sat. hum. at surface
            #     convert slrad to the civilized unit from langley minute-1 k-4

            # dlwflx has been given a negative sign for downward longwave
            # sfcnsw is the net shortwave flux (direction: dn-up)

            q0 = max(q1[0, 0, 0][0], physcons.FLOAT_EPS)
            theta1 = t1 * prslki
            rho = prsl1 / (constants.RDGAS * t1 * (1.0 + constants.ZVIR * q0))
            qs1 = fpvsx(t1)
            qs1 = max(constants.EPS * qs1 / (
                prsl1 + (constants.EPS - 1) * qs1
            ), physcons.FLOAT_EPS)
            q0 = min(qs1, q0)

            if fice < physcons.CIMIN:
                fice = physcons.CIMIN
                tice = physcons.TSICE
                tskin = physcons.TSICE
            ffw = 1.0 - fice

            qssi = fpvsx(tice)
            qssw = fpvsx(physcons.TSICE)
            qssi = constants.EPS * qssi / (ps + (constants.EPS - 1) * qssi)
            qssw = constants.EPS * qssw / (ps + (constants.EPS - 1) * qssw)

            # snow depth in water equivalent is converted from mm to m unit

            if mom4ice:
                snowd = weasd * 0.001 / fice
            else:
                snowd = weasd * 0.001

            # when snow depth is less than 1 mm, a patchy snow is assumed and
            #           soil is allowed to interact with the atmosphere.
            #           we should eventually move to a linear combination of soil and
            #           snow under the condition of patchy snow.

            # rcp = rho CP ch v

            cmm = cm * wind
            chh = rho * ch * wind
            rch = chh * constants.CP_AIR

            # sensible and latent heat flux over open water & sea ice

            evapi = physcons.HOCP * rch * (qssi - q0)
            evapw = physcons.HOCP * rch * (qssw - q0)

            snetw = sfcdsw * (1.0 - physcons.ALBFW)
            snetw = min(3.0 * sfcnsw / (1.0 + 2.0 * ffw), snetw)
            sneti = (sfcnsw - ffw * snetw) / fice

            t12 = tice * tice
            t14 = t12 * t12

            # hfi = net non-solar and upir heat flux @ ice surface

            hfi = -dlwflx + sfcemis * constants.SBC * t14 + evapi + rch * (
                tice - theta1
            )
            hfd = (
                4.0 * sfcemis * constants.SBC * tice * t12
                + (1.0 + physcons.HOCP * constants.EPS * constants.HLV * qs1 / (
                    constants.RDGAS * t12
                )) * rch
            )

            t12 = physcons.TSICE * physcons.TSICE
            t14 = t12 * t12

            # hfw = net heat flux @ water surface (within ice)

            focn = 2.0  # heat flux from ocean - should be from ocn model
            snof = 0.0  # snowfall rate - snow accumulates in gbphys

            hice = max(min(hice, physcons.HIMAX), physcons.HIMIN)
            snowd = min(snowd, physcons.HSMAX)

            if snowd > 2.0 * hice:
                snowd = hice + hice

            # run the 3-layer ice model
            snowd, hice, stc0, stc1, tice, snof, snowmt, gflux = ice3lay(
                fice,
                hfi,
                hfd,
                sneti,
                focn,
                snowd,
                hice,
                stc0,
                stc1,
                tice,
                snof,
                snowmt,
                gflux,
            )

            if mom4ice:
                hice = hi_save
                snowd = hs_save

            if tice < physcons.TIMIN:
                tice = physcons.TIMIN

            if stc0 < physcons.TIMIN:
                stc0 = physcons.TIMIN

            if stc1 < physcons.TIMIN:
                stc1 = physcons.TIMIN

            tskin = tice * fice + physcons.TSICE * ffw
            stc0 = min(stc0, constants.TICE0)
            stc1 = min(stc1, constants.TICE0)

            # calculate sensible heat flux (& evap over sea ice)

            hflxi = rch * (tice - theta1)
            hflxw = rch * (physcons.TSICE - theta1)
            hflx = fice * hflxi + ffw * hflxw
            evap = fice * evapi + ffw * evapw

            # the rest of the output

            qsurf = q1[0, 0, 0][0] + evap / (physcons.HOCP * rch)

            # convert snow depth back to mm of water equivalent

            weasd = snowd * 1000.0
            snwdph = weasd * physcons.DSI  # snow depth in mm

            hflx = hflx / rho * 1.0 / constants.CP_AIR
            evap = evap / rho * 1.0 / constants.HLV


class SurfaceSeaIce:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        mom4ice: Bool,
        lsm: Int,
        dt_atmos: Float,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._sfc_sice = stencil_factory.from_origin_domain(
            sfc_sice,
            externals={
                "delt": dt_atmos,
                "mom4ice": mom4ice,
                "lsm": lsm,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        ps: FloatFieldIJ,
        wind: FloatFieldIJ,
        t1: FloatField,
        q1: FloatFieldTracer,
        sfcemis: FloatFieldIJ,
        dlwflx: FloatFieldIJ,
        sfcnsw: FloatFieldIJ,
        sfcdsw: FloatFieldIJ,
        srflag: FloatFieldIJ,
        cm: FloatFieldIJ,
        ch: FloatFieldIJ,
        prsl1: FloatFieldIJ,
        prslki: FloatFieldIJ,
        islimsk: IntFieldIJ,
        flag_iter: BoolFieldIJ,
        hice: FloatFieldIJ,
        fice: FloatFieldIJ,
        tice: FloatFieldIJ,
        weasd: FloatFieldIJ,
        tskin: FloatFieldIJ,
        tprcp: FloatFieldIJ,
        stc0: FloatFieldIJ,
        stc1: FloatFieldIJ,
        ep: FloatFieldIJ,
        snwdph: FloatFieldIJ,
        qsurf: FloatFieldIJ,
        cmm: FloatFieldIJ,
        chh: FloatFieldIJ,
        evap: FloatFieldIJ,
        hflx: FloatFieldIJ,
        gflux: FloatFieldIJ,
        snowmt: FloatFieldIJ,
    ):
        """
        This file contains the GFS thermodynamics surface ice model.

        Fortran description:
        ! ===================================================================== !
        !  description:                                                         !
        !                                                                       !
        !  usage:                                                               !
        !                                                                       !
        !    call sfc_sice                                                      !
        !       inputs:                                                         !
        !          ( im, km, ps, u1, v1, t1, q1, delt,                          !
        !            sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,                   !
        !            cm, ch, prsl1, prslki, islimsk,                            !
        !            flag_iter, mom4ice, lsm,                                   !
        !       input/outputs:                                                  !
        !            hice, fice, tice, weasd, tskin, tprcp, stc, ep,            !
        !       outputs:                                                        !
        !            snwdph, qsurf, snowmt, gflux, cmm, chh, evap, hflx )       !
        !                                                                       !
        !  subprogram called:  ice3lay.                                         !
        !                                                                       !
        !  program history log:                                                 !
        !         2005  --  xingren wu created  from original progtm and added  !
        !                     two-layer ice model                               !
        !         200x  -- sarah lu    added flag_iter                          !
        !    oct  2006  -- h. wei      added cmm and chh to output              !
        !         2007  -- x. wu modified for mom4 coupling (i.e. mom4ice)      !
        !         2007  -- s. moorthi micellaneous changes                      !
        !    may  2009  -- y.-t. hou   modified to include surface emissivity   !
        !                     effect on lw radiation. replaced the confusing    !
        !                     slrad with sfc net sw sfcnsw (dn-up). reformatted !
        !                     the code and add program documentation block.     !
        !    sep  2009 -- s. moorthi removed rcl, changed pressure units and    !
        !                     further optimized                                 !
        !    jan  2015 -- x. wu change "cimin = 0.15" for both                  !
        !                              uncoupled and coupled case               !
        !                                                                       !
        !                                                                       !
        !  ====================  defination of variables  ====================  !
        !                                                                       !
        !  inputs:                                                       size   !
        !     im, km   - integer, horiz dimension and num of soil layers   1    !
        !     ps       - real, surface pressure                            im   !
        !     wind     - real, surface layer wind                          im   !
        !     t1       - real, surface layer mean temperature ( k )        im   !
        !     q1       - real, surface layer mean specific humidity        im   !
        !     delt     - real, time interval (second)                      1    !
        !     sfcemis  - real, sfc lw emissivity ( fraction )              im   !
        !     dlwflx   - real, total sky sfc downward lw flux ( w/m**2 )   im   !
        !     sfcnsw   - real, total sky sfc netsw flx into ground(w/m**2) im   !
        !     sfcdsw   - real, total sky sfc downward sw flux ( w/m**2 )   im   !
        !     srflag   - real, snow/rain flag for precipitation            im   !
        !     cm       - real, surface exchange coeff for momentum (m/s)   im   !
        !     ch       - real, surface exchange coeff heat & moisture(m/s) im   !
        !     prsl1    - real, surface layer mean pressure                 im   !
        !     prslki   - real,                                             im   !
        !     islimsk  - integer, sea/land/ice mask (=0/1/2)               im   !
        !     flag_iter- logical,                                          im   !
        !     mom4ice  - logical,                                          im   !
        !     lsm      - integer, flag for land surface model scheme       1    !
        !                =0: use osu scheme; =1: use noah scheme                !
        !                                                                       !
        !  input/outputs:                                                       !
        !     hice     - real, sea-ice thickness                           im   !
        !     fice     - real, sea-ice concentration                       im   !
        !     tice     - real, sea-ice surface temperature                 im   !
        !     weasd    - real, water equivalent accumulated snow depth (mm)im   !
        !     tskin    - real, ground surface skin temperature ( k )       im   !
        !     tprcp    - real, total precipitation                         im   !
        !     stc      - real, soil temp (k)                              im,km !
        !     ep       - real, potential evaporation                       im   !
        !                                                                       !
        !  outputs:                                                             !
        !     snwdph   - real, water equivalent snow depth (mm)            im   !
        !     qsurf    - real, specific humidity at sfc                    im   !
        !     snowmt   - real, snow melt (m)                               im   !
        !     gflux    - real, soil heat flux (w/m**2)                     im   !
        !     cmm      - real,                                             im   !
        !     chh      - real,                                             im   !
        !     evap     - real, evaperation from latent heat flux           im   !
        !     hflx     - real, sensible heat flux                          im   !
        !                                                                       !
        ! ===================================================================== !
        """
        # TODO: make this fully 3D
        self._sfc_sice(
            ps,
            wind,
            t1,
            q1,
            sfcemis,
            dlwflx,
            sfcnsw,
            sfcdsw,
            srflag,
            cm,
            ch,
            prsl1,
            prslki,
            islimsk,
            flag_iter,
            hice,
            fice,
            tice,
            weasd,
            tskin,
            tprcp,
            stc0,
            stc1,
            ep,
            snwdph,
            qsurf,
            cmm,
            chh,
            evap,
            hflx,
            gflux,
            snowmt,
        )
