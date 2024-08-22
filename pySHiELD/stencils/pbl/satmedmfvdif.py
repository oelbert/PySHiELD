from gt4py.cartesian.gtscript import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    sqrt,
)

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
    Int,
    IntField,
    IntFieldIJ,
    set_4d_field_size,
)
from ndsl.initialization.allocator import QuantityFactory
from pySHiELD._config import COND_DIM, TRACER_DIM, PBLConfig
from pySHiELD.functions.physics_functions import fpvs
from pySHiELD.stencils.pbl.mfpblt import PBLMassFlux
from pySHiELD.stencils.pbl.mfscu import StratocumulusMassFlux
from pySHiELD.stencils.pbl.tridiag import tridi2, tridin, tridit


FloatFieldTracer = None

def init_turbulence(
    zi: FloatField,
    zl: FloatField,
    zm: FloatField,
    phii: FloatField,
    phil: FloatField,
    chz: FloatField,
    ckz: FloatField,
    area: FloatFieldIJ,
    gdx: FloatFieldIJ,
    tke: FloatField,
    q1: FloatFieldTracer,
    rdzt: FloatField,
    prn: FloatField,
    kx1: IntField,
    prsi: FloatField,
    k_mask: IntField,
    kinver: IntFieldIJ,
    tx1: FloatFieldIJ,
    tx2: FloatFieldIJ,
    xkzo: FloatField,
    xkzmo: FloatField,
    kpblx: IntFieldIJ,
    hpblx: FloatFieldIJ,
    pblflg: BoolFieldIJ,
    sfcflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    scuflg: BoolFieldIJ,
    zorl: FloatFieldIJ,
    dusfc: FloatFieldIJ,
    dvsfc: FloatFieldIJ,
    dtsfc: FloatFieldIJ,
    dqsfc: FloatFieldIJ,
    kpbl: IntFieldIJ,
    hpbl: FloatFieldIJ,
    rbsoil: FloatFieldIJ,
    radmin: FloatFieldIJ,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    lcld: IntFieldIJ,
    kcld: IntFieldIJ,
    theta: FloatField,
    prslk: FloatField,
    psk: FloatFieldIJ,
    t1: FloatField,
    pix: FloatField,
    qlx: FloatField,
    slx: FloatField,
    thvx: FloatField,
    qtx: FloatField,
    thlx: FloatField,
    thlvx: FloatField,
    svx: FloatField,
    thetae: FloatField,
    gotvx: FloatField,
    prsl: FloatField,
    plyr: FloatField,
    rhly: FloatField,
    qstl: FloatField,
    bf: FloatField,
    cfly: FloatField,
    crb: FloatFieldIJ,
    dtdz1: FloatField,
    evap: FloatFieldIJ,
    heat: FloatFieldIJ,
    hlw: FloatField,
    radx: FloatField,
    sflux: FloatFieldIJ,
    shr2: FloatField,
    stress: FloatFieldIJ,
    hsw: FloatField,
    thermal: FloatFieldIJ,
    tsea: FloatFieldIJ,
    u10m: FloatFieldIJ,
    ustar: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    v10m: FloatFieldIJ,
    xmu: FloatFieldIJ,
    ptop: FloatFieldIJ,
    pbot: FloatFieldIJ,
):
    from __externals__ import (
        dt2, km1, ntcw, ntiw, ntke, xkzm_h, xkzm_m, xkzm_s, cap_k0_land
    )

    with computation(FORWARD), interval(0, 1):
        pcnvflg = False
        scuflg = True
        dusfc = 0.0
        dvsfc = 0.0
        dtsfc = 0.0
        dqsfc = 0.0
        kpbl = 1
        hpbl = 0.0
        kpblx = 1
        hpblx = 0.0
        pblflg = True
        lcld = km1 - 1
        kcld = km1 - 1
        mrad = km1
        krad = 0
        radmin = 0.0
        ptop = phii
        sfcflg = True
        if rbsoil[0, 0] > 0.0:
            sfcflg = False
        gdx = sqrt(area[0, 0])

    with computation(PARALLEL), interval(...):
        zi = phii[0, 0, 0] * constants.RGRAV
        zl = phil[0, 0, 0] * constants.RGRAV
        tke = max(q1[0, 0, 0][ntke], physcons.TKMIN)
    with computation(PARALLEL), interval(0, -1):
        ckz = physcons.CK1
        chz = physcons.CH1
        prn = 1.0
        kx1 = 0.0
        zm = zi[0, 0, 1]
        rdzt = 1.0 / (zl[0, 0, 1] - zl[0, 0, 0])

        #  set background diffusivities as a function of
        #  horizontal grid size with xkzm_h & xkzm_m for gdx >= 25km
        #  and 0.01 for gdx=5m
        if gdx[0, 0] >= physcons.XKGDX:
            xkzm_hx = xkzm_h
            xkzm_mx = xkzm_m
        else:
            xkzm_hx = 0.01 + ((xkzm_h - 0.01) * (1.0 / (physcons.XKGDX - 5.0))) * (
                gdx[0, 0] - 5.0
            )
            xkzm_mx = 0.01 + ((xkzm_m - 0.01) * (1.0 / (physcons.XKGDX - 5.0))) * (
                gdx[0, 0] - 5.0
            )

        if k_mask[0, 0, 0] < kinver[0, 0]:
            ptem = prsi[0, 0, 1] * tx1[0, 0]
            xkzo = xkzm_hx * min(1.0, exp(-((1.0 - ptem) * (1.0 - ptem) * 10.0)))

            if ptem >= xkzm_s:
                xkzmo = xkzm_mx
                kx1 = k_mask[0, 0, 0] + 1
            else:
                tem1 = min(
                    1.0,
                    exp(
                        -(
                            (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * (1.0 - prsi[0, 0, 1] * tx2[0, 0])
                            * 5.0
                        )
                    ),
                )
                xkzmo = xkzm_mx * tem1

        pix = psk[0, 0] / prslk[0, 0, 0]
        theta = t1[0, 0, 0] * pix[0, 0, 0]
        if (ntiw + 1) > 0:
            tem = max(q1[0, 0, 0][ntcw], physcons.QLMIN)
            tem1 = max(q1[0, 0, 0][ntiw], physcons.QLMIN)
            ptem = constants.HLV * tem + (constants.HLV + constants.HLF) * tem1
            qlx = tem + tem1
            slx = constants.CP_AIR * t1[0, 0, 0] + phil[0, 0, 0] - ptem
        else:
            qlx = max(q1[0, 0, 0][ntcw], physcons.QLMIN)
            slx = (
                constants.CP_AIR * t1[0, 0, 0]
                + phil[0, 0, 0]
                - constants.HLV * qlx[0, 0, 0]
            )

        tem = 1.0 + constants.ZVIR * max(q1[0, 0, 0][0], physcons.QMIN) - qlx[0, 0, 0]
        thvx = theta[0, 0, 0] * tem
        qtx = max(q1[0, 0, 0][0], physcons.QMIN) + qlx[0, 0, 0]
        thlx = theta[0, 0, 0] - pix[0, 0, 0] * physcons.ELOCP * qlx[0, 0, 0]
        thlvx = thlx[0, 0, 0] * (1.0 + constants.ZVIR * qtx[0, 0, 0])
        svx = constants.CP_AIR * t1[0, 0, 0] * tem
        thetae = theta[0, 0, 0] + physcons.ELOCP * pix[0, 0, 0] * max(
            q1[0, 0, 0][0], physcons.QMIN
        )
        gotvx = constants.GRAV / (t1[0, 0, 0] * tem)

        tem = (t1[0, 0, 1] - t1[0, 0, 0]) * tem * rdzt[0, 0, 0]
        if cap_k0_land:
            if tem > 1.0e-5:
                xkzo = min(xkzo[0, 0, 0], physcons.XKZINV)
                xkzmo = min(xkzmo[0, 0, 0], physcons.XKZINV)

        #  Compute empirical cloud fraction based on Xu & Randall (1996, JAS)
        plyr = 0.01 * prsl[0, 0, 0]
        es = 0.01 * fpvs(t1)
        qs = max(
            physcons.QMIN,
            constants.EPS * es / (plyr[0, 0, 0] + (constants.EPS - 1) * es),
        )
        rhly = max(0.0, min(1.0, max(physcons.QMIN, q1[0, 0, 0][0]) / qs))
        qstl = qs

    with computation(FORWARD), interval(...):
        cfly = 0.0
        clwt = 1.0e-6 * (plyr[0, 0, 0] * 0.001)
        if qlx[0, 0, 0] > clwt:
            onemrh = max(1.0e-10, 1.0 - rhly[0, 0, 0])
            tem1 = physcons.CQL / min(
                max((onemrh * qstl[0, 0, 0]) ** 0.49, 0.0001), 1.0
            )
            val = max(min(tem1 * qlx[0, 0, 0], 50.0), 0.0)
            cfly = min(max(sqrt(sqrt(rhly[0, 0, 0])) * (1.0 - exp(-val)), 0.0), 1.0)

    #  Compute buoyancy modified by clouds
    with computation(PARALLEL), interval(0, -2):
        tem1 = 0.5 * (t1[0, 0, 0] + t1[0, 0, 1])
        cfh = min(cfly[0, 0, 1], 0.5 * (cfly[0, 0, 0] + cfly[0, 0, 1]))
        alp = constants.GRAV / (0.5 * (svx[0, 0, 0] + svx[0, 0, 1]))
        gamma = physcons.EL2ORC * (0.5 * (qstl[0, 0, 0] + qstl[0, 0, 1])) / (tem1 ** 2)
        epsi = tem1 / physcons.ELOCP
        beta = (1.0 + gamma * epsi * (1.0 + constants.ZVIR)) / (1.0 + gamma)
        chx = cfh * alp * beta + (1.0 - cfh) * alp
        cqx = cfh * alp * constants.HLV * (beta - epsi)
        cqx = cqx + (1.0 - cfh) * constants.ZVIR * constants.GRAV
        bf = chx * ((slx[0, 0, 1] - slx[0, 0, 0]) * rdzt[0, 0, 0]) + cqx * (
            (qtx[0, 0, 1] - qtx[0, 0, 0]) * rdzt[0, 0, 0]
        )
        radx = (zi[0, 0, 1] - zi[0, 0, 0]) * (hsw[0, 0, 0] * xmu[0, 0] + hlw[0, 0, 0])

    with computation(FORWARD):
        #  Compute critical bulk richardson number
        with interval(0, 1):
            sflux = heat[0, 0] + evap[0, 0] * constants.ZVIR * theta[0, 0, 0]

            if (not sfcflg[0, 0]) or (sflux[0, 0] <= 0.0):
                pblflg = False

            if pblflg[0, 0]:
                thermal = thlvx[0, 0, 0]
                crb = physcons.RBCR
            else:
                tem1 = 1e-7 * (
                    max(sqrt(u10m[0, 0] ** 2 + v10m[0, 0] ** 2), 1.0)
                    / (physcons.F0 * 0.01 * zorl[0, 0])
                )
                thermal = tsea[0, 0] * (
                    1.0 + constants.ZVIR * max(q1[0, 0, 0][0], physcons.QMIN)
                )
                crb = max(
                    min(0.16 * (tem1 ** (-0.18)), physcons.CRBMAX), physcons.CRBMIN
                )

            dtdz1 = dt2 / (zi[0, 0, 1] - zi[0, 0, 0])
            ustar = sqrt(stress[0, 0])
    #  Compute buoyancy (bf) and winshear square
    with computation(PARALLEL):
        with interval(0, -2):
            dw2 = (u1[0, 0, 0] - u1[0, 0, 1]) ** 2 + (v1[0, 0, 0] - v1[0, 0, 1]) ** 2
            shr2 = max(dw2, physcons.DW2MIN) * rdzt[0, 0, 0] * rdzt[0, 0, 0]
        with interval(-2, -1):
            pbot = phii


def mrf_pbl_scheme_part1(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    kpblx: IntFieldIJ,
    k_mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    rbsoil: FloatFieldIJ,
    thermal: FloatFieldIJ,
    thlvx: FloatField,
    thlvx_0: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    zl: FloatField,
):

    with computation(FORWARD):
        with interval(0, 1):
            rbup = rbsoil[0, 0]
            thlvx_0 = thlvx[0, 0, 0]

            if not flg[0, 0]:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (constants.GRAV * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = k_mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(1, None):
            if not flg[0, 0]:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (constants.GRAV * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpblx = k_mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]


def mrf_pbl_2_thermal_excess(
    crb: FloatFieldIJ,
    evap: FloatFieldIJ,
    fh: FloatFieldIJ,
    flg: BoolFieldIJ,
    fm: FloatFieldIJ,
    gotvx: FloatField,
    heat: FloatFieldIJ,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    k_mask: IntField,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    phih: FloatFieldIJ,
    phim: FloatFieldIJ,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    rbsoil: FloatFieldIJ,
    sfcflg: BoolFieldIJ,
    sflux: FloatFieldIJ,
    thermal: FloatFieldIJ,
    theta: FloatField,
    ustar: FloatFieldIJ,
    vpert: FloatFieldIJ,
    zi: FloatField,
    zl: FloatField,
    zol: FloatFieldIJ,
):

    with computation(FORWARD), interval(1, None):
        if k_mask[0, 0, 0] == kpblx[0, 0]:
            if kpblx[0, 0] > 0:
                if rbdn[0, 0] >= crb[0, 0]:
                    rbint = 0.0
                elif rbup[0, 0] <= crb[0, 0]:
                    rbint = 1.0
                else:
                    rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])
                hpblx = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

                if hpblx[0, 0] < zi[0, 0, 0]:
                    kpblx = kpblx[0, 0] - 1
            else:
                hpblx = zl[0, 0, 0]
                kpblx = 0

            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]

            if kpbl[0, 0] <= 0:
                pblflg = False

    with computation(FORWARD), interval(0, 1):
        zol = max(rbsoil[0, 0] * fm[0, 0] * fm[0, 0] / fh[0, 0], physcons.RIMIN)
        if sfcflg[0, 0]:
            zol = min(zol[0, 0], -physcons.ZFMIN)
        else:
            zol = max(zol[0, 0], physcons.ZFMIN)

        zol1 = zol[0, 0] * physcons.SFCFRAC * hpbl[0, 0] / zl[0, 0, 0]

        if sfcflg[0, 0]:
            phih = sqrt(1.0 / (1.0 - physcons.APHI16 * zol1))
            phim = sqrt(phih[0, 0])
        else:
            phim = 1.0 + physcons.APHI5 * zol1
            phih = phim[0, 0]

        pcnvflg = pblflg[0, 0] and (zol[0, 0] < physcons.ZOLCRU)

        wst3 = gotvx[0, 0, 0] * sflux[0, 0] * hpbl[0, 0]
        ust3 = ustar[0, 0] ** 3.0

        if pblflg[0, 0]:
            wscale = max(
                (ust3 + physcons.WFAC * physcons.VK * wst3 * physcons.SFCFRAC)
                ** physcons.H1,
                ustar[0, 0] / physcons.APHI5,
            )

        flg = 1

        if pcnvflg[0, 0]:
            hgamt = heat[0, 0] / wscale
            hgamq = evap[0, 0] / wscale
            vpert = max(hgamt + hgamq * constants.ZVIR * theta[0, 0, 0], 0.0)
            thermal = thermal[0, 0] + min(
                physcons.CFAC * vpert[0, 0], physcons.GAMCRT
            )
            flg = 0
            rbup = rbsoil[0, 0]


def thermal_excess_2(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    kpbl: IntFieldIJ,
    k_mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    thermal: FloatFieldIJ,
    thlvx: FloatField,
    thlvx_0: FloatFieldIJ,
    u1: FloatField,
    v1: FloatField,
    zl: FloatField,
):

    with computation(FORWARD):
        with interval(1, 2):
            thlvx_0 = thlvx[0, 0, -1]
            if flg[0, 0]:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (constants.GRAV * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = k_mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]

        with interval(2, None):
            if flg[0, 0]:
                rbdn = rbup[0, 0]
                rbup = (
                    (thlvx[0, 0, 0] - thermal[0, 0])
                    * (constants.GRAV * zl[0, 0, 0] / thlvx_0[0, 0])
                    / max(u1[0, 0, 0] ** 2 + v1[0, 0, 0] ** 2, 1.0)
                )
                kpbl = k_mask[0, 0, 0]
                flg = rbup[0, 0] > crb[0, 0]


def enhance_pbl_height_thermal(
    crb: FloatFieldIJ,
    flg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    lcld: IntFieldIJ,
    k_mask: IntField,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    scuflg: BoolFieldIJ,
    zi: FloatField,
    zl: FloatField,
):

    with computation(FORWARD), interval(1, None):
        if pcnvflg[0, 0] and (kpbl[0, 0] == k_mask[0, 0, 0]):
            if rbdn[0, 0] >= crb[0, 0]:
                rbint = 0.0
            elif rbup[0, 0] <= crb[0, 0]:
                rbint = 1.0
            else:
                rbint = (crb[0, 0] - rbdn[0, 0]) / (rbup[0, 0] - rbdn[0, 0])

            hpbl[0, 0] = zl[0, 0, -1] + rbint * (zl[0, 0, 0] - zl[0, 0, -1])

            if hpbl[0, 0] < zi[0, 0, 0]:
                kpbl[0, 0] = kpbl[0, 0] - 1

            if kpbl[0, 0] <= 0:
                pblflg[0, 0] = False
                pcnvflg[0, 0] = False

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]
            if flg[0, 0] and (zl[0, 0, 0] >= physcons.ZSTBLMAX):
                lcld = k_mask[0, 0, 0]
                flg = 0
        with interval(1, -1):
            if flg[0, 0] and (zl[0, 0, 0] >= physcons.ZSTBLMAX):
                lcld = k_mask[0, 0, 0]
                flg = 0


def stratocumulus(
    flg: BoolFieldIJ,
    kcld: IntFieldIJ,
    krad: IntFieldIJ,
    lcld: IntFieldIJ,
    k_mask: IntField,
    radmin: FloatFieldIJ,
    radx: FloatField,
    qlx: FloatField,
    scuflg: BoolFieldIJ,
):
    from __externals__ import km1

    with computation(FORWARD):
        with interval(0, 1):
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if (
                flg[0, 0]
                and (k_mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= physcons.QLCR)
            ):
                kcld = k_mask[0, 0, 0]
                flg = 0

        with interval(0, -1):
            if (
                flg[0, 0]
                and (k_mask[0, 0, 0] <= lcld[0, 0])
                and (qlx[0, 0, 0] >= physcons.QLCR)
            ):
                kcld = k_mask[0, 0, 0]
                flg = 0

    with computation(FORWARD):
        with interval(0, 1):
            if scuflg[0, 0] and (kcld[0, 0] == (km1 - 1)):
                scuflg = False
            flg = scuflg[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and (k_mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= physcons.QLCR:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = k_mask[0, 0, 0]
                else:
                    flg = 0

        with interval(0, -1):
            if flg[0, 0] and (k_mask[0, 0, 0] <= kcld[0, 0]):
                if qlx[0, 0, 0] >= physcons.QLCR:
                    if radx[0, 0, 0] < radmin[0, 0]:
                        radmin = radx[0, 0, 0]
                        krad = k_mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if scuflg[0, 0] and krad[0, 0] <= 0:
            scuflg = False
        if scuflg[0, 0] and radmin[0, 0] >= 0.0:
            scuflg = False


def compute_mass_flux_prelim(
    pcnvflg: BoolFieldIJ,
    scuflg: BoolFieldIJ,
    t1: FloatField,
    tcdo: FloatField,
    tcko: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
):
    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            tcko = t1[0, 0, 0]
            ucko = u1[0, 0, 0]
            vcko = v1[0, 0, 0]
        if scuflg[0, 0]:
            tcdo = t1[0, 0, 0]
            ucdo = u1[0, 0, 0]
            vcdo = v1[0, 0, 0]

def compute_mass_flux_tracer_prelim(
    qcko: FloatField,
    qcdo: FloatField,
    q1: FloatFieldTracer,
    pcnvflg: BoolFieldIJ,
    scuflg: BoolFieldIJ,
    n_extra: int
):
    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            qcko[0, 0, 0][n_extra] = q1[0, 0, 0][n_extra]
        if scuflg[0, 0]:
            qcdo[0, 0, 0][n_extra] = q1[0, 0, 0][n_extra]


def compute_prandtl_num_exchange_coeff(
    chz: FloatField,
    ckz: FloatField,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    k_mask: IntField,
    pcnvflg: BoolFieldIJ,
    phih: FloatFieldIJ,
    phim: FloatFieldIJ,
    prn: FloatField,
    zi: FloatField,
):

    with computation(PARALLEL), interval(...):
        tem1 = max(zi[0, 0, 1] - physcons.SFCFRAC * hpbl[0, 0], 0.0)
        ptem = -3.0 * (tem1 ** 2.0) / (hpbl[0, 0] ** 2.0)
        if k_mask[0, 0, 0] < kpbl[0, 0]:
            if pcnvflg[0, 0]:
                prn = 1.0 + ((phih[0, 0] / phim[0, 0]) - 1.0) * exp(ptem)
            else:
                prn = phih[0, 0] / phim[0, 0]

        if k_mask[0, 0, 0] < kpbl[0, 0]:
            prn = max(min(prn[0, 0, 0], physcons.PRMAX), physcons.PRMIN)
            ckz = max(
                min(
                    physcons.CK1 + (physcons.CK0 - physcons.CK1) * exp(ptem),
                    physcons.CK0,
                ),
                physcons.CK1,
            )
            chz = max(
                min(
                    physcons.CH1 + (physcons.CH0 - physcons.CH1) * exp(ptem),
                    physcons.CH0,
                ),
                physcons.CH1,
            )


def compute_asymptotic_mixing_length(
    zldn: FloatFieldIJ,
    thvx: FloatField,
    tke: FloatField,
    gotvx: FloatField,
    zl: FloatField,
    tsea: FloatFieldIJ,
    q1_gt: FloatField,
    zi: FloatField,
    rlam: FloatField,
    ele: FloatField,
    zol: FloatFieldIJ,
    gdx: FloatFieldIJ,
    phii: FloatField,
    ptop: FloatFieldIJ,
    pbot: FloatFieldIJ,
):
    with computation(FORWARD), interval(...):
        mlenflg = True
        zlup = 0.0
        bsum = 0.0
        lev = 0
        while phii[0, 0, lev] < pbot:  # strictly less-than to prevent illegal access
            if mlenflg:
                dz = zl[0, 0, lev + 1] - zl[0, 0, lev]
                ptem = gotvx[0, 0, lev] * (thvx[0, 0, lev + 1] - thvx) * dz
                bsum = bsum + ptem
                zlup = zlup + dz
                if bsum >= tke:
                    if ptem >= 0.0:
                        tem2 = max(ptem, physcons.ZFMIN)
                    else:
                        tem2 = min(ptem, -physcons.ZFMIN)
                    ptem1 = (bsum - tke) / tem2
                    zlup = zlup - ptem1 * dz
                    zlup = max(zlup, 0.0)
                    mlenflg = False
            lev += 1

            mlenflg = True
        bsum = 0.0
        zldn = 0.0
        lev = 0
        while phii[0, 0, lev] > ptop:  # strictly less-than to prevent illegal access
            if mlenflg:
                dz = zl[0, 0, lev] - zl[0, 0, lev - 1]
                tem1 = thvx[0, 0, lev - 1]
                ptem = gotvx[0, 0, lev] * (thvx - tem1) * dz
                bsum = bsum + ptem
                zldn = zldn + dz
                if bsum >= tke:
                    if ptem >= 0.0:
                        tem2 = max(ptem, physcons.ZFMIN)
                    else:
                        tem2 = min(ptem, -physcons.ZFMIN)
                    ptem1 = (bsum - tke) / tem2
                    zldn = zldn - ptem1 * dz
                    zldn = max(zldn, 0.0)
                    mlenflg = False
            lev -= 1
        dz = zl[0, 0, lev]
        tem1 = tsea * (1.0 + constants.ZVIR * max(q1_gt[0, 0, lev][0], physcons.QMIN))
        ptem = gotvx[0, 0, lev] * (thvx - tem1) * dz
        bsum = bsum + ptem
        zldn = zldn + dz
        if bsum >= tke:
            if ptem >= 0.0:
                tem2 = max(ptem, physcons.ZFMIN)
            else:
                tem2 = min(ptem, -physcons.ZFMIN)
            ptem1 = (bsum - tke) / tem2
            zldn = zldn - ptem1 * dz
            zldn = max(zldn, 0.0)
            mlenflg = False

        tem = 0.5 * (zi[0, 0, 1] - zi)
        tem1 = min(tem, physcons.RLMN)

        ptem2 = min(zlup, zldn)
        rlam = min(max(physcons.ELMFAC * ptem2, tem1), physcons.RLMX)

        ptem2 = sqrt(zlup * zldn)
        ele = min(max(physcons.ELEFAC * ptem2, tem1), physcons.ELMX)

    with computation(FORWARD):
        with interval(0, -1):
            if zol < 0.0:
                zk = physcons.VK * zl * (1.0 - 100.0 * zol) ** 0.2
            elif zol >= 1.0:
                zk = physcons.VK * zl / 3.7
            else:
                zk = physcons.VK * zl / (1.0 + 2.7 * zol)

            elm = zk * rlam / (rlam + zk)
            dz = zi[0, 0, 1] - zi
            tem = max(gdx, dz)
            elm = min(elm, tem)
            ele = min(ele, tem)

        with interval(-1, None):
            elm = elm[0, 0, -1]
            ele = ele[0, 0, -1]


def compute_eddy_diffusivity_buoy_shear(
    bf: FloatField,
    buod: FloatField,
    buou: FloatField,
    chz: FloatField,
    ckz: FloatField,
    dku: FloatField,
    dkt: FloatField,
    dkq: FloatField,
    ele: FloatField,
    elm: FloatField,
    gotvx: FloatField,
    kpbl: IntFieldIJ,
    k_mask: IntField,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    pblflg: BoolFieldIJ,
    pcnvflg: BoolFieldIJ,
    phim: FloatFieldIJ,
    prn: FloatField,
    prod: FloatField,
    radj: FloatFieldIJ,
    rdzt: FloatField,
    rle: FloatField,
    scuflg: BoolFieldIJ,
    sflux: FloatFieldIJ,
    shr2: FloatField,
    stress: FloatFieldIJ,
    tke: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    ustar: FloatFieldIJ,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
    xkzo: FloatField,
    xkzmo: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
    zl: FloatField,
):
    with computation(PARALLEL), interval(0, -1):
        tem = (
            0.5
            * (elm[0, 0, 0] + elm[0, 0, 1])
            * sqrt(0.5 * (tke[0, 0, 0] + tke[0, 0, 1]))
        )
        ri = max(bf[0, 0, 0] / shr2[0, 0, 0], physcons.RIMIN)

        if k_mask[0, 0, 0] < kpbl[0, 0]:
            if pblflg[0, 0]:
                dku = ckz[0, 0, 0] * tem
                dkt = dku[0, 0, 0] / prn[0, 0, 0]
            else:
                dkt = chz[0, 0, 0] * tem
                dku = dkt[0, 0, 0] * prn[0, 0, 0]
        else:
            if ri < 0.0:
                dku = physcons.CK1 * tem
                dkt = physcons.RCHCK * dku[0, 0, 0]
            else:
                dkt = physcons.CH1 * tem
                dku = dkt[0, 0, 0] * min(1.0 + 2.1 * ri, physcons.PRMAX)

        tem = ckz[0, 0, 0] * tem
        dku_tmp = max(dku[0, 0, 0], tem)
        dkt_tmp = max(dkt[0, 0, 0], tem / physcons.PRSCU)

        if scuflg[0, 0]:
            if k_mask[0, 0, 0] >= mrad[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                dku = dku_tmp
                dkt = dkt_tmp

        dkq = physcons.PRTKE * dkt[0, 0, 0]

        dkt = max(min(dkt[0, 0, 0], physcons.DKMAX), xkzo[0, 0, 0])

        dkq = max(min(dkq[0, 0, 0], physcons.DKMAX), xkzo[0, 0, 0])

        dku = max(min(dku[0, 0, 0], physcons.DKMAX), xkzmo[0, 0, 0])

    with computation(PARALLEL), interval(...):
        if k_mask[0, 0, 0] == krad[0, 0]:
            if scuflg[0, 0]:
                tem1 = bf[0, 0, 0] / gotvx[0, 0, 0]
                if tem1 < physcons.TDZMIN:
                    tem1 = physcons.TDZMIN
                ptem = radj[0, 0] / tem1
                dkt = dkt[0, 0, 0] + ptem
                dku = dku[0, 0, 0] + ptem
                dkq = dkq[0, 0, 0] + ptem

    with computation(PARALLEL):
        with interval(0, 1):
            if scuflg[0, 0] and mrad[0, 0] == 0:
                ptem = xmfd[0, 0, 0] * buod[0, 0, 0]
                ptem1 = (
                    0.5
                    * (u1[0, 0, 1] - u1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (ucdo[0, 0, 0] + ucdo[0, 0, 1] - u1[0, 0, 0] - u1[0, 0, 1])
                )
                ptem2 = (
                    0.5
                    * (v1[0, 0, 1] - v1[0, 0, 0])
                    * rdzt[0, 0, 0]
                    * xmfd[0, 0, 0]
                    * (vcdo[0, 0, 0] + vcdo[0, 0, 1] - v1[0, 0, 0] - v1[0, 0, 1])
                )
            else:
                ptem = 0.0
                ptem1 = 0.0
                ptem2 = 0.0

            buop = 0.5 * (
                gotvx[0, 0, 0] * sflux[0, 0] + (-dkt[0, 0, 0] * bf[0, 0, 0] + ptem)
            )

            shrp = 0.5 * (
                dku[0, 0, 0] * shr2[0, 0, 0]
                + ptem1
                + ptem2
                + (
                    stress[0, 0]
                    * ustar[0, 0]
                    * phim[0, 0]
                    / (physcons.VK * zl[0, 0, 0])
                )
            )

            prod = buop + shrp

        with interval(1, -1):
            tem1_1 = (u1[0, 0, 1] - u1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_1 = (u1[0, 0, 0] - u1[0, 0, -1]) * rdzt[0, 0, -1]
            tem1_2 = (v1[0, 0, 1] - v1[0, 0, 0]) * rdzt[0, 0, 0]
            tem2_2 = (v1[0, 0, 0] - v1[0, 0, -1]) * rdzt[0, 0, -1]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] <= kpbl[0, 0]:
                ptem1_0 = 0.5 * (xmf[0, 0, -1] + xmf[0, 0, 0]) * buou[0, 0, 0]
                ptem1_1 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_1 + xmf[0, 0, -1] * tem2_1)
                    * (u1[0, 0, 0] - ucko[0, 0, 0])
                )
                ptem1_2 = (
                    0.5
                    * (xmf[0, 0, 0] * tem1_2 + xmf[0, 0, -1] * tem2_2)
                    * (v1[0, 0, 0] - vcko[0, 0, 0])
                )
            else:
                ptem1_0 = 0.0
                ptem1_1 = 0.0
                ptem1_2 = 0.0

            if scuflg[0, 0]:
                if k_mask[0, 0, 0] >= mrad[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                    ptem2_0 = 0.5 * (xmfd[0, 0, -1] + xmfd[0, 0, 0]) * buod[0, 0, 0]
                    ptem2_1 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_1 + xmfd[0, 0, -1] * tem2_1)
                        * (ucdo[0, 0, 0] - u1[0, 0, 0])
                    )
                    ptem2_2 = (
                        0.5
                        * (xmfd[0, 0, 0] * tem1_2 + xmfd[0, 0, -1] * tem2_2)
                        * (vcdo[0, 0, 0] - v1[0, 0, 0])
                    )
                else:
                    ptem2_0 = 0.0
                    ptem2_1 = 0.0
                    ptem2_2 = 0.0
            else:
                ptem2_0 = 0.0
                ptem2_1 = 0.0
                ptem2_2 = 0.0

            buop = (
                0.5 * ((-dkt[0, 0, -1] * bf[0, 0, -1]) + (-dkt[0, 0, 0] * bf[0, 0, 0]))
                + ptem1_0
                + ptem2_0
            )

            shrp = (
                (
                    0.5
                    * (
                        (dku[0, 0, -1] * shr2[0, 0, -1])
                        + (dku[0, 0, 0] * shr2[0, 0, 0])
                    )
                    + ptem1_1
                    + ptem2_1
                )
                + ptem1_2
                + ptem2_2
            )

            prod = buop + shrp

    with computation(PARALLEL), interval(0, -1):
        rle = physcons.CE0 / ele[0, 0, 0]


def predict_tke(
    diss: FloatField,
    prod: FloatField,
    rle: FloatField,
    tke: FloatField,
):
    from __externals__ import dtn

    with computation(PARALLEL), interval(...):
        diss = max(
            min(
                rle[0, 0, 0] * tke[0, 0, 0] * sqrt(tke[0, 0, 0]),
                prod[0, 0, 0] + tke[0, 0, 0] / dtn,
            ),
            0.0,
        )
        tke = max(
            tke[0, 0, 0] + dtn * (prod[0, 0, 0] - diss[0, 0, 0]), physcons.TKMIN
        )


def tke_up_down_prop(
    pcnvflg: BoolFieldIJ,
    qcdo: FloatField,
    qcko: FloatField,
    scuflg: BoolFieldIJ,
    tke: FloatField,
    kpbl: IntFieldIJ,
    k_mask: IntField,
    xlamue: FloatField,
    zl: FloatField,
    ad: FloatField,
    f1: FloatField,
    krad: IntFieldIJ,
    mrad: IntFieldIJ,
    xlamde: FloatField,
):

    with computation(PARALLEL), interval(...):
        if pcnvflg[0, 0]:
            qcko[0, 0, 0][7] = tke[0, 0, 0]
        if scuflg[0, 0]:
            qcdo[0, 0, 0][7] = tke[0, 0, 0]

    with computation(FORWARD), interval(1, None):
        if k_mask[0, 0, 0] < kpbl:
            tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
            if pcnvflg[0, 0] and k_mask[0, 0, 0] <= kpbl[0, 0]:
                qcko[0, 0, 0][7] = (
                    (1.0 - tem) * qcko[0, 0, -1][7]
                    + tem * (tke[0, 0, 0] + tke[0, 0, -1])
                ) / (1.0 + tem)

    with computation(BACKWARD), interval(...):
        if k_mask[0, 0, 0] < krad:
            tem = 0.5 * xlamde[0, 0, 0] * (zl[0, 0, 1] - zl[0, 0, 0])
            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
            ):
                qcdo[0, 0, 0][7] = (
                    (1.0 - tem) * qcdo[0, 0, 1][7] + tem * (tke[0, 0, 0] + tke[0, 0, 1])
                ) / (1.0 + tem)

    with computation(PARALLEL), interval(0, 1):
        if k_mask[0, 0, 0] < krad:
            ad = 1.0
            f1 = tke[0, 0, 0]


def tke_tridiag_matrix_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    delta: FloatField,
    dkq: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    qcdo: FloatField,
    qcko: FloatField,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    tke: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
):
    from __externals__ import dt2

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]
            tem2 = dsig * rdz

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7] + qcko[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * tem2 * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * tem2 * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7] + qcdo[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * tem2 * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * tem2 * xmfd[0, 0, 0]
        with interval(1, -1):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]

            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dkq[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                tem = (
                    qcko[0, 0, 0][7] + qcko[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] - tem * dtodsd * 0.5 * dsig * rdz * xmf[0, 0, 0]
                f1_p1 = tke[0, 0, 1] + tem * dtodsu * 0.5 * dsig * rdz * xmf[0, 0, 0]
            else:
                f1_p1 = tke[0, 0, 1]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                tem = (
                    qcdo[0, 0, 0][7] + qcdo[0, 0, 1][7] - (tke[0, 0, 0] + tke[0, 0, 1])
                )
                f1 = f1[0, 0, 0] + tem * dtodsd * 0.5 * dsig * rdz * xmfd[0, 0, 0]
                f1_p1 = f1_p1 - tem * dtodsu * 0.5 * dsig * rdz * xmfd[0, 0, 0]

        with interval(-1, None):
            ad = ad_p1[0, 0]
            f1 = f1_p1[0, 0]


def recover_tke_tendency_start_tridiag(
    rtg: FloatFieldTracer,
    f1: FloatField,
    q1: FloatFieldTracer,
    ad: FloatField,
    f2: FloatField,
    dtdz1: FloatField,
    evap: FloatFieldIJ,
    heat: FloatFieldIJ,
    t1: FloatField,
):
    from __externals__ import rdt, ntke

    with computation(PARALLEL), interval(...):
        rtg[0, 0, 0][ntke - 1] = (
            rtg[0, 0, 0][ntke - 1] + (f1[0, 0, 0] - q1[0, 0, 0][ntke - 1]) * rdt
        )

    with computation(FORWARD), interval(0, 1):
        ad = 1.0
        f1 = t1[0, 0, 0] + dtdz1[0, 0, 0] * heat[0, 0]
        f2[0, 0, 0][0] = q1[0, 0, 0][0] + dtdz1[0, 0, 0] * evap[0, 0]

def reset_tracers(
    f2: FloatField,
    q1: FloatFieldTracer,
    n_index: int,
):
    with computation(FORWARD), interval(0, 1):
        f2[0, 0, 0][n_index] = q1[0, 0, 0][n_index]

def heat_moist_tridiag_mat_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    delta: FloatField,
    dkt: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    f2: FloatField,
    f2_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    q1: FloatFieldTracer,
    qcdo: FloatField,
    qcko: FloatField,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    tcdo: FloatField,
    tcko: FloatField,
    t1: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
):
    from __externals__ import dt2

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * (constants.GRAV / constants.CP_AIR)
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = (
                    qcko[0, 0, 0][0]
                    + qcko[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = (
                    qcdo[0, 0, 0][0]
                    + qcdo[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            tem1 = dsig * dkt[0, 0, 0] * rdz
            dsdzt = tem1 * (constants.GRAV / constants.CP_AIR)
            dsdz2 = tem1 * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcko[0, 0, 0] + tcko[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + dtodsd * dsdzt - tem * ptem1
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt + tem * ptem2
                tem = (
                    qcko[0, 0, 0][0]
                    + qcko[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = q1[0, 0, 1][0] + tem * ptem2
            else:
                f1 = f1[0, 0, 0] + dtodsd * dsdzt
                f1_p1 = t1[0, 0, 1] - dtodsu * dsdzt
                f2_p1 = q1[0, 0, 1][0]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = tcdo[0, 0, 0] + tcdo[0, 0, 1] - (t1[0, 0, 0] + t1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = (
                    qcdo[0, 0, 0][0]
                    + qcdo[0, 0, 1][0]
                    - (q1[0, 0, 0][0] + q1[0, 0, 1][0])
                )
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


def setup_multi_tracer_tridiag(
    pcnvflg: BoolFieldIJ,
    k_mask: IntField,
    kpbl: IntFieldIJ,
    delta: FloatField,
    prsl: FloatField,
    rdzt: FloatField,
    xmf: FloatField,
    qcko: FloatField,
    q1: FloatFieldTracer,
    f2: FloatField,
    scuflg: BoolFieldIJ,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    xmfd: FloatField,
    qcdo: FloatField,
    n_index: int
):
    from __externals__ import dt2

    with computation(FORWARD), interval(0, 1):
        if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            tem = dsig * rdzt[0, 0, 0]
            ptem = 0.5 * tem * xmf[0, 0, 0]
            ptem1 = dtodsd * ptem
            ptem2 = dtodsu * ptem
            tem1 = qcko[0, 0, 0][n_index] + qcko[0, 0, 1][n_index]
            tem2 = q1[0, 0, 0][n_index] + q1[0, 0, 1][n_index]
            f2[0, 0, 0][n_index] = f2[0, 0, 0][n_index] - (tem1 - tem2) * ptem1

        if (
            scuflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            tem = dsig * rdzt[0, 0, 0]
            ptem = 0.5 * tem * xmfd[0, 0, 0]
            ptem1 = dtodsd * ptem
            ptem2 = dtodsu * ptem
            tem1 = qcdo[0, 0, 0][n_index] + qcdo[0, 0, 1][n_index]
            tem2 = q1[0, 0, 0][n_index] + q1[0, 0, 1][n_index]
            f2[0, 0, 0][n_index] = f2[0, 0, 0][n_index] + (tem1 - tem2) * ptem1

    with computation(FORWARD), interval(1, -1):
        if pcnvflg[0, 0] and k_mask[0, 0, -1] < kpbl[0, 0]:
            dtodsu = dt2 / delta[0, 0, 0]
            dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
            tem = dsig * rdzt[0, 0, -1]
            ptem = 0.5 * tem * xmf[0, 0, -1]
            ptem2 = dtodsu * ptem
            tem1 = qcko[0, 0, -1][n_index] + qcko[0, 0, 0][n_index]
            tem2 = q1[0, 0, -1][n_index] + q1[0, 0, 0][n_index]
            f2[0, 0, 0][n_index] = q1[0, 0, 0][n_index] + (tem1 - tem2) * ptem2
        else:
            f2[0, 0, 0][n_index] = q1[0, 0, 0][n_index]

        if (
            scuflg[0, 0]
            and k_mask[0, 0, -1] >= mrad[0, 0]
            and k_mask[0, 0, -1] < krad[0, 0]
        ):
            dtodsu = dt2 / delta[0, 0, 0]
            dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
            tem = dsig * rdzt[0, 0, -1]
            ptem = 0.5 * tem * xmfd[0, 0, -1]
            ptem2 = dtodsu * ptem
            tem1 = qcdo[0, 0, -1][n_index] + qcdo[0, 0, 0][n_index]
            tem2 = q1[0, 0, -1][n_index] + q1[0, 0, 0][n_index]
            f2[0, 0, 0][n_index] = f2[0, 0, 0][n_index] - (tem1 - tem2) * ptem2

        if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            tem = dsig * rdzt[0, 0, 0]
            ptem = 0.5 * tem * xmf[0, 0, 0]
            ptem1 = dtodsd * ptem
            ptem2 = dtodsu * ptem
            tem1 = qcko[0, 0, 0][n_index] + qcko[0, 0, 1][n_index]
            tem2 = q1[0, 0, 0][n_index] + q1[0, 0, 1][n_index]
            f2[0, 0, 0][n_index] = f2[0, 0, 0][n_index] - (tem1 - tem2) * ptem1

        if (
            scuflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            tem = dsig * rdzt[0, 0, 0]
            ptem = 0.5 * tem * xmfd[0, 0, 0]
            ptem1 = dtodsd * ptem
            ptem2 = dtodsu * ptem
            tem1 = qcdo[0, 0, 0][n_index] + qcdo[0, 0, 1][n_index]
            tem2 = q1[0, 0, 0][n_index] + q1[0, 0, 1][n_index]
            f2[0, 0, 0][n_index] = f2[0, 0, 0][n_index] + (tem1 - tem2) * ptem1

    with computation(FORWARD), interval(-1, None):
        if pcnvflg[0, 0] and k_mask[0, 0, -1] < kpbl[0, 0]:
            dtodsu = dt2 / delta[0, 0, 0]
            dsig = prsl[0, 0, -1] - prsl[0, 0, 0]
            tem = dsig * rdzt[0, 0, -1]
            ptem = 0.5 * tem * xmf[0, 0, -1]
            ptem2 = dtodsu * ptem
            tem1 = qcko[0, 0, -1][n_index] + qcko[0, 0, 0][n_index]
            tem2 = q1[0, 0, -1][n_index] + q1[0, 0, 0][n_index]
            f2[0, 0, 0][n_index] = q1[0, 0, 0][n_index] + (tem1 - tem2) * ptem2
        else:
            f2[0, 0, 0][n_index] = q1[0, 0, 0][n_index]


def recover_moisture_tendency(
    f2: FloatField,
    q1: FloatFieldTracer,
    rtg: FloatFieldTracer,
    n_index: int,
):
    from __externals__ import rdt

    with computation(PARALLEL), interval(...):
        rtg[0, 0, 0][n_index] = rtg[0, 0, 0][n_index] + (
            f2[0, 0, 0][n_index] - q1[0, 0, 0][n_index]
        ) * rdt

def recover_heat_tendency_add_diss_heat(
    tdt: FloatField,
    f1: FloatField,
    t1: FloatField,
    f2: FloatField,
    q1: FloatFieldTracer,
    dtsfc: FloatFieldIJ,
    delta: FloatField,
    dqsfc: FloatFieldIJ,
):
    from __externals__ import rdt
    with computation(FORWARD), interval(...):
        tdt = tdt[0, 0, 0] + (f1[0, 0, 0] - t1[0, 0, 0]) * rdt
        dtsfc = dtsfc[0, 0] + (constants.CP_AIR / constants.GRAV) * delta[0, 0, 0] * (
            (f1[0, 0, 0] - t1[0, 0, 0]) * rdt
        )
        dqsfc = dqsfc[0, 0] + (constants.HLV / constants.GRAV) * delta[0, 0, 0] * (
            (f2[0, 0, 0][0] - q1[0, 0, 0][0]) * rdt
        )


def moment_tridiag_mat_ele_comp(
    ad: FloatField,
    ad_p1: FloatFieldIJ,
    al: FloatField,
    au: FloatField,
    delta: FloatField,
    diss: FloatField,
    dku: FloatField,
    dtdz1: FloatField,
    f1: FloatField,
    f1_p1: FloatFieldIJ,
    f2: FloatField,
    f2_p1: FloatFieldIJ,
    kpbl: IntFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
    mrad: IntFieldIJ,
    pcnvflg: BoolFieldIJ,
    prsl: FloatField,
    rdzt: FloatField,
    scuflg: BoolFieldIJ,
    spd1: FloatFieldIJ,
    stress: FloatFieldIJ,
    tdt: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    vcko: FloatField,
    xmf: FloatField,
    xmfd: FloatField,
):
    from __externals__ import dspheat, dt2

    with computation(PARALLEL), interval(0, -1):
        if dspheat:
            tdt = tdt[0, 0, 0] + physcons.DSPFAC * (diss[0, 0, 0] / constants.CP_AIR)

    with computation(PARALLEL), interval(0, 1):
        ad = 1.0 + dtdz1[0, 0, 0] * stress[0, 0] / spd1[0, 0]
        f1 = u1[0, 0, 0]
        f2[0, 0, 0][0] = v1[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2
        with interval(1, -1):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]

            dtodsd = dt2 / delta[0, 0, 0]
            dtodsu = dt2 / delta[0, 0, 1]
            dsig = prsl[0, 0, 0] - prsl[0, 0, 1]
            rdz = rdzt[0, 0, 0]
            dsdz2 = dsig * dku[0, 0, 0] * rdz * rdz
            au = -dtodsd * dsdz2
            al = -dtodsu * dsdz2
            ad = ad[0, 0, 0] - au[0, 0, 0]
            ad_p1 = 1.0 - al[0, 0, 0]

            if pcnvflg[0, 0] and k_mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 0.5 * dsig * rdz * xmf[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucko[0, 0, 0] + ucko[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] - tem * ptem1
                f1_p1 = u1[0, 0, 1] + tem * ptem2
                tem = vcko[0, 0, 0] + vcko[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] - tem * ptem1
                f2_p1 = v1[0, 0, 1] + tem * ptem2
            else:
                f1_p1 = u1[0, 0, 1]
                f2_p1 = v1[0, 0, 1]

            if (
                scuflg[0, 0]
                and k_mask[0, 0, 0] >= mrad[0, 0]
                and k_mask[0, 0, 0] < krad[0, 0]
            ):
                ptem = 0.5 * dsig * rdz * xmfd[0, 0, 0]
                ptem1 = dtodsd * ptem
                ptem2 = dtodsu * ptem
                tem = ucdo[0, 0, 0] + ucdo[0, 0, 1] - (u1[0, 0, 0] + u1[0, 0, 1])
                f1 = f1[0, 0, 0] + tem * ptem1
                f1_p1 = f1_p1[0, 0] - tem * ptem2
                tem = vcdo[0, 0, 0] + vcdo[0, 0, 1] - (v1[0, 0, 0] + v1[0, 0, 1])
                f2[0, 0, 0][0] = f2[0, 0, 0][0] + tem * ptem1
                f2_p1 = f2_p1[0, 0] - tem * ptem2

        with interval(-1, None):
            f1 = f1_p1[0, 0]
            f2[0, 0, 0][0] = f2_p1[0, 0]
            ad = ad_p1[0, 0]


def recover_momentum_tendency(
    delta: FloatField,
    du: FloatField,
    dusfc: FloatFieldIJ,
    dv: FloatField,
    dvsfc: FloatFieldIJ,
    f1: FloatField,
    f2: FloatField,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    k_mask: IntField,
    u1: FloatField,
    v1: FloatField,
):
    from __externals__ import rdt

    with computation(FORWARD), interval(...):
        if k_mask[0, 0, 0] < 1:
            hpbl = hpblx[0, 0]
            kpbl = kpblx[0, 0]
        utend = (f1[0, 0, 0] - u1[0, 0, 0]) * rdt
        vtend = (f2[0, 0, 0][0] - v1[0, 0, 0]) * rdt
        du = du[0, 0, 0] + utend
        dv = dv[0, 0, 0] + vtend
        dusfc = dusfc[0, 0] + constants.RGRAV * delta[0, 0, 0] * utend
        dvsfc = dvsfc[0, 0] + constants.RGRAV * delta[0, 0, 0] * vtend


class ScaleAwareTKEMoistEDMF:
    """
    Scheme to compute subgrid vertical turbulence mixing
    using scale-aware TKE-based moist eddy-diffusion mass-flux (EDMF)
    parameterization

    Fortran name is satmedmfvdif
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        grid_area: Float,
        config: PBLConfig,
    ):
        # assert config.ntracers == config.ntke, (
        #     "PBL scheme satmedmfvdif requires ntracer "
        #     f"({config.ntracers}) == ntke ({config.ntke})"
        # )
        if config.do_dk_hb19:
            raise NotImplementedError("do_dk_hb19 has not been implemented")

        self._ntracers = config.ntracers
        self._ntrac1 = self._ntracers - 1

        global FloatFieldTracer
        FloatFieldTracer = set_4d_field_size(self._ntracers, Float)

        self.TRACER_DIM = TRACER_DIM
        self.COND_DIM = COND_DIM
        self.quantity_factory = quantity_factory
        self.quantity_factory.set_extra_dim_lengths(
            **{
                self.TRACER_DIM: self._ntracers,
                self.COND_DIM: self._ntrac1,
            }
        )
        idx = stencil_factory.grid_indexing

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        # Allocate internal variables:
        km1 = idx.domain[2] - 1
        self._kmpbl = idx.domain[2] / 2
        self._kmscu = idx.domain[2] / 2

        self._dt_atmos = config.dt_atmos
        self._rdt = 1.0 / self._dt_atmos
        self._kk = max(round(self._dt_atmos / physcons.CDTN), 1)
        self._dtn = self._dt_atmos / float(self._kk)

        self._area = grid_area

        self._ntiw = config.ntiw
        self._ntcw = config.ntcw
        self._ntke = config.ntke

        self._dspheat = config.dspheat

        # Layer mask:
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(idx.domain[2]):
            self._k_mask.data[:, :, k] = k

        # Internal compute variables
        self._lcld = make_quantity_2D(Int)
        self._kcld = make_quantity_2D(Int)
        self._krad = make_quantity_2D(Int)
        self._kx1 = make_quantity_2D(Int)
        self._kpblx = make_quantity_2D(Int)
        self._tke = make_quantity()
        self._tkeh = make_quantity()
        self._theta = make_quantity()
        self._thvx = make_quantity()
        self._thlvx = make_quantity()
        self._thlvx_0 = make_quantity_2D(Float)
        self._qlx = make_quantity()
        self._thetae = make_quantity()
        self._thlx = make_quantity()
        self._slx = make_quantity()
        self._svx = make_quantity()
        self._qtx = make_quantity()
        self._tvx = make_quantity()
        self._pix = make_quantity()
        self._radx = make_quantity()
        self._dku = make_quantity()
        self._dkt = make_quantity()
        self._dkq = make_quantity()
        self._cku = make_quantity()
        self._ckt = make_quantity()
        self._plyr = make_quantity()
        self._rhly = make_quantity()
        self._cfly = make_quantity()
        self._qstl = make_quantity()
        self._dtdz1 = make_quantity_2D(Float)
        self._gdx = make_quantity_2D(Float)
        self._phih = make_quantity_2D(Float)
        self._phim = make_quantity_2D(Float)
        self._prn = make_quantity()
        self._rbdn = make_quantity_2D(Float)
        self._rbup = make_quantity_2D(Float)
        self._thermal = make_quantity_2D(Float)
        self._ustar = make_quantity_2D(Float)
        self._wstar = make_quantity_2D(Float)
        self._hpblx = make_quantity_2D(Float)
        self._ust3 = make_quantity_2D(Float)
        self._wst3 = make_quantity_2D(Float)
        self._z0 = make_quantity_2D(Float)
        self._crb = make_quantity_2D(Float)
        self._hgamt = make_quantity_2D(Float)
        self._hgamq = make_quantity_2D(Float)
        self._wscale = make_quantity_2D(Float)
        self._vpert = make_quantity_2D(Float)
        self._zol = make_quantity_2D(Float)
        self._sflux = make_quantity_2D(Float)
        self._radj = make_quantity_2D(Float)
        self._tx1 = make_quantity_2D(Float)
        self._tx2 = make_quantity_2D(Float)
        self._radmin = make_quantity_2D(Float)
        self._zi = make_quantity()
        self._zl = make_quantity()
        self._zldn = make_quantity_2D(Float)
        self._zm = make_quantity()
        self._xkzo = make_quantity()
        self._xkzmo = make_quantity()
        self._xkzm_hx = make_quantity_2D(Float)
        self._xkzm_mx = make_quantity_2D(Float)
        self._rdzt = make_quantity()
        self._al = make_quantity()
        self._ad = make_quantity()
        self._au = make_quantity()
        self._f1 = make_quantity()
        self._elm = make_quantity()
        self._ele = make_quantity()
        self._rle = make_quantity()
        self._ckz = make_quantity()
        self._chz = make_quantity()
        self._diss = make_quantity()
        self._prod = make_quantity()
        self._bf = make_quantity()
        self._shr2 = make_quantity()
        self._xlamue = make_quantity()
        self._xlamde = make_quantity()
        self._gotvx = make_quantity()
        self._rlam = make_quantity()
        self._mrad = make_quantity_2D(Int)
        self._ad_p1 = make_quantity_2D(Float)
        self._f1_p1 = make_quantity_2D(Float)
        self._f2_p1 = make_quantity_2D(Float)

        # Variables for updrafts (thermals):
        self._tcko = make_quantity()
        self._ucko = make_quantity()
        self._vcko = make_quantity()
        self._buou = make_quantity()
        self._xmf = make_quantity()

        # Variables for stratocumulus-top induced downdrafts:
        self._tcdo = make_quantity()
        self._ucdo = make_quantity()
        self._vcdo = make_quantity()
        self._buod = make_quantity()
        self._xmfd = make_quantity()

        self._mlenflg = False
        self._pblflg = make_quantity_2D(Bool)
        self._sfcflg = make_quantity_2D(Bool)
        self._flg = make_quantity_2D(Bool)
        self._scuflg = quantity_factory.ones([X_DIM, Y_DIM], units="none", dtype=Bool)
        self._pcnvflg = make_quantity_2D(Bool)

        # Limiting pressures for vertical loops:
        self._ptop = make_quantity_2D(Float)
        self._pbot = make_quantity_2D(Float)

        # Allocate higher order fields
        self._f2 = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.COND_DIM],
            units="unknown",
            dtype=Int,
        )

        self._qcko = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Int,
        )

        self._qcdo = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM, self.TRACER_DIM],
            units="unknown",
            dtype=Int,
        )

        # Init stencils:
        self._init_turbulence = stencil_factory.from_origin_domain(
            func=init_turbulence,
            externals={
                "km1": km1,
                "xkzm_h": config.xkzm_h,
                "xkzm_m": config.xkzm_m,
                "xkzm_s": config.xkzm_s,
                "dt2": self._dt_atmos,
                "ntke": self._ntke,
                "ntiw": self._ntiw,
                "ntcw": self._ntcw,
                "cap_k0_land": config.cap_k0_land
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mrf_pbl_scheme_part1 = stencil_factory.from_origin_domain(
            func=mrf_pbl_scheme_part1,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmpbl),
        )

        self._mrf_pbl_2_thermal_excess = stencil_factory.from_origin_domain(
            func=mrf_pbl_2_thermal_excess,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._thermal_excess_2 = stencil_factory.from_origin_domain(
            func=thermal_excess_2,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmpbl),
        )

        self._enhance_pbl_height_thermal = stencil_factory.from_origin_domain(
            func=enhance_pbl_height_thermal,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._stratocumulus = stencil_factory.from_origin_domain(
            func=stratocumulus,
            externals={"km1": km1},
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmscu),
        )

        self._compute_mass_flux_prelim = stencil_factory.from_origin_domain(
            func=compute_mass_flux_prelim,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )
        self._compute_mass_flux_tracer_prelim = stencil_factory.from_origin_domain(
            func=compute_mass_flux_tracer_prelim,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfpblt = PBLMassFlux(
            stencil_factory,
            quantity_factory,
            self._dt_atmos,
            self._ntcw,
            self._ntrac1,
            self._kmpbl,
        )

        self._mfscu = StratocumulusMassFlux(
            stencil_factory,
            quantity_factory,
            self._dt_atmos,
            self._ntcw,
            self._ntrac1,
            self._kmscu,
        )

        self._compute_prandtl_num_exchange_coeff = stencil_factory.from_origin_domain(
            func=compute_prandtl_num_exchange_coeff,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, self._kmpbl),
        )

        self._compute_asymptotic_mixing_length = stencil_factory.from_origin_domain(
            func=compute_asymptotic_mixing_length,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(0, 0, -1)),
        )

        self._compute_eddy_diffusivity_buoy_shear = stencil_factory.from_origin_domain(
            func=compute_eddy_diffusivity_buoy_shear,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._predict_tke = stencil_factory.from_origin_domain(
            func=predict_tke,
            externals={
                "dtn": self._dtn,
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(add=(0, 0, -1)),
        )

        self._tke_up_down_prop = stencil_factory.from_origin_domain(
            func=tke_up_down_prop,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._tke_tridiag_matrix_ele_comp = stencil_factory.from_origin_domain(
            func=tke_tridiag_matrix_ele_comp,
            externals={"dt2": self._dt_atmos},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._tridit = stencil_factory.from_origin_domain(
            func=tridit,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._recover_tke_tendency_start_tridiag = stencil_factory.from_origin_domain(
            func=recover_tke_tendency_start_tridiag,
            externals={
                "rdt": self._rdt,
                "ntke": self._ntke
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        if self._ntrac1 >= 2:
            self._reset_tracers = stencil_factory.from_origin_domain(
                func=reset_tracers,
                origin=idx.origin_compute(),
                domain=idx.domain_compute(),
            )

        self._heat_moist_tridiag_mat_ele_comp = stencil_factory.from_origin_domain(
            func=heat_moist_tridiag_mat_ele_comp,
            externals={"dt2": self._dt_atmos},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        if self._ntrac1 >= 2:
            self._setup_multi_tracer_tridiag = stencil_factory.from_origin_domain(
                func=setup_multi_tracer_tridiag,
                externals={"dt2": self._dt_atmos},
                origin=idx.origin_compute(),
                domain=idx.domain_compute(),
            )

        self._tridin = stencil_factory.from_origin_domain(
            func=tridin,
            externals={"nt": self._ntrac1},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._recover_moisture_tendency = (
            stencil_factory.from_origin_domain(
                func=recover_moisture_tendency,
                externals={
                    "rdt": self._rdt,
                },
                origin=idx.origin_compute(),
                domain=idx.domain_compute(),
            )
        )

        self._recover_heat_tendency_add_diss_heat = (
            stencil_factory.from_origin_domain(
                func=recover_heat_tendency_add_diss_heat,
                externals={
                    "rdt": self._rdt,
                },
                origin=idx.origin_compute(),
                domain=idx.domain_compute(),
            )
        )

        self._moment_tridiag_mat_ele_comp = stencil_factory.from_origin_domain(
            func=moment_tridiag_mat_ele_comp,
            externals={
                "dspheat": self._dspheat,
                "dt2": self._dt_atmos,
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._tridi2 = stencil_factory.from_origin_domain(
            func=tridi2,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._recover_momentum_tendency = stencil_factory.from_origin_domain(
            func=recover_momentum_tendency,
            externals={"rdt": self._rdt},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

    def __call__(
        self,
        kpbl: IntFieldIJ,
        kinver: IntFieldIJ,
        dv: FloatField,
        du: FloatField,
        tdt: FloatField,
        rtg: FloatFieldTracer,  # FloatField with extra data dimension
        hpbl: FloatFieldIJ,
        u1: FloatField,  # ix
        v1: FloatField,  # ix
        t1: FloatField,  # ix
        q1: FloatFieldTracer,  # FloatField with extra data dimension
        hsw: FloatField,  # ix
        hlw: FloatField,  # ix
        xmu: FloatFieldIJ,
        psk: FloatFieldIJ,  # ix
        rbsoil: FloatFieldIJ,
        zorl: FloatFieldIJ,
        tsea: FloatFieldIJ,
        u10m: FloatFieldIJ,
        v10m: FloatFieldIJ,
        fm: FloatFieldIJ,
        fh: FloatFieldIJ,
        evap: FloatFieldIJ,
        heat: FloatFieldIJ,
        stress: FloatFieldIJ,
        spd1: FloatFieldIJ,
        prsi: FloatField,  # ix
        delta: FloatField,  # ix, Fortran name is del
        prsl: FloatField,  # ix
        prslk: FloatField,  # ix
        phii: FloatField,  # ix
        phil: FloatField,  # ix
        dusfc: FloatFieldIJ,
        dvsfc: FloatFieldIJ,
        dtsfc: FloatFieldIJ,
        dqsfc: FloatFieldIJ,
    ):

        """
        ix is the block size in i, for us the same as im since gt4py handles threading
        Still have to figure out what to do with:
        rtg(im,km,ntrac), q1(ix,km,ntrac),
        """

        self._init_turbulence(
            self._zi,
            self._zl,
            self._zm,
            phii,
            phil,
            self._chz,
            self._ckz,
            self._area,
            self._gdx,
            self._tke,
            q1,
            self._rdzt,
            self._prn,
            self._kx1,
            prsi,
            self._k_mask,
            kinver,
            self._tx1,
            self._tx2,
            self._xkzo,
            self._xkzmo,
            self._kpblx,
            self._hpblx,
            self._pblflg,
            self._sfcflg,
            self._pcnvflg,
            self._scuflg,
            zorl,
            dusfc,
            dvsfc,
            dtsfc,
            dqsfc,
            kpbl,
            hpbl,
            rbsoil,
            self._radmin,
            self._mrad,
            self._krad,
            self._lcld,
            self._kcld,
            self._theta,
            prslk,
            psk,
            t1,
            self._pix,
            self._qlx,
            self._slx,
            self._thvx,
            self._qtx,
            self._thlx,
            self._thlvx,
            self._svx,
            self._thetae,
            self._gotvx,
            prsl,
            self._plyr,
            self._rhly,
            self._qstl,
            self._bf,
            self._cfly,
            self._crb,
            self._dtdz1,
            evap,
            heat,
            hlw,
            self._radx,
            self._sflux,
            self._shr2,
            stress,
            hsw,
            self._thermal,
            tsea,
            u10m,
            self._ustar,
            u1,
            v1,
            v10m,
            xmu,
            self._ptop,
            self._pbot,
        )

        self._mrf_pbl_scheme_part1(
            self._crb,
            self._flg,
            self._kpblx,
            self._k_mask,
            self._rbdn,
            self._rbup,
            rbsoil,
            self._thermal,
            self._thlvx,
            self._thlvx_0,
            u1,
            v1,
            self._zl,
        )

        self._mrf_pbl_2_thermal_excess(
            self._crb,
            evap,
            fh,
            self._flg,
            fm,
            self._gotvx,
            heat,
            hpbl,
            self._hpblx,
            kpbl,
            self._kpblx,
            self._k_mask,
            self._pblflg,
            self._pcnvflg,
            self._phih,
            self._phim,
            self._rbdn,
            self._rbup,
            rbsoil,
            self._sfcflg,
            self._sflux,
            self._thermal,
            self._theta,
            self._ustar,
            self._vpert,
            self._zi,
            self._zl,
            self._zol,
        )

        self._thermal_excess_2(
            self._crb,
            self._flg,
            kpbl,
            self._k_mask,
            self._rbdn,
            self._rbup,
            self._thermal,
            self._thlvx,
            self._thlvx_0,
            u1,
            v1,
            self._zl,
        )

        self._enhance_pbl_height_thermal(
            self._crb,
            self._flg,
            hpbl,
            kpbl,
            self._lcld,
            self._k_mask,
            self._pblflg,
            self._pcnvflg,
            self._rbdn,
            self._rbup,
            self._scuflg,
            self._zi,
            self._zl,
        )

        self._stratocumulus(
            self._flg,
            self._kcld,
            self._krad,
            self._lcld,
            self._k_mask,
            self._radmin,
            self._radx,
            self._qlx,
            self._scuflg,
        )

        self._compute_mass_flux_prelim(
            self._pcnvflg,
            self._scuflg,
            t1,
            self._tcdo,
            self._tcko,
            u1,
            self._ucdo,
            self._ucko,
            v1,
            self._vcdo,
            self._vcko,
        )

        for n in range(8):
            self._compute_mass_flux_tracer_prelim(
                self._qcdo,
                self._qcko,
                q1,
                self._pcnvflg,
                self._scuflg,
                n,
            )

        self._mfpblt(
            self._pcnvflg,
            self._zl,
            self._zm,
            q1,  # I, J, K, ntracer field
            u1,
            v1,
            self._plyr,
            self._pix,
            self._thlx,
            self._thvx,
            self._gdx,
            hpbl,
            kpbl,
            self._vpert,
            self._buou,
            self._xmf,
            self._tcko,
            self._qcko,  # I, J, K, ntracer field
            self._ucko,
            self._vcko,
            self._xlamue,
            self._k_mask,
        )

        self._mfscu(
            self._pcnvflg,
            self._zl,
            self._zm,
            q1,  # I, J, K, ntracer field
            u1,
            v1,
            self._plyr,
            self._pix,
            self._thlx,
            self._thvx,
            self._thlvx,
            self._gdx,
            self._thetae,
            self._radj,
            self._krad,
            self._mrad,
            self._radmin,
            self._buod,
            self._xmfd,
            self._tcdo,
            self._qcdo,  # I, J, K, ntracer field
            self._ucdo,
            self._vcdo,
            self._xlamde,
            self._k_mask,
        )

        self._compute_prandtl_num_exchange_coeff(
            self._chz,
            self._ckz,
            hpbl,
            kpbl,
            self._k_mask,
            self._pcnvflg,
            self._phih,
            self._phim,
            self._prn,
            self._zi,
        )

        self._compute_asymptotic_mixing_length(
            self._zldn,
            self._thvx,
            self._tke,
            self._gotvx,
            self._zl,
            tsea,
            q1,
            self._zi,
            self._rlam,
            self._ele,
            self._zol,
            self._gdx,
            phii,
            self._ptop,
            self._pbot,
        )

        self._compute_eddy_diffusivity_buoy_shear(
            self._bf,
            self._buod,
            self._buou,
            self._chz,
            self._ckz,
            self._dku,
            self._dkt,
            self._dkq,
            self._ele,
            self._elm,
            self._gotvx,
            kpbl,
            self._k_mask,
            self._mrad,
            self._krad,
            self._pblflg,
            self._pcnvflg,
            self._phim,
            self._prn,
            self._prod,
            self._radj,
            self._rdzt,
            self._rle,
            self._scuflg,
            self._sflux,
            self._shr2,
            stress,
            self._tke,
            u1,
            self._ucdo,
            self._ucko,
            self._ustar,
            v1,
            self._vcdo,
            self._vcko,
            self._xkzo,
            self._xkzmo,
            self._xmf,
            self._xmfd,
            self._zl,
        )

        for n in range(self._kk):
            self._predict_tke(
                self._diss,
                self._prod,
                self._rle,
                self._tke,
            )

        self._tke_up_down_prop(
            self._pcnvflg,
            self._qcdo,
            self._qcko,
            self._scuflg,
            self._tke,
            kpbl,
            self._k_mask,
            self._xlamue,
            self._zl,
            self._ad,
            self._f1,
            self._krad,
            self._mrad,
            self._xlamde,
        )

        self._tke_tridiag_matrix_ele_comp(
            self._ad,
            self._ad_p1,
            self._al,
            self._au,
            delta,
            self._dkq,
            self._f1,
            self._f1_p1,
            kpbl,
            self._krad,
            self._k_mask,
            self._mrad,
            self._pcnvflg,
            prsl,
            self._qcdo,
            self._qcko,
            self._rdzt,
            self._scuflg,
            self._tke,
            self._xmf,
            self._xmfd,
        )

        self._tridit(
            self._au,
            self._ad,
            self._al,
            self._f1,
        )

        self._recover_tke_tendency_start_tridiag(
            rtg,
            self._f1,
            q1,
            self._ad,
            self._f2,
            self._dtdz1,
            evap,
            heat,
            t1,
        )

        if self._ntrac1 >= 2:
            for n in range(1, self._ntrac1):
                self._reset_tracers(
                    self._f1,
                    q1,
                    n,
                )

        self._heat_moist_tridiag_mat_ele_comp(
            self._ad,
            self._ad_p1,
            self._al,
            self._au,
            delta,
            self._dkt,
            self._f1,
            self._f1_p1,
            self._f2,
            self._f2_p1,
            kpbl,
            self._krad,
            self._k_mask,
            self._mrad,
            self._pcnvflg,
            prsl,
            q1,
            self._qcdo,
            self._qcko,
            self._rdzt,
            self._scuflg,
            self._tcdo,
            self._tcko,
            t1,
            self._xmf,
            self._xmfd,
        )

        if self._ntrac1 >= 2:
            for n in range(self.ntrac - 1):
                self._setup_multi_tracer_tridiag(
                    self._pcnvflg,
                    self._k_mask,
                    kpbl,
                    delta,
                    prsl,
                    self._rdzt,
                    self._xmf,
                    self._qcko,
                    q1,
                    self._f2,
                    self._scuflg,
                    self._mrad,
                    self._krad,
                    self._xmfd,
                    self._qcdo,
                    n
                )

        for n in range(self._ntrac1):
            self._tridin(
                self._al,
                self._ad,
                self._au,
                self._f1,
                self._f2,
                self._au,
                self._f1,
                self._f2,
                n
            )

            self._recover_moisture_tendency(
                self._f2,
                q1,
                rtg,
                n,
            )

        self._recover_heat_tendency_add_diss_heat(
            tdt,
            self._f1,
            t1,
            self._f2,
            q1,
            dtsfc,
            delta,
            dqsfc,
        )

        moment_tridiag_mat_ele_comp(
            self._ad,
            self._ad_p1,
            self._al,
            self._au,
            delta,
            self._diss,
            self._dku,
            self._dtdz1,
            self._f1,
            self._f1_p1,
            self._f2,
            self._f2_p1,
            kpbl,
            self._krad,
            self._k_mask,
            self._mrad,
            self._pcnvflg,
            prsl,
            self._rdzt,
            self._scuflg,
            spd1,
            stress,
            tdt,
            u1,
            self._ucdo,
            self._ucko,
            v1,
            self._vcdo,
            self._vcko,
            self._xmf,
            self._xmfd,
        )

        self._tridi2(
            self._f1,
            self._f2,
            self._au,
            self._al,
            self._ad,
            self._au,
            self._f1,
            self._f2,
        )

        self._recover_momentum_tendency(
            delta,
            du,
            dusfc,
            dv,
            dvsfc,
            self._f1,
            self._f2,
            hpbl,
            self._hpblx,
            kpbl,
            self._kpblx,
            self._k_mask,
            u1,
            v1,
        )
