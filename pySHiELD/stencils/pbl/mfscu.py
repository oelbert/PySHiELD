from gt4py.cartesian.gtscript import (
    __INLINED
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    interval,
    sqrt,
)

import ndsl.constants as constants
from ndsl.constants import X_DIM, Y_DIM, Z_DIM

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.initialization.allocator import QuantityFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from pySHiELD.functions.physics_functions import fpvs


def mfscu_s2(
    zl: FloatField,
    mask: IntField,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    zm: FloatField,
    zm_mrad: FloatFieldIJ,
    xlamde: FloatField,
    xlamdem: FloatField,
    hrad: FloatFieldIJ,
    cnvflg: BoolFieldIJ,
    ce0: Float,
    cm: Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0, 0, 0]


def mfscu_s6(
    zl: FloatField,
    mask: IntField,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    zm: FloatField,
    zm_mrad: FloatFieldIJ,
    xlamde: FloatField,
    xlamdem: FloatField,
    hrad: FloatFieldIJ,
    cnvflg: BoolFieldIJ,
    mrady: IntFieldIJ,
    mradx: IntFieldIJ,
    ce0: Float,
    cm: Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mrady[0, 0] < mradx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = ce0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = ce0 / dz
            xlamdem = cm * xlamde[0, 0, 0]


def mfscu_10(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mrad: IntFieldIJ,
    mask: IntField,
    zl: FloatField,
    xlamde: FloatField,
    qcdo: FloatField,
    q1: FloatField,
    ntcw: Int,
    ntrac1: Int,
):
    from __externals__ import ntcw, ntrac1
    with computation(BACKWARD), interval(...):
        if __INLINED(ntcw > 2):
            for n in range(1, ntcw - 1):
                if (
                    cnvflg[0, 0]
                    and mask[0, 0, 0] < krad[0, 0]
                    and mask[0, 0, 0] >= mrad[0, 0]
                ):
                    dz = zl[0, 0, 1] - zl[0, 0, 0]
                    tem = 0.5 * xlamde[0, 0, 0] * dz
                    factor = 1.0 + tem
                    qcdo[0, 0, 0][n] = (
                        (1.0 - tem) * qcdo[0, 0, 1][n]
                        + tem * (q1[0, 0, 0][n] + q1[0, 0, 1][n])
                    ) / factor

        if __INLINED(ntrac1 > ntcw):
            for n1 in range(ntcw, ntrac1):
                if (
                    cnvflg[0, 0]
                    and mask[0, 0, 0] < krad[0, 0]
                    and mask[0, 0, 0] >= mrad[0, 0]
                ):
                    dz = zl[0, 0, 1] - zl[0, 0, 0]
                    tem = 0.5 * xlamde[0, 0, 0] * dz
                    factor = 1.0 + tem
                    qcdo[0, 0, 0][n1] = (
                        (1.0 - tem) * qcdo[0, 0, 1][n1]
                        + tem * (q1[0, 0, 0][n1] + q1[0, 0, 1][n1])
                    ) / factor


def mfscu_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hrad: FloatFieldIJ,
    krad: IntFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    q1: FloatField,
    qtd: FloatField,
    qtx: FloatField,
    ra1: FloatFieldIJ,
    ra2: FloatFieldIJ,
    radmin: FloatFieldIJ,
    radj: FloatFieldIJ,
    thetae: FloatField,
    thld: FloatField,
    thlvd: FloatFieldIJ,
    thlvx: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    wd2: FloatField,
    zm: FloatField,
    a1: Float,
    a11: Float,
    a2: Float,
    a22: Float,
    actei: Float,
    cldtime: Float,
    cp: Float,
    hvap: Float,
    g: Float,
    ntcw: Int,
):

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw - 1]

    with computation(FORWARD), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = mask[0, 0, 0] - 1
                tem1 = max(cldtime * radmin[0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0)
                thld = thlx[0, 0, 0] + tem1
                qtd = qtx[0, 0, 0]
                thlvd = thlvx[0, 0, 0] + tem1
                buo = -g * tem1 / thvx[0, 0, 0]

                ra1 = a1
                ra2 = a11

                tem = thetae[0, 0, 0] - thetae[0, 0, 1]
                tem1 = qtx[0, 0, 0] - qtx[0, 0, 1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = cp * tem / (hvap * tem1)
                    if cteit > actei:
                        ra1 = a2
                        ra2 = a22

                radj = -ra2[0, 0] * radmin[0, 0]

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrad = krad[0, 0]


def mfscu_s1(
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    thlvd: FloatFieldIJ,
    thlvx: FloatField,
):

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = mask[0, 0, 0]
                else:
                    flg[0, 0] = 0

    with computation(FORWARD), interval(0, 1):
        kk = krad[0, 0] - mrad[0, 0]
        if cnvflg[0, 0]:
            if kk < 1:
                cnvflg[0, 0] = 0


def mfscu_s3(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    thld: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    qtd: FloatField,
    qtx: FloatField,
    xlamde: FloatField,
    zl: FloatField,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    fv: Float,
    g: Float,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(constants.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)
        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + fv * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * elocp * qld) * tem1
            else:
                tem1 = 1.0 + fv * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = g * (1.0 - thvd / thvx[0, 0, 0])


def mfscu_s4(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    wd2: FloatField,
    xlamde: FloatField,
    zm: FloatField,
    bb1: Float,
    bb2: Float,
):

    with computation(FORWARD), interval(...):
        if mask[0, 0, 0] == krad1[0, 0]:
            if cnvflg[0, 0]:
                dz = zm[0, 0, 1] - zm[0, 0, 0]
                wd2 = (bb2 * buo[0, 0, 1] * dz) / (
                    1.0 + (0.5 * bb1 * xlamde[0, 0, 0] * dz)
                )


def mfscu_s5(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    krad: IntFieldIJ,
    krad1: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    mradx: IntFieldIJ,
    mrady: IntFieldIJ,
    xlamde: FloatField,
    wd2: FloatField,
    zm: FloatField,
):
    with computation(BACKWARD), interval(...):
        dz = zm[0, 0, 1] - zm[0, 0, 0]
        tem = 0.25 * 2.0 * (xlamde[0, 0, 0] + xlamde[0, 0, 1]) * dz
        ptem1 = 1.0 + tem
        if cnvflg[0, 0] and mask[0, 0, 0] < krad1[0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrady = mrad[0, 0]
        if flg[0, 0]:
            mradx = krad[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            if flg[0, 0] and mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = mask[0, 0, 0]
                else:
                    flg = 0

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            if mrad[0, 0] < mradx[0, 0]:
                mrad = mradx[0, 0]
            if (krad[0, 0] - mrad[0, 0]) < 1:
                cnvflg = 0


def mfscu_s7(
    cnvflg: BoolFieldIJ,
    gdx: FloatFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    ra1: FloatFieldIJ,
    scaldfunc: FloatFieldIJ,
    sumx: FloatFieldIJ,
    wd2: FloatField,
    xlamde: FloatField,
    xlamavg: FloatFieldIJ,
    xmfd: FloatField,
    zl: FloatField,
    dt2: Float,
):
    with computation(FORWARD), interval(0, 1):
        xlamavg = 0.0
        sumx = 0.0

    with computation(BACKWARD), interval(-1, None):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(BACKWARD), interval(...):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if wd2[0, 0, 0] > 0:
                xmfd = ra1[0, 0] * sqrt(wd2[0, 0, 0])
            else:
                xmfd = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem1 = (3.14 * (0.2 / xlamavg[0, 0]) * (0.2 / xlamavg[0, 0])) / (
                    gdx[0, 0] * gdx[0, 0]
                )
                sigma = min(max(tem1, 0.001), 0.999)

            if cnvflg[0, 0]:
                if sigma > ra1[0, 0]:
                    scaldfunc = max(min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0)
                else:
                    scaldfunc = 1.0

    with computation(BACKWARD), interval(...):
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0] * xmfd[0, 0, 0], xmmx)


def mfscu_s8(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    thld: FloatField,
    thlx: FloatField,
):

    with computation(PARALLEL), interval(...):
        if krad[0, 0] == mask[0, 0, 0]:
            if cnvflg[0, 0]:
                thld = thlx[0, 0, 0]


def mfscu_s9(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mask: IntField,
    mrad: IntFieldIJ,
    pix: FloatField,
    plyr: FloatField,
    qcdo: FloatField,
    qtd: FloatField,
    qtx: FloatField,
    tcdo: FloatField,
    thld: FloatField,
    thlx: FloatField,
    u1: FloatField,
    ucdo: FloatField,
    v1: FloatField,
    vcdo: FloatField,
    xlamde: FloatField,
    xlamdem: FloatField,
    zl: FloatField,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    pgcon: Float,
    ntcw: Int,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            tem = 0.5 * xlamde[0, 0, 0] * dz
            factor = 1.0 + tem
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(constants.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
        dq = qtd[0, 0, 0] - qs
        gamma = el2orc * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)

        if cnvflg[0, 0] and mask[0, 0, 0] >= mrad[0, 0] and mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                qcdo[0, 0, 0][0] = qs
                qcdo[0, 0, 0][ntcw - 1] = qld
                tcdo = tld + elocp * qld
            else:
                qcdo[0, 0, 0] = qtd[0, 0, 0]
                qcdo[0, 0, 0][ntcw - 1] = 0.0
                tcdo = tld

        if cnvflg[0, 0] and mask[0, 0, 0] < krad[0, 0] and mask[0, 0, 0] >= mrad[0, 0]:
            tem = 0.5 * xlamdem[0, 0, 0] * dz
            factor = 1.0 + tem
            ptem = tem - pgcon
            ptem1 = tem + pgcon
            ucdo = (
                (1.0 - tem) * ucdo[0, 0, 1] + ptem * u1[0, 0, 1] + ptem1 * u1[0, 0, 0]
            ) / factor
            vcdo = (
                (1.0 - tem) * vcdo[0, 0, 1] + ptem * v1[0, 0, 1] + ptem1 * v1[0, 0, 0]
            ) / factor
            
class StratocumulusMassFlux:
    """
    A mass-flux parameterization for stratocumulus-top-induced turbulence
    mixing
    Fortran name is mfscu
    """
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory
    ):
        pass
    
    def __call__(
        self,
        im,
        ix,
        km,
        kmscu,
        ntcw,
        ntrac1,
        delt,
        cnvflg,
        zl,
        zm,
        q1,
        t1,
        u1,
        v1,
        plyr,
        pix,
        thlx,
        thvx,
        thlvx,
        gdx,
        thetae,
        radj,
        krad,
        mrad,
        radmin,
        buo,
        xmfd,
        tcdo,
        qcdo,
        ucdo,
        vcdo,
        xlamde,
        g,
        gocp,
        elocp,
        el2orc,
        mask,
        qtx,
        wd2,
        hrad,
        krad1,
        thld,
        qtd,
        thlvd,
        ra1,
        ra2,
        flg,
        xlamdem,
        mradx,
        mrady,
        sumx,
        xlamavg,
        scaldfunc,
        zm_mrad,
        ce0,
        cm,
        pgcon,
        qmin,
        qlmin,
        b1,
        f1,
        a1,
        a2,
        a11,
        a22,
        cldtime,
        actei,
        hvap,
        cp,
        eps,
        epsm1,
        fv,
    ):

        totflg = True

        for i in range(im):
            totflg = totflg and ~cnvflg[i, 0]

        if totflg:
            return

        mfscu_s0(
            buo=buo,
            cnvflg=cnvflg,
            flg=flg,
            hrad=hrad,
            krad=krad,
            krad1=krad1,
            mask=mask,
            mrad=mrad,
            q1=q1,
            qtd=qtd,
            qtx=qtx,
            ra1=ra1,
            ra2=ra2,
            radmin=radmin,
            radj=radj,
            thetae=thetae,
            thld=thld,
            thlvd=thlvd,
            thlvx=thlvx,
            thlx=thlx,
            thvx=thvx,
            wd2=wd2,
            zm=zm,
            a1=a1,
            a11=a11,
            a2=a2,
            a22=a22,
            actei=actei,
            cldtime=cldtime,
            cp=cp,
            hvap=hvap,
            g=g,
            ntcw=ntcw,
            domain=(im, 1, km),
        )

        mfscu_s1(
            cnvflg=cnvflg,
            flg=flg,
            krad=krad,
            mask=mask,
            mrad=mrad,
            thlvd=thlvd,
            thlvx=thlvx,
            domain=(im, 1, kmscu),
        )

        totflg = True

        for i in range(im):
            totflg = totflg and ~cnvflg[i, 0]

        if totflg:
            return

        for i in range(im):
            zm_mrad[i, 0] = zm[i, 0, mrad[i, 0] - 1]

        mfscu_s2(
            zl=zl,
            mask=mask,
            mrad=mrad,
            krad=krad,
            zm=zm,
            zm_mrad=zm_mrad,
            xlamde=xlamde,
            xlamdem=xlamdem,
            hrad=hrad,
            cnvflg=cnvflg,
            ce0=ce0,
            cm=cm,
            domain=(im, 1, kmscu),
        )

        mfscu_s3(
            buo=buo,
            cnvflg=cnvflg,
            el2orc=el2orc,
            elocp=elocp,
            eps=eps,
            epsm1=epsm1,
            fv=fv,
            g=g,
            krad=krad,
            mask=mask,
            pix=pix,
            plyr=plyr,
            thld=thld,
            thlx=thlx,
            thvx=thvx,
            qtd=qtd,
            qtx=qtx,
            xlamde=xlamde,
            zl=zl,
            domain=(im, 1, kmscu),
        )

        bb1 = 2.0
        bb2 = 4.0

        mfscu_s4(
            buo=buo,
            cnvflg=cnvflg,
            krad1=krad1,
            mask=mask,
            wd2=wd2,
            xlamde=xlamde,
            zm=zm,
            bb1=bb1,
            bb2=bb2,
            domain=(im, 1, km),
        )

        mfscu_s5(
            buo=buo,
            cnvflg=cnvflg,
            flg=flg,
            krad=krad,
            krad1=krad1,
            mask=mask,
            mrad=mrad,
            mradx=mradx,
            mrady=mrady,
            xlamde=xlamde,
            wd2=wd2,
            zm=zm,
            domain=(im, 1, kmscu),
        )

        totflg = True

        for i in range(im):
            totflg = totflg and ~cnvflg[i, 0]

        if totflg:
            return

        for i in range(im):
            zm_mrad[i, 0] = zm[i, 0, mrad[i, 0] - 1]

        mfscu_s6(
            zl=zl,
            mask=mask,
            mrad=mrad,
            krad=krad,
            zm=zm,
            zm_mrad=zm_mrad,
            xlamde=xlamde,
            xlamdem=xlamdem,
            hrad=hrad,
            cnvflg=cnvflg,
            mrady=mrady,
            mradx=mradx,
            ce0=ce0,
            cm=cm,
            domain=(im, 1, kmscu),
        )

        mfscu_s7(
            cnvflg=cnvflg,
            dt2=delt,
            gdx=gdx,
            krad=krad,
            mask=mask,
            mrad=mrad,
            ra1=ra1,
            scaldfunc=scaldfunc,
            sumx=sumx,
            wd2=wd2,
            xlamde=xlamde,
            xlamavg=xlamavg,
            xmfd=xmfd,
            zl=zl,
            domain=(im, 1, kmscu),
        )

        mfscu_s8(
            cnvflg=cnvflg, krad=krad, mask=mask, thld=thld, thlx=thlx, domain=(im, 1, km)
        )

        mfscu_s9(
            cnvflg=cnvflg,
            el2orc=el2orc,
            elocp=elocp,
            eps=eps,
            epsm1=epsm1,
            krad=krad,
            mask=mask,
            mrad=mrad,
            pgcon=pgcon,
            pix=pix,
            plyr=plyr,
            qcdo=qcdo,
            qtd=qtd,
            qtx=qtx,
            tcdo=tcdo,
            thld=thld,
            thlx=thlx,
            u1=u1,
            ucdo=ucdo,
            v1=v1,
            vcdo=vcdo,
            xlamde=xlamde,
            xlamdem=xlamdem,
            zl=zl,
            ntcw=ntcw,
            domain=(im, 1, kmscu),
        )

        mfscu_10(
            cnvflg=cnvflg,
            krad=krad,
            mrad=mrad,
            mask=mask,
            zl=zl,
            xlamde=xlamde,
            qcdo=qcdo,
            q1=q1,
            domain=(im, 1, kmscu),
            externals={
                "ntcw": ntcw,
                "ntrac1": ntrac1
            }
        )

        return radj, mrad, buo, xmfd, tcdo, qcdo, ucdo, vcdo, xlamde