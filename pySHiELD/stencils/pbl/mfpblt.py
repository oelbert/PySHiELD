from gt4py.cartesian.gtscript import (
    __INLINED,
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
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from pySHiELD.functions.physics_functions import fpvs


def mfpblt_s3(
    cnvflg: BoolFieldIJ,
    kpbl: IntFieldIJ,
    mask: IntField,
    xlamue: FloatField,
    qcko: FloatField,
    q1_gt: FloatField,
    zl: FloatField,
):
    from __externals__ import ntcw, ntrac1

    with computation(FORWARD), interval(1, None):
        if __INLINED(ntcw > 2):
            for n in range(1, ntcw):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0, 0, 0][n] = (
                        (1.0 - tem) * qcko[0, 0, -1][n]
                        + tem * (q1_gt[0, 0, 0][n] + q1_gt[0, 0, -1][n])
                    ) / factor

        if __INLINED(ntrac1 > ntcw):
            for n2 in range(ntcw, ntrac1):
                if cnvflg[0, 0] and mask[0, 0, 0] <= kpbl[0, 0]:
                    dz = zl[0, 0, 0] - zl[0, 0, -1]
                    tem = 0.5 * xlamue[0, 0, -1] * dz
                    factor = 1.0 + tem
                    qcko[0, 0, 0][n2] = (
                        (1.0 - tem) * qcko[0, 0, -1][n2]
                        + tem * (q1_gt[0, 0, 0][n2] + q1_gt[0, 0, -1][n2])
                    ) / factor


def mfpblt_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    q1: FloatField,
    qtu: FloatField,
    qtx: FloatField,
    thlu: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    vpert: FloatFieldIJ,
    wu2: FloatField,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    rbup: FloatFieldIJ,
    rbdn: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    xlamavg: FloatFieldIJ,
    sumx: FloatFieldIJ,
    alp: Float,
    g: Float,
    ntcw: Int,
):

    with computation(PARALLEL), interval(0, -1):
        if cnvflg[0, 0]:
            buo = 0.0
            wu2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw]

    with computation(FORWARD), interval(0, 1):
        kpblx = 0
        kpbly = 0
        rbup = 0.0
        rbdn = 0.0
        hpblx = 0.0
        xlamavg = 0.0
        sumx = 0.0
        if cnvflg[0, 0]:
            ptem = min(alp * vpert[0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = g * ptem / thvx[0, 0, 0]


def mfpblt_s1(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    qtu: FloatField,
    qtx: FloatField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    thlu: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    wu2: FloatField,
    xlamue: FloatField,
    xlamuem: FloatField,
    zl: FloatField,
    zm: FloatField,
    ce0: Float,
    cm: Float,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    fv: Float,
    g: Float,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                xlamue = ce0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(1, None):
            if cnvflg[0, 0]:
                tem = 0.5 * xlamue[0, 0, -1] * (zl[0, 0, 0] - zl[0, 0, -1])
                factor = 1.0 + tem
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

                tlu = thlu[0, 0, 0] / pix[0, 0, 0]
                es = 0.01 * fpvs(tlu)
                qs = max(constants.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
                dq = qtu[0, 0, 0] - qs

                if dq > 0.0:
                    gamma = el2orc * qs / (tlu ** 2)
                    qlu = dq / (1.0 + gamma)
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * elocp * qlu) * (
                        1.0 + fv * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + fv * qtu[0, 0, 0])
                buo = g * (thvu / thvx[0, 0, 0] - 1.0)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                wu2 = (4.0 * buo[0, 0, 0] * zm[0, 0, 0]) / (
                    1.0 + (0.5 * 2.0 * xlamue[0, 0, 0] * zm[0, 0, 0])
                )
        with interval(1, None):
            if cnvflg[0, 0]:
                dz = zm[0, 0, 0] - zm[0, 0, -1]
                tem = 0.25 * 2.0 * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
                wu2 = (((1.0 - tem) * wu2[0, 0, -1]) + (4.0 * buo[0, 0, 0] * dz)) / (
                    1.0 + tem
                )

    with computation(FORWARD):
        with interval(0, 1):
            flg = True
            kpbly = kpbl[0, 0]
            if cnvflg[0, 0]:
                flg = False
                rbup = wu2[0, 0, 0]

        with interval(1, None):
            if not flg[0, 0]:
                rbdn = rbup[0, 0]
                rbup = wu2[0, 0, 0]
                kpblx = mask[0, 0, 0]
                flg = rbup[0, 0] <= 0.0


def mfpblt_s1a(
    cnvflg: BoolFieldIJ,
    hpblx: FloatFieldIJ,
    kpblx: IntFieldIJ,
    mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    zm: FloatField,
):

    with computation(FORWARD), interval(...):
        rbint = 0.0

        if mask[0, 0, 0] == kpblx[0, 0]:
            if cnvflg[0, 0]:
                if rbdn[0, 0] <= 0.0:
                    rbint = 0.0
                elif rbup[0, 0] >= 0.0:
                    rbint = 1.0
                else:
                    rbint = rbdn[0, 0] / (rbdn[0, 0] - rbup[0, 0])

                hpblx = zm[0, 0, -1] + rbint * (zm[0, 0, 0] - zm[0, 0, -1])


def mfpblt_s2(
    cnvflg: BoolFieldIJ,
    gdx: FloatFieldIJ,
    hpbl: FloatFieldIJ,
    hpblx: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    qcko: FloatField,
    qtu: FloatField,
    qtx: FloatField,
    scaldfunc: FloatFieldIJ,
    sumx: FloatFieldIJ,
    tcko: FloatField,
    thlu: FloatField,
    thlx: FloatField,
    u1: FloatField,
    ucko: FloatField,
    v1: FloatField,
    vcko: FloatField,
    xmf: FloatField,
    xlamavg: FloatFieldIJ,
    xlamue: FloatField,
    xlamuem: FloatField,
    wu2: FloatField,
    zl: FloatField,
    zm: FloatField,
    a1: Float,
    dt2: Float,
    ce0: Float,
    cm: Float,
    el2orc: Float,
    elocp: Float,
    eps: Float,
    epsm1: Float,
    pgcon: Float,
):

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                if kpbl[0, 0] > kpblx[0, 0]:
                    kpbl = kpblx[0, 0]
                    hpbl = hpblx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (kpbly[0, 0] > kpblx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = ce0 * (ptem + ptem1)
            else:
                xlamue = ce0 / dz
            xlamuem = cm * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        with interval(1, None):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = a1 * sqrt(wu2[0, 0, 0])
            else:
                xmf = 0.0

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                tem = 0.2 / xlamavg[0, 0]
                sigma = min(
                    max((3.14 * tem * tem) / (gdx[0, 0] * gdx[0, 0]), 0.001),
                    0.999,
                )

                if sigma > a1:
                    scaldfunc = max(min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0)
                else:
                    scaldfunc = 1.0

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0] and (mask[0, 0, 0] < kpbl[0, 0]):
            xmf = min(scaldfunc[0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

            tlu = thlu[0, 0, 0] / pix[0, 0, 0]
            es = 0.01 * fpvs(tlu)
            qs = max(constants.QMIN, eps * es / (plyr[0, 0, 0] + epsm1 * es))
            dq = qtu[0, 0, 0] - qs
            qlu = dq / (1.0 + (el2orc * qs / (tlu ** 2)))

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko[0, 0, 0][0] = qs
                    qcko[0, 0, 0][1] = qlu
                    tcko = tlu + elocp * qlu
                else:
                    qcko[0, 0, 0][0] = qtu[0, 0, 0]
                    qcko[0, 0, 0][1] = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (mask[0, 0, 0] <= kpbl[0, 0]):
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1]
                    + (tem + pgcon) * u1[0, 0, 0]
                    + (tem - pgcon) * u1[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1]
                    + (tem + pgcon) * v1[0, 0, 0]
                    + (tem - pgcon) * v1[0, 0, -1]
                ) / factor


class PBLMassFlux:
    """
    EDMF parameterization for the convective boundary layer (Siebesma et al., 2007)
    to take into account nonlocal transport by large eddies
    Fortran name is mfpblt
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        pass

    def __call__(
        self,
        im,
        ix,
        km,
        kmpbl,
        ntcw,
        ntrac1,
        delt,
        cnvflg,
        zl,
        zm,
        q1_gt,
        t1,
        u1,
        v1,
        plyr,
        pix,
        thlx,
        thvx,
        gdx,
        hpbl,
        kpbl,
        vpert,
        buo,
        xmf,
        tcko,
        qcko,
        ucko,
        vcko,
        xlamue,
        g,
        gocp,
        elocp,
        el2orc,
        mask,
        qtx,
        wu2,
        qtu,
        xlamuem,
        thlu,
        kpblx,
        kpbly,
        rbup,
        rbdn,
        flg,
        hpblx,
        xlamavg,
        sumx,
        scaldfunc,
        ce0,
        cm,
        qmin,
        qlmin,
        alp,
        pgcon,
        a1,
        b1,
        f1,
        fv,
        eps,
        epsm1,
    ):
        totflag = True

        for i in range(im):
            totflag = totflag and ~cnvflg[i, 0]

        if totflag:
            return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue

        mfpblt_s0(
            alp=alp,
            buo=buo,
            cnvflg=cnvflg,
            g=g,
            hpbl=hpbl,
            kpbl=kpbl,
            q1=q1_gt,
            qtu=qtu,
            qtx=qtx,
            thlu=thlu,
            thlx=thlx,
            thvx=thvx,
            vpert=vpert,
            wu2=wu2,
            kpblx=kpblx,
            kpbly=kpbly,
            rbup=rbup,
            rbdn=rbdn,
            hpblx=hpblx,
            xlamavg=xlamavg,
            sumx=sumx,
            ntcw=ntcw - 1,
        )

        mfpblt_s1(
            buo=buo,
            ce0=ce0,
            cm=cm,
            cnvflg=cnvflg,
            elocp=elocp,
            el2orc=el2orc,
            eps=eps,
            epsm1=epsm1,
            flg=flg,
            fv=fv,
            g=g,
            hpbl=hpbl,
            kpbl=kpbl,
            kpblx=kpblx,
            kpbly=kpbly,
            mask=mask,
            pix=pix,
            plyr=plyr,
            qtu=qtu,
            qtx=qtx,
            rbdn=rbdn,
            rbup=rbup,
            thlu=thlu,
            thlx=thlx,
            thvx=thvx,
            wu2=wu2,
            xlamue=xlamue,
            xlamuem=xlamuem,
            zl=zl,
            zm=zm,
            domain=(im, 1, kmpbl),
        )

        mfpblt_s1a(
            cnvflg=cnvflg,
            hpblx=hpblx,
            kpblx=kpblx,
            mask=mask,
            rbdn=rbdn,
            rbup=rbup,
            zm=zm,
            domain=(im, 1, km),
        )

        mfpblt_s2(
            a1=a1,
            ce0=ce0,
            cm=cm,
            cnvflg=cnvflg,
            dt2=delt,
            el2orc=el2orc,
            elocp=elocp,
            eps=eps,
            epsm1=epsm1,
            gdx=gdx,
            hpbl=hpbl,
            hpblx=hpblx,
            kpbl=kpbl,
            kpblx=kpblx,
            kpbly=kpbly,
            mask=mask,
            pgcon=pgcon,
            pix=pix,
            plyr=plyr,
            qcko=qcko,
            qtu=qtu,
            qtx=qtx,
            scaldfunc=scaldfunc,
            sumx=sumx,
            tcko=tcko,
            thlu=thlu,
            thlx=thlx,
            u1=u1,
            ucko=ucko,
            v1=v1,
            vcko=vcko,
            xlamue=xlamue,
            xlamuem=xlamuem,
            xlamavg=xlamavg,
            xmf=xmf,
            wu2=wu2,
            zl=zl,
            zm=zm,
            domain=(im, 1, kmpbl),
        )

        mfpblt_s3(
            cnvflg=cnvflg,
            kpbl=kpbl,
            mask=mask,
            xlamue=xlamue,
            qcko=qcko,
            q1_gt=q1_gt,
            zl=zl,
            ntcw=ntcw,
            ntrac1=ntrac1,
            domain=(im, 1, kmpbl),
            externals={
                "ntcw": ntcw,
                "ntrac1": ntrac1,
            },
        )

        return kpbl, hpbl, buo, xmf, tcko, qcko, ucko, vcko, xlamue
