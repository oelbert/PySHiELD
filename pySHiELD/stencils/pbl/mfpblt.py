from gt4py.cartesian.gtscript import (
    __INLINED,
    FORWARD,
    PARALLEL,
    computation,
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
)
from ndsl.initialization.allocator import QuantityFactory
from pySHiELD._config import FloatFieldTracer
from pySHiELD.functions.physics_functions import fpvs


def mfpblt_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    q1: FloatFieldTracer,
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
):
    from __externals__ import ntcw

    with computation(PARALLEL), interval(...):
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
            ptem = min(physcons.ALP * vpert[0, 0], 3.0)
            thlu = thlx[0, 0, 0] + ptem
            qtu = qtx[0, 0, 0]
            buo = constants.GRAV * ptem / thvx[0, 0, 0]


def mfpblt_s1(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hpbl: FloatFieldIJ,
    kpbl: IntFieldIJ,
    kpblx: IntFieldIJ,
    kpbly: IntFieldIJ,
    k_mask: IntField,
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
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if k_mask[0, 0, 0] < kpbl[0, 0]:
                xlamue = physcons.CE0 * (
                    1.0 / (zm[0, 0, 0] + dz)
                    + 1.0 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                )
            else:
                xlamue = physcons.CE0 / dz
            xlamuem = physcons.CM * xlamue[0, 0, 0]

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
                qs = max(
                    physcons.QMIN,
                    constants.EPS * es / (plyr[0, 0, 0] + constants.EPSM1 * es),
                )
                dq = qtu[0, 0, 0] - qs

                if dq > 0.0:
                    gamma = physcons.EL2ORC * qs / (tlu ** 2)
                    qlu = dq / (1.0 + gamma)
                    qtu = qs + qlu
                    thvu = (thlu[0, 0, 0] + pix[0, 0, 0] * physcons.ELOCP * qlu) * (
                        1.0 + constants.ZVIR * qs - qlu
                    )
                else:
                    thvu = thlu[0, 0, 0] * (1.0 + constants.ZVIR * qtu[0, 0, 0])
                buo = constants.GRAV * (thvu / thvx[0, 0, 0] - 1.0)

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
                kpblx = k_mask[0, 0, 0]
                flg = rbup[0, 0] <= 0.0


def mfpblt_s1a(
    cnvflg: BoolFieldIJ,
    hpblx: FloatFieldIJ,
    kpblx: IntFieldIJ,
    k_mask: IntField,
    rbdn: FloatFieldIJ,
    rbup: FloatFieldIJ,
    zm: FloatField,
):

    with computation(FORWARD), interval(1, None):
        rbint = 0.0

        if k_mask[0, 0, 0] == kpblx[0, 0]:
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
    k_mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    qcko: FloatFieldTracer,
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
):
    from __externals__ import dt2

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                if kpbl[0, 0] > kpblx[0, 0]:
                    kpbl = kpblx[0, 0]
                    hpbl = hpblx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (kpbly[0, 0] > kpblx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if k_mask[0, 0, 0] < kpbl[0, 0]:
                ptem = 1 / (zm[0, 0, 0] + dz)
                ptem1 = 1 / max(hpbl[0, 0] - zm[0, 0, 0] + dz, dz)
                xlamue = physcons.CE0 * (ptem + ptem1)
            else:
                xlamue = physcons.CE0 / dz
            xlamuem = physcons.CM * xlamue[0, 0, 0]

    with computation(FORWARD):
        with interval(0, 1):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (k_mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz
        with interval(1, None):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if cnvflg[0, 0] and (k_mask[0, 0, 0] < kpbl[0, 0]):
                xlamavg = xlamavg[0, 0] + xlamue[0, 0, 0] * dz
                sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (k_mask[0, 0, 0] < kpbl[0, 0]):
            if wu2[0, 0, 0] > 0.0:
                xmf = physcons.A1 * sqrt(wu2[0, 0, 0])
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

                if sigma > physcons.A1:
                    scaldfunc = max(min((1.0 - sigma) * (1.0 - sigma), 1.0), 0.0)
                else:
                    scaldfunc = 1.0

    with computation(PARALLEL), interval(...):
        xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
        if cnvflg[0, 0] and (k_mask[0, 0, 0] < kpbl[0, 0]):
            xmf = min(scaldfunc[0, 0] * xmf[0, 0, 0], xmmx)

    with computation(FORWARD):
        with interval(0, 1):
            if cnvflg[0, 0]:
                thlu = thlx[0, 0, 0]
        with interval(1, None):
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (k_mask[0, 0, 0] <= kpbl[0, 0]):
                thlu = (
                    (1.0 - tem) * thlu[0, 0, -1]
                    + tem * (thlx[0, 0, -1] + thlx[0, 0, 0])
                ) / factor
                qtu = (
                    (1.0 - tem) * qtu[0, 0, -1] + tem * (qtx[0, 0, -1] + qtx[0, 0, 0])
                ) / factor

            tlu = thlu[0, 0, 0] / pix[0, 0, 0]
            es = 0.01 * fpvs(tlu)
            qs = max(
                physcons.QMIN,
                constants.EPS * es / (plyr[0, 0, 0] + constants.EPSM1 * es),
            )
            dq = qtu[0, 0, 0] - qs
            qlu = dq / (1.0 + (physcons.EL2ORC * qs / (tlu ** 2)))

            if cnvflg[0, 0] and (k_mask[0, 0, 0] <= kpbl[0, 0]):
                if dq > 0.0:
                    qtu = qs + qlu
                    qcko[0, 0, 0][0] = qs
                    qcko[0, 0, 0][1] = qlu
                    tcko = tlu + physcons.ELOCP * qlu
                else:
                    qcko[0, 0, 0][0] = qtu[0, 0, 0]
                    qcko[0, 0, 0][1] = 0.0
                    qcko_track = 1
                    tcko = tlu

            tem = 0.5 * xlamuem[0, 0, -1] * dz
            factor = 1.0 + tem

            if cnvflg[0, 0] and (k_mask[0, 0, 0] <= kpbl[0, 0]):
                ucko = (
                    (1.0 - tem) * ucko[0, 0, -1]
                    + (tem + physcons.PGCON) * u1[0, 0, 0]
                    + (tem - physcons.PGCON) * u1[0, 0, -1]
                ) / factor
                vcko = (
                    (1.0 - tem) * vcko[0, 0, -1]
                    + (tem + physcons.PGCON) * v1[0, 0, 0]
                    + (tem - physcons.PGCON) * v1[0, 0, -1]
                ) / factor


def mfpblt_s3(
    cnvflg: BoolFieldIJ,
    kpbl: IntFieldIJ,
    k_mask: IntField,
    xlamue: FloatField,
    qcko: FloatFieldTracer,
    q1: FloatFieldTracer,
    zl: FloatField,
    n_tracer: int,
):
    with computation(FORWARD), interval(1, None):
        if cnvflg[0, 0] and k_mask[0, 0, 0] <= kpbl[0, 0]:
            dz = zl[0, 0, 0] - zl[0, 0, -1]
            tem = 0.5 * xlamue[0, 0, -1] * dz
            factor = 1.0 + tem
            qcko[0, 0, 0][n_tracer] = (
                (1.0 - tem) * qcko[0, 0, -1][n_tracer]
                + tem * (q1[0, 0, 0][n_tracer] + q1[0, 0, -1][n_tracer])
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
        dt2: Float,
        ntcw: Int,
        ntrac1: Int,
        kmpbl: Int,
        ntke: Int,
    ):
        idx = stencil_factory.grid_indexing
        self._im = idx.iec - idx.isc
        self._jm = idx.jec - idx.jsc

        self._ntcw = ntcw
        self._ntrac1 = ntrac1
        self._ntke = ntke

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        self._xlamuem = make_quantity()
        self._wu2 = make_quantity()
        self._thlu = make_quantity()
        self._qtx = make_quantity()
        self._qtu = make_quantity()
        self._sigma = make_quantity_2D(Float)
        self._kpblx = make_quantity_2D(Int)
        self._kpbly = make_quantity_2D(Int)
        self._rbdn = make_quantity_2D(Float)
        self._rbup = make_quantity_2D(Float)
        self._hpblx = make_quantity_2D(Float)
        self._xlamavg = make_quantity_2D(Float)
        self._scaldfunc = make_quantity_2D(Float)
        self._sumx = make_quantity_2D(Float)
        self._flg = make_quantity_2D(Bool)

        self._mfpblt_s0 = stencil_factory.from_origin_domain(
            func=mfpblt_s0,
            externals={"ntcw": ntcw},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfpblt_s1 = stencil_factory.from_origin_domain(
            func=mfpblt_s1,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmpbl),
        )

        self._mfpblt_s1a = stencil_factory.from_origin_domain(
            func=mfpblt_s1a,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfpblt_s2 = stencil_factory.from_origin_domain(
            func=mfpblt_s2,
            externals={
                "dt2": dt2,
            },
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmpbl),
        )

        if (self._ntcw > 2) or (self._ntrac1 > self._ntcw):
            self._mfpblt_s3 = stencil_factory.from_origin_domain(
                func=mfpblt_s3,
                origin=idx.origin_compute(),
                domain=(idx.iec, idx.jec, kmpbl),
            )

    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        zl: FloatField,
        zm: FloatField,
        q1: FloatFieldTracer,  # I, J, K, ntracer field
        u1: FloatField,
        v1: FloatField,
        plyr: FloatField,
        pix: FloatField,
        thlx: FloatField,
        thvx: FloatField,
        gdx: FloatFieldIJ,
        hpbl: FloatFieldIJ,
        kpbl: IntFieldIJ,
        vpert: FloatFieldIJ,
        buo: FloatField,
        xmf: FloatField,
        tcko: FloatField,
        qcko: FloatFieldTracer,  # I, J, K, ntracer field
        ucko: FloatField,
        vcko: FloatField,
        xlamue: FloatField,
        k_mask: IntField,
    ):
        totflag = True

        for i in range(self._im):
            for j in range(self._jm):
                totflag = totflag and (not cnvflg.view[i, j])

        if totflag:
            return

        self._mfpblt_s0(
            buo,
            cnvflg,
            hpbl,
            kpbl,
            q1,
            self._qtu,
            self._qtx,
            self._thlu,
            thlx,
            thvx,
            vpert,
            self._wu2,
            self._kpblx,
            self._kpbly,
            self._rbup,
            self._rbdn,
            self._hpblx,
            self._xlamavg,
            self._sumx,
        )

        self._mfpblt_s1(
            buo,
            cnvflg,
            self._flg,
            hpbl,
            kpbl,
            self._kpblx,
            self._kpbly,
            k_mask,
            pix,
            plyr,
            self._qtu,
            self._qtx,
            self._rbdn,
            self._rbup,
            self._thlu,
            thlx,
            thvx,
            self._wu2,
            xlamue,
            self._xlamuem,
            zl,
            zm,
        )

        self._mfpblt_s1a(
            cnvflg,
            self._hpblx,
            self._kpblx,
            k_mask,
            self._rbdn,
            self._rbup,
            zm,
        )

        self._mfpblt_s2(
            cnvflg,
            gdx,
            hpbl,
            self._hpblx,
            kpbl,
            self._kpblx,
            self._kpbly,
            k_mask,
            pix,
            plyr,
            qcko,
            self._qtu,
            self._qtx,
            self._scaldfunc,
            self._sumx,
            tcko,
            self._thlu,
            thlx,
            u1,
            ucko,
            v1,
            vcko,
            xmf,
            self._xlamavg,
            xlamue,
            self._xlamuem,
            self._wu2,
            zl,
            zm,
        )

        if self._ntcw > 2:
            for n in range(1, self._ntcw):
                self._mfpblt_s3(
                    cnvflg,
                    kpbl,
                    k_mask,
                    xlamue,
                    qcko,
                    q1,
                    zl,
                    n,
                )
        if self._ntrac1 > self._ntcw:
            for n in range(self._ntcw, self._ntrac1):
                dim_n = n if n < self._ntke else n + 1
                self._mfpblt_s3(
                    cnvflg,
                    kpbl,
                    k_mask,
                    xlamue,
                    qcko,
                    q1,
                    zl,
                    dim_n,
                )
