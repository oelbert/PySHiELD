from gt4py.cartesian.gtscript import (
    __INLINED,
    BACKWARD,
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
from pySHiELD.functions.physics_functions import fpvs


def mfscu_s2(
    zl: FloatField,
    k_mask: IntField,
    mrad: IntFieldIJ,
    krad: IntFieldIJ,
    zm: FloatField,
    zm_mrad: FloatFieldIJ,
    xlamde: FloatField,
    xlamdem: FloatField,
    hrad: FloatFieldIJ,
    cnvflg: BoolFieldIJ,
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if k_mask[0, 0, 0] >= mrad[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = physcons.CE0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = physcons.CE0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = physcons.CE0 / dz
            xlamdem = physcons.CM * xlamde[0, 0, 0]


def mfscu_s6(
    zl: FloatField,
    k_mask: IntField,
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
):
    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0] and (mrady[0, 0] < mradx[0, 0]):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            if k_mask[0, 0, 0] >= mrad[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if mrad[0, 0] == 0:
                    xlamde = physcons.CE0 * (
                        (1.0 / (zm[0, 0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
                else:
                    xlamde = physcons.CE0 * (
                        (1.0 / (zm[0, 0, 0] - zm_mrad[0, 0] + dz))
                        + 1.0 / max(hrad[0, 0] - zm[0, 0, 0] + dz, dz)
                    )
            else:
                xlamde = physcons.CE0 / dz
            xlamdem = physcons.CM * xlamde[0, 0, 0]


def mfscu_10(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    mrad: IntFieldIJ,
    k_mask: IntField,
    zl: FloatField,
    xlamde: FloatField,
    qcdo: FloatField,
    q1: FloatField,
    n_tracer: int
):
    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
        ):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            tem = 0.5 * xlamde[0, 0, 0] * dz
            factor = 1.0 + tem
            qcdo[0, 0, 0][n_tracer] = (
                (1.0 - tem) * qcdo[0, 0, 1][n_tracer]
                + tem * (q1[0, 0, 0][n_tracer] + q1[0, 0, 1][n_tracer])
            ) / factor


def mfscu_s0(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    hrad: FloatFieldIJ,
    krad: IntFieldIJ,
    krad1: IntFieldIJ,
    k_mask: IntField,
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
):
    from __externals__ import ntcw

    with computation(PARALLEL), interval(...):
        if cnvflg[0, 0]:
            buo = 0.0
            wd2 = 0.0
            qtx = q1[0, 0, 0][0] + q1[0, 0, 0][ntcw - 1]

    with computation(FORWARD), interval(...):
        if krad[0, 0] == k_mask[0, 0, 0]:
            if cnvflg[0, 0]:
                hrad = zm[0, 0, 0]
                krad1 = k_mask[0, 0, 0] - 1
                tem1 = max(
                    constants.CLDTIME * radmin[0, 0] / (zm[0, 0, 1] - zm[0, 0, 0]), -3.0
                )
                thld = thlx[0, 0, 0] + tem1
                qtd = qtx[0, 0, 0]
                thlvd = thlvx[0, 0, 0] + tem1
                buo = -constants.GRAV * tem1 / thvx[0, 0, 0]

                ra1 = physcons.A1
                ra2 = physcons.A11

                tem = thetae[0, 0, 0] - thetae[0, 0, 1]
                tem1 = qtx[0, 0, 0] - qtx[0, 0, 1]
                if (tem > 0.0) and (tem1 > 0.0):
                    cteit = constants.CP_AIR * tem / (constants.HLV * tem1)
                    if cteit > constants.ACTEI:
                        ra1 = physcons.A2
                        ra2 = physcons.A22

                radj = -ra2[0, 0] * radmin[0, 0]

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrad = krad[0, 0]


def mfscu_s1(
    cnvflg: BoolFieldIJ,
    flg: BoolFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
    mrad: IntFieldIJ,
    thlvd: FloatFieldIJ,
    thlvx: FloatField,
):
    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = k_mask[0, 0, 0]
                else:
                    flg[0, 0] = 0
        with interval(0, -1):
            if flg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if thlvd[0, 0] <= thlvx[0, 0, 0]:
                    mrad[0, 0] = k_mask[0, 0, 0]
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
    k_mask: IntField,
    pix: FloatField,
    plyr: FloatField,
    thld: FloatField,
    thlx: FloatField,
    thvx: FloatField,
    qtd: FloatField,
    qtx: FloatField,
    xlamde: FloatField,
    zl: FloatField,
):

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        tem = 0.5 * xlamde[0, 0, 0] * dz
        factor = 1.0 + tem
        if cnvflg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
            thld = (
                (1.0 - tem) * thld[0, 0, 1] + tem * (thlx[0, 0, 0] + thlx[0, 0, 1])
            ) / factor
            qtd = (
                (1.0 - tem) * qtd[0, 0, 1] + tem * (qtx[0, 0, 0] + qtx[0, 0, 1])
            ) / factor

        tld = thld[0, 0, 0] / pix[0, 0, 0]
        es = 0.01 * fpvs(tld)
        qs = max(
            constants.QMIN, constants.EPS * es / (plyr[0, 0, 0] + constants.EPSM1 * es)
        )
        dq = qtd[0, 0, 0] - qs
        gamma = constants.EL2ORC * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)
        if cnvflg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
            if dq > 0.0:
                qtd = qs + qld
                tem1 = 1.0 + constants.ZVIR * qs - qld
                thvd = (thld[0, 0, 0] + pix[0, 0, 0] * constants.ELOCP * qld) * tem1
            else:
                tem1 = 1.0 + constants.ZVIR * qtd[0, 0, 0]
                thvd = thld[0, 0, 0] * tem1
            buo = constants.GRAV * (1.0 - thvd / thvx[0, 0, 0])


def mfscu_s4(
    buo: FloatField,
    cnvflg: BoolFieldIJ,
    krad1: IntFieldIJ,
    k_mask: IntField,
    wd2: FloatField,
    xlamde: FloatField,
    zm: FloatField,
):
    from __externals__ import bb1, bb2

    with computation(FORWARD), interval(...):
        if k_mask[0, 0, 0] == krad1[0, 0]:
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
    k_mask: IntField,
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
        if cnvflg[0, 0] and k_mask[0, 0, 0] < krad1[0, 0]:
            wd2 = (((1.0 - tem) * wd2[0, 0, 1]) + (4.0 * buo[0, 0, 1] * dz)) / ptem1

    with computation(FORWARD), interval(0, 1):
        flg = cnvflg[0, 0]
        mrady = mrad[0, 0]
        if flg[0, 0]:
            mradx = krad[0, 0]

    with computation(BACKWARD):
        with interval(-1, None):
            if flg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = k_mask[0, 0, 0]
                else:
                    flg = 0
        with interval(0, -1):
            if flg[0, 0] and k_mask[0, 0, 0] < krad[0, 0]:
                if wd2[0, 0, 0] > 0.0:
                    mradx = k_mask[0, 0, 0]
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
    k_mask: IntField,
    mrad: IntFieldIJ,
    ra1: FloatFieldIJ,
    scaldfunc: FloatFieldIJ,
    sumx: FloatFieldIJ,
    wd2: FloatField,
    xlamde: FloatField,
    xlamavg: FloatFieldIJ,
    xmfd: FloatField,
    zl: FloatField,
):
    from __externals__ import dt2

    with computation(FORWARD), interval(0, 1):
        xlamavg = 0.0
        sumx = 0.0

    with computation(BACKWARD), interval(-1, None):
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            dz = zl[0, 0, 1] - zl[0, 0, 0]
            xlamavg = xlamavg[0, 0] + xlamde[0, 0, 0] * dz
            sumx = sumx[0, 0] + dz

    with computation(FORWARD), interval(0, 1):
        if cnvflg[0, 0]:
            xlamavg = xlamavg[0, 0] / sumx[0, 0]

    with computation(BACKWARD), interval(...):
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
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
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            xmmx = (zl[0, 0, 1] - zl[0, 0, 0]) / dt2
            xmfd = min(scaldfunc[0, 0] * xmfd[0, 0, 0], xmmx)


def mfscu_s8(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
    thld: FloatField,
    thlx: FloatField,
):
    with computation(PARALLEL), interval(...):
        if krad[0, 0] == k_mask[0, 0, 0]:
            if cnvflg[0, 0]:
                thld = thlx[0, 0, 0]


def mfscu_s9(
    cnvflg: BoolFieldIJ,
    krad: IntFieldIJ,
    k_mask: IntField,
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
):
    from __externals__ import ntcw

    with computation(BACKWARD), interval(...):
        dz = zl[0, 0, 1] - zl[0, 0, 0]
        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
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
        qs = max(
            constants.QMIN, constants.EPS * es / (plyr[0, 0, 0] + constants.EPSM1 * es)
        )
        dq = qtd[0, 0, 0] - qs
        gamma = constants.EL2ORC * qs / (tld ** 2)
        qld = dq / (1.0 + gamma)

        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
        ):
            if dq > 0.0:
                qtd = qs + qld
                qcdo[0, 0, 0][0] = qs
                qcdo[0, 0, 0][ntcw - 1] = qld
                tcdo = tld + constants.ELOCP * qld
            else:
                qcdo[0, 0, 0][0] = qtd[0, 0, 0]
                qcdo[0, 0, 0][ntcw - 1] = 0.0
                tcdo = tld

        if (
            cnvflg[0, 0]
            and k_mask[0, 0, 0] < krad[0, 0]
            and k_mask[0, 0, 0] >= mrad[0, 0]
        ):
            tem = 0.5 * xlamdem[0, 0, 0] * dz
            factor = 1.0 + tem
            ptem = tem - constants.PGCON
            ptem1 = tem + constants.PGCON
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
        quantity_factory: QuantityFactory,
        dt2: Float,
        ntcw: Int,
        ntrac1: Int,
        kmscu: Int,
    ):

        idx = stencil_factory.grid_indexing
        self._im = idx.iec - idx.isc
        self._jm = idx.jec - idx.jsc

        self._kmscu = kmscu
        self._ntcw = ntcw
        self._ntrac1 = ntrac1
        self._dt2 = dt2

        # From our tuning:
        self._bb1 = 2.0
        self._bb2 = 4.0

        # Allocate internal storages:
        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        self._qtx = make_quantity()
        self._qtd = make_quantity()
        self._wd2 = make_quantity()
        self._thld = make_quantity()
        self._xlamdem = make_quantity()
        self._flg = make_quantity_2D(Bool)
        self._krad1 = make_quantity_2D(Int)
        self._mradx = make_quantity_2D(Int)
        self._mrady = make_quantity_2D(Int)
        self._hrad = make_quantity_2D(Float)
        self._thlvd = make_quantity_2D(Float)
        self._ra1 = make_quantity_2D(Float)
        self._ra2 = make_quantity_2D(Float)
        self._xlamavg = make_quantity_2D(Float)
        self._sigma = make_quantity_2D(Float)
        self._scaldfunc = make_quantity_2D(Float)
        self._sumx = make_quantity_2D(Float)
        self._zm_mrad = make_quantity_2D(Float)

        # Compile stencils:
        self._mfscu_s0 = stencil_factory.from_origin_domain(
            func=mfscu_s0,
            externals={"ntcw": ntcw},
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfscu_s1 = stencil_factory.from_origin_domain(
            func=mfscu_s1,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s2 = stencil_factory.from_origin_domain(
            func=mfscu_s2,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s3 = stencil_factory.from_origin_domain(
            func=mfscu_s3,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s4 = stencil_factory.from_origin_domain(
            func=mfscu_s4,
            externals={
                "bb1": self._bb1,
                "bb2": self._bb2,
            },
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfscu_s5 = stencil_factory.from_origin_domain(
            func=mfscu_s5,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s6 = stencil_factory.from_origin_domain(
            func=mfscu_s6,
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s7 = stencil_factory.from_origin_domain(
            func=mfscu_s7,
            externals={
                "dt2": self._dt2,
            },
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        self._mfscu_s8 = stencil_factory.from_origin_domain(
            func=mfscu_s8,
            origin=idx.origin_compute(),
            domain=idx.domain_compute(),
        )

        self._mfscu_s9 = stencil_factory.from_origin_domain(
            func=mfscu_s9,
            externals={
                "ntcw": ntcw,
            },
            origin=idx.origin_compute(),
            domain=(idx.iec, idx.jec, kmscu),
        )

        if (self._ntcw > 2) or (self._ntrac1 > self._ntcw):
            self._mfscu_10 = stencil_factory.from_origin_domain(
                func=mfscu_10,
                origin=idx.origin_compute(),
                domain=(idx.iec, idx.jec, kmscu),
            )

    def __call__(
        self,
        cnvflg: BoolFieldIJ,
        zl: FloatField,
        zm: FloatField,
        q1,  # I, J, K, ntracer field
        u1: FloatField,
        v1: FloatField,
        plyr: FloatField,
        pix: FloatField,
        thlx: FloatField,
        thvx: FloatField,
        thlvx: FloatField,
        gdx: FloatFieldIJ,
        thetae: FloatField,
        radj: FloatFieldIJ,
        krad: IntFieldIJ,
        mrad: IntFieldIJ,
        radmin: FloatFieldIJ,
        buo: FloatField,
        xmfd: FloatField,
        tcdo: FloatField,
        qcdo,  # I, J, K, ntracer field
        ucdo: FloatField,
        vcdo: FloatField,
        xlamde: FloatField,
        k_mask: IntField,
    ):

        totflg = True

        for i in range(self._im):
            for j in range(self._jm):
                totflg = totflg and (not cnvflg.view[i, j])

        if totflg:
            return

        mfscu_s0(
            buo,
            cnvflg,
            self._flg,
            self._hrad,
            krad,
            self._krad1,
            k_mask,
            mrad,
            q1,
            self._qtd,
            self._qtx,
            self._ra1,
            self._ra2,
            radmin,
            radj,
            thetae,
            self._thld,
            self._thlvd,
            thlvx,
            thlx,
            thvx,
            self._wd2,
            zm,
        )

        self._mfscu_s1(
            cnvflg,
            self._flg,
            krad,
            k_mask,
            mrad,
            self._thlvd,
            thlvx,
        )

        totflg = True

        for i in range(self._im):
            for j in range(self._jm):
                totflg = totflg and (not cnvflg.view[i, j])

        if totflg:
            return

        for i in range(self._im):
            for j in range(self._jm):
                self._zm_mrad.view[i, j] = zm.view[i, 0, mrad.view[i, j] - 1]

        self._mfscu_s2(
            zl,
            k_mask,
            mrad,
            krad,
            zm,
            self._zm_mrad,
            xlamde,
            self._xlamdem,
            self._hrad,
            cnvflg,
        )

        self._mfscu_s3(
            buo,
            cnvflg,
            krad,
            k_mask,
            pix,
            plyr,
            self._thld,
            thlx,
            thvx,
            self._qtd,
            self._qtx,
            xlamde,
            zl,
        )

        self._mfscu_s4(
            buo,
            cnvflg,
            self._krad1,
            k_mask,
            self._wd2,
            xlamde,
            zm,
        )

        self._mfscu_s5(
            buo,
            cnvflg,
            self._flg,
            krad,
            self._krad1,
            k_mask,
            mrad,
            self._mradx,
            self._mrady,
            xlamde,
            self._wd2,
            zm,
        )

        totflg = True

        for i in range(self._im):
            for j in range(self._jm):
                totflg = totflg and (not cnvflg.view[i, j])

        if totflg:
            return

        for i in range(self._im):
            for j in range(self._jm):
                self._zm_mrad.view[i, j] = zm.view[i, j, mrad.view[i, j] - 1]

        self._mfscu_s6(
            zl,
            k_mask,
            mrad,
            krad,
            zm,
            self._zm_mrad,
            xlamde,
            self._xlamdem,
            self._hrad,
            cnvflg,
            self._mrady,
            self._mradx,
        )

        self._mfscu_s7(
            cnvflg,
            gdx,
            krad,
            k_mask,
            mrad,
            self._ra1,
            self._scaldfunc,
            self._sumx,
            self._wd2,
            xlamde,
            self._xlamavg,
            xmfd,
            zl,
        )

        self._mfscu_s8(
            cnvflg,
            krad,
            k_mask,
            self._thld,
            thlx,
        )

        self._mfscu_s9(
            cnvflg,
            krad,
            k_mask,
            mrad,
            pix,
            plyr,
            qcdo,
            self._qtd,
            self._qtx,
            tcdo,
            self._thld,
            thlx,
            u1,
            ucdo,
            v1,
            vcdo,
            xlamde,
            self._xlamdem,
            zl,
        )

        if self._ntcw > 2:
            for n in range(1, self._ntcw):
                mfscu_10(
                    cnvflg,
                    krad,
                    mrad,
                    k_mask,
                    zl,
                    xlamde,
                    qcdo,
                    q1,
                    n,
                )

        if self._ntrac1 > self._ntcw:
            for n in range(self._ntcw, self._ntrac1):
                mfscu_10(
                    cnvflg,
                    krad,
                    mrad,
                    k_mask,
                    zl,
                    xlamde,
                    qcdo,
                    q1,
                    n,
                )
