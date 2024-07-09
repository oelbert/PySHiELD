from gt4py.cartesian.gtscript import PARALLEL, computation, interval, sqrt

import ndsl.constants as constants
import pySHiELD.constants as physcons

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    BoolFieldIJ,
    FloatFieldIJ,
    IntFieldIJ,
)
from pySHiELD.functions.physics_functions import fpvsx


def sfc_ocean(
    ps: FloatFieldIJ,
    u1: FloatFieldIJ,
    v1: FloatFieldIJ,
    t1: FloatFieldIJ,
    q1: FloatFieldIJ,
    tskin: FloatFieldIJ,
    cm: FloatFieldIJ,
    ch: FloatFieldIJ,
    prsl1: FloatFieldIJ,
    prslki: FloatFieldIJ,
    ddvel: FloatFieldIJ,
    qsurf: FloatFieldIJ,
    cmm: FloatFieldIJ,
    chh: FloatFieldIJ,
    gflux: FloatFieldIJ,
    evap: FloatFieldIJ,
    hflx: FloatFieldIJ,
    ep: FloatFieldIJ,
    islimsk: IntFieldIJ,
    flag_iter: BoolFieldIJ,
):
    with computation(PARALLEL), interval(0, 1):
        if (islimsk == 0) and (flag_iter):
            wind = max(sqrt(u1**2 + v1**2) + max(0., min(ddvel, 30)), 1.0)
            q0 = max(q1, 1.e-8)
            rho = prsl1 / (constants.RDGAS * t1 * (1.0 + constants.ZVIR * q0))

            qss = fpvsx(tskin)
            qss = constants.EPS * qss / (ps + (constants.EPS - 1.) * qss)

            evap = 0.0
            hflx = 0.0
            ep = 0.0
            gflux = 0.0

            # rcp  = rho cp ch v
            rch = rho * constants.CP_AIR * ch * wind
            cmm = cm * wind
            chh = rho * ch * wind

            # sensible and latent heat flux over open water:
            hflx = rch * (tskin - t1 * prslki)

            evap = physcons.HOCP * rch * (qss - q0)
            qsurf = qss

            tem = 1.0 / rho
            hflx = hflx * tem / constants.CP
            evap = evap * tem / constants.HLV


class SurfaceOcean:
    def __init__(self, stencil_factory: StencilFactory,):
        grid_indexing = stencil_factory.grid_indexing
        self._sfc_ocean = stencil_factory.from_origin_domain(
            sfc_ocean,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        ps: FloatFieldIJ,
        u1: FloatFieldIJ,
        v1: FloatFieldIJ,
        t1: FloatFieldIJ,
        q1: FloatFieldIJ,
        tskin: FloatFieldIJ,
        cm: FloatFieldIJ,
        ch: FloatFieldIJ,
        prsl1: FloatFieldIJ,
        prslki: FloatFieldIJ,
        ddvel: FloatFieldIJ,
        qsurf: FloatFieldIJ,
        cmm: FloatFieldIJ,
        chh: FloatFieldIJ,
        gflux: FloatFieldIJ,
        evap: FloatFieldIJ,
        hflx: FloatFieldIJ,
        ep: FloatFieldIJ,
        islimsk: IntFieldIJ,
        flag_iter: BoolFieldIJ,
    ):
        self._sfc_ocean(
            ps,
            u1,
            v1,
            t1,
            q1,
            tskin,
            cm,
            ch,
            prsl1,
            prslki,
            ddvel,
            qsurf,
            cmm,
            chh,
            gflux,
            evap,
            hflx,
            ep,
            islimsk,
            flag_iter,
        )
