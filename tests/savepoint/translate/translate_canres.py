from ndsl import Namelist, StencilFactory
from ndsl.quantity import Quantity
from ndsl.dsl.stencil import GridIndexing
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl.dsl.typing import (
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntFieldIJ,
)
from ndsl.constants import X_DIM, Y_DIM, Z_DIM, Z_INTERFACE_DIM
from gt4py.cartesian.gtscript import FORWARD, computation, interval
from pySHiELD.stencils.surface.noah_lsm.lsm_driver import canres
from pySHiELD.stencils.surface.noah_lsm.lsm_2d import canres_fn
import pySHiELD.constants as physcons
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


def canres_stencil(
    nroot: IntFieldIJ,
    swdn: FloatFieldIJ,
    ch: FloatFieldIJ,
    q2: FloatFieldIJ,
    q2sat: FloatFieldIJ,
    dqsdt2: FloatFieldIJ,
    sfctmp: FloatFieldIJ,
    sfcprs: FloatFieldIJ,
    sfcems: FloatFieldIJ,
    sh2o0: FloatFieldIJ,
    sh2o1: FloatFieldIJ,
    sh2o2: FloatFieldIJ,
    sh2o3: FloatFieldIJ,
    smcwlt: FloatFieldIJ,
    smcref: FloatFieldIJ,
    zsoil0: FloatFieldIJ,
    zsoil1: FloatFieldIJ,
    zsoil2: FloatFieldIJ,
    zsoil3: FloatFieldIJ,
    rsmin: FloatFieldIJ,
    rgl: FloatFieldIJ,
    hs: FloatFieldIJ,
    xlai: FloatFieldIJ,
    zroot: FloatFieldIJ,
    rc: FloatFieldIJ,
    pc: FloatFieldIJ,
    rcs: FloatFieldIJ,
    rct: FloatFieldIJ,
    rcq: FloatFieldIJ,
    rcsoil: FloatFieldIJ,
    lsm_mask: BoolFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        if lsm_mask:
            rc, pc, rcs, rct, rcq, rcsoil = canres_fn(
                nroot,
                swdn,
                ch,
                q2,
                q2sat,
                dqsdt2,
                sfctmp,
                physcons.CP1,
                sfcprs,
                sfcems,
                sh2o0,
                sh2o1,
                sh2o2,
                sh2o3,
                smcwlt,
                smcref,
                zsoil0,
                zsoil1,
                zsoil2,
                zsoil3,
                rsmin,
                physcons.RSMAX,
                physcons.TOPT,
                rgl,
                hs,
                xlai,
                zroot,
            )

def set_2d_fields(
    sh2o: FloatField,
    zsoil: FloatField,
    sh2o0: FloatFieldIJ,
    sh2o1: FloatFieldIJ,
    sh2o2: FloatFieldIJ,
    sh2o3: FloatFieldIJ,
    zsoil0: FloatFieldIJ,
    zsoil1: FloatFieldIJ,
    zsoil2: FloatFieldIJ,
    zsoil3: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        sh2o0 = sh2o[0, 0, 0]
        sh2o1 = sh2o[0, 0, 1]
        sh2o2 = sh2o[0, 0, 2]
        sh2o3 = sh2o[0, 0, 3]
        zsoil0 = zsoil[0, 0, 0]
        zsoil1 = zsoil[0, 0, 1]
        zsoil2 = zsoil[0, 0, 2]
        zsoil3 = zsoil[0, 0, 3]

def set_3d_fields(
    sh2o: FloatField,
    zsoil: FloatField,
    sh2o0: FloatFieldIJ,
    sh2o1: FloatFieldIJ,
    sh2o2: FloatFieldIJ,
    sh2o3: FloatFieldIJ,
    zsoil0: FloatFieldIJ,
    zsoil1: FloatFieldIJ,
    zsoil2: FloatFieldIJ,
    zsoil3: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        sh2o[0, 0, 0] = sh2o0
        zsoil[0, 0, 0] = zsoil0
    with computation(FORWARD), interval(1, 2):
        sh2o[0, 0, 0] = sh2o1
        zsoil[0, 0, 0] = zsoil1
    with computation(FORWARD), interval(2, 3):
        sh2o[0, 0, 0] = sh2o2
        zsoil[0, 0, 0] = zsoil2
    with computation(FORWARD), interval(3, 4):
        sh2o[0, 0, 0] = sh2o3
        zsoil[0, 0, 0] = zsoil3

class Canres2D:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._im = grid_indexing.iec - grid_indexing.isc
        self._jm = grid_indexing.jec - grid_indexing.jsc

        domain = grid_indexing.domain
        domain2D = (domain[0], domain[1], 1)

        def make_quantity_2d() -> Quantity:
            return quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Float,
            )
        self._sh2o0 = make_quantity_2d()
        self._sh2o1 = make_quantity_2d()
        self._sh2o2 = make_quantity_2d()
        self._sh2o3 = make_quantity_2d()
        self._zsoil0 = make_quantity_2d()
        self._zsoil1 = make_quantity_2d()
        self._zsoil2 = make_quantity_2d()
        self._zsoil3 = make_quantity_2d()
        self._zroot = make_quantity_2d()

        self._canres = stencil_factory.from_origin_domain(
            func=canres_stencil,
            origin=grid_indexing.origin_compute(),
            domain=domain2D,
        )

        self._set_2d_fields = stencil_factory.from_origin_domain(
            func=set_2d_fields,
            origin=grid_indexing.origin_compute(),
            domain=domain2D,
        )

        self._set_3d_fields = stencil_factory.from_origin_domain(
            func=set_3d_fields,
            origin=grid_indexing.origin_compute(),
            domain=domain,
        )
        pass

    def __call__(
        self,
        nroot,
        swdn,
        ch,
        q2,
        q2sat,
        dqsdt2,
        sfctmp,
        sfcprs,
        sfcems,
        sh2o,
        smcwlt,
        smcref,
        zsoil,
        rsmin,
        rsmax,
        topt,
        rgl,
        hs,
        xlai,
        rc,
        pc,
        rcs,
        rct,
        rcq,
        rcsoil,
        lsm_mask,
    ):
        for i in range(self._im):
            for j in range(self._jm):
                self._zroot.view[i, j] = zsoil[i, j, nroot[i, j]]
        
        self._set_2d_fields(
            sh2o,
            zsoil,
            self._sh2o0,
            self._sh2o1,
            self._sh2o2,
            self._sh2o3,
            self._zsoil0,
            self._zsoil1,
            self._zsoil2,
            self._zsoil3,
        )

        self._canres(
            nroot,
            swdn,
            ch,
            q2,
            q2sat,
            dqsdt2,
            sfctmp,
            sfcprs,
            sfcems,
            self._sh2o0,
            self._sh2o1,
            self._sh2o2,
            self._sh2o3,
            smcwlt,
            smcref,
            self._zsoil0,
            self._zsoil1,
            self._zsoil2,
            self._zsoil3,
            rsmin,
            rgl,
            hs,
            xlai,
            self._zroot,
            rc,
            pc,
            rcs,
            rct,
            rcq,
            rcsoil,
            lsm_mask,
        )

        self._set_3d_fields(
            sh2o,
            zsoil,
            self._sh2o0,
            self._sh2o1,
            self._sh2o2,
            self._sh2o3,
            self._zsoil0,
            self._zsoil1,
            self._zsoil2,
            self._zsoil3,
        )

class Canres:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        nsoil,
    ):
        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )
        self._zroot = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(nsoil):
            self._k_mask.data[:, :, k] = k

        grid_indexing = stencil_factory.grid_indexing
        self._canres = stencil_factory.from_origin_domain(
            func=canres,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        nroot,
        swdn,
        ch,
        q2,
        q2sat,
        dqsdt2,
        sfctmp,
        sfcprs,
        sfcems,
        sh2o,
        smcwlt,
        smcref,
        zsoil,
        rsmin,
        rgl,
        hs,
        xlai,
        shdfac,
        rc,
        pc,
        rcs,
        rct,
        rcq,
        rcsoil,
        lsm_mask,
    ):
        for i in range():
            for j in range():
                self._zroot.view[i, j] = zsoil[i, j, nroot[i, j]]

        self._canres(
            nroot,
            swdn,
            ch,
            q2,
            q2sat,
            dqsdt2,
            sfctmp,
            sfcprs,
            sfcems,
            sh2o,
            smcwlt,
            smcref,
            zsoil,
            self._zroot,
            rsmin,
            rgl,
            hs,
            xlai,
            self._k_mask,
            shdfac,
            rc,
            pc,
            rcs,
            rct,
            rcq,
            rcsoil,
            lsm_mask,
        )


class TranslateCanres3D(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        grid_domain = (
            stencil_factory.grid_indexing.domain[0],
            stencil_factory.grid_indexing.domain[1],
            int(namelist.lsoil),
        )
        surface_grid_index = GridIndexing(
            grid_domain,
            stencil_factory.grid_indexing.n_halo,
            stencil_factory.grid_indexing.south_edge,
            stencil_factory.grid_indexing.north_edge,
            stencil_factory.grid_indexing.west_edge,
            stencil_factory.grid_indexing.east_edge,
        )
        surface_factory = StencilFactory(
            stencil_factory.config,
            surface_grid_index,
            stencil_factory.comm,
        )
        super().__init__(grid, namelist, surface_factory)
        self.stencil_factory = surface_factory
        self.in_vars["data_vars"] = {
            "nroot": {"shield": True},
            "swdn": {"shield": True},
            "ch": {"shield": True},
            "q2": {"shield": True},
            "q2sat": {"shield": True},
            "dqsdt2": {"shield": True},
            "sfctmp": {"shield": True},
            "sfcprs": {"shield": True},
            "sfcems": {"shield": True},
            "sh2o": {"shield": True},
            "smcwlt": {"shield": True},
            "smcref": {"shield": True},
            "shdfac": {"shield": True},
            "zsoil": {"shield": True},
            "rsmin": {"shield": True},
            "rsmax": {"shield": True},
            "topt": {"shield": True},
            "rgl": {"shield": True},
            "hs": {"shield": True},
            "xlai": {"shield": True},
            "rc": {"shield": True},
            "pc": {"shield": True},
            "rcs": {"shield": True},
            "rct": {"shield": True},
            "rcq": {"shield": True},
            "rcsoil": {"shield": True},
            "lsm_mask": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "nsoil",
        ]
        self.out_vars = {
            "rc": {"shield": True},
            "pc": {"shield": True},
            "rcs": {"shield": True},
            "rct": {"shield": True},
            "rcq": {"shield": True},
            "rcsoil": {"shield": True},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs.pop("nsoil"),
        inputs.pop("rsmax"),
        inputs.pop("topt"),
        self.compute_func = Canres(
            self.stencil_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class Translate2dCanres(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        grid_domain = (
            stencil_factory.grid_indexing.domain[0],
            stencil_factory.grid_indexing.domain[1],
            int(namelist.lsoil),
        )
        surface_grid_index = GridIndexing(
            grid_domain,
            stencil_factory.grid_indexing.n_halo,
            stencil_factory.grid_indexing.south_edge,
            stencil_factory.grid_indexing.north_edge,
            stencil_factory.grid_indexing.west_edge,
            stencil_factory.grid_indexing.east_edge,
        )
        surface_factory = StencilFactory(
            stencil_factory.config,
            surface_grid_index,
            stencil_factory.comm,
        )
        super().__init__(grid, namelist, surface_factory)
        self.stencil_factory = surface_factory
        self.in_vars["data_vars"] = {
            "nroot": {"shield": True},
            "swdn": {"shield": True},
            "ch": {"shield": True},
            "q2": {"shield": True},
            "q2sat": {"shield": True},
            "dqsdt2": {"shield": True},
            "sfctmp": {"shield": True},
            "sfcprs": {"shield": True},
            "sfcems": {"shield": True},
            "sh2o": {"shield": True},
            "smcwlt": {"shield": True},
            "smcref": {"shield": True},
            "shdfac": {"shield": True},
            "zsoil": {"shield": True},
            "rsmin": {"shield": True},
            "rsmax": {"shield": True},
            "topt": {"shield": True},
            "rgl": {"shield": True},
            "hs": {"shield": True},
            "xlai": {"shield": True},
            "rc": {"shield": True},
            "pc": {"shield": True},
            "rcs": {"shield": True},
            "rct": {"shield": True},
            "rcq": {"shield": True},
            "rcsoil": {"shield": True},
            "lsm_mask": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "nsoil",
        ]
        self.out_vars = {
            "rc": {"shield": True},
            "pc": {"shield": True},
            "rcs": {"shield": True},
            "rct": {"shield": True},
            "rcq": {"shield": True},
            "rcsoil": {"shield": True},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.lsoil,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )
        inputs.pop("nsoil")

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        self.compute_func = Canres2D(
            self.stencil_factory,
            quantity_factory
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateCanres1(TranslateCanres3D):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)


class TranslateCanres2(TranslateCanres1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
