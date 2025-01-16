from gt4py.cartesian.gtscript import FORWARD, computation, interval

import pySHiELD.constants as physcons
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntFieldIJ,
)
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from ndsl import Namelist, StencilFactory
from ndsl.dsl.stencil import GridIndexing
from pySHiELD.stencils.surface.noah_lsm.nopac import NOPAC
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class NopacTest:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        nsoil: Int,
        ivegsrc: Int,
        lheatstrg: Bool,
        dt: Float,
    ):

        grid_indexing = stencil_factory.grid_indexing

        domain = grid_indexing.domain
        domain_2d = (domain[0], domain[1], 1)

        self._k_mask = quantity_factory.zeros(
            [X_DIM, Y_DIM, Z_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(nsoil):
            self._k_mask.data[:, :, k] = k

        self._nopac = NOPAC(
            stencil_factory,
            quantity_factory,
            ivegsrc,
            lheatstrg,
            dt
        )

    def __call__(
        self,
        nroot,
        etp,
        prcp,
        smcmax,
        smcwlt,
        smcref,
        smcdry,
        shdfac,
        sfctmp,
        sfcems,
        t24,
        th2,
        fdown,
        epsca,
        bexp,
        pc,
        rch,
        rr,
        slope,
        kdt,
        frzx,
        psisat,
        zsoil,
        dksat,
        dwsat,
        ice,
        rtdis,
        quartz,
        vegtype,
        cmc,
        t1,
        stc,
        sh2o,
        tbot,
        smc,
        eta,
        ssoil,
        runoff1,
        runoff2,
        runoff3,
        edir,
        ec,
        et,
        ett,
        beta,
        drip,
        dew,
        flx1,
        flx3,
        nopac_mask,
    ):
        self._nopac(
            nroot,
            etp,
            prcp,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            shdfac,
            sfctmp,
            sfcems,
            t24,
            th2,
            fdown,
            epsca,
            bexp,
            pc,
            rch,
            rr,
            slope,
            kdt,
            frzx,
            psisat,
            zsoil,
            dksat,
            dwsat,
            ice,
            rtdis,
            quartz,
            vegtype,
            cmc,
            t1,
            stc,
            sh2o,
            tbot,
            smc,
            eta,
            ssoil,
            runoff1,
            runoff2,
            runoff3,
            edir,
            ec,
            et,
            ett,
            beta,
            drip,
            dew,
            flx1,
            flx3,
            nopac_mask,
            self._k_mask,
        )

class TranslateNopack1(TranslatePhysicsFortranData2Py):
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
            "nopac_mask": {"shield": True},
            "nroot": {"shield": True},
            "ice": {"shield": True},
            "etp": {"shield": True},
            "prcp": {"shield": True},
            "smcmax": {"shield": True},
            "smcwlt": {"shield": True},
            "smcref": {"shield": True},
            "smcdry": {"shield": True},
            "cmcmax": {"shield": True},
            "shdfac": {"shield": True},
            "sbeta": {"shield": True},
            "sfctmp": {"shield": True},
            "sfcems": {"shield": True},
            "t24": {"shield": True},
            "th2": {"shield": True},
            "fdown": {"shield": True},
            "epsca": {"shield": True},
            "bexp": {"shield": True},
            "pc": {"shield": True},
            "rch": {"shield": True},
            "rr": {"shield": True},
            "cfactr": {"shield": True},
            "slope": {"shield": True},
            "kdt": {"shield": True},
            "frzx": {"shield": True},
            "psisat": {"shield": True},
            "dksat": {"shield": True},
            "dwsat": {"shield": True},
            "zbot": {"shield": True},
            "quartz": {"shield": True},
            "fxexp": {"shield": True},
            "csoil": {"shield": True},
            "cmc": {"shield": True},
            "t1": {"shield": True},
            "tbot": {"shield": True},
            "beta": {"shield": True},
            "ssoil": {"shield": True},
            "runoff1": {"shield": True},
            "runoff2": {"shield": True},
            "runoff3": {"shield": True},
            "edir": {"shield": True},
            "ec": {"shield": True},
            "ett": {"shield": True},
            "drip": {"shield": True},
            "dew": {"shield": True},
            "flx1": {"shield": True},
            "flx3": {"shield": True},
            "eta": {"shield": True},
            "zsoil": {"shield": True},
            "rtdis": {"shield": True},
            "stc": {"shield": True},
            "sh2o": {"shield": True},
            "smc": {"shield": True},
            "et": {"shield": True},
            "vegtype": {"shield": True, "index_field": True},
        }
        self.in_vars["parameters"] = [
            "nsoil",
            "dt",
            "lheatstrg",
        ]
        self.out_vars = {
            "cmc": {"shield": True},
            "t1": {"shield": True},
            "tbot": {"shield": True},
            "beta": {"shield": True},
            "et": {"shield": True},
            "stc": {"shield": True},
            "sh2o": {"shield": True},
            "eta": {"shield": True},
            "smc": {"shield": True},
            "ssoil": {"shield": True},
            "runoff1": {"shield": True},
            "runoff2": {"shield": True},
            "runoff3": {"shield": True},
            "edir": {"shield": True},
            "ec": {"shield": True},
            "ett": {"shield": True},
            "drip": {"shield": True},
            "dew": {"shield": True},
            "flx1": {"shield": True},
            "flx3": {"shield": True},
        }

        self.stencil_factory = surface_factory
        init_sizer = grid.quantity_factory.sizer
        sizer = SubtileGridSizer(
            init_sizer.nx,
            init_sizer.ny,
            namelist.lsoil,
            init_sizer.n_halo,
            init_sizer.extra_dim_lengths,
        )
        sizer.nz = namelist.lsoil
        self.quantity_factory = QuantityFactory(
            sizer,
            self.grid.quantity_factory._numpy,
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = NopacTest(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("nsoil"),
            self.namelist.ivegsrc,
            inputs.pop("lheatstrg"),
            inputs.pop("dt"),
        )
        inputs.pop("sbeta")
        inputs.pop("cmcmax")
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateNopack2(TranslateNopack1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
