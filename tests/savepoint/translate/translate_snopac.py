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
from pySHiELD.stencils.surface.noah_lsm.snopac import SNOPAC
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

def invert_bool(
    bool_in: BoolFieldIJ,
    bool_out: BoolFieldIJ,
):
    with computation(FORWARD), interval(...):
        if bool_in:
            bool_out = False
        else:
            bool_out = True

class SnopacTest:
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

        self._snopack_mask = quantity_factory.zeros(
                [X_DIM, Y_DIM],
                units="unknown",
                dtype=Bool,
            )

        self._k_mask = quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Int,
            )

        for k in range(nsoil):
            self._k_mask.data[:, :, k] = k
            
        self._invert = stencil_factory.from_origin_domain(
            func=invert_bool,
            origin=grid_indexing.origin_compute(),
            domain=domain_2d,
        )
        
        self._snopac = SNOPAC(
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
        df1,
        sfcems,
        sfctmp,
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
        dwsat,
        dksat,
        shdfac,
        ice,
        rtdis,
        quartz,
        flx2,
        snowng,
        ffrozp,
        vegtype,
        prcp1,
        cmc,
        t1,
        stc,
        sncovr,
        sneqv,
        sndens,
        snowh,
        sh2o,
        tbot,
        smc,
        ssoil,
        runoff1,
        runoff2,
        runoff3,
        edir,
        ec,
        et,
        ett,
        snomlt,
        drip,
        dew,
        flx1,
        flx3,
        esnow,
        nopac_mask,
    ):
        self._invert(nopac_mask, self._snopack_mask)
        self._snopac(
            nroot,
            etp,
            prcp,
            smcmax,
            smcwlt,
            smcref,
            smcdry,
            df1,
            sfcems,
            sfctmp,
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
            dwsat,
            dksat,
            shdfac,
            ice,
            rtdis,
            quartz,
            flx2,
            snowng,
            ffrozp,
            vegtype,
            prcp1,
            cmc,
            t1,
            stc,
            sncovr,
            sneqv,
            sndens,
            snowh,
            sh2o,
            tbot,
            smc,
            ssoil,
            runoff1,
            runoff2,
            runoff3,
            edir,
            ec,
            et,
            ett,
            snomlt,
            drip,
            dew,
            flx1,
            flx3,
            esnow,
            self._snopack_mask,
            self._k_mask,
        )

class TranslateSnopack1(TranslatePhysicsFortranData2Py):
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
            "snowng": {"shield": True},
            "nroot": {"shield": True},
            "ice": {"shield": True},
            "etp": {"shield": True},
            "prcp": {"shield": True},
            "smcmax": {"shield": True},
            "smcwlt": {"shield": True},
            "smcref": {"shield": True},
            "smcdry": {"shield": True},
            "cmcmax": {"shield": True},
            "df1": {"shield": True},
            "shdfac": {"shield": True},
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
            "flx2": {"shield": True},
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
            "prcp1": {"shield": True},
            "sncovr": {"shield": True},
            "sneqv": {"shield": True},
            "sndens": {"shield": True},
            "snowh": {"shield": True},
            "zsoil": {"shield": True},
            "rtdis": {"shield": True},
            "stc": {"shield": True},
            "sh2o": {"shield": True},
            "smc": {"shield": True},
            "et": {"shield": True},
            "snomlt": {"shield": True},
            "esnow": {"shield": True},
            "ffrozp": {"shield": True},
            "vegtype": {"shield": True, "index_varible": True},
        }
        self.in_vars["parameters"] = [
            "nsoil",
            "dt",
            "lheatstrg",
        ]
        self.out_vars = {
            "prcp1": {"shield": True},
            "cmc": {"shield": True},
            "t1": {"shield": True},
            "stc": {"shield": True},
            "sncovr": {"shield": True},
            "sneqv": {"shield": True},
            "sndens": {"shield": True},
            "snowh": {"shield": True},
            "sh2o": {"shield": True},
            "tbot": {"shield": True},
            "beta": {"shield": True},
            "smc": {"shield": True},
            "ssoil": {"shield": True},
            "runoff1": {"shield": True},
            "runoff2": {"shield": True},
            "runoff3": {"shield": True},
            "edir": {"shield": True},
            "ec": {"shield": True},
            "et": {"shield": True},
            "ett": {"shield": True},
            "snomlt": {"shield": True},
            "drip": {"shield": True},
            "dew": {"shield": True},
            "flx1": {"shield": True},
            "flx3": {"shield": True},
            "esnow": {"shield": True},
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
        self.compute_func = SnopacTest(
            self.stencil_factory,
            self.quantity_factory,
            inputs.pop("nsoil"),
            self.namelist.ivegsrc,
            inputs.pop("lheatstrg"),
            inputs.pop("dt"),
        )
        inputs.pop("beta")
        self.compute_func(**inputs)
        return self.slice_output(inputs)

class TranslateSnopack2(TranslateSnopack1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
