from ndsl import Namelist, QuantityFactory, StencilFactory
from ndsl.dsl.stencil import GridIndexing
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD._config import LSMConfig
from pySHiELD.stencils.surface.noah_lsm.sfc_params import set_soil_veg
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class SoilVeg:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: LSMConfig,
    ):
        pass

    def __call__(
        self,
    ):
        pass


class TranslateSoilVeg(TranslatePhysicsFortranData2Py):
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
        self.compute_func = SoilVeg(
            self.stencil_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)