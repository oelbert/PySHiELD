from ndsl import Namelist, QuantityFactory, StencilFactory
from ndsl.dsl.stencil import GridIndexing
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD._config import LSMConfig
from pySHiELD.stencils.surface.noah_lsm.lsm_driver import NoahLSM
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateNoahLSM_iter1(TranslatePhysicsFortranData2Py):
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
        self.in_vars["data_vars"] = {
            "ps": {"shield": True},
            "t1": {"shield": True},
            "q1": {"shield": True},
            "soil_data": {"serialname": "soiltyp", "shield": True},
            "veg_data": {"serialname": "vegtype", "shield": True},
            "vegfrac_data": {"serialname": "vegfpert", "shield": True},
            "sfcemis": {"shield": True},
            "dlwflx": {"shield": True},
            "dswflx": {"serialname": "dswsfc", "shield": True},
            "snet": {"shield": True},
            "tg3": {"shield": True},
            "cm": {"shield": True},
            "ch": {"shield": True},
            "prsl1": {"shield": True},
            "prslki": {"shield": True},
            "zf": {"shield": True},
            "land_data": {"serialname": "land", "shield": True},
            "wind": {"shield": True},
            "slope_data": {"serialname": "slopetyp", "shield": True},
            "shdmin": {"shield": True},
            "shdmax": {"shield": True},
            "snoalb": {"shield": True},
            "sfalb": {"shield": True},
            "flag_iter": {"shield": True},
            "flag_guess": {"shield": True},
            "bexppert": {"shield": True},
            "xlaipert": {"shield": True},
            "weasd": {"shield": True},
            "snwdph": {"shield": True},
            "tskin": {"shield": True},
            "tprcp": {"shield": True},
            "srflag": {"shield": True},
            "smc": {"shield": True, "kend": namelist.lsoil},
            "stc": {"shield": True, "kend": namelist.lsoil},
            "slc": {"shield": True, "kend": namelist.lsoil},
            "canopy": {"shield": True},
            "trans": {"shield": True},
            "tsurf": {"shield": True},
            "z0rl": {"shield": True, "serialname": "zorl"},
            "sncovr1": {"shield": True},
            "qsurf": {"shield": True},
            "gflux": {"shield": True},
            "drain": {"shield": True},
            "evap": {"shield": True},
            "hflx": {"shield": True},
            "ep": {"shield": True},
            "runoff": {"shield": True},
            "cmm": {"shield": True},
            "chh": {"shield": True},
            "evbs": {"shield": True},
            "evcw": {"shield": True},
            "sbsno": {"shield": True},
            "snowc": {"shield": True},
            "stm": {"shield": True},
            "snohf": {"shield": True},
            "smcwlt2": {"shield": True},
            "smcref2": {"shield": True},
            "wet1": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "ivegsrc",
            "km",
            "lheatstrg",
            "pertvegf",
            "isot",
            "delt",
        ]
        self.out_vars = {
            "weasd": {"shield": True},
            "snwdph": {"shield": True},
            "tskin": {"shield": True},
            "tprcp": {"shield": True},
            "srflag": {"shield": True},
            "smc": {"shield": True, "kend": namelist.lsoil},
            "stc": {"shield": True, "kend": namelist.lsoil},
            "slc": {"shield": True, "kend": namelist.lsoil},
            "canopy": {"shield": True},
            "trans": {"shield": True},
            "tsurf": {"shield": True},
            "z0rl": {"shield": True, "serialname": "zorl"},
            "sncovr1": {"shield": True},
            "qsurf": {"shield": True},
            "gflux": {"shield": True},
            "drain": {"shield": True},
            "evap": {"shield": True},
            "hflx": {"shield": True},
            "ep": {"shield": True},
            "runoff": {"shield": True},
            "cmm": {"shield": True},
            "chh": {"shield": True},
            "evbs": {"shield": True},
            "evcw": {"shield": True},
            "sbsno": {"shield": True},
            "snowc": {"shield": True},
            "stm": {"shield": True},
            "snohf": {"shield": True},
            "smcwlt2": {"shield": True},
            "smcref2": {"shield": True},
            "wet1": {"shield": True},
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
        inputs.pop("shdmin")
        inputs.pop("shdmax")
        config = LSMConfig(
            lsoil=inputs.pop("km"),
            isot=inputs.pop("isot"),
            ivegsrc=inputs.pop("ivegsrc"),
            lheatstrg=inputs.pop("lheatstrg"),
            pertvegf=inputs.pop("pertvegf"),
        )
        self.compute_func = NoahLSM(
            self.stencil_factory,
            self.quantity_factory,
            config,
            inputs.pop("land_data"),
            inputs.pop("veg_data"),
            inputs.pop("soil_data"),
            inputs.pop("slope_data"),
            inputs.pop("vegfrac_data"),
            inputs.pop("delt"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateNoahLSM_iter2(TranslateNoahLSM_iter1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
