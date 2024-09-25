from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.noah_lsm.lsm_driver import NoahLSM
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py
from pySHiELD._config import LSMConfig


class TranslateNoahLSM_iter1(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ps": {"shield": True},
            "t1": {"shield": True},
            "q1": {"shield": True},
            "sfcemis": {"shield": True},
            "dlwflx": {"shield": True},
            "dswflx": {"shield": True},
            "snet": {"shield": True},
            "tg3": {"shield": True},
            "cm": {"shield": True},
            "ch": {"shield": True},
            "prsl1": {"shield": True},
            "prslki": {"shield": True},
            "zf": {"shield": True},
            "wind": {"shield": True},
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
            "smc": {"shield": True},
            "stc": {"shield": True},
            "slc": {"shield": True},
            "canopy": {"shield": True},
            "trans": {"shield": True},
            "tsurf": {"shield": True},
            "zorl": {"shield": True},
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
        ]
        self.out_vars = {
            "weasd": {"shield": True},
            "snwdph": {"shield": True},
            "tskin": {"shield": True},
            "tprcp": {"shield": True},
            "srflag": {"shield": True},
            "smc": {"shield": True},
            "stc": {"shield": True},
            "slc": {"shield": True},
            "canopy": {"shield": True},
            "trans": {"shield": True},
            "tsurf": {"shield": True},
            "zorl": {"shield": True},
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
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        config = LSMConfig(
            lsoil=inputs.pop("lsoil"),
            isot=inputs.pop("isot"),
            ivegsrc=inputs.pop("ivegsrc"),
            lheatstrg=inputs.pop("lheatstrg"),
            pertvegf=inputs.pop("pertvegf"),
        )
        self.compute_func = NoahLSM(
            self.stencil_factory,
            self.grid.quantity_factory,
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
