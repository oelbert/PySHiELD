from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_diff import SurfaceExchange
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSurfaceExchange_iter1(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u1": {"shield": True},
            "v1": {"shield": True},
            "t1": {"shield": True},
            "q1": {"shield": True},
            "ddvel": {"shield": True},
            "tsurf": {"shield": True},
            "tsfc": {"shield": True},
            "prslki": {"shield": True},
            "prsl1": {"shield": True},
            "z0rl": {"shield": True},
            "z1": {"shield": True},
            "shdmax": {"shield": True},
            "sigmaf": {"shield": True},
            "ustar": {"shield": True},
            "snowdepth": {"shield": True},
            "ztrl": {"shield": True},
            "cm": {"shield": True},
            "ch": {"shield": True},
            "rb": {"shield": True},
            "stress": {"shield": True},
            "fm": {"shield": True},
            "fh": {"shield": True},
            "wind": {"shield": True},
            "fm10": {"shield": True},
            "fh2": {"shield": True},
            "islimsk": {"shield": True},
            "vegtype": {"shield": True},
            "flag_iter": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "ivegsrc",
            "do_z0_hwrf15",
            "do_z0_hwrf17",
            "do_z0_hwrf17_hwonly",
            "do_z0_moon",
            "redrag",
            "wind_th_hwrf",
            "z0s_max",
        ]
        self.out_vars = {
            "wind": {"shield": True},
            "z0rl": {"shield": True},
            "ztrl": {"shield": True},
            "cm": {"shield": True},
            "ch": {"shield": True},
            "stress": {"shield": True},
            "fm": {"shield": True},
            "fh": {"shield": True},
            "ustar": {"shield": True},
            "fm10": {"shield": True},
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = SurfaceExchange(
            self.stencil_factory,
            inputs.pop("ivegsrc"),
            inputs.pop("do_z0_hwrf15"),
            inputs.pop("do_z0_hwrf17"),
            inputs.pop("do_z0_hwrf17_hwonly"),
            inputs.pop("do_z0_moon"),
            inputs.pop("redrag"),
            inputs.pop("wind_th_hwrf"),
            inputs.pop("z0s_max"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateSurfaceExchange_iter2(TranslateSurfaceExchange_iter1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
