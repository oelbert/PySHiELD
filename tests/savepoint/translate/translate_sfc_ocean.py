from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_ocean import SurfaceOcean
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TranslateSurfaceOcean_iter1(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ps": {"serialname": "ocean_ps", "shield": True},
            "u1": {"serialname": "ocean_u1", "shield": True},
            "v1": {"serialname": "ocean_v1", "shield": True},
            "t1": {"serialname": "ocean_t1", "shield": True},
            "q1": {"serialname": "ocean_q1", "shield": True},
            "tskin": {"serialname": "ocean_tskin", "shield": True},
            "cm": {"serialname": "ocean_cm", "shield": True},
            "ch": {"serialname": "ocean_ch", "shield": True},
            "prsl1": {"serialname": "ocean_prsl1", "shield": True},
            "prslki": {"serialname": "ocean_prslki", "shield": True},
            "qsurf": {"serialname": "ocean_qsurf", "shield": True},
            "cmm": {"serialname": "ocean_cmm", "shield": True},
            "chh": {"serialname": "ocean_chh", "shield": True},
            "gflux": {"serialname": "ocean_gflux", "shield": True},
            "evap": {"serialname": "ocean_evap", "shield": True},
            "hflx": {"serialname": "ocean_hflx", "shield": True},
            "ep": {"serialname": "ocean_ep", "shield": True},
            "islimsk": {"serialname": "ocean_islmsk", "shield": True},
            "flag_iter": {"serialname": "ocean_flag_iter", "shield": True},
        }
        self.in_vars["parameters"] = [
            "ocean_ddvel",
        ]
        self.out_vars = {
            "qsurf": {"serialname": "ocean_qsurf", "shield": True},
            "cmm": {"serialname": "ocean_cmm", "shield": True},
            "chh": {"serialname": "ocean_chh", "shield": True},
            "gflux": {"serialname": "ocean_gflux", "shield": True},
            "evap": {"serialname": "ocean_evap", "shield": True},
            "hflx": {"serialname": "ocean_hflx", "shield": True},
            "ep": {"serialname": "ocean_ep", "shield": True},
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.compute_func = SurfaceOcean(
            self.stencil_factory,
        )
        self.make_storage_data_input_vars(inputs)
        inputs["ddvel"] = inputs.pop("ocean_ddvel")
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateSurfaceOcean_iter2(TranslateSurfaceOcean_iter1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
