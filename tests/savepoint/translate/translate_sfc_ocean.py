from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_ocean import SurfaceOcean
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

!$ser data ocean_cmm=Diag%cmm ocean_chh=Diag%chh ocean_gflux=gflx ocean_evap=evap ocean_hflx=hflx
class TranslateSurfaceOcean(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ps": {"serialname": "ocean_ps", "SHiELD": True},
            "u1": {"serialname": "ocean_u1", "SHiELD": True},
            "v1": {"serialname": "ocean_v1", "SHiELD": True},
            "t1": {"serialname": "ocean_t1", "SHiELD": True},
            "q1": {"serialname": "ocean_q1", "SHiELD": True},
            "tskin": {"serialname": "ocean_tskin", "SHiELD": True},
            "cm": {"serialname": "ocean_cm", "SHiELD": True},
            "ch": {"serialname": "ocean_ch", "SHiELD": True},
            "prsl1": {"serialname": "ocean_prsl1", "SHiELD": True},
            "prslki": {"serialname": "ocean_prslki", "SHiELD": True},
            "ddvel": {"serialname": "ocean_ddvel", "SHiELD": True},
            "qsurf": {"serialname": "ocean_qsurf", "SHiELD": True},
            "cmm": {"serialname": "ocean_cmm", "SHiELD": True},
            "chh": {"serialname": "ocean_chh", "SHiELD": True},
            "gflux": {"serialname": "ocean_gflux", "SHiELD": True},
            "evap": {"serialname": "ocean_evap", "SHiELD": True},
            "hflx": {"serialname": "ocean_hflx", "SHiELD": True},
            "ep": {"serialname": "ocean_ep", "SHiELD": True},
            "islimsk": {"serialname": "ocean_islmsk", "SHiELD": True},
            "flag_iter": {"serialname": "ocean_flag_iter", "SHiELD": True},
        }
        self.out_vars = {
            "qsurf": {"serialname": "ocean_qsurf", "SHiELD": True},
            "cmm": {"serialname": "ocean_cmm", "SHiELD": True},
            "chh": {"serialname": "ocean_chh", "SHiELD": True},
            "gflux": {"serialname": "ocean_gflux", "SHiELD": True},
            "evap": {"serialname": "ocean_evap", "SHiELD": True},
            "hflx": {"serialname": "ocean_hflx", "SHiELD": True},
            "ep": {"serialname": "ocean_ep", "SHiELD": True},
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.compute_func = SurfaceOcean(
            self.stencil_factory,
        )
        self.make_storage_data_input_vars(inputs)
        inputs["gq0"] = inputs["gq0"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
