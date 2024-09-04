from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_sice import SurfaceSeaIce
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSurfaceSeaIce_iter1(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u1": {"serialname": "sice_u1", "shield": True},
            "v1": {"serialname": "sice_v1", "shield": True},
            "t1": {"serialname": "sice_t1", "shield": True},
            "ps": {"serialname": "sice_ps", "shield": True},
            "wind": {"serialname": "sice_wind", "shield": True},
            "q1": {"serialname": "sice_q1", "shield": True},
            "sfcemis": {"serialname": "sfcemis", "shield": True},
            "dlwflx": {"serialname": "sice_dlwflx", "shield": True},
            "sfcnsw": {"serialname": "sice_sfcnsw", "shield": True},
            "sfcdsw": {"serialname": "sice_sfcdsw", "shield": True},
            "srflag": {"serialname": "sice_srflag", "shield": True},
            "cm": {"serialname": "sice_cm", "shield": True},
            "ch": {"serialname": "sice_ch", "shield": True},
            "prsl1": {"serialname": "sice_prsl1", "shield": True},
            "prslki": {"serialname": "sice_prslki", "shield": True},
            "islimsk": {"serialname": "sice_islmsk", "shield": True},
            "flag_iter": {"serialname": "sice_flag_iter", "shield": True},
            "hice": {"serialname": "sice_hice", "shield": True},
            "fice": {"serialname": "sice_fice", "shield": True},
            "tice": {"serialname": "sice_tice", "shield": True},
            "weasd": {"serialname": "sice_weasd", "shield": True},
            "tskin": {"serialname": "sice_tskin", "shield": True},
            "tprcp": {"serialname": "sice_tprcp", "shield": True},
            "stc0": {"serialname": "sice_stc0", "shield": True},
            "stc1": {"serialname": "sice_stc1", "shield": True},
            "ep": {"serialname": "sice_ep", "shield": True},
            "snwdph": {"serialname": "sice_snowd", "shield": True},
            "qsurf": {"serialname": "sice_qsurf", "shield": True},
            "cmm": {"serialname": "sice_cmm", "shield": True},
            "chh": {"serialname": "sice_chh", "shield": True},
            "evap": {"serialname": "sice_evap", "shield": True},
            "hflx": {"serialname": "sice_hflx", "shield": True},
            "gflux": {"serialname": "sice_gflux", "shield": True},
            "snowmt": {"serialname": "sice_snowmt", "shield": True},
        }
        self.in_vars["parameters"] = [
            "sice_delt",
            "sice_mom4ice",
            "sice_lsm",
        ]
        self.out_vars = {
            "hice": {"serialname": "sice_hice", "shield": True},
            "fice": {"serialname": "sice_fice", "shield": True},
            "tice": {"serialname": "sice_tice", "shield": True},
            "weasd": {"serialname": "sice_weasd", "shield": True},
            "tskin": {"serialname": "sice_tskin", "shield": True},
            "tprcp": {"serialname": "sice_tprcp", "shield": True},
            "stc0": {"serialname": "sice_stc0", "shield": True},
            "stc1": {"serialname": "sice_stc1", "shield": True},
            "ep": {"serialname": "sice_ep", "shield": True},
            "snwdph": {"serialname": "sice_snowd", "shield": True},
            "qsurf": {"serialname": "sice_qsurf", "shield": True},
            "snowmt": {"serialname": "sice_snowmt", "shield": True},
            "gflux": {"serialname": "sice_gflux", "shield": True},
            "cmm": {"serialname": "sice_cmm", "shield": True},
            "chh": {"serialname": "sice_chh", "shield": True},
            "evap": {"serialname": "sice_evap", "shield": True},
            "hflx": {"serialname": "sice_hflx", "shield": True},
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.compute_func = SurfaceSeaIce(
            self.stencil_factory,
            mom4ice=self.inputs.pop("sice_mom4ice"),
            lsm=self.inputs.pop("sice_lsm"),
            dt_atmos=self.inputs.pop("sice_delt"),
        )
        self.make_storage_data_input_vars(inputs)
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateSurfaceSeaIce_iter2(TranslateSurfaceSeaIce_iter1):
    def __init__(
        self,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(namelist, stencil_factory)
