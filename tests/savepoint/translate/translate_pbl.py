from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD.stencils.pbl import ScaleAwareTKEMoistEDMF
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TranslatePBL(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ntrac": {"serialname": "pbl_ntrac", "shield": True},
            "ntcw": {"serialname": "pbl_ntcw", "shield": True},
            "ntiw": {"serialname": "pbl_ntiw", "shield": True},
            "ntke": {"serialname": "pbl_ntke", "shield": True},
            "dv": {"serialname": "pbl_dv", "shield": True},
            "du": {"serialname": "pbl_du", "shield": True},
            "tdt": {"serialname": "pbl_tdt", "shield": True},
            "rtg": {"serialname": "pbl_rtg", "shield": True},
            "u1": {"serialname": "pbl_u1", "shield": True},
            "v1": {"serialname": "pbl_v1", "shield": True},
            "t1": {"serialname": "pbl_t1", "shield": True},
            "q1": {"serialname": "pbl_q1", "shield": True},
            "swh": {"serialname": "pbl_swh", "shield": True},
            "hlw": {"serialname": "pbl_hlw", "shield": True},
            "xmu": {"serialname": "pbl_xmu", "shield": True},
            "garea": {"serialname": "pbl_garea", "shield": True},
            "islmsk": {"serialname": "pbl_islmsk", "shield": True},
            "psk": {"serialname": "pbl_psk", "shield": True},
            "rbsoil": {"serialname": "pbl_rbsoil", "shield": True},
            "zorl": {"serialname": "pbl_zorl", "shield": True},
            "u10m": {"serialname": "pbl_u10m", "shield": True},
            "v10m": {"serialname": "pbl_v10m", "shield": True},
            "fm": {"serialname": "pbl_fm", "shield": True},
            "fh": {"serialname": "pbl_fh", "shield": True},
            "tsea": {"serialname": "pbl_tsea", "shield": True},
            "heat": {"serialname": "pbl_heat", "shield": True},
            "evap": {"serialname": "pbl_evap", "shield": True},
            "stress": {"serialname": "pbl_stress", "shield": True},
            "wind": {"serialname": "pbl_wind", "shield": True},
            "kpbl": {"serialname": "pbl_kpbl", "shield": True},
            "prsi": {"serialname": "pbl_prsi", "shield": True},
            "delta": {"serialname": "pbl_delta", "shield": True},
            "prsl": {"serialname": "pbl_prsl", "shield": True},
            "prslk": {"serialname": "pbl_prslk", "shield": True},
            "phii": {"serialname": "pbl_phii", "shield": True},
            "phil": {"serialname": "pbl_phil", "shield": True},
            "dtp": {"serialname": "pbl_dtp", "shield": True},
            "dspheat": {"serialname": "pbl_dspheat", "shield": True},
            "dusfc": {"serialname": "pbl_dusfc", "shield": True},
            "dvsfc": {"serialname": "pbl_dvsfc", "shield": True},
            "dtsfc": {"serialname": "pbl_dtsfc", "shield": True},
            "dqsfc": {"serialname": "pbl_dqsfc", "shield": True},
            "hpbl": {"serialname": "pbl_hpbl", "shield": True},
            "kinver": {"serialname": "pbl_kinver", "shield": True},
            "xkzm_m": {"serialname": "pbl_xkzm_m", "shield": True},
            "xkzm_h": {"serialname": "pbl_xkzm_h", "shield": True},
            "xkzm_ml": {"serialname": "pbl_xkzm_ml", "shield": True},
            "xkzm_hl": {"serialname": "pbl_xkzm_hl", "shield": True},
            "xkzm_mi": {"serialname": "pbl_xkzm_mi", "shield": True},
            "xkzm_hi": {"serialname": "pbl_xkzm_hi", "shield": True},
            "xkzm_s": {"serialname": "pbl_xkzm_s", "shield": True},
            "xkzminv": {"serialname": "pbl_xkzminv", "shield": True},
            "do_dk_hb19": {"serialname": "pbl_do_dk_hb19", "shield": True},
            "xkzm_lim": {"serialname": "pbl_xkzm_lim", "shield": True},
            "xkgdx": {"serialname": "pbl_xkgdx", "shield": True},
            "rlmn": {"serialname": "pbl_rlmn", "shield": True},
            "rlmx": {"serialname": "pbl_rlmx", "shield": True},
            "dkt": {"serialname": "pbl_dkt", "shield": True},
            "cap_k0_land": {"serialname": "pbl_cap_k0_land", "shield": True},
        }

        self.out_vars = {
            "du": {"serialname": "pbl_du", "shield": True},
            "dv": {"serialname": "pbl_dv", "shield": True},
            "tdt": {"serialname": "pbl_tdt", "shield": True},
            "rtg": {"serialname": "pbl_rtg", "shield": True},
            "kpbl": {"serialname": "pbl_kpbl", "shield": True},
            "dusfc": {"serialname": "pbl_dusfc", "shield": True},
            "dvsfc": {"serialname": "pbl_dvsfc", "shield": True},
            "dtsfc": {"serialname": "pbl_dtsfc", "shield": True},
            "dqsfc": {"serialname": "pbl_dqsfc", "shield": True},
            "hpbl": {"serialname": "pbl_hpbl", "shield": True},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npx - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )

        compute_func = ScaleAwareTKEMoistEDMF(
            self.stencil_factory,
            quantity_factory,
            self.grid.grid_data.area,
            self.namelist.pbl,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
