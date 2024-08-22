from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD.stencils.pbl import ScaleAwareTKEMoistEDMF
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TranslatePBL(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ntrac": {"serialname": "pbl_ntrac", "mp3": True},
            "ntcw": {"serialname": "pbl_ntcw", "mp3": True},
            "ntiw": {"serialname": "pbl_ntiw", "mp3": True},
            "ntke": {"serialname": "pbl_ntke", "mp3": True},
            "dv": {"serialname": "pbl_dv", "mp3": True},
            "du": {"serialname": "pbl_du", "mp3": True},
            "tdt": {"serialname": "pbl_tdt", "mp3": True},
            "rtg": {"serialname": "pbl_rtg", "mp3": True},
            "u1": {"serialname": "pbl_u1", "mp3": True},
            "v1": {"serialname": "pbl_v1", "mp3": True},
            "t1": {"serialname": "pbl_t1", "mp3": True},
            "q1": {"serialname": "pbl_q1", "mp3": True},
            "swh": {"serialname": "pbl_swh", "mp3": True},
            "hlw": {"serialname": "pbl_hlw", "mp3": True},
            "xmu": {"serialname": "pbl_xmu", "mp3": True},
            "garea": {"serialname": "pbl_garea", "mp3": True},
            "islmsk": {"serialname": "pbl_islmsk", "mp3": True},
            "psk": {"serialname": "pbl_psk", "mp3": True},
            "rbsoil": {"serialname": "pbl_rbsoil", "mp3": True},
            "zorl": {"serialname": "pbl_zorl", "mp3": True},
            "u10m": {"serialname": "pbl_u10m", "mp3": True},
            "v10m": {"serialname": "pbl_v10m", "mp3": True},
            "fm": {"serialname": "pbl_fm", "mp3": True},
            "fh": {"serialname": "pbl_fh", "mp3": True},
            "tsea": {"serialname": "pbl_tsea", "mp3": True},
            "heat": {"serialname": "pbl_heat", "mp3": True},
            "evap": {"serialname": "pbl_evap", "mp3": True},
            "stress": {"serialname": "pbl_stress", "mp3": True},
            "wind": {"serialname": "pbl_wind", "mp3": True},
            "kpbl": {"serialname": "pbl_kpbl", "mp3": True},
            "prsi": {"serialname": "pbl_prsi", "mp3": True},
            "delta": {"serialname": "pbl_delta", "mp3": True},
            "prsl": {"serialname": "pbl_prsl", "mp3": True},
            "prslk": {"serialname": "pbl_prslk", "mp3": True},
            "phii": {"serialname": "pbl_phii", "mp3": True},
            "phil": {"serialname": "pbl_phil", "mp3": True},
            "dtp": {"serialname": "pbl_dtp", "mp3": True},
            "dspheat": {"serialname": "pbl_dspheat", "mp3": True},
            "dusfc": {"serialname": "pbl_dusfc", "mp3": True},
            "dvsfc": {"serialname": "pbl_dvsfc", "mp3": True},
            "dtsfc": {"serialname": "pbl_dtsfc", "mp3": True},
            "dqsfc": {"serialname": "pbl_dqsfc", "mp3": True},
            "hpbl": {"serialname": "pbl_hpbl", "mp3": True},
            "kinver": {"serialname": "pbl_kinver", "mp3": True},
            "xkzm_m": {"serialname": "pbl_xkzm_m", "mp3": True},
            "xkzm_h": {"serialname": "pbl_xkzm_h", "mp3": True},
            "xkzm_ml": {"serialname": "pbl_xkzm_ml", "mp3": True},
            "xkzm_hl": {"serialname": "pbl_xkzm_hl", "mp3": True},
            "xkzm_mi": {"serialname": "pbl_xkzm_mi", "mp3": True},
            "xkzm_hi": {"serialname": "pbl_xkzm_hi", "mp3": True},
            "xkzm_s": {"serialname": "pbl_xkzm_s", "mp3": True},
            "xkzminv": {"serialname": "pbl_xkzminv", "mp3": True},
            "do_dk_hb19": {"serialname": "pbl_do_dk_hb19", "mp3": True},
            "xkzm_lim": {"serialname": "pbl_xkzm_lim", "mp3": True},
            "xkgdx": {"serialname": "pbl_xkgdx", "mp3": True},
            "rlmn": {"serialname": "pbl_rlmn", "mp3": True},
            "rlmx": {"serialname": "pbl_rlmx", "mp3": True},
            "dkt": {"serialname": "pbl_dkt", "mp3": True},
            "cap_k0_land": {"serialname": "pbl_cap_k0_land", "mp3": True},
        }

        self.out_vars = {
            "du": {"serialname": "pbl_du", "mp3": True},
            "dv": {"serialname": "pbl_dv", "mp3": True},
            "tdt": {"serialname": "pbl_tdt", "mp3": True},
            "rtg": {"serialname": "pbl_rtg", "mp3": True},
            "kpbl": {"serialname": "pbl_kpbl", "mp3": True},
            "dusfc": {"serialname": "pbl_dusfc", "mp3": True},
            "dvsfc": {"serialname": "pbl_dvsfc", "mp3": True},
            "dtsfc": {"serialname": "pbl_dtsfc", "mp3": True},
            "dqsfc": {"serialname": "pbl_dqsfc", "mp3": True},
            "hpbl": {"serialname": "pbl_hpbl", "mp3": True},
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
