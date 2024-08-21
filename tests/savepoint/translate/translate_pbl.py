from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD.stencils.pbl import ScaleAwareTKEMoistEDMF
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TranslatePBL(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "ntrac": {"serialname": "pbl_ntrac"},
            "ntcw": {"serialname": "pbl_ntcw"},
            "ntiw": {"serialname": "pbl_ntiw"},
            "ntke": {"serialname": "pbl_ntke"},
            "dv": {"serialname": "pbl_dv"},
            "du": {"serialname": "pbl_du"},
            "tdt": {"serialname": "pbl_tdt"},
            "rtg": {"serialname": "pbl_rtg"},
            "u1": {"serialname": "pbl_u1"},
            "v1": {"serialname": "pbl_v1"},
            "t1": {"serialname": "pbl_t1"},
            "q1": {"serialname": "pbl_q1"},
            "swh": {"serialname": "pbl_swh"},
            "hlw": {"serialname": "pbl_hlw"},
            "xmu": {"serialname": "pbl_xmu"},
            "garea": {"serialname": "pbl_garea"},
            "islmsk": {"serialname": "pbl_islmsk"},
            "psk": {"serialname": "pbl_psk"},
            "rbsoil": {"serialname": "pbl_rbsoil"},
            "zorl": {"serialname": "pbl_zorl"},
            "u10m": {"serialname": "pbl_u10m"},
            "v10m": {"serialname": "pbl_v10m"},
            "fm": {"serialname": "pbl_fm"},
            "fh": {"serialname": "pbl_fh"},
            "tsea": {"serialname": "pbl_tsea"},
            "heat": {"serialname": "pbl_heat"},
            "evap": {"serialname": "pbl_evap"},
            "stress": {"serialname": "pbl_stress"},
            "wind": {"serialname": "pbl_wind"},
            "kpbl": {"serialname": "pbl_kpbl"},
            "prsi": {"serialname": "pbl_prsi"},
            "delta": {"serialname": "pbl_delta"},
            "prsl": {"serialname": "pbl_prsl"},
            "prslk": {"serialname": "pbl_prslk"},
            "phii": {"serialname": "pbl_phii"},
            "phil": {"serialname": "pbl_phil"},
            "dtp": {"serialname": "pbl_dtp"},
            "dspheat": {"serialname": "pbl_dspheat"},
            "dusfc": {"serialname": "pbl_dusfc"},
            "dvsfc": {"serialname": "pbl_dvsfc"},
            "dtsfc": {"serialname": "pbl_dtsfc"},
            "dqsfc": {"serialname": "pbl_dqsfc"},
            "hpbl": {"serialname": "pbl_hpbl"},
            "kinver": {"serialname": "pbl_kinver"},
            "xkzm_m": {"serialname": "pbl_xkzm_m"},
            "xkzm_h": {"serialname": "pbl_xkzm_h"},
            "xkzm_ml": {"serialname": "pbl_xkzm_ml"},
            "xkzm_hl": {"serialname": "pbl_xkzm_hl"},
            "xkzm_mi": {"serialname": "pbl_xkzm_mi"},
            "xkzm_hi": {"serialname": "pbl_xkzm_hi"},
            "xkzm_s": {"serialname": "pbl_xkzm_s"},
            "xkzminv": {"serialname": "pbl_xkzminv"},
            "do_dk_hb19": {"serialname": "pbl_do_dk_hb19"},
            "xkzm_lim": {"serialname": "pbl_xkzm_lim"},
            "xkgdx": {"serialname": "pbl_xkgdx"},
            "rlmn": {"serialname": "pbl_rlmn"},
            "rlmx": {"serialname": "pbl_rlmx"},
            "dkt": {"serialname": "pbl_dkt"},
            "cap_k0_land": {"serialname": "pbl_cap_k0_land"},
        }

        self.out_vars = {
            "du": {"serialname": "pbl_du"},
            "dv": {"serialname": "pbl_dv"},
            "tdt": {"serialname": "pbl_tdt"},
            "rtg": {"serialname": "pbl_rtg"},
            "kpbl": {"serialname": "pbl_kpbl"},
            "dusfc": {"serialname": "pbl_dusfc"},
            "dvsfc": {"serialname": "pbl_dvsfc"},
            "dtsfc": {"serialname": "pbl_dtsfc"},
            "dqsfc": {"serialname": "pbl_dqsfc"},
            "hpbl": {"serialname": "pbl_hpbl"},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
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
