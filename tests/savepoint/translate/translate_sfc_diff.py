from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_diff import SurfaceExchange
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateSurfaceExchange(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"dycore": True},
            "qliquid": {"dycore": True},
            "qrain": {"dycore": True},
            "qsnow": {"dycore": True},
            "qice": {"dycore": True},
            "qgraupel": {"dycore": True},
            "qo3mr": {"dycore": True},
            "qsgs_tke": {"dycore": True},
            "qcld": {"dycore": True},
            "pt": {"dycore": True},
            "delp": {"dycore": True},
            "delz": {"dycore": True},
            "ua": {"dycore": True},
            "va": {"dycore": True},
            "w": {"dycore": True},
            "omga": {"dycore": True},
        }
        self.in_vars["parameters"] = [
            "do_z0_hwrf15",
            "do_z0_hwrf17",
            "do_z0_hwrf17_hwonly",
            "do_z0_moon"
        ]
        self.out_vars = {
            "gt0": {
                "serialname": "IPD_gt0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gu0": {
                "serialname": "IPD_gu0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gv0": {
                "serialname": "IPD_gv0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qvapor": {
                "serialname": "IPD_qvapor",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qliquid": {
                "serialname": "IPD_qliquid",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qrain": {
                "serialname": "IPD_rain",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qice": {
                "serialname": "IPD_qice",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qsnow": {
                "serialname": "IPD_snow",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qgraupel": {
                "serialname": "IPD_qgraupel",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qcld": {
                "serialname": "IPD_qcld",
                "kend": namelist.npz - 1,
                "order": "F",
            },
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.compute_func = SurfaceExchange(
            self.stencil_factory,
            self.inputs["do_z0_hwrf15"],
            self.inputs["do_z0_hwrf17"],
            self.inputs["do_z0_hwrf17_hwonly"],
            self.inputs["do_z0_moon"],
        )
        self.make_storage_data_input_vars(inputs)
        inputs["gq0"] = inputs["gq0"]["qvapor"]
        self.compute_func(**inputs)
        return self.slice_output(inputs)
