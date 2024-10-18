from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.typing import Int
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD.stencils.pbl.mfscu import StratocumulusMassFlux
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class TranslateMFSCU(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "cnvflg": {"shield": True, "serialname": "scuflg"},
            "zl": {"shield": True},
            "zm": {"shield": True},
            "q1": {"shield": True},
            "t1": {"shield": True},
            "u1": {"shield": True},
            "v1": {"shield": True},
            "plyr": {"shield": True},
            "pix": {"shield": True},
            "thlx": {"shield": True},
            "thvx": {"shield": True},
            "gdx": {"shield": True},
            "thetae": {"shield": True},
            "radj": {"shield": True},
            "krad": {"shield": True},
            "mrad": {"shield": True},
            "radmin": {"shield": True},
            "buo": {"shield": True},
            "xmfd": {"shield": True},
            "tcdo": {"shield": True},
            "qcdo": {"shield": True},
            "ucdo": {"shield": True},
            "vcdo": {"shield": True},
            "xlamde": {"shield": True},
        }
        self.in_vars["parameters"] = [
            "kmscu" "ntcw",
            "dt2",
            "ntrac1",
        ]

        self.out_vars = {
            "radj": {"shield": True},
            "krad": {"shield": True},
            "mrad": {"shield": True},
            "buo": {"shield": True},
            "xmfd": {"shield": True},
            "tcdo": {"shield": True},
            "qcdo": {"shield": True},
            "ucdo": {"shield": True},
            "vcdo": {"shield": True},
            "xlamde": {"shield": True},
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

        k_mask = quantity_factory.zeros(
            [Z_DIM],
            units="unknown",
            dtype=Int,
        )
        for k in range(1, self.stencil_factory.grid_indexing.domain[2] + 1):
            k_mask.data[k] = k
        self.make_storage_data_input_vars(inputs)
        inputs.pop("t1")
        cnvflg = quantity_factory.from_array(
            data=inputs.pop('cnvflg'),
            dims=[X_DIM, Y_DIM],
            units="",
        )
        mrad = quantity_factory.from_array(
            data=inputs.pop('mrad'),
            dims=[X_DIM, Y_DIM],
            units="",
        )
        zm = quantity_factory.from_array(
            data=inputs.pop('zm'),
            dims=[X_DIM, Y_DIM, Z_DIM],
            units="",
        )

        compute_func = StratocumulusMassFlux(
            self.stencil_factory,
            quantity_factory,
            dt2=inputs.pop("dt2"),
            ntcw=inputs.pop("ntcw"),
            ntrac1=inputs.pop("ntrac1"),
            kmpbl=inputs.pop("kmpbl"),
            ntke=inputs.pop("ntke"),
        )

        compute_func(**inputs, k_mask=k_mask, cnvflg=cnvflg, mrad=mrad, zm=zm)

        return self.slice_output(inputs)
