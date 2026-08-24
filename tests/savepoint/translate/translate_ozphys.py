from ndsl import QuantityFactory, StencilFactory, SubtileGridSizer
from ndsl.constants import I_DIM, J_DIM, K_DIM
from pyshield.stencils.ozone import ozphys_2015
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py
from ndsl.dsl.typing import (
    Float,
    Int,
)

class OzPhys:
    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        oz_coeff,
        levozp,
        dt_phys,
        npz: int,
    ):
        def make_quantity():
            return self.quantity_factory.zeros(
                dims=[I_DIM, J_DIM, K_DIM], units="unknown"
            )

        def make_quantity_2d(type=Float):
            return quantity_factory.zeros([I_DIM, J_DIM], units="unknown", dtype=type)

        self._k_val = quantity_factory.zeros(
            [K_DIM],
            units="unknown",
            dtype=Int,
        )

        for k in range(npz):
            self._k_val[k] = k

        self._colo3 = make_quantity()
        self._coloz = make_quantity()
        self._kmin = make_quantity_2d(Int)
        self._kmax = make_quantity_2d(Int)

        self._prod = self.quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM, self.OZONE_DIM],
            units="unknown",
            dtype=Float,
        )

        ldiag3d = True
        self._ozphys_2015 = stencil_factory.from_dims_halo(
            func=ozphys_2015,
            externals={
                "dt_phys": dt_phys,
                "ldiag3d": ldiag3d,
                "pl_coeff": len(oz_coeff),
                "ko3": levozp,
            },
            compute_dims=[I_DIM, J_DIM, K_DIM],
        )
        pass

    def __call__(
        self,
        ozi,
        ozo,
        tin,
        oz_pres,
        prsl,
        ozpl,
        delp,
        ozp,
    ):
        self._ozphys_2015(
            ozi,
            ozo,
            tin,
            oz_pres,
            prsl,
            ozpl,
            self._prod,
            delp,
            ozp,
            self._colo3,
            self._coloz,
            self._kmin,
            self._kmax,
            self._k_val,
        )

class TranslateOZPhys15(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, config, stencil_factory):
        super().__init__(grid, config, stencil_factory)
        self.in_vars["data_vars"] = {
            "ozi": {"serialname": "ozi", "shield": True},
            "ozo": {"serialname": "ozo", "shield": True},
            "tin": {"serialname": "tin", "shield": True},
            "oz_pres": {"serialname": "oz_pres", "shield": True},
            "prsl": {"serialname": "prsl", "shield": True},
            "ozpl": {"serialname": "ozpl", "shield": True},
            "delp": {"serialname": "delp", "shield": True},
            "ozp": {"serialname": "ozp", "shield": True},
        }
        self.in_vars["parameters"] = ["ozcoef", "delt", "levozp"]
        self.out_vars = {
            "ozo": {"serialname": "ozo", "shield": True},
            "ozp": {"serialname": "ozp", "shield": True},
        }

        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        sizer = SubtileGridSizer.from_tile_params(
            nx_tile=self.config.npx - 1,
            ny_tile=self.config.npy - 1,
            nz=self.config.npz,
            n_halo=3,
            data_dimensions={},
            layout=self.config.layout,
            backend=self.stencil_factory.backend,
        )

        quantity_factory = QuantityFactory(sizer, backend=self.stencil_factory.backend)

        self.make_storage_data_input_vars(inputs)

        compute_func = OzPhys(
            self.stencil_factory,
            quantity_factory,
            inputs.pop("ozcoef"),
            levozp,
            inputs.pop("delt"),
            self.config.npz,
        )

        compute_func(**inputs)

        return self.slice_output(inputs)
