import copy

import numpy as np

import ndsl.dsl.gt4py_utils as utils
from ndsl.dsl.typing import Float
from ndsl.initialization.allocator import QuantityFactory
from ndsl.initialization.sizer import SubtileGridSizer
from pySHiELD import PHYSICS_PACKAGES, PhysicsState
from pySHiELD.stencils.pbl import ScaleAwareTKEMoistEDMF
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py

class TranslatePBL(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "mph_qv1", "microph": True},
        }

        self.out_vars = {
            "pt_dt": {"serialname": "mph_pt_dt", "kend": namelist.npz - 1},
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

        self.pbl = ScaleAwareTKEMoistEDMF(
            self.stencil_factory,
            quantity_factory,
            self.grid.grid_data,
            self.namelist.pbl,
        )

        pass