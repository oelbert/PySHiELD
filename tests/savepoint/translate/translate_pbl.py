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
        pass