from ._config import PHYSICS_PACKAGES, PhysicsConfig
from .physics_state import PhysicsState
from .radiation import RTE_RRTMGPState
from .stencils.physics import Physics
from .stencils.surface import SurfaceState


__all__ = [
    "PHYSICS_PACKAGES",
    "PhysicsConfig",
    "PhysicsState",
    "Physics",
    "RTE_RRTMGPState",
    "SurfaceState",
]
__version__ = "0.2.0"

"""
PHYSICS_PACKAGES: Enum of available physics packages
PhysicsConfig: Class containing physics variables and model configurations
PhysicsState: Class containing the atmospheric state for the physics schemes
SurfaceState: Class containing the surface state for the physics schemes
RTE_RRTMGPState: Class containing the radiation variables used in the RTE-RRTMGP solver
Physics: Class that specifies and executes the subgrid physics schemes
"""
