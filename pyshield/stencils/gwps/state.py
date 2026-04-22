from dataclasses import dataclass, field

from ndsl import Quantity
from ndsl.constants import I_DIM, J_DIM, K_DIM, K_INTERFACE_DIM


@dataclass()
class OrographicGravityWaveDragState:
    prsi: Quantity = field(
        metadata={
            "name": "interface_pressure",
            "dims": [I_DIM, J_DIM, K_INTERFACE_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    prsl: Quantity = field(
        metadata={
            "name": "layer_mean_pressure",
            "dims": [I_DIM, J_DIM, K_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
