import dataclasses


DEFAULT_FLOAT = 0.0


@dataclasses.dataclass
class OrographicGravityWaveDragConfig:
    dt_atmos: float = DEFAULT_FLOAT
    """timestep length (s)"""
