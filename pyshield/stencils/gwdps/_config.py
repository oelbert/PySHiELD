import dataclasses


DEFAULT_FLOAT = 0.0
DEFAULT_INT = 0


@dataclasses.dataclass
class OrographicGravityWaveDragConfig:
    dt_atmos: float = DEFAULT_FLOAT
    """timestep length (s)"""
    cdmbgwd: list = [2.0, 0.25]
    """multiplication factors for cdmb and gwd"""
    npx: int = DEFAULT_INT
    """number of horizontal points per tile in the x-direction"""
    npy: int = DEFAULT_INT
    """number of horizontal points per tile in the y-direction"""
    lonr: int = DEFAULT_INT
    """number of longitude points per tile"""
    nmtvr: int = 14
    """number of topographic variables such as variance
        used in the GWD parameterization"""

    def __post_init__(self):
        if self.lonr == DEFAULT_INT:
            self.lonr = self.npx - 1
