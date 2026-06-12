import ndsl.constants as constants
from ndsl.dsl.typing import Float


PI = Float(3.1415926535897931)
RAD_TO_DEG = Float(180.0 / PI)
DEG_TO_RAD = Float(PI / 180.0)
DW2MIN = Float(1.0)
RIMIN = Float(-100.0)
RIC = Float(0.25)
BNV2MIN = Float(1.0e-5)
EFMIN = Float(0.0)
EFMAX = Float(10.0)
HPMAX = Float(2400.0)
HPMIN = Float(1.0)
FRC = Float(1.0)
CE = Float(0.8)
CEOFRC = Float(CE / FRC)
FRMAX = Float(100.0)
CG = Float(0.5)
GMAX = Float(1.0)
VELEPS = Float(1.0)
FACTOP = Float(0.5)
RLOLEV = Float(50000.0)
HNCRIT = 8000.0  # Max value in meters for ELVMAX (*j*)
SIGFAC = 4.0  # MB3a expt test for ELVMAX factor (*j*)
HMINMT = 50.0  # min mtn height (*j*)
MINWIND = 0.1  # min wind component (*j*)
DPMIN = 5000.0  # Minimum thickness of the reference layer
RDI = Float(1.0 / constants.RDGAS)
GOR = Float(constants.GRAV / constants.RDGAS)
GR2 = Float(constants.GRAV * GOR)
GOCP = Float(constants.GRAV / constants.CP_AIR)
FV = Float(constants.RVGAS / constants.RDGAS - 1)
MDIR = 8
FDIR = MDIR / (PI + PI)
NWDIR = [6, 7, 5, 8, 2, 3, 1, 4]
