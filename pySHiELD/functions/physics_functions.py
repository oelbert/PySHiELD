import ndsl.constants as constants
import pySHiELD.constants as physcons

from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import exp, min, max, floor


@gtscript.function
def fpvsx(t):
    """
    Computes saturation water vapor pressure, adapted from Fortran:
    ! Subprogram: fpvsx        Compute saturation vapor pressure
    !   Author: N Phillips            w/NMC2X2   Date: 30 dec 82
    !
    ! Abstract: Exactly compute saturation vapor pressure from temperature.
    !   The saturation vapor pressure over either liquid and ice is computed
    !   over liquid for temperatures above the triple point,
    !   over ice for temperatures 20 degress below the triple point,
    !   and a linear combination of the two for temperatures in between.
    !   The water model assumes a perfect gas, constant specific heats
    !   for gas, liquid and ice, and neglects the volume of the condensate.
    !   The model does account for the variation of the latent heat
    !   of condensation and sublimation with temperature.
    !   The Clausius-Clapeyron equation is integrated from the triple point
    !   to get the formula
    !       pvsl=con_psat*(tr**xa)*exp(xb*(1.-tr))
    !   where tr is ttp/t and other values are physical constants.
    !   The reference for this computation is Emanuel(1994), pages 116-117.
    !   This function should be expanded inline in the calling routine.
    !
    ! Program History Log:
    !   91-05-07  Iredell             made into inlinable function
    !   94-12-30  Iredell             exact computation
    ! 1999-03-01  Iredell             f90 module
    ! 2001-02-26  Iredell             ice phase
    ! 2024-07-09  Oelbert             Python version
    !
    ! Usage:   pvs=fpvsx(t)
    !
    !   Input argument list:
    !     t          Real(krealfp) temperature in Kelvin
    !
    !   Output argument list:
    !     fpvsx      Real(krealfp) saturation vapor pressure in Pascals
    """
    dldtl = physcons.CPVAP - physcons.CPH2O1
    xponal = -dldtl / constants.RVGAS
    xponbl = -dldtl / constants.RVGAS + constants.HLV / (
        constants.RVGAS * constants.TICE
    )
    dldti = physcons.CPVAP - physcons.CPICE
    heati = constants.HLV + constants.HLF
    xponai = -dldti / constants.RVGAS
    xponbi = -dldti / constants.RVGAS + heati / (
        constants.RVGAS * constants.TICE
    )
    tr = constants.TICE / t
    tice2 = constants.TICE - 20.0
    if t > constants.TICE:
        fpvsx = physcons.PSAT * (tr**xponal) * exp(xponbl * (1. - tr))
    elif t < tice2:
        fpvsx = physcons.PSAT * (tr**xponai) * exp(xponbi * (1. - tr))
    else:
        w = (t - tice2) / (constants.TICE - tice2)
        pvl = physcons.PSAT * (tr**xponal) * exp(xponbl * (1. - tr))
        pvi = physcons.PSAT * (tr**xponai) * exp(xponbi * (1. - tr))
        fpvsx = w * pvl + (1. - w) * pvi
    return fpvsx


@gtscript.function
def fpvs(t):
    """
    Adapted from a lookup table version in Fortran for use in GT4Py
    """
    nxpvs = 7501.0
    xmin = 180.0
    xmax = 330.0
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1. / xinc
    c1xpvs = 1. - xmin * c2xpvs
    xj = min(max(c1xpvs + c2xpvs * t, 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    tx = floor(jx)
    fpvs = fpvsx(jx - 1) + (xj - jx) * (fpvsx(jx) - fpvsx(jx - 1))
