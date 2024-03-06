from gt4py.cartesian import gtscript
from gt4py.cartesian.gtscript import exp, floor

import ndsl.constants as constants


@gtscript.function
def fpvsx(t):
    tliq = constants.TICE
    tice = constants.TICE - 20.0
    dldtl = constants.CP_VAP - constants.C_LIQ
    heatl = constants.HLV
    xponal = -dldtl / constants.RVGAS
    xponbl = -dldtl / constants.RVGAS + heatl / (constants.RVGAS * constants.TICE)
    dldti = constants.CP_VAP - constants.C_ICE_0
    heati = constants.HLV + constants.HLF
    xponai = -dldti / constants.RVGAS
    xponbi = -dldti / constants.RVGAS + heati / (constants.RVGAS * constants.TICE)

    tr = constants.TICE / t

    fpvsx = 0.0
    if t > tliq:
        fpvsx = constants.PSAT * tr ** xponal * exp(xponbl * (1.0 - tr))
    elif t < tice:
        fpvsx = constants.PSAT * tr ** xponai * exp(xponbi * (1.0 - tr))
    else:
        w = (t - tice) / (tliq - tice)
        pvl = constants.PSAT * (tr ** xponal) * exp(xponbl * (1.0 - tr))
        pvi = constants.PSAT * (tr ** xponai) * exp(xponbi * (1.0 - tr))
        fpvsx = w * pvl + (1.0 - w) * pvi

    return fpvsx


@gtscript.function
def fpvs(t):
    # gpvs function variables
    xmin = 180.0
    xmax = 330.0
    nxpvs = 7501
    xinc = (xmax - xmin) / (nxpvs - 1)
    c2xpvs = 1.0 / xinc
    c1xpvs = 1.0 - (xmin * c2xpvs)

    xj = min(max(c1xpvs + c2xpvs * t[0, 0, 0], 1.0), nxpvs)
    jx = min(xj, nxpvs - 1.0)
    jx = floor(jx)

    # Convert jx to "x"
    x = xmin + (jx * xinc)
    xm = xmin + ((jx - 1) * xinc)

    fpvs = fpvsx(xm) + (xj - jx) * (fpvsx(x) - fpvsx(xm))

    return fpvs
