import ndsl.constants as constants
from ndsl.dsl.gt4py import BACKWARD, FORWARD, computation, interval, log
from ndsl.dsl.typing import FloatField, FloatFieldK, IntFieldIJ, IntFieldK

from pyshield._config import FloatFieldOzone


def ozphys(
    ozi: FloatField,
    ozo: FloatField,
    tin: FloatField,
    po3: FloatFieldK,
    prsl: FloatField,
    prdout: FloatFieldOzone,
    prod: FloatFieldOzone,
    delp: FloatField,
    ozp: FloatFieldOzone,
    colo3: FloatField,
    kmin: IntFieldIJ,
    kmax: IntFieldIJ,
    k_index: IntFieldK,
):
    """
    !! The operational GFS currently parameterizes ozone production and
    !! destruction based on monthly mean coefficients provided by Naval
    !! Research Laboratory through CHEM2D chemistry model
    !! (McCormack et al. 2006).
    !! Monthly and zonal mean ozone production rate and ozone destruction
    !! rate per unit ozone mixing ratio were provided by NRL based on
    !! CHEM2D model.
    !! Original version of these terms were provided by NASA/DAO based on
    !! NASA 2D Chemistry model - GSM is capable of running both versions
    !!
    !! section intra_oz Intraphysics Cummunication
    !! - Routine OZPHYS is called from GBPHYS after call to RAYLEIGH_DAMP
    !! @{
    !!
    !! param[in] ix,im     integer, horizontal dimension and num of used pts
    !! param[in] levs      integer, vertical layer dimension
    !! param[in] ko3       integer, number of layers for ozone data
    !! param[in] dt        real, physics time step in seconds
    !! param[in] ozi       real, updated ozone
    !! param[in] ozo       real, updated ozone
    !! param[in] tin       real, updated temperature
    !! param[in] po3       real, (ko3), ozone forcing data level pressure
    !!                      (ln(Pa))
    !! param[in] prsl      real, (ix,levs),mean layer pressure
    !! param[in] prdout    real, (ix,ko3,pl_coeff),ozone forcing data
    !! param[in] pl_coeff  integer, number coefficients in ozone forcing
    !! param[in] delp      real, (ix,levs)
    !! param[in] ldiag3d   logical, flag for 3d diagnostic fields
    !! param[out] ozp       real, ozone change due to physics
    !! param[in] me        integer, pe number - used for debug prints

    ! this code assumes that both prsl and po3 are from bottom to top
    ! as are all other variables
    """
    from __externals__ import dt_phys, ldiag3d, pl_coeff, ko3

    with computation(BACKWARD):
        with interval(-1, None):
            colo3 = ozi * delp / constants.GRAV
        with interval(0, -1):
            colo3 = colo3[0, 0, 1] + ozi * delp / constants.GRAV

    with computation(FORWARD), interval(0, -1):
        pmin = 1.0e10
        pmax = -1.0e10

        wk1 = log(prsl)
        pmin = min(wk1, pmin)
        pmax = max(wk1, pmax)
        prod = 0.0
        kmax = 0
        kmin = 0
        kval = -k_index
        k_offset_limit = ko3 - k_index
        while kval < k_offset_limit:
            if pmin < po3[0, 0, kval]:
                kmax = kval
            if pmax < po3[0, 0, kval]:
                kmin = kval
            kval += 1
        k2 = kmin
        while k2 < kmax:
            temp = 1.0 / (po3[0, 0, k2] - po3[0, 0, k2+1])
            if wk1 < po3[0, 0, k2] and wk1 >= po3[0, 0, k2+1]:
                wk2 = (wk1 - po3[0, 0, k2+1]) * temp
                wk3 = 1.0 - wk2
                prod = wk2 * prdout[0, 0, k2] + wk3 * prdout[0, 0, k2+1]

        if wk1 >= po3[0, 0, k_offset_limit]:
            prod = prdout[0, 0, k_offset_limit]
        if wk1 >= po3[0, 0, -k_index]:
            prod = prdout[0, 0, -k_index]

        if pl_coeff == 2:
            ozib = ozi
            ozo = (ozib + prod[0, 0, 0][0] * dt_phys) / (1.0 + prod[0, 0, 0][1] * dt_phys)
            if ldiag3d:

                ozp[0, 0, 0][0] = ozp[0, 0, 0][0] + prod[0, 0, 0][0] * dt_phys
                ozp[0, 0, 0][1] = ozp[0, 0, 0][1] + (ozo - ozib)
        if pl_coeff == 4:
            ozib = ozi
            tem = prod[0, 0, 0][0] + prod[0, 0, 0][2] * tin + prod[0, 0, 0][3] * colo3[0, 0, 1]
            ozo = (ozib + tem * dt_phys) / (1.0 / prod[0, 0, 0][1] * dt_phys)
            if ldiag3d:
                ozp[0, 0, 0][0] = ozp[0, 0, 0][0] + prod[0, 0, 0][0] * dt_phys
                ozp[0, 0, 0][1] = ozp[0, 0, 0][1] + (ozo - ozib)
                ozp[0, 0, 0][2] = ozp[0, 0, 0][2] + prod[0, 0, 0][2] * tin * dt_phys
                ozp[0, 0, 0][3] = ozp[0, 0, 0][3] + prod[0, 0, 0][3] * colo3[0, 0, 1] * dt_phys

def ozphys_2015(
    ozi: FloatField,
    ozo: FloatField,
    tin: FloatField,
    po3: FloatFieldK,
    prsl: FloatField,
    prdout: FloatFieldOzone,
    prod: FloatFieldOzone,
    delp: FloatField,
    ozp: FloatFieldOzone,
    colo3: FloatField,
    coloz: FloatField,
    kmin: IntFieldIJ,
    kmax: IntFieldIJ,
    k_index: IntFieldK,
):
    """
    this code assumes that both prsl and po3 are from bottom to top
    as are all other variables
    This code is specifically for NRL parameterization and
    climatological T and O3 are in location 5 and 6 of prdout array
        June 2015 - Shrinivas Moorthi
    """

    from __externals__ import dt_phys, ldiag3d, pl_coeff, ko3

    with computation(BACKWARD):
        with interval(-1, None):
            colo3 = 0.0
            coloz = 0.0
        with interval(0, -1):
            pmin = 1.0e10
            pmax = -1.0e10

            wk1 = log(prsl)
            pmin = min(wk1, pmin)
            pmax = max(wk1, pmax)
            prod = 0.0
            kmax = 0
            kmin = 0
            kval = -k_index
            k_offset_limit = ko3 - k_index
            while kval < k_offset_limit:
                if pmin < po3[0, 0, kval]:
                    kmax = kval
                if pmax < po3[0, 0, kval]:
                    kmin = kval
                kval += 1
            k2 = kmin
            while k2 < kmax:
                temp = 1.0 / (po3[0, 0, k2] - po3[0, 0, k2+1])
                if wk1 < po3[0, 0, k2] and wk1 >= po3[0, 0, k2+1]:
                    wk2 = (wk1 - po3[0, 0, k2+1]) * temp
                    wk3 = 1.0 - wk2
                    prod = wk2 * prdout[0, 0, k2] + wk3 * prdout[0, 0, k2+1]

            if wk1 >= po3[0, 0, k_offset_limit]:
                prod = prdout[0, 0, k_offset_limit]
            if wk1 >= po3[0, 0, -k_index]:
                prod = prdout[0, 0, -k_index]
            colo3[0, 0, 0] = colo3[0, 0, 1] + ozi * delp / constants.GRAV
            coloz[0, 0, 0] = coloz[0, 0, 1] + prod[0, 0, 0][5] * delp / constants.GRAV
            prod[0, 0, 0][1] = min(prod[0, 0, 0][1], 0.0)

            ozib = ozi
            tem = prod[0, 0, 0][0] - prod[0, 0, 0][1] * prod[0, 0, 0][5] + prod[0, 0, 0][2] * (tin - prod[0, 0, 0][4]) + prod[0, 0, 0][3] * (colo3[0, 0, 0]-coloz[0, 0, 0])
            ozo = (ozib + tem * dt_phys) / (1.0 - prod[0, 0, 0][1] * dt_phys)

            if ldiag3d:
                ozp[0, 0, 0][0] = ozp[0, 0, 0][0] + (prod[0, 0, 0][0] - prod[0, 0, 0][1] * prod[0, 0, 0][5]) * dt_phys
                ozp[0, 0, 0][1] = ozp[0, 0, 0][1] + (ozo - ozib)
                ozp[0, 0, 0][2] = ozp[0, 0, 0][2] + prod[0, 0, 0][2] * (tin-prod[0, 0, 0][4]) * dt_phys
                ozp[0, 0, 0][3] = ozp[0, 0, 0][3] + prod[0, 0, 0][3] * (colo3-coloz) * dt_phys
