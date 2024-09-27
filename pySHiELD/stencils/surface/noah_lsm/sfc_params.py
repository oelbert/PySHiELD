import numpy as np
from ndsl.dsl.typing import Float, Int

import pySHiELD.constants as physcons


# Assuming isot = ivet = 1

BARE = 16
DEFINED_SLOPE = 9
DEFINED_SOIL = 19
DEFINED_VEG = 20

ZSOIL_DATA = np.array([-0.1, -0.4, -1.0, -2.0])


# Vegetation tables, assuming ivet = 1 from Fortran

SLOPE_DATA = np.zeros(30)
SLOPE_DATA[:20] = 1.0
SNUPX = np.array(
    [
        0.080,
        0.080,
        0.080,
        0.080,
        0.080,
        0.020,
        0.020,
        0.060,
        0.040,
        0.020,
        0.010,
        0.020,
        0.020,
        0.020,
        0.013,
        0.013,
        0.010,
        0.020,
        0.020,
        0.020,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
RSMTBL = np.array(
    [
        300.0,
        300.0,
        70.0,
        175.0,
        175.0,
        70.0,
        70.0,
        70.0,
        70.0,
        20.0,
        40.0,
        20.0,
        400.0,
        35.0,
        200.0,
        70.0,
        100.0,
        70.0,
        150.0,
        200.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)
RGLTBL = np.array(
    [
        30.0,
        30.0,
        30.0,
        30.0,
        30.0,
        100.0,
        100.0,
        65.0,
        65.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        100.0,
        30.0,
        100.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)
HSTBL = np.array(
    [
        47.35,
        41.69,
        47.35,
        54.53,
        51.93,
        42.00,
        42.00,
        42.00,
        42.00,
        36.35,
        60.00,
        36.25,
        42.00,
        36.25,
        42.00,
        42.00,
        51.75,
        42.00,
        42.00,
        42.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)
Z0_DATA = np.array(
    [
        1.089,
        2.653,
        0.854,
        0.826,
        0.80,
        0.05,
        0.03,
        0.856,
        0.856,
        0.15,
        0.04,
        0.13,
        1.00,
        0.25,
        0.011,
        0.011,
        0.001,
        0.076,
        0.05,
        0.03,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
LAI_DATA = np.array(
    [
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        3.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)

NROOT_DATA = np.array(
    [
        4,
        4,
        4,
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        1,
        3,
        2,
        3,
        1,
        3,
        3,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)


# Soil tables, assuming isot = 1 from Fortran
REFSMC = np.array(
    [
        0.236,
        0.283,
        0.312,
        0.360,
        0.360,
        0.329,
        0.315,
        0.387,
        0.382,
        0.338,
        0.404,
        0.403,
        0.348,
        0.283,
        0.133,
        0.283,
        0.403,
        0.133,
        0.236,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
BB = np.array(
    [
        4.05,
        4.26,
        4.74,
        5.33,
        5.33,
        5.25,
        6.77,
        8.72,
        8.17,
        10.73,
        10.39,
        11.55,
        5.25,
        4.26,
        4.05,
        4.26,
        11.55,
        4.05,
        4.05,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)
DRYSMC = np.array(
    [
        0.010,
        0.025,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.010,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
F11 = np.array(
    [
        -1.090,
        -1.041,
        -0.568,
        0.162,
        0.162,
        -0.327,
        -1.535,
        -1.118,
        -1.297,
        -3.211,
        -1.916,
        -2.258,
        -0.201,
        -1.041,
        -2.287,
        -1.041,
        -2.258,
        -2.287,
        -1.090,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
MAXSMC = np.array(
    [
        0.395,
        0.421,
        0.434,
        0.476,
        0.476,
        0.439,
        0.404,
        0.464,
        0.465,
        0.406,
        0.468,
        0.457,
        0.464,
        0.421,
        0.200,
        0.421,
        0.457,
        0.200,
        0.395,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.00,
    ]
)
SATPSI = np.array(
    [
        0.035,
        0.0363,
        0.1413,
        0.7586,
        0.7586,
        0.3548,
        0.1349,
        0.6166,
        0.2630,
        0.0977,
        0.3236,
        0.4677,
        0.3548,
        0.0363,
        0.0350,
        0.0363,
        0.4677,
        0.0350,
        0.0350,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)
SATDK = np.array(
    [
        1.41e-5,
        0.20e-5,
        0.10e-5,
        0.52e-5,
        0.72e-5,
        0.25e-5,
        0.45e-5,
        0.34e-5,
        1.41e-5,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)
WLTSMC = np.array(
    [
        0.023,
        0.028,
        0.047,
        0.084,
        0.084,
        0.066,
        0.069,
        0.120,
        0.103,
        0.100,
        0.126,
        0.135,
        0.069,
        0.028,
        0.012,
        0.028,
        0.135,
        0.012,
        0.023,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
        0.000,
    ]
)
SATDW = np.array(
    [
        0.6316e-4,
        0.5171e-5,
        0.8072e-5,
        0.2386e-4,
        0.2386e-4,
        0.1433e-4,
        0.1006e-4,
        0.2358e-4,
        0.1130e-4,
        0.1864e-04,
        0.9658e-05,
        0.1151e-04,
        0.1356e-04,
        0.5171e-05,
        0.9978e-05,
        0.5171e-05,
        0.1151e-04,
        0.9978e-05,
        0.6316e-04,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)
QTZ = np.array(
    [
        0.92,
        0.82,
        0.25,
        0.15,
        0.10,
        0.20,
        0.60,
        0.10,
        0.35,
        0.52,
        0.10,
        0.25,
        0.05,
        0.25,
        0.07,
        0.25,
        0.60,
        0.52,
        0.92,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
    ]
)


def parse_veg_data(
    veg_data: np.ndarray,
    NROOT_DATA: np.ndarray,
):
    nroot = NROOT_DATA[veg_data]  # and similar for other arrays

    return nroot


def set_soil_veg(
    isl_mask: np.ndarray,
    veg_data: np.ndarray,
    soil_data: np.ndarray,
    vegfrac_data: np.ndarray,
):
    """
    Handles initialization of LSM fields. Includes redprm Fortran subroutine
    idl_mask: mask for surface types. 0 is sea, 1, is land, 2 is ice
    ! ===================================================================== !
    !  description:                                                         !
    !                                                                       !
    !  subroutine redprm internally sets(default valuess), or optionally    !
    !  read-in via namelist i/o, all soil and vegetation parameters         !
    !  required for the execusion of the noah lsm.                          !
    !                                                                       !
    !  optional non-default parameters can be read in, accommodating up to  !
    !  30 soil, veg, or slope classes, if the default max number of soil,   !
    !  veg, and/or slope types is reset.                                    !
    !                                                                       !
    !  future upgrades of routine redprm must expand to incorporate some    !
    !  of the empirical parameters of the frozen soil and snowpack physics  !
    !  (such as in routines frh2o, snowpack, and snow_new) not yet set in   !
    !  this redprm routine, but rather set in lower level subroutines.      !
    !                                                                       !
    !  all soil, veg, slope, and universal parameters values are defined    !
    !  externally (in subroutine "set_soilveg.f") and then accessed via     !
    !  "use namelist_soilveg" (below) and then set here.                    !
    !                                                                       !
    !  soil types   zobler (1986)      cosby et al (1984) (quartz cont.(1)) !
    !      1         coarse            loamy sand            (0.82)         !
    !      2         medium            silty clay loam       (0.10)         !
    !      3         fine              light clay            (0.25)         !
    !      4         coarse-medium     sandy loam            (0.60)         !
    !      5         coarse-fine       sandy clay            (0.52)         !
    !      6         medium-fine       clay loam             (0.35)         !
    !      7         coarse-med-fine   sandy clay loam       (0.60)         !
    !      8         organic           loam                  (0.40)         !
    !      9         glacial land ice  loamy sand            (na using 0.82)!
    !     13: <old>- glacial land ice -<old>                                !
    !     13:        glacial-ice (no longer use these parameters), now      !
    !                treated as ice-only surface and sub-surface            !
    !                (in subroutine hrtice)                                 !
    !  upgraded to statsgo (19-type)
    !     1: sand
    !     2: loamy sand
    !     3: sandy loam
    !     4: silt loam
    !     5: silt
    !     6:loam
    !     7:sandy clay loam
    !     8:silty clay loam
    !     9:clay loam
    !     10:sandy clay
    !     11: silty clay
    !     12: clay
    !     13: organic material
    !     14: water
    !     15: bedrock
    !     16: other (land-ice)
    !     17: playa
    !     18: lava
    !     19: white sand
    !                                                                       !
    !  ssib vegetation types (dorman and sellers, 1989; jam)                !
    !      1:  broadleaf-evergreen trees  (tropical forest)                 !
    !      2:  broadleaf-deciduous trees                                    !
    !      3:  broadleaf and needleleaf trees (mixed forest)                !
    !      4:  needleleaf-evergreen trees                                   !
    !      5:  needleleaf-deciduous trees (larch)                           !
    !      6:  broadleaf trees with groundcover (savanna)                   !
    !      7:  groundcover only (perennial)                                 !
    !      8:  broadleaf shrubs with perennial groundcover                  !
    !      9:  broadleaf shrubs with bare soil                              !
    !     10:  dwarf trees and shrubs with groundcover (tundra)             !
    !     11:  bare soil                                                    !
    !     12:  cultivations (the same parameters as for type 7)             !
    !     13: <old>- glacial (the same parameters as for type 11) -<old>    !
    !     13:  glacial-ice (no longer use these parameters), now treated as !
    !          ice-only surface and sub-surface (in subroutine hrtice)      !
    !  upgraded to IGBP (20-type)
    !      1:Evergreen Needleleaf Forest
    !      2:Evergreen Broadleaf Forest
    !      3:Deciduous Needleleaf Forest
    !      4:Deciduous Broadleaf Forest
    !      5:Mixed Forests
    !      6:Closed Shrublands
    !      7:Open Shrublands
    !      8:Woody Savannas
    !      9:Savannas
    !      10:Grasslands
    !      11:Permanent wetlands
    !      12:Croplands
    !      13:Urban and Built-Up
    !      14:Cropland/natural vegetation mosaic
    !      15:Snow and Ice
    !      16:Barren or Sparsely Vegetated
    !      17:Water
    !      18:Wooded Tundra
    !      19:Mixed Tundra
    !      20:Bare Ground Tundra
    !                                                                       !
    !  slopetyp is to estimate linear reservoir coefficient slope to the    !
    !  baseflow runoff out of the bottom layer. lowest class (slopetyp=0)   !
    !  means highest slope parameter = 1.                                   !
    !                                                                       !
    !  slope class       percent slope                                      !
    !      1                0-8                                             !
    !      2                8-30                                            !
    !      3                > 30                                            !
    !      4                0-30                                            !
    !      5                0-8 & > 30                                      !
    !      6                8-30 & > 30                                     !
    !      7                0-8, 8-30, > 30                                 !
    !      9                glacial ice                                     !
    !    blank              ocean/sea                                       !
    !                                                                       !
    !  note: class 9 from zobler file should be replaced by 8 and 'blank' 9 !
    !                                                                       !
    !                                                                       !
    !  subprogram called:  none                                             !
    !                                                                       !
    !  ====================  defination of variables  ====================  !
    !                                                                       !
    !  inputs from calling program:                                  size   !
    !     nsoil    - integer, number of soil layers                    1    !
    !     vegtyp   - integer, vegetation type (integer index)          1    !
    !     soiltyp  - integer, soil type (integer index)                1    !
    !     slopetyp - integer, class of sfc slope (integer index)       1    !
    !     sldpth   - integer, thickness of each soil layer (m)       nsoil  !
    !     zsoil    - integer, soil depth (negative sign) (m)         nsoil  !
    !                                                                       !
    !  outputs to the calling program:                                      !
    !     cfactr   - real, canopy water parameters                     1    !
    !     cmcmax   - real, maximum canopy water parameters             1    !
    !     rsmin    - real, mimimum stomatal resistance                 1    !
    !     rsmax    - real, maximum stomatal resistance                 1    !
    !     topt     - real, optimum transpiration air temperature       1    !
    !     refkdt   - real, =2.e-6 the sat. dk. val for soil type 2     1    !
    !     kdt      - real,                                             1    !
    !     sbeta    - real, param to cal veg effect on soil heat flux   1    !
    !     shdfac   - real, vegetation greenness fraction               1    !
    !     rgl      - real, canopy resistance func (in solar rad term)  1    !
    !     hs       - real, canopy resistance func (vapor deficit term) 1    !
    !     zbot     - real, specify depth of lower bd soil temp (m)     1    !
    !     frzx     - real, frozen ground parameter, ice content        1    !
    !                      threshold above which frozen soil is impermeable !
    !     psisat   - real, saturated soil potential                    1    !
    !     slope    - real, linear reservoir coefficient                1    !
    !     snup     - real, threshold snow depth (water equi m)         1    !
    !     salp     - real, snow cover shape parameter                  1    !
    !                      from anderson's hydro-17 best fit salp = 2.6     !
    !     bexp     - real, the 'b' parameter                           1    !
    !     dksat    - real, saturated soil hydraulic conductivity       1    !
    !     dwsat    - real, saturated soil diffusivity                  1    !
    !     smcmax   - real, max soil moisture content (porosity)        1    !
    !     smcwlt   - real, wilting pt soil moisture contents           1    !
    !     smcref   - real, reference soil moisture (onset stress)      1    !
    !     smcdry   - real, air dry soil moist content limits           1    !
    !     f1       - real, used to comp soil diffusivity/conductivity  1    !
    !     quartz   - real, soil quartz content                         1    !
    !     fxexp    - real, bare soil evaporation exponent              1    !
    !     rtdis    - real, root distribution                         nsoil  !
    !     nroot    - integer, number of root layers                    1    !
    !     z0       - real, roughness length (m)                        1    !
    !     czil     - real, param to cal roughness length of heat       1    !
    !     xlai     - real, leaf area index                             1    !
    !     csoil    - real, soil heat capacity (j m-3 k-1)              1    !
    !                                                                       !
    !  ====================    end of description    =====================  !
    """

    bexp = np.zeros_like(soil_data, dtype=Float)
    dksat = np.zeros_like(soil_data, dtype=Float)
    dwsat = np.zeros_like(soil_data, dtype=Float)
    f1 = np.zeros_like(soil_data, dtype=Float)
    psisat = np.zeros_like(soil_data, dtype=Float)
    quartz = np.zeros_like(soil_data, dtype=Float)
    smcdry = np.zeros_like(soil_data, dtype=Float)
    smcmax = np.zeros_like(soil_data, dtype=Float)
    smcref = np.zeros_like(soil_data, dtype=Float)
    smcwlt = np.zeros_like(soil_data, dtype=Float)
    frzfact = np.zeros_like(soil_data, dtype=Float)

    nroot = np.zeros_like(veg_data, dtype=Int)
    zroot = np.zeros_like(veg_data, dtype=Float)
    snup = np.zeros_like(veg_data, dtype=Float)
    rsmin = np.zeros_like(veg_data, dtype=Float)
    rgl = np.zeros_like(veg_data, dtype=Float)
    hs = np.zeros_like(veg_data, dtype=Float)
    xlai = np.zeros_like(veg_data, dtype=Float)
    shdfac = np.zeros_like(veg_data, dtype=Float)
    land_mask = isl_mask >= 1
    ice_mask = isl_mask == 2
    ice_mask = ice_mask.astype(int)

    bexp[land_mask] = BB[soil_data[land_mask]]
    dksat[land_mask] = SATDK[soil_data[land_mask]]
    dwsat[land_mask] = SATDW[soil_data[land_mask]]
    f1[land_mask] = F11[soil_data[land_mask]]

    psisat[land_mask] = SATPSI[soil_data[land_mask]]
    quartz[land_mask] = QTZ[soil_data[land_mask]]
    smcdry[land_mask] = DRYSMC[soil_data[land_mask]]
    smcmax[land_mask] = MAXSMC[soil_data[land_mask]]
    smcref[land_mask] = REFSMC[soil_data[land_mask]]
    smcwlt[land_mask] = WLTSMC[soil_data[land_mask]]

    kdt = physcons.REFKDT * dksat / physcons.REFDK

    frzfact[land_mask] = (smcmax[land_mask] / smcref[land_mask]) * (0.412 / 0.468)

    # to adjust frzk parameter to actual soil type: frzk * frzfact
    frzx = physcons.FRZK * frzfact

    nroot[isl_mask == 1] = NROOT_DATA[veg_data[isl_mask == 1]]
    zroot[isl_mask == 1] = ZSOIL_DATA[nroot[isl_mask == 1] - 1]

    snup[land_mask] = SNUPX[veg_data[land_mask]]
    rsmin[land_mask] = RSMTBL[veg_data[land_mask]]

    rgl[land_mask] = RGLTBL[veg_data[land_mask]]
    hs[land_mask] = HSTBL[veg_data[land_mask]]
    # roughness length is not set here
    # z0[land_mask] = Z0_DATA[veg_data[land_mask]]
    xlai[land_mask] = LAI_DATA[veg_data[land_mask]]

    shdfac = np.maximum(vegfrac_data, 0.01)
    shdfac[veg_data == BARE] = 0.0

    # calculate root distribution.  present version assumes uniform
    # distribution based on soil layer depths.

    rtdis = np.zeros((nroot.shape[0], nroot.shape[1], ZSOIL_DATA.shape[0] + 1))
    sldpth = np.zeros(
        (nroot.shape[0], nroot.shape[1], ZSOIL_DATA.shape[0] + 1), dtype=Float
    )
    zsoil = np.zeros((ZSOIL_DATA.shape[0] + 1), dtype=Float)

    for i in range(nroot.shape[0]):
        for j in range(nroot.shape[1]):
            for k in range(len(ZSOIL_DATA)):
                if isl_mask[i, j] == 1:
                    sldpth[i, j, k + 1] = ZSOIL_DATA[k] - ZSOIL_DATA[k - 1]
                    zsoil[k] = ZSOIL_DATA[k]
                    if k < nroot[i, j]:
                        rtdis[i, j, k] = -sldpth[i, j, k] * zroot[i, j]
                elif isl_mask[i, j] == 2:
                    zsoil[i, j, k] = -3.0 * (k + 1) / (len(ZSOIL_DATA) + 1)

    return (
        nroot,
        zroot,
        sldpth,
        zsoil,
        snup,
        rsmin,
        rgl,
        hs,
        xlai,
        bexp,
        dksat,
        dwsat,
        f1,
        kdt,
        psisat,
        quartz,
        smcdry,
        smcmax,
        smcref,
        smcwlt,
        shdfac,
        frzx,
        rtdis,
        land_mask,
        ice_mask,
    )
