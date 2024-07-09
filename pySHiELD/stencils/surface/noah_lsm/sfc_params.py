import ndsl.constants as constants
import numpy as np
import pySHiELD.constants as physcons
from pySHiELD._config import SurfaceConfig


BARE = 15
DEFINED_SLOPE = 9
DEFINED_SOIL = 19
DEFINED_VEG = 20

ZSOIL = np.array([-0.1, -0.4, -1.0, -2.0])

NROOT_DATA = np.array([4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2,
                       3, 1, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

SATDK = np.array(
    1.41e-5, 0.20e-5, 0.10e-5, 0.52e-5, 0.72e-5,
    0.25e-5, 0.45e-5, 0.34e-5, 1.41e-5, 0.00,
    0.00   , 0.00   , 0.00   , 0.00   , 0.00,
    0.00   , 0.00   , 0.00   , 0.00   , 0.00,
    0.00   , 0.00   , 0.00   , 0.00   , 0.00,
    0.00   , 0.00   , 0.00   , 0.00   , 0.00
)

def parse_veg_data(
    veg_data: np.ndarray,
    NROOT_DATA: np.ndarray,
):
    nroot = NROOT_DATA[veg_data]  # and similar for other arrays

    return nroot

def redprm(
    config: SurfaceConfig,
    veg_data: np.ndarray,
    soil_data: np.ndarray,
    zbot_data: np.ndarray,
    salp_data: np.ndarray,
    cfactr_data: np.ndarray,
    cmcmax_data: np.ndarray,
    sbeta_data: np.ndarray,
    rsmax_data: np.ndarray,
    topt_data: np.ndarray,
    refdk_data: np.ndarray,
    frzk_data: np.ndarray,
    fxexp_data: np.ndarray,
    refkdt_data: np.ndarray,
    czil_data: np.ndarray,
    csoil_data: np.ndarray,
    vegfrac_data: np.ndarray,
):
    '''
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
    '''

    zbot = zbot_data
    salp = salp_data
    cfactr = cfactr_data
    cmcmax = cmcmax_data
    sbeta = sbeta_data
    rsmax = rsmax_data
    topt = topt_data
    refdk = refdk_data
    frzk = frzk_data
    fxexp = fxexp_data
    refkdt = refkdt_data
    czil = czil_data
    csoil = csoil_data

    nroot = config.nroot_data[veg_data]
    zroot = ZSOIL[nroot - 1]
    sldpth = [ZSOIL[k + 1] - ZSOIL[k] for k in range(len(ZSOIL) - 1)]
    sldpth = np.array(
        sldpth.insert(0, -ZSOIL[0])
    )

    bexp = config.bb[soil_data]
    dksat = config.satdk[soil_data]
    dwsat = config.satdw[soil_data]
    f1 = config.f11[soil_data]
    kdt = refkdt * dksat / refdk

    psisat = config.satpsi[soil_data]
    quartz = config.qtz[soil_data]
    smcdry = config.drysmc[soil_data]
    smcmax = config.maxsmc[soil_data]
    smcref = config.refsmc[soil_data]
    smcwlt = config.wltsmc[soil_data]

    shdfac = max(vegfrac_data, 0.01)

    kdt = physcons.REFKDT * dksat / physcons.REFDK

    frzfact = (smcmax / smcref) * (0.412 / 0.468)

    # to adjust frzk parameter to actual soil type: frzk * frzfact
    frzx = physcons.FRZK * frzfact

    shdfac[veg_data == BARE] = 0.0

    # calculate root distribution.  present version assumes uniform
    # distribution based on soil layer depths.

    rtdis = np.zeros((nroot.shape[0], nroot.shape[1], ZSOIL.shape))

    for i in range(nroot.shape[0]):
        for j in range(nroot.shape[1]):
            for k in range[nroot[i, j]]:
                rtdis[i, j, k] = -sldpth[k] * zroot[k]

    pass

    return (
        zbot,
        salp,
        cfactr,
        cmcmax,
        sbeta,
        rsmax,
        topt,
        refdk,
        frzk,
        fxexp,
        refkdt,
        czil,
        csoil,
        nroot,
        zroot,
        sldpth,
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
        kdt,
        shdfac,
        frzx,
        rtdis,
    )
