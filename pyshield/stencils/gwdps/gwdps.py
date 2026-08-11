import numpy as np

import ndsl.constants as constants
import pyshield.stencils.gwdps.constants as gwdpscons
from ndsl.constants import X_DIM, Y_DIM, Z_DIM
from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, atan, computation, cos, exp, interval, sin, sqrt, max, min

# from pace.dsl.dace.orchestration import orchestrate
from ndsl.dsl.stencil import StencilFactory
from ndsl.dsl.typing import (
    Bool,
    BoolField,
    BoolFieldIJ,
    Float,
    FloatField,
    FloatFieldIJ,
    Int,
    IntField,
    IntFieldIJ,
    IntFieldK,
)
from ndsl.initialization.allocator import QuantityFactory
from pyshield.stencils.gwdps._config import OrographicGravityWaveDragConfig
from pyshield.stencils.gwdps.state import OrographicGravityWaveDragState


def init_columns(
    dusfc: FloatFieldIJ,
    dvsfc: FloatFieldIJ,
    db: FloatField,
    ang: FloatField,
    uds: FloatField,
):
    with computation(FORWARD), interval(0, 1):
        dusfc = 0.0
        dvsfc = 0.0

    with computation(PARALLEL), interval(...):
        db = 0.0
        ang = 0.0
        uds = 0.0


def find_mountain_blocking_points(
    ipt: BoolFieldIJ,
    npt: IntFieldIJ,
    rdxzb: FloatField,
    elvmax: FloatFieldIJ,
    hprime: FloatFieldIJ,
):
    from __externals__ import nmtvr
    with computation(FORWARD), interval(0, 1):
        ipt = False
        npt = 0
        if hprime > gwdpscons.gwdpscons.HPMIN:
            if nmtvr == 14:  # mb present
                if elvmax > gwdpscons.HMINMT:
                    ipt = True
                    npt = 1
                else:
                    ipt = False
        else:
            ipt = False
    with computation(PARALLEL), interval(...):
        rdxzb = 0.0

# if np.sum(npt[:] == 0):
#     return

# if nmtvr == 14:

# kmll = kmm1
# domain[2] = kmll
# cdmb = Float(4.0 * 192.0 / float(lonr))
# if cdmbgwd[0] >= 0:
#     cdmb = cdmb * cdmbgwd[0]
def mountain_blocking(
    ipt: BoolFieldIJ,
    iwklm: IntFieldIJ,
    idxzb: IntFieldIJ,
    kreflm: IntFieldIJ,
    elvmax: FloatFieldIJ,
    hprime: FloatFieldIJ,
    phil: FloatField,
    phii: FloatField,
    wk: FloatFieldIJ,
    vtj: FloatField,
    u1: FloatField,
    v1: FloatField,
    t1: FloatField,
    q1: FloatField,
    vtk: FloatField,
    prsi: FloatField,
    prsl: FloatField,
    prslk: FloatField,
    bnv2lm: FloatField,
    ro: FloatField,
    prsi_wk: FloatFieldIJ,
    prsl_wk: FloatFieldIJ,
    delks: FloatFieldIJ,
    delks1: FloatFieldIJ,
    ubar: FloatFieldIJ,
    vbar: FloatFieldIJ,
    roll: FloatFieldIJ,
    pe: FloatFieldIJ,
    ek: FloatFieldIJ,
    bnv2bar: FloatFieldIJ,
    delp: FloatField,
    theta: FloatFieldIJ,
    ang: FloatField,
    uds: FloatField,
    up: FloatFieldIJ,
    zbk: FloatFieldIJ,
    phil_zb: FloatFieldIJ,
    gamma: FloatFieldIJ,
    db: FloatField,
    sigma: FloatFieldIJ,
    k_index: IntFieldK,
):
    from __externals__ import cdmb
    with computation(FORWARD), interval(0, 1):
        if ipt:
            # --- iwklm is the level above the height of the of the mountain.
            # --- idxzb is the level of the dividing streamline.
            # INITIALIZE DIVIDING STREAMLINE (DS) CONTROL VECTOR

            iwklm = 2
            idxzb = 0
            kreflm = 0

            # > --- Subgrid Mountain Blocking Section
            #
            # ..............................
            # ..............................
            #
            #  (*j*)  11/03:  test upper limit on KMLL=km - 1
            #      then do not need hncrit -- test with large hncrit first.
            # KMLL  = km / 2 # maximum mtnlm height : # of vertical levels / 2

            # --- No mtn should be as high as KMLL (so we do not have to start at
            # --- the top of the model but could do calc for all levels).
            elvmax = min(
                elvmax + gwdpscons.SIGFAC * hprime, gwdpscons.HNCRIT
            )

    with computation(PARALLEL), interval(0, -1):  # for k in range(kmll):
        if ipt:
            # --- interpolate to max mtn height for index, iwklm[mm] wk[gz]
            # --- ELVMAX is limited to hncrit because to hi res topo30 orog.
            pkp1log = phil[0, 0, 1] / constants.GRAV
            pklog = phil / constants.GRAV
            if (elvmax <= pkp1log) and (elvmax >= pklog):
                wk = (
                    constants.GRAV
                    * elvmax
                    / (phil[0, 0, 1] - phil)
                )
                iwklm = max(iwklm, k_index + 1)
                prsi_wk = prsi[0, 0, 1]
                prsl_wk: prsl[0, 0, 1]
            # ---        find at prsl levels large scale environment variables
            # ---        these cover all possible mtn max heights
            vtj = t1 * (1.0 + gwdpscons.FV * q1)
            vtk = vtj / prslk

            # DENSITY Kg/M**3
            ro = gwdpscons.RDI * prsl / vtj

    # klevm1 = kmll - 1
    with computation(PARALLEL), interval(0, -2):  # for k in range(klevm1):
        if ipt:
            rdz = constants.GRAV / (phil[0, 0, 1] - phil)
            # Brunt-Vaisala Frequency
            # > - Compute Brunt-Vaisala Frequency \f$N\f$.
            bnv2lm = (
                (constants.GRAV + constants.GRAV)
                * rdz
                * (vtk[0, 0, 1] - vtk)
                / (vtk[0, 0, 1] + vtk)
            )
            bnv2lm = max(bnv2lm, gwdpscons.BNV2MIN)

    with computation(FORWARD), interval(0, 1):
        if ipt:
            delks = 1.0 / (prsi - prsi_wk)
            delks1 = 1.0 / (prsl - prsl_wk)
            ubar = 0.0
            vbar = 0.0
            roll = 0.0
            pe = 0.0
            ek = 0.0
            bnv2bar = (
                (prsl - prsl[0, 0, 1]) * delks1 * bnv2lm
            )

    # --- find the dividing stream line height
    # --- starting from the level above the max mtn downward
    # --- iwklm[mm] is the k-index of mtn elvmax elevation
    # > - Find the dividing streamline height starting from the level above
    # the maximum mountain height and processing downward.
    with computation(BACKWARD), interval(0, -2):  # for ktrial in range(kmll - 1, -1, -1):
        if ipt:
            if (k_index < iwklm) and (kreflm == 0):
                kreflm = k_index

    # --- in the layer kreflm[mm] to 1 find PE (which needs N, ELVMAX)
    # ---  make averages, guess dividing stream (DS) line layer.
    # ---  This is not used in the first cut except for testing and
    # --- is the vert ave of quantities from the surface to mtn top.
    with computation(FORWARD), interval(0, -2):
        if k_index <= kreflm:
            rdelks = delp * delks
            ubar = ubar + rdelks * u1  # trial Mean U below
            vbar = vbar + rdelks * v1  # trial Mean V below
            # trial Mean ro below:
            roll = roll + rdelks * ro
            rdelks = (prsl - prsl[0, 0, 1]) * delks1
            bnv2bar = bnv2bar + bnv2lm * rdelks
            # --- these vert ave are for diags, testing and GWD to follow (*j*).

    # --- integrate to get PE in the trial layer.
    # --- Need the first layer where PE>EK - as soon as
    # --- idxzb is not 0 we have a hit and Zb is found.
    with computation(BACKWARD), interval(...):
        if k_index <= iwklm:
            phiang = atan(v1, u1) * gwdpscons.RAD_TO_DEG
            ang = theta - phiang
            ang = (
                ang - 180.0 if (ang > 90.0) else ang
            )
            ang = (
                ang + 180.0 if (ang < -90.0) else ang
            )
            ang = ang * gwdpscons.DEG_TO_RAD

            # > - Compute wind speed UDS
            # # \f[
            # #     UDS=\max(\sqrt{u1^2+v1^2},minwnd)
            # # \f]
            # #  where \f$ minwnd=0.1 \f$, \f$u1\f$ and \f$v1\f$ are zonal and
            # #  meridional wind components of model layer wind.
            uds = max(
                sqrt(u1 * u1 + v1 * v1),
                gwdpscons.MINWIND,
            )
            # --- Test to see if we found Zb previously
            if idxzb == 0:
                pe = pe + bnv2lm * (
                    constants.GRAV * elvmax - phil
                ) * (phii[0, 0, 1] - phii) / (
                    constants.GRAV * constants.GRAVG
                )

                # --- KE
                # --- Wind projected on the line perpendicular
                #         to mtn range, U(Zb(K)).
                # --- kenetic energy is at the layer Zb
                # --- THETA ranges from -+90deg |_
                #         to the mtn "largest topo variations"
                up = uds * cos(ang)
                ek = 0.5 * up * up

                # --- Dividing Stream lime  is found when PE =exceeds EK.
                if pe >= ek:
                    idxzb = k_index
                    rdxzb = k_index
            # --- Then mtn blocked flow is between Zb=k(idxzb[i, j]) and surface

            # > - The dividing streamline height (idxzb), of a subgrid scale
            # #  obstable, is found by comparing the potential (PE) and kinetic
            # #  energies (EK) of the upstream large scale wind and subgrid
            # #  scale air parcel movements. the dividing streamline is found
            # #  when \f$PE\geq EK\f$. Mountain-blocked flow is defined to
            # #  exist between the surface and the dividing streamline height
            # #  (\f$h_d\f$), which can be found by solving an integral
            # #  equation for \f$h_d\f$:
            # # \f[
            # #  \frac{U^{2}(h_{d})}{2}=\int_{h_{d}}^{H} N^{2}(z)(H-z)dz
            # # \f]
            # #  where \f$H\f$ is the maximum subgrid scale elevation within the
            # #  grid box of actual orography, \f$h\f$, obtained from the
            # #  GTOPO30 dataset from the U.S. Geological Survey.

    with computation(FORWARD), interval(0, 1):
        if ipt:
            # --- Calc if N constant in layers (Zb guess) - a diagnostic only.
            zbk = (
                elvmax - sqrt(ubar * ubar + vbar * vbar) / bnv2bar
            )
            zlen = 0.0

    # --- The drag for mtn blocked flow
    with computation(BACKWARD), interval(...):
        if idxzb >= 0:
            if k_index == idxzb:
                phil_zb = phil
            if k_index <= idxzb:
                if phil_zb > phil:
                    # > - Calculate \f$ZLEN\f$, which sums up a number of
                    # #  contributions of elliptic obstables.
                    # # \f[
                    # #     ZLEN=\sqrt{[\frac{h_{d}-z}{z+h'}]}
                    # # \f]
                    # #  where \f$z\f$ is the height, \f$h'\f$ is the orographic
                    # #  standard deviation (HPRIME).
                    zlen = sqrt(
                        (phil_zb - phil)
                        / (phil + constants.GRAV * hprime)
                    )
                    # --- lm eq 14:
                    # > - Calculate the drag coefficient to vary with the aspect
                    # #  ratio of the obstable as seen by the incident flow
                    # #  (see eq.14 in Lott and Miller (1997)
                    # #  \cite lott_and_miller_1997)
                    # # \f[
                    # #  R=\frac{\cos^{2}\psi+\gamma\sin^{2}\psi}
                    # #      {\gamma\cos^{2}\psi+\sin^{2}\psi}
                    # # \f]
                    # #  where \f$\psi\f$, which is derived from THETA, is the
                    # #  angle between the incident flow direction and the
                    # #  normal ridge direcion. \f$\gamma\f$ is the orographic
                    # #  anisotropy (GAMMA).
                    r = (
                        cos(ang) ** 2
                        + gamma * sin(ang) ** 2
                    )
                    if abs(r) < 1.0e-20:
                        db = 0.0
                    else:
                        r = (
                            gamma * cos(ang) ** 2
                            + sin(ang) ** 2
                        ) / r
                        # --- (negitive of db -- see sign at tendency)
                        # > - In each model layer below the dividing
                        # #  streamlines, a drag from the blocked flow is
                        # #  exerted by the obstacle on the large scale flow.
                        # #  The drag per unit area and per unit height is
                        # #  written (eq.15 in Lott and Miller (1997)
                        # #  \cite lott_and_miller_1997):
                        # # \f[
                        # #  D_{b}(z)=-C_{d}\max(2-\frac{1}{R},0)\rho
                        # #      \frac{\sigma}{2h'}ZLEN
                        # #      \max(\cos\psi,\gamma\sin\psi)\frac{UDS}{2}
                        # # \f]
                        # #  where \f$C_{d}\f$ is a specified constant,
                        # #  \f$\sigma\f$ is the orographic slope.
                        dbtmp = (
                            0.25
                            * cdmb * max(2.0 - r, 0.0) * sigma * max(
                                np.cos(ang),
                                gamma * sin(ang),
                            )
                            * zlen
                            / hprime
                        )
                        db = dbtmp * uds
        # .............................
        # .............................
        #  end  mtn blocking section

# if # nmtvr != 14:
def no_mountain_blocking(
    idxzb: IntFieldIJ,
    rdxzb: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        idxzb = 0
        rdxzb = 0.0

# .............................
# .............................

# > --- Orographic Gravity Wave Drag Section
kmpbl = km / 2  # maximum pbl height : # of vertical levels / 2

#  Scale cleff between IM=384*2 and 192*2 for T126/T170 and T62
if lonr > 0:
    # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/384.0)  # this is inverse of CLEFF!
    # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
    # cleff = 0.5E-5 * np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
    # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/192)/float(lonr/192)
    # cleff = 1.0E-5 / np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
    cleff = 0.5e-5 / np.sqrt(float(lonr) / 192.0)  # this is inverse of CLEFF!
    # hmhj for ndsl
    # jw    cleff = 0.1E-5 / np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
    #       cleff = 2.0E-5 * np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
    #       cleff = 2.5E-5 * np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
cleff = cleff * cdmbgwd[1] if cdmbgwd[1] >= 0.0 else cleff

kbps = 1
kmps = km

def start_orographic_gwd_and_find_indices(
    u1: FloatField,
    v1: FloatField,
    t1: FloatField,
    q1: FloatField,
    prsi: FloatField,
    prsi0: FloatFieldIJ,
    prsl: FloatField,
    prslk: FloatField,
    phil: FloatField,
    ipt: BoolFieldIJ,
    vtj: FloatField,
    vtk: FloatField,
    taup: FloatField,
    ro: FloatField,
    ri_n: FloatField,
    bnv2: FloatField,
    delks: FloatField,
    delks1: FloatFieldIJ,
    ubar: FloatFieldIJ,
    vbar: FloatFieldIJ,
    roll: FloatFieldIJ,
    bnv2bar: FloatFieldIJ,
    iwk: IntFieldIJ,
    kref: IntFieldIJ,
    kpbl: IntFieldIJ,
    k_index: IntFieldK,
):
    from __externals__ import kmpbl
    with computation(PARALLEL), interval(...):
        if ipt:
            vtj = t1 * (1.0 + gwdpscons.FV * q1)
            vtk = vtj / prslk
            # DENSITY TONS/M**3:
            ro = gwdpscons.RDI * prsl / vtj
            taup = 0.0

    with computation(FORWARD), interval(0, -1):
        if ipt:
            ti = 2.0 / (t1 + t1[0, 0, 1])
            tem = ti / (prsl - prsl[0, 0, 1])
            rdz = constants.GRAV / (phil[0, 0, 1] - phil)
            tem1 = u1 - u1[0, 0, 1]
            tem2 = v1 - v1[0, 0, 1]
            dw2 = tem1 * tem1 + tem2 * tem2
            shr2 = max(dw2, gwdpscons.DW2MIN) * rdz * rdz
            bvf2 = (
                constants.GRAV
                * (gwdpscons.GOCP + rdz * (vtj[0, 0, 1] - vtj))
                * ti
            )
            ri_n = max(bvf2 / shr2, gwdpscons.RIMIN)  # Richardson number
            # Brunt-Vaisala Frequency
            # tem       = GR2 * (PRSL[i, j, k]+PRSL[i, j, k+1]) * tem
            # bnv2[i,j,k]=tem*(VTK[i,j,k+1]-VTK[i,j,k])/(VTK[i,j,k+1]+VTK[i,j,k])
            bnv2 = (
                (constants.GRAV + constants.GRAV)
                * rdz
                * (vtk[0, 0, 1] - vtk)
                / (vtk[0, 0, 1] + vtk)
            )
            bnv2 = max(bnv2, gwdpscons.BNV2MIN)

    # Finding the first interface index above 50 hPa level
    with computation(FORWARD):
        with interval(0, 1):
            if ipt:
                iwk = 2
                prsi0 = prsi
        with interval(1, kmpbl):
            if ipt:
                tem = prsi0 - prsi
                iwk = k_index if tem < gwdpscons.DPMIN else iwk

    with computation(FORWARD), interval(0, 1):
        if ipt:
            kref = max(iwk, kpbl + 1)  # reference level
            delks = 1.0 / (prsi - prsi[0, 0, kref])
            delks1 = 1.0 / (prsl - prsl[0, 0, kref])
            ubar = 0.0
            vbar = 0.0
            roll = 0.0

            bnv2bar = (
                (prsl - prsl[0, 0, 1]) * delks1 * bnv2
            )

kbps = max(kbps, max(kref[:]))
kmps = min(kmps, min(kref[:]))
kbpsp1 = kbps + 1
kbpsm1 = kbps - 1

def means_below_kref(
    u1: FloatField,
    v1: FloatField,
    prsl: FloatField,
    delp: FloatField,
    delks: FloatFieldIJ,
    delks1: FloatFieldIJ,
    ubar: FloatFieldIJ,
    vbar: FloatFieldIJ,
    bnv2bar: FloatFieldIJ,
    bnv2: FloatField,
    kref: IntFieldIJ,
    ro: FloatField,
    roll: FloatFieldIJ,
    ipt: BoolFieldIJ,
    k_index: IntFieldK,
):
    from __externals__ import kbps
    with computation(PARALLEL), interval(0, kbps):
        if ipt:
            if k_index < kref:
                rdelks = delp * delks
                ubar = ubar + rdelks * u1  # Mean U below kref
                vbar = vbar + rdelks * v1  # Mean V below kref
                # Mean ro below kref:
                roll = roll + rdelks * ro
                rdelks = (prsl - prsl[0, 0, 1]) * delks1
                bnv2bar = bnv2bar + bnv2 * rdelks

def find_low_level_properties(
    ubar: FloatFieldIJ,
    vbar: FloatFieldIJ,
    ipt: BoolFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        if ipt:
            wdir = atan(ubar, vbar) + gwdpscons.PI
            idir = Int(gwdpscons.FDIR * wdir) % gwdpscons.MDIR
            nwd = gwdpscons.NWDIR[idir]
            oa[i, j] = (1 - 2 * int((nwd - 1) / 4)) * oa4[i, j][nwd - 1 % 4]
            clx[i, j] = clx4[i, j][nwd - 1 % 4]


def gwps_py(
    ix: int,
    iy: int,
    km: int,
    a,
    b,
    c,
    u1,
    v1,
    t1,
    q1,
    kpbl,
    prsi,
    delp,
    prsl,
    prslk,
    phii,
    phil,
    deltim,
    kdt,
    hprime,
    oc,
    oa4,
    clx4,
    theta,
    sigma,
    gamma,
    elvmax,
    dusfc,
    dvsfc,
    nmtvr,
    cdmbgwd,
    p_crit,
    lonr,
):
    cdmb = Float(4.0 * 192.0 / float(lonr))
    if cdmbgwd[0] >= 0:
        cdmb = cdmb * cdmbgwd[0]
    kmm1 = km - 1
    kmm2 = km - 2
    lcap = km
    lcapp1 = lcap + 1
    db = np.zeros((ix, iy, km))
    ang = np.zeros((ix, iy, km))
    uds = np.zeros((ix, iy, km))
    rdxzb = np.zeros((ix, iy, km))
    npt = 0
    ipt = np.zeros((ix * iy))
    iwklm = np.zeros((ix * iy))
    idxzb = np.zeros((ix * iy))
    kreflm = np.zeros((ix * iy))
    wk = np.zeros((ix * iy))
    vtj = np.zeros((ix * iy, km))
    vtk = np.zeros((ix * iy, km))
    ro = np.zeros((ix * iy, km))
    bnv2lm = np.zeros((ix * iy, km))
    delks = np.zeros((ix * iy))
    delks1 = np.zeros((ix * iy))
    ubar = np.zeros((ix * iy))
    vbar = np.zeros((ix * iy))
    roll = np.zeros((ix * iy))
    pe = np.zeros((ix * iy))
    ek = np.zeros((ix * iy))
    bnv2bar = np.zeros((ix * iy))
    up = np.zeros((ix * iy))
    zbk = np.zeros((ix * iy))
    taup = np.zeros((ix * iy, km))
    ri_n = np.zeros((ix * iy))
    bnv2 = np.zeros((ix * iy))
    iwk = np.zeros((ix * iy))
    kref = np.zeros((ix * iy))
    oa = np.zeros((ix * iy))
    clx = np.zeros((ix * iy))
    xn = np.zeros((ix * iy))
    yn = np.zeros((ix * iy))
    taub = np.zeros((ix * iy))
    ulow = np.zeros((ix * iy))
    dtfac = np.zeros((ix * iy))
    icrilv = np.zeros((ix * iy))
    uloi = np.zeros((ix * iy))
    velco = np.zeros((ix * iy, km))
    kint = np.zeros((ix * iy))
    xlinv = np.zeros((ix * iy))
    scor = np.zeros((ix * iy))
    taud = np.zeros((ix * iy))

    for i in range(ix):
        for j in range(iy):
            dusfc[i, j] = 0.0
            dvsfc[i, j] = 0.0
            for k in range(km):
                db[i, j, k] = 0.0
                ang[i, j, k] = 0.0
                uds[i, j, k] = 0.0

    if nmtvr == 14:
        rdxzb[:] = 0.0
        ipt[:] = 0
        npt = 0
        for i in range(ix):
            for j in range(iy):
                if (elvmax[i, j] > gwdpscons.HMINMT) and (
                    hprime[i, j] > gwdpscons.gwdpscons.HPMIN
                ):
                    ipt[npt] = (i, j)
                    npt += 1
        if npt == 0:
            return

        for mm in range(npt):
            # --- iwklm is the level above the height of the of the mountain.
            # --- idxzb is the level of the dividing streamline.
            # INITIALIZE DIVIDING STREAMLINE (DS) CONTROL VECTOR

            iwklm[mm] = 2
            idxzb[mm] = 0
            kreflm[mm] = 0

        # > --- Subgrid Mountain Blocking Section
        #
        # ..............................
        # ..............................
        #
        #  (*j*)  11/03:  test upper limit on KMLL=km - 1
        #      then do not need hncrit -- test with large hncrit first.
        # KMLL  = km / 2 ! maximum mtnlm height : # of vertical levels / 2
        kmll = kmm1
        # --- No mtn should be as high as KMLL (so we do not have to start at
        # --- the top of the model but could do calc for all levels).

        for mm in range(npt):
            i, j = ipt[mm]
            elvmax[i, j] = min(
                elvmax[i, j] + gwdpscons.SIGFAC * hprime[i, j], gwdpscons.HNCRIT
            )

        for k in range(kmll):
            for mm in range(npt):
                i, j = ipt[mm]
                # --- interpolate to max mtn height for index, iwklm[mm] wk[gz]
                # --- ELVMAX is limited to hncrit because to hi res topo30 orog.
                pkp1log = phil[i, j, k + 1] / constants.GRAV
                pklog = phil[i, j, k] / constants.GRAV
                if (elvmax[i, j] <= pkp1log) and (elvmax[i, j] >= pklog):
                    wk[mm] = (
                        constants.GRAV
                        * elvmax[i, j]
                        / (phil[i, j, k + 1] - phil[i, j, k])
                    )
                    iwklm[mm] = max(iwklm[mm], k + 1)
                # ---        find at prsl levels large scale environment variables
                # ---        these cover all possible mtn max heights
                vtj[mm, k] = t1[i, j, k] * (1.0 + gwdpscons.FV * q1[i, j, k])
                vtk[mm, k] = vtj[mm, k] / prslk[i, j, k]

                # DENSITY Kg/M**3
                ro[mm, k] = gwdpscons.RDI * prsl[i, j, k] / vtj[mm, k]

        klevm1 = kmll - 1
        for k in range(klevm1):
            for mm in range(npt):
                (i, j) = ipt[mm]
                rdz = constants.GRAV / (phil[i, j, k + 1] - phil[i, j, k])
                # Brunt-Vaisala Frequency
                # > - Compute Brunt-Vaisala Frequency \f$N\f$.
                bnv2lm[mm, k] = (
                    (constants.GRAV + constants.GRAV)
                    * rdz
                    * (vtk[mm, k + 1] - vtk[mm, k])
                    / (vtk[mm, k + 1] + vtk[mm, k])
                )
                bnv2lm[mm, k] = max(bnv2lm[mm, k], gwdpscons.BNV2MIN)

        for mm in range(npt):
            (i, j) = ipt[mm]
            delks[mm] = 1.0 / (prsi[i, j, 1] - prsi[i, j, iwklm[mm]])
            delks1[mm] = 1.0 / (prsl[i, j, 1] - prsl[i, j, iwklm[mm]])
            ubar[mm] = 0.0
            vbar[mm] = 0.0
            roll[mm] = 0.0
            pe[mm] = 0.0
            ek[mm] = 0.0
            bnv2bar[mm] = (prsl[i, j, 1] - prsl[i, j, 2]) * delks1[mm] * bnv2lm[mm, 1]

        # --- find the dividing stream line height
        # --- starting from the level above the max mtn downward
        # --- iwklm[mm] is the k-index of mtn elvmax elevation
        # > - Find the dividing streamline height starting from the level above
        # the maximum mountain height and processing downward.
        for ktrial in range(kmll - 1, -1, -1):
            for mm in range(npt):
                if (ktrial < iwklm[mm]) and (kreflm[mm] == 0):
                    kreflm[mm] = ktrial

        # --- in the layer kreflm[mm] to 1 find PE (which needs N, ELVMAX)
        # ---  make averages, guess dividing stream (DS) line layer.
        # ---  This is not used in the first cut except for testing and
        # --- is the vert ave of quantities from the surface to mtn top.

        for mm in range(npt):
            for k in range(kreflm[mm]):
                (i, j) = ipt[mm]
                rdelks = delp[i, j, k] * delks[mm]
                ubar[mm] = ubar[mm] + rdelks * u1[i, j, k]  # trial Mean U below
                vbar[mm] = vbar[mm] + rdelks * v1[i, j, k]  # trial Mean V below
                roll[mm] = roll[mm] + rdelks * ro[mm, k]  # trial Mean ro below
                rdelks = (prsl[i, j, k] - prsl[i, j, k + 1]) * delks1[mm]
                bnv2bar[mm] = bnv2bar[mm] + bnv2lm[mm, k] * rdelks
                # --- these vert ave are for diags, testing and GWD to follow (*j*).

        # --- integrate to get PE in the trial layer.
        # --- Need the first layer where PE>EK - as soon as
        # --- idxzb is not 0 we have a hit and Zb is found.

        for mm in range(npt):
            (i, j) = ipt[mm]
            for k in range(iwklm[mm] - 1, -1, -1):
                phiang = np.atan2(v1[i, j, k], u1[i, j, k]) * gwdpscons.RAD_TO_DEG
                ang[mm, k] = theta[i, j] - phiang
                ang[mm, k] = ang[mm, k] - 180.0 if (ang[mm, k] > 90.0) else ang[mm, k]
                ang[mm, k] = ang[mm, k] + 180.0 if (ang[mm, k] < -90.0) else ang[mm, k]
                ang[mm, k] = ang[mm, k] * gwdpscons.DEG_TO_RAD

                # > - Compute wind speed UDS
                # # \f[
                # #     UDS=\max(\sqrt{u1^2+v1^2},minwnd)
                # # \f]
                # #  where \f$ minwnd=0.1 \f$, \f$u1\f$ and \f$v1\f$ are zonal and
                # #  meridional wind components of model layer wind.
                uds[mm, k] = max(
                    np.sqrt(u1[i, j, k] * u1[i, j, k] + v1[i, j, k] * v1[i, j, k]),
                    gwdpscons.MINWIND,
                )
                # --- Test to see if we found Zb previously
                if idxzb[mm] == 0:
                    pe[mm] = pe[mm] + bnv2lm[mm, k] * (
                        constants.GRAV * elvmax[i, j] - phil[i, j, k]
                    ) * (phii[i, j, k + 1] - phii[i, j, k]) / (
                        constants.GRAV * constants.GRAVG
                    )

                    # --- KE
                    # --- Wind projected on the line perpendicular
                    #         to mtn range, U(Zb(K)).
                    # --- kenetic energy is at the layer Zb
                    # --- THETA ranges from -+90deg |_
                    #         to the mtn "largest topo variations"
                    up[mm] = uds[mm, k] * np.cos(ang[mm, k])
                    ek[mm] = 0.5 * up[mm] * up[mm]

                    # --- Dividing Stream lime  is found when PE =exceeds EK.
                    if pe[mm] >= ek[mm]:
                        idxzb[mm] = k
                        rdxzb[i, j] = Float(k)
                # --- Then mtn blocked flow is between Zb=k(idxzb[mm]) and surface

                # > - The dividing streamline height (idxzb), of a subgrid scale
                # #  obstable, is found by comparing the potential (PE) and kinetic
                # #  energies (EK) of the upstream large scale wind and subgrid scale
                # #  air parcel movements. the dividing streamline is found when
                # #  \f$PE\geq EK\f$. Mountain-blocked flow is defined to exist between
                # #  the surface and the dividing streamline height (\f$h_d\f$), which
                # #  can be found by solving an integral equation for \f$h_d\f$:
                # # \f[
                # #  \frac{U^{2}(h_{d})}{2}=\int_{h_{d}}^{H} N^{2}(z)(H-z)dz
                # # \f]
                # #  where \f$H\f$ is the maximum subgrid scale elevation within the
                # #  grid box of actual orography, \f$h\f$, obtained from the GTOPO30
                # #  dataset from the U.S. Geological Survey.

        for mm in range(npt):
            (i, j) = ipt[mm]
            # --- Calc if N constant in layers (Zb guess) - a diagnostic only.
            zbk[mm] = (
                elvmax[i, j]
                - np.sqrt(ubar[mm] * ubar[mm] + vbar[mm] * vbar[mm]) / bnv2bar[mm]
            )

        # --- The drag for mtn blocked flow
        for mm in range(npt):
            (i, j) = ipt[mm]
            zlen = 0.0
            if idxzb[mm] > 0:
                for k in range(idxzb[mm] - 1, -1, -1):
                    if phil[i, j, idxzb[mm]] > phil[i, j, k]:
                        # > - Calculate \f$ZLEN\f$, which sums up a number of
                        # #  contributions of elliptic obstables.
                        # # \f[
                        # #     ZLEN=\sqrt{[\frac{h_{d}-z}{z+h'}]}
                        # # \f]
                        # #  where \f$z\f$ is the height, \f$h'\f$ is the orographic
                        # #  standard deviation (HPRIME).
                        ZLEN = np.sqrt(
                            (phil[i, j, idxzb[mm]] - phil[i, j, k])
                            / (phil[i, j, k] + constants.GRAV * hprime[i, j])
                        )
                        # --- lm eq 14:
                        # > - Calculate the drag coefficient to vary with the aspect
                        # #  ratio of the obstable as seen by the incident flow
                        # #  (see eq.14 in Lott and Miller (1997)
                        # #  \cite lott_and_miller_1997)
                        # # \f[
                        # #  R=\frac{\cos^{2}\psi+\gamma\sin^{2}\psi}
                        # #      {\gamma\cos^{2}\psi+\sin^{2}\psi}
                        # # \f]
                        # #  where \f$\psi\f$, which is derived from THETA, is the
                        # #  angle between the incident flow direction and the normal
                        # #  ridge direcion. \f$\gamma\f$ is the orographic
                        # #  anisotropy (GAMMA).
                        r = (
                            np.cos(ang[mm, k]) ** 2
                            + gamma[i, j] * np.sin(ang[mm, k]) ** 2
                        )
                        if abs(r) < 1.0e-20:
                            db[mm, k] = 0.0
                        else:
                            r = (
                                gamma[i, j] * np.cos(ang[mm, k]) ** 2
                                + np.sin(ang[mm, k]) ** 2
                            ) / r
                            # --- (negitive of db -- see sign at tendency)
                            # > - In each model layer below the dividing streamlines, a
                            # #  drag from the blocked flow is exerted by the obstacle
                            # #  on the large scale flow. The drag per unit area and
                            # #  per unit height is written (eq.15 in Lott and
                            # #  Miller (1997) \cite lott_and_miller_1997):
                            # # \f[
                            # #  D_{b}(z)=-C_{d}\max(2-\frac{1}{R},0)\rho\frac{\sigma}
                            # #      {2h'}ZLEN\max(\cos\psi,\gamma\sin\psi)\frac{UDS}{2}
                            # # \f]
                            # #  where \f$C_{d}\f$ is a specified constant,
                            # #  \f$\sigma\f$ is the orographic slope.

                            dbtmp = (
                                0.25
                                * cdmb
                                * max(2.0 - r, 0.0)
                                * sigma[i, j]
                                * max(
                                    np.cos(ang[mm, k]), gamma[i, j] * np.sin(ang[mm, k])
                                )
                                * zlen
                                / hprime[i, j]
                            )
                            db[mm, k] = dbtmp * uds[mm, k]

        # .............................
        # .............................
        #  end  mtn blocking section

    else:  # nmtvr != 14:
        # ----  for mb not present and  gwd (nmtvr .ne .14)
        ipt = 0
        npt = 0
        for i in range(ix):
            for j in range(iy):
                if hprime[i, j] > gwdpscons.HPMIN:
                    ipt[npt] = i
                    npt += 1
        if npt == 0:
            return  # No gwd/mb calculation done!

        for m in range(npt):
            idxzb[mm] = 0
            rdxzb[mm] = 0.0

    # .............................
    # .............................

    # > --- Orographic Gravity Wave Drag Section
    kmpbl = km / 2  # maximum pbl height : # of vertical levels / 2

    #  Scale cleff between IM=384*2 and 192*2 for T126/T170 and T62
    if lonr > 0:
        # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/384.0)  # this is inverse of CLEFF!
        # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
        # cleff = 0.5E-5 * np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
        # cleff = 1.0E-5 * np.sqrt(FLOAT(lonr)/192)/float(lonr/192)
        # cleff = 1.0E-5 / np.sqrt(FLOAT(lonr)/192.0)  # this is inverse of CLEFF!
        cleff = 0.5e-5 / np.sqrt(float(lonr) / 192.0)  # this is inverse of CLEFF!
        # hmhj for ndsl
        # jw    cleff = 0.1E-5 / np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
        #       cleff = 2.0E-5 * np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
        #       cleff = 2.5E-5 * np.sqrt(FLOAT(lonr)/192.0) !  this is inverse of CLEFF!
    cleff = cleff * cdmbgwd[1] if cdmbgwd[1] >= 0.0 else cleff

    for k in range(km):
        for mm in range(npt):
            (i, j) = ipt[mm]
            vtj[mm, k] = t1[i, j, k] * (1.0 + gwdpscons.FV * q1[i, j, k])
            vtk[mm, k] = vtj[mm, k] / prslk[i, j, k]
            ro[mm, k] = gwdpscons.RDI * prsl[i, j, k] / vtj[mm, k]  # DENSITY TONS/M**3
            taup[mm, k] = 0.0

    for k in range(kmm1):
        for mm in range(npt):
            (i, j) = ipt[mm]
            ti = 2.0 / (t1[i, j, k] + t1[i, j, k + 1])
            tem = ti / (prsl[i, j, k] - prsl[i, j, k + 1])
            rdz = constants.GRAV / (phil[i, j, k + 1] - phil[i, j, k])
            tem1 = u1[i, j, k] - u1[i, j, k + 1]
            tem2 = v1[i, j, k] - v1[i, j, k + 1]
            dw2 = tem1 * tem1 + tem2 * tem2
            shr2 = max(dw2, gwdpscons.DW2MIN) * rdz * rdz
            bvf2 = (
                constants.GRAV
                * (gwdpscons.GOCP + rdz * (vtj[mm, k + 1] - vtj[mm, k]))
                * ti
            )
            ri_n[mm, k] = max(bvf2 / shr2, gwdpscons.RIMIN)  # Richardson number
            # Brunt-Vaisala Frequency
            # tem       = GR2 * (PRSL[i, j, k]+PRSL[i, j, k+1]) * tem
            # bnv2[mm, k] = tem * (VTK[mm, k+1]-VTK[mm, k])/(VTK[mm, k+1]+VTK[mm, k])
            bnv2[mm, k] = (
                (constants.GRAV + constants.GRAV)
                * rdz
                * (vtk[mm, k + 1] - vtk[mm, k])
                / (vtk[mm, k + 1] + vtk[mm, k])
            )
            bnv2[mm, k] = max(bnv2[mm, k], gwdpscons.BNV2MIN)

    # Finding the first interface index above 50 hPa level
    for mm in range(npt):
        iwk[mm] = 2
    for k in range(2, kmpbl):
        for mm in range(npt):
            (i, j) = ipt[mm]
            tem = prsi[i, j, 0] - prsi[i, j, k]
            iwk[mm] = k if tem < gwdpscons.DPMIN else iwk[mm]

    # > - Calculate the reference level index: kref=max(2,KPBL+1). where
    # # KPBL is the index for the PBL top layer.
    kbps = 1
    kmps = km
    for mm in range(npt):
        (i, j) = ipt[mm]
        kref[mm] = max(iwk[mm], kpbl[i, j] + 1)  # reference level
        delks[mm] = 1.0 / (prsi[i, j, 0] - prsi[i, j, kref[mm]])
        delks1[mm] = 1.0 / (prsl[i, j, 0] - prsl[i, j, kref[mm]])
        ubar[mm] = 0.0
        vbar[mm] = 0.0
        roll[mm] = 0.0
        kbps = max(kbps, kref[mm])
        kmps = min(kmps, kref[mm])

        bnv2bar[mm] = (prsl[i, j, 0] - prsl[i, j, 1]) * delks1[mm] * bnv2[mm, 0]

    kbpsp1 = kbps + 1
    kbpsm1 = kbps - 1
    for k in range(kbps):
        for mm in range(npt):
            if k < kref[mm]:
                (i, j) = ipt[mm]
                rdelks = delp[i, j, k] * delks[mm]
                ubar[mm] = ubar[mm] + rdelks * u1[i, j, k]  # Mean U below kref
                vbar[mm] = vbar[mm] + rdelks * v1[i, j, k]  # Mean V below kref

                roll[mm] = roll[mm] + rdelks * ro[mm, k]  # Mean ro below kref
                rdelks = (prsl[i, j, k] - prsl[i, j, k + 1]) * delks1[mm]
                bnv2bar[mm] = bnv2bar[mm] + bnv2[mm, k] * rdelks

    # FIGURE OUT LOW-LEVEL HORIZONTAL WIND DIRECTION AND FIND 'oa'
    # NWD  1   2   3   4   5   6   7   8
    #  WD  W   S  SW  NW   E   N  NE  SE

    # > - Calculate low-level horizontal wind direction, the derived
    # # orographic asymmetry parameter (oa), and the derived Lx (CLX).
    for mm in range(npt):
        (i, j) = ipt[mm]
        wdir = np.atan2(ubar[mm], vbar[mm]) + gwdpscons.PI
        idir = int(gwdpscons.FDIR * wdir) % gwdpscons.MDIR
        nwd = gwdpscons.NWDIR(idir)
        oa[mm] = (1 - 2 * int((nwd - 1) / 4)) * oa4[i, j, nwd - 1 % 4]
        clx[mm] = clx4[i, j, nwd - 1 % 4]

    # -----XN,YN            "LOW-LEVEL" WIND PROJECTIONS IN ZONAL & MERIDIONAL
    #                           DIRECTIONS
    # -----ULOW             "LOW-LEVEL" WIND MAGNITUDE -        (= U)
    # -----bnv2             bnv2 = N**2
    # -----TAUB             BASE MOMENTUM FLUX
    # -----= -(ro * U**3/(N*XL)*GF(FR) FOR N**2 > 0
    # -----= 0.                        FOR N**2 < 0
    # -----FR               FROUDE    =   N*HPRIME / U
    # -----constants.GRAV                GMAX*FR**2/(FR**2+CG/OC)

    # -----INITIALIZE SOME ARRAYS

    for mm in range(npt):
        xn[mm] = 0.0
        yn[mm] = 0.0
        taub[mm] = 0.0
        ulow[mm] = 0.0
        dtfac[mm] = 1.0
        icrilv[mm] = False  # INITIALIZE CRITICAL LEVEL CONTROL VECTOR

        # ----COMPUTE THE "LOW LEVEL" WIND MAGNITUDE (M/S)
        ulow[mm] = max(np.sqrt(ubar[mm] * ubar[mm] + vbar[mm] * vbar[mm]), 1.0)
        uloi[mm] = 1.0 / ulow[mm]

    for k in range(kmm1):
        for mm in range(npt):
            (i, j) = ipt[mm]
            velco[mm, k] = 0.5 * (
                (u1[i, j, k] + u1[i, j, k + 1]) * ubar[mm]
                + (v1[i, j, k] + v1[i, j, k + 1]) * vbar[mm]
            )
            velco[mm, k] = velco[mm, k] * uloi[mm]

    #   find the interface level of the projected wind where
    #   low levels & upper levels meet above pbl
    #
    #     do i=1,npt
    #       kint(i) = km
    #     enddo
    #     do k = 1,kmm1
    #       do i = 1,npt
    #         if (K > kref[mm]) THEN
    #           if(velco[mm, k] < veleps and (prsi(i) .eq. km) then
    #             kint(i) = k+1
    #           endif
    #         endif
    #       enddo
    #     enddo
    #  WARNING  KINT = KREF !!!!!!!!!
    for mm in range(npt):
        kint[mm] = kref[mm]

    for mm in range(npt):
        (i, j) = ipt[mm]
        bnv = np.sqrt(bnv2bar[mm])
        fr = bnv * uloi[mm] * min(hprime[i, j], gwdpscons.HPMAX)
        fr = min(fr, gwdpscons.FRMAX)
        xn[mm] = ubar[mm] * uloi[mm]
        yn[mm] = vbar[mm] * uloi[mm]

        #     Compute the base level stress and store it in TAUB
        #     CALCULATE ENHANCEMENT FACTOR, NUMBER OF MOUNTAINS & ASPECT
        #     RATIO CONST. USE SIMPLIFIED RELATIONSHIP BETWEEN STANDARD
        #     DEVIATION & CRITICAL HGT
        #
        # > - Calculate enhancement factor (E),number of mountans (m') and
        # #  aspect ratio constant.
        # # \n As in eq.(4.9),(4.10),(4.11) in Kim and Arakawa (1995)
        # #  \cite kim_and_arakawa_1995, we define m' and E in such a way that they
        # #  depend on the geometry and location of the subgrid-scale orography
        # #  through oa and the nonlinearity of flow above the orography through
        # #  Fr. OC, which is the orographic convexity, and statistically
        # #  determine how protruded (sharp) the subgrid-scale orography is, is
        # #  included in the saturation flux constants.GRAV' in such a way that
        # #  constants.GRAV' is proportional to OC. The forms of E,m' and
        # #  constants.GRAV' are:
        # # \f[
        # #   E(oa,F_{r_{0}})=(oa+2)^{\delta}
        # # \f]
        # # \f[
        # #   \delta=C_{E}F_{r_{0}}/F_{r_{c}}
        # # \f]
        # # \f[
        # #   m'(oa,CLX)=C_{m}\triangle x(1+CLX)^{oa+1}
        # # \f]
        # # \f[
        # #   constants.GRAV'(OC,F_{r_{0}})=\frac{F_{r_{0}}^2}{F_{r_{0}}^2+a^{2}}
        # # \f]
        # # \f[
        # #  a^{2}=C_{constants.GRAV}OC^{-1}
        # # \f]
        # #  where \f$F_{r_{c}}(=1)\f$ is the critical Froude number,
        # #  \f$F_{r_{0}}\f$ is the Froude number. \f$C_{E}\f$,\f$C_{m}\f$,
        # #  \f$C_{constants.GRAV}\f$ are constants.

        # > - Calculate the reference-level drag \f$\tau_{0}\f$ (eq.(4.8) in
        # #  Kim and Arakawa (1995) \cite kim_and_arakawa_1995):
        # # \f[
        # #  \tau_0=E\frac{m'}{\triangle x}\frac{\rho_{0}U_0^3}{N_{0}}constants.GRAV'
        # # \f]
        # #  where \f$E\f$,\f$m'\f$, and \f$constants.GRAV'\f$ are the enhancement
        # #  factor, "the number of mountains", and the flux function defined above,
        # #  respectively.

        efact = (oa[mm] + 2.0) ** (gwdpscons.CEOFRC * fr)
        efact = min(max(efact, gwdpscons.EFMIN), gwdpscons.EFMAX)

        coefm = (1.0 + clx[mm]) ** (oa[mm] + 1.0)

        xlinv[mm] = coefm * cleff

        tem = fr * fr * oc[i, j]
        # constants.GRAV/N0
        gfobnv = gwdpscons.GMAX * tem / ((tem + gwdpscons.CG) * bnv)

        taub[mm] = (
            xlinv[mm] * roll[mm] * ulow[mm] * ulow[mm] * ulow[mm] * (gfobnv * efact)
        )  # BASE FLUX Tau0

        # tem = min(HPRIME[mm],hpmax)
        # taub[mm] = xlinv[mm] * roll[mm] * ulow[mm] * BNV * tem * tem

        k = max(1, kref[mm] - 1)
        tem = max(velco[mm, k] * velco[mm, k], 0.1)
        scor[mm] = bnv2[mm, k] / tem  # Scorer parameter below ref level

    # ----SET UP BOTTOM VALUES OF STRESS
    for k in range(kbps):
        for mm in range(npt):
            if k <= kref[mm]:
                taup[mm, k] = taub[mm]

        #   Now compute vertical structure of the stress.
        for k in range(kmps, kmm1):  # Vertical Level K Loop!
            kp1 = k + 1
            for mm in range(npt):
                # -UNSTABLE LAYER if RI < RIC
                # -UNSTABLE LAYER if UPPER AIR VEL COMP ALONG SURF VEL <=0 (CRIT LAY)
                # - AT (U-C)=0. CRIT LAYER EXISTS AND BIT VECTOR SHOULD BE SET (<=)
                if k >= kref[mm]:
                    icrilv[mm] = (
                        icrilv[mm]
                        or (ri_n[mm, k] < gwdpscons.RIC)
                        or (velco[mm, k] <= 0.0)
                    )
        # > - Compute the drag above the reference level (\f$k\geq kref\f$):
        # #  - Calculate the ratio of the Scorer parameter (\f$R_{scor}\f$).
        # # \n From a series of experiments, Kim and Arakawa (1995)
        # # \cite kim_and_arakawa_1995 found that the magnitude of drag divergence
        # # tends to be underestimated by the revised scheme in low-level
        # # downstream regions with wave breaking. Therefore, at low levels when
        # # oa > 0 (i.e., in the "downstream" region) the saturation hypothesis
        # # is replaced by the following formula based on the ratio of the
        # # the Scorer parameter:
        # #\f[
        # # R_{scor}=\min \left[\frac{\tau_i}{\tau_{i+1}},1\right]
        # #\f]
        for mm in range(npt):
            if k >= kref[mm]:
                if not icrilv[mm] and taup[mm, k] > 0.0:
                    temv = 1.0 / max(velco[mm, k], 0.01)
                    # if (oa[mm] > 0. and  prsi(ipt(i),KP1)>RLOLEV) THEN
                    if oa[mm] > 0.0 and kp1 < kint[mm]:
                        scork = bnv2[mm, k] * temv * temv
                        rscor = min(1.0, scork / scor[mm])
                        scor[mm] = scork
                    else:
                        rscor = 1.0

                    # >  - The drag above the reference level is expressed as:
                    # #\f[
                    # # \tau=\frac{m'}{\triangle x}\rho NUh_d^2
                    # #\f]
                    # # where \f$h_{d}\f$ is the displacement wave amplitude. In the
                    # # absence of wave breaking, the displacement amplitude for the
                    # # \f$i^{th}\f$ layer can be expressed using the drag for the
                    # # layer immediately below. Thus, assuming \f$\tau_i=\tau_{i+1}\f$,
                    # # we can get:
                    # #\f[
                    # # h_{d_i}^2=\frac{\triangle x}{m'}\frac{\tau_{i+1}}
                    # #     {\rho_{i}N_{i}U_{i}}
                    # #\f]

                    brvf = np.sqrt(bnv2[mm, k])  # Brunt-Vaisala Frequency
                    # tem1 = xlinv[mm]*(ro[mm, kp1]+ro[mm, k])*brvf*velco[mm, k]*0.5
                    tem1 = (
                        xlinv[mm]
                        * (ro[mm, kp1] + ro[mm, k])
                        * brvf
                        * 0.5
                        * max(velco[mm, k], 0.01)
                    )
                    hd = np.sqrt(taup[mm, k] / tem1)
                    fro = brvf * hd * temv

                    # rim is the  MINIMUM-RICHARDSON NUMBER BY SHUTTS (1985)

                    # > - The minimum Richardson number (\f$Ri_{m}\f$) or local
                    # # wave-modified Richardson number, which determines the onset of
                    # # wave breaking, is expressed in terms of \f$R_{i}\f$ and
                    # # \f$F_{r_{d}}=Nh_{d}/U\f$:
                    # #\f[
                    # # Ri_{m}=\frac{Ri(1-Fr_{d})}{(1+\sqrt{Ri}\cdot Fr_{d})^{2}}
                    # #\f]
                    # # see eq.(4.6) in
                    # # Kim and Arakawa (1995) \cite kim_and_arakawa_1995.

                    tem2 = np.sqrt(ri_n[mm, k])
                    tem = 1.0 + tem2 * fro
                    rim = ri_n[mm, k] * (1.0 - fro) / (tem * tem)

                    # CHECK STABILITY TO EMPLOY THE 'SATURATION HYPOTHESIS'
                    # OF LINDZEN (1981) EXCEPT AT TROPOSPHERIC DOWNSTREAM REGIONS

                    # >  - Check stability to employ the 'saturation hypothesis' of
                    # # Lindzen (1981) \cite lindzen_1981 except at tropospheric
                    # # downstream regions.
                    # # \n Wave breaking occurs when \f$Ri_{m}<Ri_{c}=0.25\f$. Then
                    # # Lindzen's wave saturation hypothesis resets the displacement
                    # # amplitude \f$h_{d}\f$ to that corresponding to
                    # # \f$Ri_{m}=0.25\f$, we obtain the critical \f$h_{d}\f$
                    # # (or \f$h_{c}\f$) expressed in
                    # # terms of the mean values of \f$U\f$, \f$N\f$, and \f$Ri\f$ (
                    # # eq.(4.7) in Kim and Arakawa (1995) \cite kim_and_arakawa_1995):
                    # #\f[
                    # # h_{c}=\frac{U}{N}\left\{2(2+\frac{1}
                    # #     {\sqrt{Ri}})^{1/2}-(2+\frac{1}{\sqrt{Ri}})\right\}
                    # #\f]
                    # # if \f$Ri_{m}\leq Ri_{c}\f$, obtain \f$\tau\f$ from the drag
                    # # above the reference level by using \f$h_{c}\f$ computed above;
                    # # otherwise \f$\tau\f$ is unchanged (note: scaled by the ratio of
                    # # the Scorer paramter).

                    # if (RIM <= RIC and (
                    #     OA[mm] <= 0. .OR. prsi(ipt[mm],KP1)<=RLOLEV)
                    # ) THEN
                    if rim <= gwdpscons.RIC and (oa[mm] <= 0.0 or kp1 >= kint[mm]):
                        temc = 2.0 + 1.0 / tem2
                        hd = velco[mm, k] * (2.0 * np.sqrt(temc) - temc) / brvf
                        taup[mm, kp1] = tem1 * hd * hd
                    else:
                        taup[mm, kp1] = taup[mm, k] * rscor
                    taup[mm, kp1] = min(taup[mm, kp1], taup[mm, k])

    if lcap <= km:
        for klcap in range(lcapp1, km + 1):
            for mm in range(npt):
                (i, j) = ipt[mm]
                sira = prsi[i, j, klcap] / prsi[i, j, lcap]
                taup[mm, klcap] = sira * taup[mm, lcap]

    #  SJL: linear decay above p_crit, becoming constant at 1 mb
    #  Angular momentum conservation is ensured, except the top leakage
    # ----------------------- SJL mod ------------------------------
    if p_crit > 1.0e-10:
        for mm in range(npt):
            (i, j) = ipt[mm]
            for k in range(km / 2, km + 1):
                if prsi[i, j, k] < p_crit:  # scale it to zero @ top
                    taup[mm, k] = (
                        taup[mm, k]
                        * (prsi[i, j, k] - prsi[i, j, km + 1])
                        / (p_crit - prsi[i, j, km + 1])
                    )
                elif prsi[i, j, k] < 1.0e2:
                    taup[mm, k] = taup[mm, k - 1]  # constant stress-> zero Drag

    # ----------------------- SJL mod ------------------------------

    #     Calculate - (g/p*)*d(tau)/d(sigma) and Decel terms dtaux, dtauy
    for k in range(km):
        for mm in range(npt):
            (i, j) = ipt[mm]
            taud[mm, k] = (
                constants.GRAV * (taup[mm, k + 1] - taup[mm, k]) / delp[i, j, k]
            )

    # ------LIMIT DE-ACCELERATION (MOMENTUM DEPOSITION ) AT TOP TO 1/2 VALUE
    # ------THE IDEA IS SOME STUFF MUST GO OUT THE 'TOP'
    if p_crit <= 1.0e-10:
        for klcap in range(lcap, km):
            for mm in range(npt):
                taud[mm, klcap] = taud[mm, klcap] * gwdpscons.FACTOP

    # ------IF THE GRAVITY WAVE DRAG WOULD FORCE A CRITICAL LINE IN THE
    # ------LAYERS BELOW SIGMA=RLOLEV DURING THE NEXT deltim TIMESTEP,
    # ------THEN ONLY APPLY DRAG UNTIL THAT CRITICAL LINE IS REACHED.
    for k in range(kmm1):
        for mm in range(npt):
            (i, j) = ipt[mm]
            if (k < kref[mm]) and (prsi[i, j, k] >= gwdpscons.RLOLEV):
                if taud[mm, k] != 0.0:
                    tem = deltim * taud[mm, k]
                    dtfac[mm] = min(dtfac[mm], np.abs(velco[mm, k] / tem))

    # > - Calculate outputs: A, B, dusfc, dvsfc (see parameter description).
    # #  - Below the dividing streamline height (k < idxzb), mountain
    # #    blocking(\f$D_{b}\f$) is applied.
    # #  - Otherwise (k>= idxzb), orographic GWD (\f$\tau\f$) is applied.
    for k in range(km):
        for mm in range(npt):
            (i, j) = ipt[mm]
            taud[mm, k] = taud[mm, k] * dtfac[mm]
            dtaux = taud[mm, k] * xn[mm]
            dtauy = taud[mm, k] * yn[mm]
            eng0 = 0.5 * (u1[i, j, k] * u1[i, j, k] + v1[i, j, k] * v1[i, j, k])
            # ---  lm mb (*j*)  changes overwrite GWD
            if (k < idxzb[mm]) and (idxzb[mm] != 0):
                dbim = db[mm, k] / (1.0 + db[mm, k] * deltim)
                a[i, j, k] = -dbim * v1[i, j, k] + a[i, j, k]
                b[i, j, k] = -dbim * u1[i, j, k] + b[i, j, k]
                eng1 = eng0 * (1.0 - dbim * deltim) * (1.0 - dbim * deltim)
                dusfc[i, j] = dusfc[i, j] - dbim * u1[i, j, k] * delp[i, j, k]
                dvsfc[i, j] = dvsfc[i, j] - dbim * v1[i, j, k] * delp[i, j, k]
            else:
                a[i, j, k] = dtauy + a[i, j, k]
                b[i, j, k] = dtaux + b[i, j, k]
                eng1 = 0.5 * (
                    (u1[i, j, k] + dtaux * deltim) * (u1[i, j, k] + dtaux * deltim)
                    + (v1[i, j, k] + dtauy * deltim) * (v1[i, j, k] + dtauy * deltim)
                )
                dusfc[i, j] = dusfc[i, j] + dtaux * delp[i, j, k]
                dvsfc[i, j] = dvsfc[i, j] + dtauy * delp[i, j, k]
            c[i, j, k] = c[i, j, k] + max(eng0 - eng1, 0.0) / constants.CP_AIR / deltim

    tem = -1.0 / constants.GRAV
    for mm in range(npt):
        (i, j) = ipt[mm]
        # tem    = (-1.E3/G)
        dusfc[i, j] = tem * dusfc[i, j]
        dvsfc[i, j] = tem * dvsfc[i, j]

    return


class OrographicGravityWaveDrag:
    """
    Fortran name is gwdps
    """

    def __init__(
        self,
        stencil_factory: StencilFactory,
        quantity_factory: QuantityFactory,
        config: OrographicGravityWaveDragConfig,
    ):
        self._cdmb = Float(4.0 * 192.0 / float(config.lonr))
        if config.cdmbgwd[0] >= 0:
            self._cdmb = self._cdmb * config.cdmbgwd[0]

        idx = stencil_factory.grid_indexing

        def make_quantity():
            return quantity_factory.zeros(
                [X_DIM, Y_DIM, Z_DIM],
                units="unknown",
                dtype=Float,
            )

        def make_quantity_2D(type):
            return quantity_factory.zeros([X_DIM, Y_DIM], units="unknown", dtype=type)

        # Allocate internal variables:
        nz = idx.domain[2]

        self._lcap = make_quantity_2D(Int)
        self._lcapp1 = make_quantity_2D(Int)
        self._lcap[:] = nz
        self._lcapp1[:] = nz + 1
        pass

    def __call__(self, state: OrographicGravityWaveDragState):
        pass
