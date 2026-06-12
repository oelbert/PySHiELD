import numpy as np

import ndsl.constants as constants
import pyshield.stencils.gwdps.constants as gwdpscons
from ndsl.dsl.typing import Float


def gwps_reshape(
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
    ipt = np.zeros((ix, iy))
    iwklm = np.zeros((ix, iy))
    idxzb = np.zeros((ix, iy))
    kreflm = np.zeros((ix, iy))
    wk = np.zeros((ix, iy))
    vtj = np.zeros((ix, iy, km))
    vtk = np.zeros((ix, iy, km))
    ro = np.zeros((ix, iy, km))
    bnv2lm = np.zeros((ix, iy, km))
    delks = np.zeros((ix, iy))
    delks1 = np.zeros((ix, iy))
    ubar = np.zeros((ix, iy))
    vbar = np.zeros((ix, iy))
    roll = np.zeros((ix, iy))
    pe = np.zeros((ix, iy))
    ek = np.zeros((ix, iy))
    bnv2bar = np.zeros((ix, iy))
    up = np.zeros((ix, iy))
    zbk = np.zeros((ix, iy))
    taup = np.zeros((ix, iy, km))
    ri_n = np.zeros((ix, iy))
    bnv2 = np.zeros((ix, iy))
    iwk = np.zeros((ix, iy))
    kref = np.zeros((ix, iy))
    oa = np.zeros((ix, iy))
    clx = np.zeros((ix, iy))
    xn = np.zeros((ix, iy))
    yn = np.zeros((ix, iy))
    taub = np.zeros((ix, iy))
    ulow = np.zeros((ix, iy))
    dtfac = np.zeros((ix, iy))
    icrilv = np.zeros((ix, iy))
    uloi = np.zeros((ix, iy))
    velco = np.zeros((ix, iy, km))
    kint = np.zeros((ix, iy))
    xlinv = np.zeros((ix, iy))
    scor = np.zeros((ix, iy))
    taud = np.zeros((ix, iy))

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
                    ipt[i, j] = True
                    npt += 1
                else:
                    ipt[i, j] = False
        if npt == 0:
            return

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                # --- iwklm is the level above the height of the of the mountain.
                # --- idxzb is the level of the dividing streamline.
                # INITIALIZE DIVIDING STREAMLINE (DS) CONTROL VECTOR

                iwklm[i, j] = 2
                idxzb[i, j] = 0
                kreflm[i, j] = 0

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

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                elvmax[i, j] = min(
                    elvmax[i, j] + gwdpscons.SIGFAC * hprime[i, j], gwdpscons.HNCRIT
                )

        for k in range(kmll):
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    # --- interpolate to max mtn height for index, iwklm[mm] wk[gz]
                    # --- ELVMAX is limited to hncrit because to hi res topo30 orog.
                    pkp1log = phil[i, j, k + 1] / constants.GRAV
                    pklog = phil[i, j, k] / constants.GRAV
                    if (elvmax[i, j] <= pkp1log) and (elvmax[i, j] >= pklog):
                        wk[i, j] = (
                            constants.GRAV
                            * elvmax[i, j]
                            / (phil[i, j, k + 1] - phil[i, j, k])
                        )
                        iwklm[i, j] = max(iwklm[i, j], k + 1)
                    # ---        find at prsl levels large scale environment variables
                    # ---        these cover all possible mtn max heights
                    vtj[i, j, k] = t1[i, j, k] * (1.0 + gwdpscons.FV * q1[i, j, k])
                    vtk[i, j, k] = vtj[i, j, k] / prslk[i, j, k]

                    # DENSITY Kg/M**3
                    ro[i, j, k] = gwdpscons.RDI * prsl[i, j, k] / vtj[i, j, k]

        klevm1 = kmll - 1
        for k in range(klevm1):
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    rdz = constants.GRAV / (phil[i, j, k + 1] - phil[i, j, k])
                    # Brunt-Vaisala Frequency
                    # > - Compute Brunt-Vaisala Frequency \f$N\f$.
                    bnv2lm[i, j, k] = (
                        (constants.GRAV + constants.GRAV)
                        * rdz
                        * (vtk[i, j, k + 1] - vtk[i, j, k])
                        / (vtk[i, j, k + 1] + vtk[i, j, k])
                    )
                    bnv2lm[i, j, k] = max(bnv2lm[i, j, k], gwdpscons.BNV2MIN)

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                delks[i, j] = 1.0 / (prsi[i, j, 1] - prsi[i, j, iwklm[i, j]])
                delks1[i, j] = 1.0 / (prsl[i, j, 1] - prsl[i, j, iwklm[i, j]])
                ubar[i, j] = 0.0
                vbar[i, j] = 0.0
                roll[i, j] = 0.0
                pe[i, j] = 0.0
                ek[i, j] = 0.0
                bnv2bar[i, j] = (
                    (prsl[i, j, 1] - prsl[i, j, 2]) * delks1[i, j] * bnv2lm[i, j, 1]
                )

        # --- find the dividing stream line height
        # --- starting from the level above the max mtn downward
        # --- iwklm[mm] is the k-index of mtn elvmax elevation
        # > - Find the dividing streamline height starting from the level above
        # the maximum mountain height and processing downward.
        for ktrial in range(kmll - 1, -1, -1):
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    if (ktrial < iwklm[i, j]) and (kreflm[i, j] == 0):
                        kreflm[i, j] = ktrial

        # --- in the layer kreflm[mm] to 1 find PE (which needs N, ELVMAX)
        # ---  make averages, guess dividing stream (DS) line layer.
        # ---  This is not used in the first cut except for testing and
        # --- is the vert ave of quantities from the surface to mtn top.

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                for k in range(kreflm[i, j]):
                    rdelks = delp[i, j, k] * delks[i, j]
                    ubar[i, j] = ubar[i, j] + rdelks * u1[i, j, k]  # trial Mean U below
                    vbar[i, j] = vbar[i, j] + rdelks * v1[i, j, k]  # trial Mean V below
                    # trial Mean ro below:
                    roll[i, j] = roll[i, j] + rdelks * ro[i, j, k]
                    rdelks = (prsl[i, j, k] - prsl[i, j, k + 1]) * delks1[i, j]
                    bnv2bar[i, j] = bnv2bar[i, j] + bnv2lm[i, j, k] * rdelks
                    # --- these vert ave are for diags, testing and GWD to follow (*j*).

        # --- integrate to get PE in the trial layer.
        # --- Need the first layer where PE>EK - as soon as
        # --- idxzb is not 0 we have a hit and Zb is found.

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                for k in range(iwklm[i, j] - 1, -1, -1):
                    phiang = np.atan2(v1[i, j, k], u1[i, j, k]) * gwdpscons.RAD_TO_DEG
                    ang[i, j, k] = theta[i, j] - phiang
                    ang[i, j, k] = (
                        ang[i, j, k] - 180.0 if (ang[i, j, k] > 90.0) else ang[i, j, k]
                    )
                    ang[i, j, k] = (
                        ang[i, j, k] + 180.0 if (ang[i, j, k] < -90.0) else ang[i, j, k]
                    )
                    ang[i, j, k] = ang[i, j, k] * gwdpscons.DEG_TO_RAD

                    # > - Compute wind speed UDS
                    # # \f[
                    # #     UDS=\max(\sqrt{u1^2+v1^2},minwnd)
                    # # \f]
                    # #  where \f$ minwnd=0.1 \f$, \f$u1\f$ and \f$v1\f$ are zonal and
                    # #  meridional wind components of model layer wind.
                    uds[i, j, k] = max(
                        np.sqrt(u1[i, j, k] * u1[i, j, k] + v1[i, j, k] * v1[i, j, k]),
                        gwdpscons.MINWIND,
                    )
                    # --- Test to see if we found Zb previously
                    if idxzb[i, j] == 0:
                        pe[i, j] = pe[i, j] + bnv2lm[i, j, k] * (
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
                        up[i, j] = uds[i, j, k] * np.cos(ang[i, j, k])
                        ek[i, j] = 0.5 * up[i, j] * up[i, j]

                        # --- Dividing Stream lime  is found when PE =exceeds EK.
                        if pe[i, j] >= ek[i, j]:
                            idxzb[i, j] = k
                            rdxzb[i, j] = Float(k)
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

        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                # --- Calc if N constant in layers (Zb guess) - a diagnostic only.
                zbk[i, j] = (
                    elvmax[i, j]
                    - np.sqrt(ubar[i, j] * ubar[i, j] + vbar[i, j] * vbar[i, j])
                    / bnv2bar[i, j]
                )

        # --- The drag for mtn blocked flow
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                zlen = 0.0
                if idxzb[i, j] > 0:
                    for k in range(idxzb[i, j] - 1, -1, -1):
                        if phil[i, j, idxzb[i, j]] > phil[i, j, k]:
                            # > - Calculate \f$ZLEN\f$, which sums up a number of
                            # #  contributions of elliptic obstables.
                            # # \f[
                            # #     ZLEN=\sqrt{[\frac{h_{d}-z}{z+h'}]}
                            # # \f]
                            # #  where \f$z\f$ is the height, \f$h'\f$ is the orographic
                            # #  standard deviation (HPRIME).
                            ZLEN = np.sqrt(
                                (phil[i, j, idxzb[i, j]] - phil[i, j, k])
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
                            # #  angle between the incident flow direction and the
                            # #  normal ridge direcion. \f$\gamma\f$ is the orographic
                            # #  anisotropy (GAMMA).
                            r = (
                                np.cos(ang[i, j, k]) ** 2
                                + gamma[i, j] * np.sin(ang[i, j, k]) ** 2
                            )
                            if abs(r) < 1.0e-20:
                                db[i, j, k] = 0.0
                            else:
                                r = (
                                    gamma[i, j] * np.cos(ang[i, j, k]) ** 2
                                    + np.sin(ang[i, j, k]) ** 2
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
                                    * cdmb
                                    * max(2.0 - r, 0.0)
                                    * sigma[i, j]
                                    * max(
                                        np.cos(ang[i, j, k]),
                                        gamma[i, j] * np.sin(ang[i, j, k]),
                                    )
                                    * zlen
                                    / hprime[i, j]
                                )
                                db[i, j, k] = dbtmp * uds[i, j, k]

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
            idxzb[i, j] = 0
            rdxzb[i, j] = 0.0

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
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                vtj[i, j, k] = t1[i, j, k] * (1.0 + gwdpscons.FV * q1[i, j, k])
                vtk[i, j, k] = vtj[i, j, k] / prslk[i, j, k]
                # DENSITY TONS/M**3:
                ro[i, j, k] = gwdpscons.RDI * prsl[i, j, k] / vtj[i, j, k]
                taup[i, j, k] = 0.0

    for k in range(kmm1):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                ti = 2.0 / (t1[i, j, k] + t1[i, j, k + 1])
                tem = ti / (prsl[i, j, k] - prsl[i, j, k + 1])
                rdz = constants.GRAV / (phil[i, j, k + 1] - phil[i, j, k])
                tem1 = u1[i, j, k] - u1[i, j, k + 1]
                tem2 = v1[i, j, k] - v1[i, j, k + 1]
                dw2 = tem1 * tem1 + tem2 * tem2
                shr2 = max(dw2, gwdpscons.DW2MIN) * rdz * rdz
                bvf2 = (
                    constants.GRAV
                    * (gwdpscons.GOCP + rdz * (vtj[i, j, k + 1] - vtj[i, j, k]))
                    * ti
                )
                ri_n[i, j, k] = max(bvf2 / shr2, gwdpscons.RIMIN)  # Richardson number
                # Brunt-Vaisala Frequency
                # tem       = GR2 * (PRSL[i, j, k]+PRSL[i, j, k+1]) * tem
                # bnv2[i,j,k]=tem*(VTK[i,j,k+1]-VTK[i,j,k])/(VTK[i,j,k+1]+VTK[i,j,k])
                bnv2[i, j, k] = (
                    (constants.GRAV + constants.GRAV)
                    * rdz
                    * (vtk[i, j, k + 1] - vtk[i, j, k])
                    / (vtk[i, j, k + 1] + vtk[i, j, k])
                )
                bnv2[i, j, k] = max(bnv2[i, j, k], gwdpscons.BNV2MIN)

    # Finding the first interface index above 50 hPa level
    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            iwk[i, j] = 2
    for k in range(2, kmpbl):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                tem = prsi[i, j, 0] - prsi[i, j, k]
                iwk[i, j] = k if tem < gwdpscons.DPMIN else iwk[i, j]

    # > - Calculate the reference level index: kref=max(2,KPBL+1). where
    # # KPBL is the index for the PBL top layer.
    kbps = 1
    kmps = km
    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            kref[i, j] = max(iwk[i, j], kpbl[i, j] + 1)  # reference level
            delks[i, j] = 1.0 / (prsi[i, j, 0] - prsi[i, j, kref[i, j]])
            delks1[i, j] = 1.0 / (prsl[i, j, 0] - prsl[i, j, kref[i, j]])
            ubar[i, j] = 0.0
            vbar[i, j] = 0.0
            roll[i, j] = 0.0
            kbps = max(kbps, kref[i, j])
            kmps = min(kmps, kref[i, j])

            bnv2bar[i, j] = (
                (prsl[i, j, 0] - prsl[i, j, 1]) * delks1[i, j] * bnv2[i, j, 0]
            )

    kbpsp1 = kbps + 1
    kbpsm1 = kbps - 1
    for k in range(kbps):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                if k < kref[i, j]:
                    rdelks = delp[i, j, k] * delks[i, j]
                    ubar[i, j] = ubar[i, j] + rdelks * u1[i, j, k]  # Mean U below kref
                    vbar[i, j] = vbar[i, j] + rdelks * v1[i, j, k]  # Mean V below kref
                    # Mean ro below kref:
                    roll[i, j] = roll[i, j] + rdelks * ro[i, j, k]
                    rdelks = (prsl[i, j, k] - prsl[i, j, k + 1]) * delks1[i, j]
                    bnv2bar[i, j] = bnv2bar[i, j] + bnv2[i, j, k] * rdelks

    # FIGURE OUT LOW-LEVEL HORIZONTAL WIND DIRECTION AND FIND 'oa'
    # NWD  1   2   3   4   5   6   7   8
    #  WD  W   S  SW  NW   E   N  NE  SE

    # > - Calculate low-level horizontal wind direction, the derived
    # # orographic asymmetry parameter (oa), and the derived Lx (CLX).
    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            wdir = np.atan2(ubar[i, j], vbar[i, j]) + gwdpscons.PI
            idir = int(gwdpscons.FDIR * wdir) % gwdpscons.MDIR
            nwd = gwdpscons.NWDIR(idir)
            oa[i, j] = (1 - 2 * int((nwd - 1) / 4)) * oa4[i, j, nwd - 1 % 4]
            clx[i, j] = clx4[i, j, nwd - 1 % 4]

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

    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            xn[[i, j]] = 0.0
            yn[[i, j]] = 0.0
            taub[[i, j]] = 0.0
            ulow[[i, j]] = 0.0
            dtfac[[i, j]] = 1.0
            icrilv[[i, j]] = False  # INITIALIZE CRITICAL LEVEL CONTROL VECTOR

            # ----COMPUTE THE "LOW LEVEL" WIND MAGNITUDE (M/S)
            ulow[[i, j]] = max(
                np.sqrt(ubar[[i, j]] * ubar[[i, j]] + vbar[[i, j]] * vbar[[i, j]]), 1.0
            )
            uloi[[i, j]] = 1.0 / ulow[[i, j]]

    for k in range(kmm1):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                velco[i, j, k] = 0.5 * (
                    (u1[i, j, k] + u1[i, j, k + 1]) * ubar[i, j]
                    + (v1[i, j, k] + v1[i, j, k + 1]) * vbar[i, j]
                )
                velco[i, j, k] = velco[i, j, k] * uloi[i, j]

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
    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            kint[i, j] = kref[i, j]

    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            bnv = np.sqrt(bnv2bar[i, j])
            fr = bnv * uloi[i, j] * min(hprime[i, j], gwdpscons.HPMAX)
            fr = min(fr, gwdpscons.FRMAX)
            xn[i, j] = ubar[i, j] * uloi[i, j]
            yn[i, j] = vbar[i, j] * uloi[i, j]

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
            # #  \tau_0=E\frac{m'}{\triangle x}\frac{\rho_{0}U_0^3}
            # #      {N_{0}}constants.GRAV'
            # # \f]
            # #  where \f$E\f$,\f$m'\f$, and \f$constants.GRAV'\f$ are the enhancement
            # #  factor, "the number of mountains", and the flux function defined above,
            # #  respectively.

            efact = (oa[i, j] + 2.0) ** (gwdpscons.CEOFRC * fr)
            efact = min(max(efact, gwdpscons.EFMIN), gwdpscons.EFMAX)

            coefm = (1.0 + clx[i, j]) ** (oa[i, j] + 1.0)

            xlinv[i, j] = coefm * cleff

            tem = fr * fr * oc[i, j]
            # constants.GRAV/N0
            gfobnv = gwdpscons.GMAX * tem / ((tem + gwdpscons.CG) * bnv)

            taub[i, j] = (
                xlinv[i, j]
                * roll[i, j]
                * ulow[i, j]
                * ulow[i, j]
                * (ulow[i, j] * (gfobnv * efact))
            )  # BASE FLUX Tau0

            # tem = min(HPRIME[i, j],hpmax)
            # taub[i, j] = xlinv[i, j] * roll[i, j] * ulow[i, j] * BNV * tem * tem

            k = max(1, kref[i, j] - 1)
            tem = max(velco[i, j, k] * velco[i, j, k], 0.1)
            scor[i, j] = bnv2[i, j, k] / tem  # Scorer parameter below ref level

    # ----SET UP BOTTOM VALUES OF STRESS
    for k in range(kbps):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                if k <= kref[i, j]:
                    taup[i, j, k] = taub[i, j]

        #   Now compute vertical structure of the stress.
        for k in range(kmps, kmm1):  # Vertical Level K Loop!
            kp1 = k + 1
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    # -UNSTABLE LAYER if RI < RIC
                    # -UNSTABLE LAYER if
                    #      UPPER AIR VEL COMP ALONG SURF VEL <=0 (CRIT LAY)
                    # - AT (U-C)=0. CRIT LAYER EXISTS AND BIT VECTOR SHOULD BE SET (<=)
                    if k >= kref[i, j]:
                        icrilv[i, j] = (
                            icrilv[i, j]
                            or (ri_n[i, j, k] < gwdpscons.RIC)
                            or (velco[i, j, k] <= 0.0)
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
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                if k >= kref[i, j]:
                    if not icrilv[i, j] and taup[i, j, k] > 0.0:
                        temv = 1.0 / max(velco[i, j, k], 0.01)
                        # if (oa[i, j] > 0. and  prsi(ipt(i),KP1)>RLOLEV) THEN
                        if oa[i, j] > 0.0 and kp1 < kint[i, j]:
                            scork = bnv2[i, j, k] * temv * temv
                            rscor = min(1.0, scork / scor[i, j])
                            scor[i, j] = scork
                        else:
                            rscor = 1.0

                        # >  - The drag above the reference level is expressed as:
                        # #\f[
                        # # \tau=\frac{m'}{\triangle x}\rho NUh_d^2
                        # #\f]
                        # # where \f$h_{d}\f$ is the displacement wave amplitude. In the
                        # # absence of wave breaking, the displacement amplitude for the
                        # # \f$i^{th}\f$ layer can be expressed using the drag for the
                        # # layer immediately below. Thus, assuming
                        # # \f$\tau_i=\tau_{i+1}\f$, we can get:
                        # #\f[
                        # # h_{d_i}^2=\frac{\triangle x}{m'}\frac{\tau_{i+1}}
                        # #     {\rho_{i}N_{i}U_{i}}
                        # #\f]

                        brvf = np.sqrt(bnv2[i, j, k])  # Brunt-Vaisala Frequency
                        # tem1 = xlinv[i, j]*(ro[i,j,kp1]+ro[i,j,k])*brvf*velco[i,j,k]/2
                        tem1 = (
                            xlinv[i, j]
                            * (ro[i, j, kp1] + ro[i, j, k])
                            * brvf
                            * 0.5
                            * max(velco[i, j, k], 0.01)
                        )
                        hd = np.sqrt(taup[i, j, k] / tem1)
                        fro = brvf * hd * temv

                        # rim is the  MINIMUM-RICHARDSON NUMBER BY SHUTTS (1985)

                        # > - The minimum Richardson number (\f$Ri_{m}\f$) or local
                        # # wave-modified Richardson number, which determines the onset
                        # # of wave breaking, is expressed in terms of \f$R_{i}\f$ and
                        # # \f$F_{r_{d}}=Nh_{d}/U\f$:
                        # #\f[
                        # # Ri_{m}=\frac{Ri(1-Fr_{d})}{(1+\sqrt{Ri}\cdot Fr_{d})^{2}}
                        # #\f]
                        # # see eq.(4.6) in
                        # # Kim and Arakawa (1995) \cite kim_and_arakawa_1995.

                        tem2 = np.sqrt(ri_n[i, j, k])
                        tem = 1.0 + tem2 * fro
                        rim = ri_n[i, j, k] * (1.0 - fro) / (tem * tem)

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
                        # # eq.(4.7) in Kim and Arakawa (1995)
                        # # \cite kim_and_arakawa_1995):
                        # #\f[
                        # # h_{c}=\frac{U}{N}\left\{2(2+\frac{1}
                        # #     {\sqrt{Ri}})^{1/2}-(2+\frac{1}{\sqrt{Ri}})\right\}
                        # #\f]
                        # # if \f$Ri_{m}\leq Ri_{c}\f$, obtain \f$\tau\f$ from the drag
                        # # above the reference level by using \f$h_{c}\f$ computed
                        # # above; otherwise \f$\tau\f$ is unchanged (note: scaled by
                        # # the ratio of the Scorer paramter).

                        # if (RIM <= RIC and (
                        #     OA[i, j] <= 0. .OR. prsi(ipt[i, j],KP1)<=RLOLEV)
                        # ) THEN
                        if rim <= gwdpscons.RIC and (
                            oa[i, j] <= 0.0 or kp1 >= kint[i, j]
                        ):
                            temc = 2.0 + 1.0 / tem2
                            hd = velco[i, j, k] * (2.0 * np.sqrt(temc) - temc) / brvf
                            taup[i, j, kp1] = tem1 * hd * hd
                        else:
                            taup[i, j, kp1] = taup[i, j, k] * rscor
                        taup[i, j, kp1] = min(taup[i, j, kp1], taup[i, j, k])

    if lcap <= km:
        for klcap in range(lcapp1, km + 1):
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    sira = prsi[i, j, klcap] / prsi[i, j, lcap]
                    taup[i, j, klcap] = sira * taup[i, j, lcap]

    #  SJL: linear decay above p_crit, becoming constant at 1 mb
    #  Angular momentum conservation is ensured, except the top leakage
    # ----------------------- SJL mod ------------------------------
    if p_crit > 1.0e-10:
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                for k in range(km / 2, km + 1):
                    if prsi[i, j, k] < p_crit:  # scale it to zero @ top
                        taup[i, j, k] = (
                            taup[i, j, k]
                            * (prsi[i, j, k] - prsi[i, j, km + 1])
                            / (p_crit - prsi[i, j, km + 1])
                        )
                    elif prsi[i, j, k] < 1.0e2:
                        taup[i, j, k] = taup[i, j, k - 1]  # constant stress-> zero Drag

    # ----------------------- SJL mod ------------------------------

    #     Calculate - (g/p*)*d(tau)/d(sigma) and Decel terms dtaux, dtauy
    for k in range(km):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                taud[i, j, k] = (
                    constants.GRAV * (taup[i, j, k + 1] - taup[i, j, k]) / delp[i, j, k]
                )

    # ------LIMIT DE-ACCELERATION (MOMENTUM DEPOSITION ) AT TOP TO 1/2 VALUE
    # ------THE IDEA IS SOME STUFF MUST GO OUT THE 'TOP'
    if p_crit <= 1.0e-10:
        for klcap in range(lcap, km):
            for i, j in np.ndindex((ix, iy)):
                if ipt[i, j] is True:
                    taud[i, j, klcap] = taud[i, j, klcap] * gwdpscons.FACTOP

    # ------IF THE GRAVITY WAVE DRAG WOULD FORCE A CRITICAL LINE IN THE
    # ------LAYERS BELOW SIGMA=RLOLEV DURING THE NEXT deltim TIMESTEP,
    # ------THEN ONLY APPLY DRAG UNTIL THAT CRITICAL LINE IS REACHED.
    for k in range(kmm1):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                if (k < kref[i, j]) and (prsi[i, j, k] >= gwdpscons.RLOLEV):
                    if taud[i, j, k] != 0.0:
                        tem = deltim * taud[i, j, k]
                        dtfac[i, j] = min(dtfac[i, j], np.abs(velco[i, j, k] / tem))

    # > - Calculate outputs: A, B, dusfc, dvsfc (see parameter description).
    # #  - Below the dividing streamline height (k < idxzb), mountain
    # #    blocking(\f$D_{b}\f$) is applied.
    # #  - Otherwise (k>= idxzb), orographic GWD (\f$\tau\f$) is applied.
    for k in range(km):
        for i, j in np.ndindex((ix, iy)):
            if ipt[i, j] is True:
                taud[i, j, k] = taud[i, j, k] * dtfac[i, j]
                dtaux = taud[i, j, k] * xn[i, j]
                dtauy = taud[i, j, k] * yn[i, j]
                eng0 = 0.5 * (u1[i, j, k] * u1[i, j, k] + v1[i, j, k] * v1[i, j, k])
                # ---  lm mb (*j*)  changes overwrite GWD
                if (k < idxzb[i, j]) and (idxzb[i, j] != 0):
                    dbim = db[i, j, k] / (1.0 + db[i, j, k] * deltim)
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
                        + (v1[i, j, k] + dtauy * deltim)
                        * (v1[i, j, k] + dtauy * deltim)
                    )
                    dusfc[i, j] = dusfc[i, j] + dtaux * delp[i, j, k]
                    dvsfc[i, j] = dvsfc[i, j] + dtauy * delp[i, j, k]
                c[i, j, k] = (
                    c[i, j, k] + max(eng0 - eng1, 0.0) / constants.CP_AIR / deltim
                )

    tem = -1.0 / constants.GRAV
    for i, j in np.ndindex((ix, iy)):
        if ipt[i, j] is True:
            # tem    = (-1.E3/G)
            dusfc[i, j] = tem * dusfc[i, j]
            dvsfc[i, j] = tem * dvsfc[i, j]

    return
