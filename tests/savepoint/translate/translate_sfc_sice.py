from ndsl import Namelist, StencilFactory
from pySHiELD.stencils.surface.sfc_sice import SurfaceSeaIce
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py
from gt4py.cartesian.gtscript import FORWARD, computation, interval, sqrt
from ndsl.constants import X_DIM, Y_DIM
from ndsl.dsl.typing import Float, FloatFieldIJ  # noqa: F401


def calc_wind(
    wind: FloatFieldIJ,
    u1: FloatFieldIJ,
    v1: FloatFieldIJ,
    ddvel: FloatFieldIJ,
):
    with computation(FORWARD), interval(0, 1):
        wind = max(sqrt(u1 ** 2 + v1 ** 2) + max(0.0, min(ddvel, 30.0)), 1.0)

class TestSurfaceSeaIce:
    def __init__(
        self,
        stencil_factory,
        quantity_factory,
        mom4ice,
        lsm,
        dt_atmos,
    ):
        grid_indexing = stencil_factory.grid_indexing
        self._wind = quantity_factory.zeros(
            [X_DIM, Y_DIM],
            units="unknown",
            dtype=Float,
        )
        self._calc_wind = stencil_factory.from_origin_domain(
            calc_wind,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        self._sice = SurfaceSeaIce(
            stencil_factory,
            mom4ice,
            lsm,
            dt_atmos,
        )

    def __call__(
        self,
        ps,
        u1,
        v1,
        ddvel,
        t1,
        q1,
        sfcemis,
        dlwflx,
        sfcnsw,
        sfcdsw,
        srflag,
        cm,
        ch,
        prsl1,
        prslki,
        islimsk,
        flag_iter,
        hice,
        fice,
        tice,
        weasd,
        tskin,
        tprcp,
        stc0,
        stc1,
        ep,
        snwdph,
        qsurf,
        cmm,
        chh,
        evap,
        hflx,
        gflux,
        snowmt,
    ):
        self._calc_wind(
            self._wind,
            u1,
            v1,
            ddvel,
        )
        self._sice(
            ps,
            self._wind,
            t1,
            q1,
            sfcemis,
            dlwflx,
            sfcnsw,
            sfcdsw,
            srflag,
            cm,
            ch,
            prsl1,
            prslki,
            islimsk,
            flag_iter,
            hice,
            fice,
            tice,
            weasd,
            tskin,
            tprcp,
            stc0,
            stc1,
            ep,
            snwdph,
            qsurf,
            cmm,
            chh,
            evap,
            hflx,
            gflux,
            snowmt,
        )


class TranslateSurfaceSeaIce_iter1(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u1": {"serialname": "sice_u1", "shield": True},
            "v1": {"serialname": "sice_v1", "shield": True},
            "t1": {"serialname": "sice_t1", "shield": True},
            "ps": {"serialname": "sice_ps", "shield": True},
            "wind": {"serialname": "sice_wind", "shield": True},
            "q1": {"serialname": "sice_q1", "shield": True},
            "sfcemis": {"serialname": "sfcemis", "shield": True},
            "dlwflx": {"serialname": "sice_dlwflx", "shield": True},
            "sfcnsw": {"serialname": "sice_sfcnsw", "shield": True},
            "sfcdsw": {"serialname": "sice_sfcdsw", "shield": True},
            "srflag": {"serialname": "sice_srflag", "shield": True},
            "cm": {"serialname": "sice_cm", "shield": True},
            "ch": {"serialname": "sice_ch", "shield": True},
            "prsl1": {"serialname": "sice_prsl1", "shield": True},
            "prslki": {"serialname": "sice_prslki", "shield": True},
            "islimsk": {"serialname": "sice_islmsk", "shield": True},
            "flag_iter": {"serialname": "sice_flag_iter", "shield": True},
            "hice": {"serialname": "sice_hice", "shield": True},
            "fice": {"serialname": "sice_fice", "shield": True},
            "tice": {"serialname": "sice_tice", "shield": True},
            "weasd": {"serialname": "sice_weasd", "shield": True},
            "tskin": {"serialname": "sice_tskin", "shield": True},
            "tprcp": {"serialname": "sice_tprcp", "shield": True},
            "stc0": {"serialname": "sice_stc0", "shield": True},
            "stc1": {"serialname": "sice_stc1", "shield": True},
            "ep": {"serialname": "sice_ep", "shield": True},
            "snwdph": {"serialname": "sice_snowd", "shield": True},
            "qsurf": {"serialname": "sice_qsurf", "shield": True},
            "cmm": {"serialname": "sice_cmm", "shield": True},
            "chh": {"serialname": "sice_chh%chh", "shield": True},
            "evap": {"serialname": "sice_evap", "shield": True},
            "hflx": {"serialname": "sice_hflx", "shield": True},
            "gflux": {"serialname": "sice_gflux", "shield": True},
            "snowmt": {"serialname": "sice_snowmt", "shield": True},
        }
        self.in_vars["parameters"] = [
            "sice_delt",
            "sice_mom4ice",
            "sice_lsm",
        ]
        self.out_vars = {
            "hice": {"serialname": "sice_hice", "shield": True},
            "fice": {"serialname": "sice_fice", "shield": True},
            "tice": {"serialname": "sice_tice", "shield": True},
            "weasd": {"serialname": "sice_weasd", "shield": True},
            "tskin": {"serialname": "sice_tskin", "shield": True},
            "tprcp": {"serialname": "sice_tprcp", "shield": True},
            "stc0": {"serialname": "sice_stc0", "shield": True},
            "stc1": {"serialname": "sice_stc1", "shield": True},
            "ep": {"serialname": "sice_ep", "shield": True},
            "snwdph": {"serialname": "sice_snowd", "shield": True},
            "qsurf": {"serialname": "sice_qsurf", "shield": True},
            "snowmt": {"serialname": "sice_snowmt", "shield": True},
            "gflux": {"serialname": "sice_gflux", "shield": True},
            "cmm": {"serialname": "sice_cmm", "shield": True},
            "chh": {"serialname": "sice_chh%chh", "shield": True},
            "evap": {"serialname": "sice_evap", "shield": True},
            "hflx": {"serialname": "sice_hflx", "shield": True},
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.compute_func = TestSurfaceSeaIce(
            self.stencil_factory,
            self.grid.quantity_factory,
            mom4ice=inputs.pop("sice_mom4ice"),
            lsm=inputs.pop("sice_lsm"),
            dt_atmos=inputs.pop("sice_delt"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateSurfaceSeaIce_iter2(TranslateSurfaceSeaIce_iter1):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        super().__init__(namelist, stencil_factory)
