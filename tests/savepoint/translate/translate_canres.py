from ndsl import Namelist, StencilFactory
from ndsl.dsl.stencil import GridIndexing
from pySHiELD.stencils.surface.noah_lsm.lsm_driver import canres
from tests.savepoint.translate.translate_physics import TranslatePhysicsFortranData2Py


class Canres:
    def __init__(self, stencil_factory: StencilFactory):
        grid_indexing = stencil_factory.grid_indexing
        self._canres = stencil_factory.from_origin_domain(
            func=canres,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def __call__(
        self,
        nroot,
        swdn,
        ch,
        q2,
        q2sat,
        dqsdt2,
        sfctmp,
        sfcprs,
        sfcems,
        sh2o,
        smcwlt,
        smcref,
        zsoil,
        zroot,
        rsmin,
        rgl,
        hs,
        xlai,
        k_mask,
        shdfac,
        rc,
        pc,
        rcs,
        rct,
        rcq,
        rcsoil,
        lsm_mask,
    ):
        self._canres(
            nroot,
            swdn,
            ch,
            q2,
            q2sat,
            dqsdt2,
            sfctmp,
            sfcprs,
            sfcems,
            sh2o,
            smcwlt,
            smcref,
            zsoil,
            zroot,
            rsmin,
            rgl,
            hs,
            xlai,
            k_mask,
            shdfac,
            rc,
            pc,
            rcs,
            rct,
            rcq,
            rcsoil,
            lsm_mask,
        )


class TranslateCanres(TranslatePhysicsFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: Namelist,
        stencil_factory: StencilFactory,
    ):
        grid_domain = (
            stencil_factory.grid_indexing.domain[0],
            stencil_factory.grid_indexing.domain[1],
            int(namelist.lsoil),
        )
        surface_grid_index = GridIndexing(
            grid_domain,
            stencil_factory.grid_indexing.n_halo,
            stencil_factory.grid_indexing.south_edge,
            stencil_factory.grid_indexing.north_edge,
            stencil_factory.grid_indexing.west_edge,
            stencil_factory.grid_indexing.east_edge,
        )
        surface_factory = StencilFactory(
            stencil_factory.config,
            surface_grid_index,
            stencil_factory.comm,
        )
        super().__init__(grid, namelist, surface_factory)
        self.stencil_factory = surface_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs.pop("nsoil"),
        self.compute_func = Canres(
            self.stencil_factory,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
