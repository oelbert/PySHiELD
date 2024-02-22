from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, Mapping

from ndsl.quantity import Quantity
from ndsl.initialization.allocator import QuantityFactory, GridSizer
from ndsl.constants import X_DIM, Y_DIM, Z_DIM


@dataclass()
class MicrophysicsState:
    qvapor: Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qliquid: Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: Quantity = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: Quantity = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: Quantity = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qcld: Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    qcon: Quantity = field(
        metadata={
            "name": "condensate_mixing_ratio",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qcloud_cond_nuclei: Quantity = field(
        metadata={
            "name": "cloud_condensate_nuclei_fraction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    qcloud_ice_nuclei: Quantity = field(
        metadata={
            "name": "cloud_ice_nuclei_fraction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    ua: Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    wa: Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    pt: Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    delp: Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    geopotential_surface_height: Quantity = field(
        metadata={
            "name": "geopotential_surface_height",
            "dims": [X_DIM, Y_DIM],
            "units": "m",
            "intent": "in",
        }
    )
    preflux_water: Quantity = field(
        metadata={
            "name": "cloud_water_flux",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_ice: Quantity = field(
        metadata={
            "name": "cloud_ice_flux",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_rain: Quantity = field(
        metadata={
            "name": "rain_flux",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_snow: Quantity = field(
        metadata={
            "name": "snow_flux",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_graupel: Quantity = field(
        metadata={
            "name": "graupel_flux",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    column_water: Quantity = field(
        metadata={
            "name": "cloud_water_precipitated_to_ground",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_ice: Quantity = field(
        metadata={
            "name": "cloud_ice_precipitated_to_ground",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_rain: Quantity = field(
        metadata={
            "name": "rain_precipitated_to_ground",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_snow: Quantity = field(
        metadata={
            "name": "snow_precipitated_to_ground",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_graupel: Quantity = field(
        metadata={
            "name": "graupel_precipitated_to_ground",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    condensation: Quantity = field(
        metadata={
            "name": "total_column_condensation",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    deposition: Quantity = field(
        metadata={
            "name": "total_column_deposition",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    sublimation: Quantity = field(
        metadata={
            "name": "total_column_sublimation",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    evaporation: Quantity = field(
        metadata={
            "name": "total_column_evaporation",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    total_energy: Quantity = field(
        metadata={
            "name": "total_energy",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    column_energy_change: Quantity = field(
        metadata={
            "name": "energy_change_in_column",
            "dims": [X_DIM, Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    cappa: Quantity = field(
        metadata={
            "name": "cappa",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    adj_vmr: Quantity = field(
        metadata={
            "name": "mixing_ratio_adjustment",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "",
            "intent": "out",
        }
    )
    particle_concentration_w: Quantity = field(
        metadata={
            "name": "cloud_water_particle_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_w: Quantity = field(
        metadata={
            "name": "cloud_water_effective_diameter",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_w: Quantity = field(
        metadata={
            "name": "cloud_water_optical_extinction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_w: Quantity = field(
        metadata={
            "name": "cloud_water_radar_reflectivity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_w: Quantity = field(
        metadata={
            "name": "cloud_water_terminal_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_r: Quantity = field(
        metadata={
            "name": "rain_particle_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_r: Quantity = field(
        metadata={
            "name": "rain_effective_diameter",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_r: Quantity = field(
        metadata={
            "name": "rain_optical_extinction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_r: Quantity = field(
        metadata={
            "name": "rain_radar_reflectivity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_r: Quantity = field(
        metadata={
            "name": "rain_terminal_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_i: Quantity = field(
        metadata={
            "name": "cloud_ice_particle_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_i: Quantity = field(
        metadata={
            "name": "cloud_ice_effective_diameter",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_i: Quantity = field(
        metadata={
            "name": "cloud_ice_optical_extinction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_i: Quantity = field(
        metadata={
            "name": "cloud_ice_radar_reflectivity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_i: Quantity = field(
        metadata={
            "name": "cloud_ice_terminal_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_s: Quantity = field(
        metadata={
            "name": "snow_particle_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_s: Quantity = field(
        metadata={
            "name": "snow_effective_diameter",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_s: Quantity = field(
        metadata={
            "name": "snow_optical_extinction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_s: Quantity = field(
        metadata={
            "name": "snow_radar_reflectivity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_s: Quantity = field(
        metadata={
            "name": "snow_terminal_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_g: Quantity = field(
        metadata={
            "name": "graupel_particle_concentration",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_g: Quantity = field(
        metadata={
            "name": "graupel_effective_diameter",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_g: Quantity = field(
        metadata={
            "name": "graupel_optical_extinction",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_g: Quantity = field(
        metadata={
            "name": "graupel_radar_reflectivity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_g: Quantity = field(
        metadata={
            "name": "graupel_terminal_velocity",
            "dims": [X_DIM, Y_DIM, Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    quantity_factory: InitVar[QuantityFactory]

    @classmethod
    def init_zeros(cls, quantity_factory) -> "MicrophysicsState":
        initial_arrays = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                initial_arrays[_field.name] = quantity_factory.zeros(
                    _field.metadata["dims"], _field.metadata["units"], dtype=float
                ).data
        return cls(**initial_arrays, quantity_factory=quantity_factory)

    @classmethod
    def init_from_storages(
        cls,
        storages: Mapping[str, Any],
        sizer: GridSizer,
        quantity_factory: QuantityFactory,
    ) -> "MicrophysicsState":
        inputs: Dict[str, Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                if _field.metadata["intent"] == "out":
                    inputs[_field.name] = quantity_factory.zeros(
                        _field.metadata["dims"], _field.metadata["units"], dtype=float
                    ).data
                else:  # intent is in or inout
                    inputs[_field.name] = Quantity(
                        storages[_field.name],
                        _field.metadata["dims"],
                        _field.metadata["units"],
                        origin=sizer.get_origin(_field.metadata["dims"]),
                        extent=sizer.get_extent(_field.metadata["dims"]),
                    )
        return cls(**inputs, quantity_factory=quantity_factory)

    # TODO Will we want "from physics" and "from dycore" methods?
    # Or do init_zeros and then populate?
