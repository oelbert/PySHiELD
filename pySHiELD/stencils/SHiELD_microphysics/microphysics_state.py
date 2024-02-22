from dataclasses import InitVar, dataclass, field, fields
from typing import Any, Dict, Mapping

import pace.util


@dataclass()
class MicrophysicsState:
    qvapor: pace.util.Quantity = field(
        metadata={
            "name": "specific_humidity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qliquid: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qice: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qrain: pace.util.Quantity = field(
        metadata={
            "name": "rain_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qsnow: pace.util.Quantity = field(
        metadata={
            "name": "snow_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qgraupel: pace.util.Quantity = field(
        metadata={
            "name": "graupel_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qcld: pace.util.Quantity = field(
        metadata={
            "name": "cloud_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    qcon: pace.util.Quantity = field(
        metadata={
            "name": "condensate_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
            "intent": "inout",
        }
    )
    qcloud_cond_nuclei: pace.util.Quantity = field(
        metadata={
            "name": "cloud_condensate_nuclei_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    qcloud_ice_nuclei: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_nuclei_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "inout",
        }
    )
    ua: pace.util.Quantity = field(
        metadata={
            "name": "eastward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    va: pace.util.Quantity = field(
        metadata={
            "name": "northward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    wa: pace.util.Quantity = field(
        metadata={
            "name": "vertical_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
            "intent": "inout",
        }
    )
    pt: pace.util.Quantity = field(
        metadata={
            "name": "air_temperature",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "degK",
            "intent": "inout",
        }
    )
    delp: pace.util.Quantity = field(
        metadata={
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa",
            "intent": "inout",
        }
    )
    delz: pace.util.Quantity = field(
        metadata={
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m",
            "intent": "inout",
        }
    )
    geopotential_surface_height: pace.util.Quantity = field(
        metadata={
            "name": "geopotential_surface_height",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "m",
            "intent": "in",
        }
    )
    preflux_water: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_ice: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_rain: pace.util.Quantity = field(
        metadata={
            "name": "rain_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_snow: pace.util.Quantity = field(
        metadata={
            "name": "snow_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    preflux_graupel: pace.util.Quantity = field(
        metadata={
            "name": "graupel_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    column_water: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_precipitated_to_ground",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_ice: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_precipitated_to_ground",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_rain: pace.util.Quantity = field(
        metadata={
            "name": "rain_precipitated_to_ground",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_snow: pace.util.Quantity = field(
        metadata={
            "name": "snow_precipitated_to_ground",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    column_graupel: pace.util.Quantity = field(
        metadata={
            "name": "graupel_precipitated_to_ground",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    condensation: pace.util.Quantity = field(
        metadata={
            "name": "total_column_condensation",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    deposition: pace.util.Quantity = field(
        metadata={
            "name": "total_column_deposition",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    sublimation: pace.util.Quantity = field(
        metadata={
            "name": "total_column_sublimation",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    evaporation: pace.util.Quantity = field(
        metadata={
            "name": "total_column_evaporation",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    total_energy: pace.util.Quantity = field(
        metadata={
            "name": "total_energy",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    column_energy_change: pace.util.Quantity = field(
        metadata={
            "name": "energy_change_in_column",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    cappa: pace.util.Quantity = field(
        metadata={
            "name": "cappa",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "inout",
        }
    )
    adj_vmr: pace.util.Quantity = field(
        metadata={
            "name": "mixing_ratio_adjustment",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "intent": "out",
        }
    )
    particle_concentration_w: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_particle_concentration",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_w: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_effective_diameter",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_w: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_optical_extinction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_w: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_radar_reflectivity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_w: pace.util.Quantity = field(
        metadata={
            "name": "cloud_water_terminal_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_r: pace.util.Quantity = field(
        metadata={
            "name": "rain_particle_concentration",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_r: pace.util.Quantity = field(
        metadata={
            "name": "rain_effective_diameter",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_r: pace.util.Quantity = field(
        metadata={
            "name": "rain_optical_extinction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_r: pace.util.Quantity = field(
        metadata={
            "name": "rain_radar_reflectivity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_r: pace.util.Quantity = field(
        metadata={
            "name": "rain_terminal_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_i: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_particle_concentration",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_i: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_effective_diameter",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_i: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_optical_extinction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_i: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_radar_reflectivity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_i: pace.util.Quantity = field(
        metadata={
            "name": "cloud_ice_terminal_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_s: pace.util.Quantity = field(
        metadata={
            "name": "snow_particle_concentration",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_s: pace.util.Quantity = field(
        metadata={
            "name": "snow_effective_diameter",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_s: pace.util.Quantity = field(
        metadata={
            "name": "snow_optical_extinction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_s: pace.util.Quantity = field(
        metadata={
            "name": "snow_radar_reflectivity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_s: pace.util.Quantity = field(
        metadata={
            "name": "snow_terminal_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    particle_concentration_g: pace.util.Quantity = field(
        metadata={
            "name": "graupel_particle_concentration",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    effective_diameter_g: pace.util.Quantity = field(
        metadata={
            "name": "graupel_effective_diameter",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    optical_extinction_g: pace.util.Quantity = field(
        metadata={
            "name": "graupel_optical_extinction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    radar_reflectivity_g: pace.util.Quantity = field(
        metadata={
            "name": "graupel_radar_reflectivity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    terminal_velocity_g: pace.util.Quantity = field(
        metadata={
            "name": "graupel_terminal_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "intent": "out",
        }
    )
    quantity_factory: InitVar[pace.util.QuantityFactory]

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
        sizer: pace.util.GridSizer,
        quantity_factory: pace.util.QuantityFactory,
    ) -> "MicrophysicsState":
        inputs: Dict[str, pace.util.Quantity] = {}
        for _field in fields(cls):
            if "dims" in _field.metadata.keys():
                if _field.metadata["intent"] == "out":
                    inputs[_field.name] = quantity_factory.zeros(
                        _field.metadata["dims"], _field.metadata["units"], dtype=float
                    ).data
                else:  # intent is in or inout
                    inputs[_field.name] = pace.util.Quantity(
                        storages[_field.name],
                        _field.metadata["dims"],
                        _field.metadata["units"],
                        origin=sizer.get_origin(_field.metadata["dims"]),
                        extent=sizer.get_extent(_field.metadata["dims"]),
                    )
        return cls(**inputs, quantity_factory=quantity_factory)

    # TODO Will we want "from physics" and "from dycore" methods?
    # Or do init_zeros and then populate?
