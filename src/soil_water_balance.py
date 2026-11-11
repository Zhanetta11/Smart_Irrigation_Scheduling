"""
Soil-Water Balance Model

This module simulates daily changes in soil moisture
based on rainfall, irrigation, evapotranspiration,
surface runoff, and drainage.

Units:
- Water storage: mm
- Rainfall: mm/day
- Irrigation: mm/day
- Evapotranspiration: mm/day
- Soil moisture: m3/m3
"""

import math
from dataclasses import dataclass


@dataclass
class SoilConfig:
    """Configuration for the soil-water simulation."""

    # Provisional loam soil parameters
    root_depth_m: float = 0.5
    field_capacity: float = 0.30
    wilting_point: float = 0.15
    saturation: float = 0.45

    # Initial volumetric soil moisture
    initial_moisture: float = 0.27

    # Simplified crop coefficient
    crop_coefficient: float = 1.0


class SoilWaterBalance:
    """Daily soil-water balance simulation."""

    def __init__(self, config=None):

        self.config = config or SoilConfig()
        c = self.config

        # Validate configuration values
        parameters = [
            c.root_depth_m,
            c.field_capacity,
            c.wilting_point,
            c.saturation,
            c.initial_moisture,
            c.crop_coefficient
        ]

        if not all(math.isfinite(x) for x in parameters):
            raise ValueError(
                "Soil parameters must be finite numbers"
            )

        if not (
            0 <= c.wilting_point
            < c.field_capacity
            < c.saturation <= 1
        ):
            raise ValueError("Invalid soil parameters")

        if not (
            0 <= c.initial_moisture <= c.saturation
        ):
            raise ValueError("Invalid initial moisture")

        if c.root_depth_m <= 0:
            raise ValueError(
                "Root depth must be positive"
            )

        if c.crop_coefficient < 0:
            raise ValueError(
                "Crop coefficient must be nonnegative"
            )

        # Convert root depth from meters to millimeters
        self.root_depth_mm = c.root_depth_m * 1000

        # Calculate soil water storage thresholds
        self.field_capacity_mm = (
            c.field_capacity * self.root_depth_mm
        )

        self.wilting_point_mm = (
            c.wilting_point * self.root_depth_mm
        )

        self.saturation_mm = (
            c.saturation * self.root_depth_mm
        )

        # Initialize soil water storage
        self.reset()

    def reset(self):
        """Reset soil to its initial water storage."""

        self.water_storage = (
            self.config.initial_moisture
            * self.root_depth_mm
        )

        return self.get_soil_moisture()

    def get_soil_moisture(self):
        """Return volumetric soil moisture (m3/m3)."""

        return self.water_storage / self.root_depth_mm

    def step(self, rainfall, irrigation, et0):
        """
        Simulate one day of soil-water balance.

        Parameters:
            rainfall: Daily precipitation (mm)
            irrigation: Applied irrigation (mm)
            et0: Reference evapotranspiration (mm/day)

        Returns:
            Dictionary containing soil moisture,
            water fluxes, and balance error.
        """

        # Validate daily inputs
        inputs = [rainfall, irrigation, et0]

        if not all(math.isfinite(x) for x in inputs):
            raise ValueError(
                "Water inputs must be finite numbers"
            )

        if min(inputs) < 0:
            raise ValueError(
                "Water inputs must be nonnegative"
            )

        # Store previous water storage
        previous_storage = self.water_storage

        # 1. Calculate crop evapotranspiration demand
        etc_demand = (
            et0 * self.config.crop_coefficient
        )

        # 2. Add rainfall and irrigation
        incoming_water = rainfall + irrigation

        available_water = (
            previous_storage + incoming_water
        )

        # 3. Calculate surface runoff
        # Water exceeding saturation is removed
        runoff = max(
            0.0,
            available_water - self.saturation_mm
        )

        available_water -= runoff

        # 4. Calculate water stress factor
        # Simplified approximation based on soil moisture
        stress_factor = max(
            0.0,
            min(
                1.0,
                (
                    available_water - self.wilting_point_mm
                ) / (
                    self.field_capacity_mm
                    - self.wilting_point_mm
                )
            )
        )

        # 5. Calculate actual evapotranspiration
        actual_et = min(
            available_water,
            etc_demand * stress_factor
        )

        available_water -= actual_et

        # 6. Calculate drainage above field capacity
        drainage = max(
            0.0,
            available_water - self.field_capacity_mm
        )

        available_water -= drainage

        # 7. Update soil water storage
        self.water_storage = max(
            0.0,
            available_water
        )

        moisture = self.get_soil_moisture()

        # 8. Verify water balance conservation
        expected_storage = (
            previous_storage
            + rainfall
            + irrigation
            - actual_et
            - drainage
            - runoff
        )

        balance_error = (
            self.water_storage - expected_storage
        )

        if abs(balance_error) > 1e-6:
            raise ValueError(
                f"Water balance error: "
                f"{balance_error:.8f} mm"
            )

        # 9. Return simulation results
        return {
            "soil_moisture": moisture,
            "water_storage_mm": self.water_storage,
            "rainfall_mm": rainfall,
            "irrigation_mm": irrigation,
            "etc_demand_mm": etc_demand,
            "actual_et_mm": actual_et,
            "drainage_mm": drainage,
            "runoff_mm": runoff,
            "stress_factor": stress_factor,
            "balance_error_mm": balance_error
        }