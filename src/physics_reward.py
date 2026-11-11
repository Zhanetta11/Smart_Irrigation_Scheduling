
"""
Physics-Informed Reward Function

The reward encourages:
1. Healthy soil moisture
2. Lower irrigation consumption
3. Reduced drainage
4. Reduced crop water stress

All coefficients are experimental.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicsRewardConfig:
    target_min: float = 0.25
    target_max: float = 0.30

    water_penalty_weight: float = 0.12
    drainage_penalty_weight: float = 0.08
    stress_penalty_weight: float = 2.0

    dry_soil_penalty_weight: float = 20.0
    wet_soil_penalty_weight: float = 10.0

    target_reward: float = 1.0


def calculate_physics_reward(
    soil_moisture,
    irrigation_mm,
    drainage_mm,
    stress_factor,
    config=None
):
    """
    Calculate a physics-informed irrigation reward.

    Parameters
    ----------
    soil_moisture : float
        Volumetric soil moisture (m3/m3).

    irrigation_mm : float
        Daily irrigation amount in mm.

    drainage_mm : float
        Daily drainage amount in mm.

    stress_factor : float
        Simplified crop water availability factor (0-1).

    Returns
    -------
    float
        Total reward for the current step.
    """

    c = config or PhysicsRewardConfig()

    values = [
        soil_moisture,
        irrigation_mm,
        drainage_mm,
        stress_factor
    ]

    if not all(math.isfinite(v) for v in values):
        raise ValueError("Reward inputs must be finite")

    if irrigation_mm < 0 or drainage_mm < 0:
        raise ValueError(
            "Irrigation and drainage must be nonnegative"
        )

    if not 0 <= stress_factor <= 1:
        raise ValueError(
            "Stress factor must be between 0 and 1"
        )

    if not 0 <= soil_moisture <= 1:
        raise ValueError("Invalid soil moisture")

    if not 0 <= c.target_min <= c.target_max <= 1:
        raise ValueError("Invalid target moisture range")

    # 1. Soil moisture reward
    if c.target_min <= soil_moisture <= c.target_max:
        moisture_reward = c.target_reward

    elif soil_moisture < c.target_min:
        moisture_reward = (
            -c.dry_soil_penalty_weight
            * (c.target_min - soil_moisture)
        )

    else:
        moisture_reward = (
            -c.wet_soil_penalty_weight
            * (soil_moisture - c.target_max)
        )

    # 2. Irrigation water penalty
    water_penalty = (
        c.water_penalty_weight * irrigation_mm
    )

    # 3. Drainage penalty
    drainage_penalty = (
        c.drainage_penalty_weight * drainage_mm
    )

    # 4. Crop water stress penalty
    stress_penalty = (
        c.stress_penalty_weight
        * (1.0 - stress_factor)
    )

    # 5. Final reward
    reward = (
        moisture_reward
        - water_penalty
        - drainage_penalty
        - stress_penalty
    )

    return float(reward)