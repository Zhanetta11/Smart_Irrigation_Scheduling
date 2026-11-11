"""
Physics-Informed Reward Tests
"""

from pathlib import Path
import pandas as pd

from gymnasium.utils.env_checker import check_env

from src.physics_reward import calculate_physics_reward
from src.irrigation_env import SmartIrrigationEnv


print("PHYSICS-INFORMED REWARD TEST")
print("=" * 50)

# Test 1: No irrigation
reward_no_water = calculate_physics_reward(
    soil_moisture=0.27,
    irrigation_mm=0,
    drainage_mm=0,
    stress_factor=1.0
)

# Test 2: Moderate irrigation
reward_moderate = calculate_physics_reward(
    soil_moisture=0.27,
    irrigation_mm=5,
    drainage_mm=0,
    stress_factor=1.0
)

# Test 3: Excessive irrigation and drainage
reward_excessive = calculate_physics_reward(
    soil_moisture=0.27,
    irrigation_mm=8,
    drainage_mm=5,
    stress_factor=1.0
)

# Test 4: Dry soil and water stress
reward_dry = calculate_physics_reward(
    soil_moisture=0.20,
    irrigation_mm=0,
    drainage_mm=0,
    stress_factor=0.5
)

print("No irrigation reward:", reward_no_water)
print("Moderate irrigation reward:", reward_moderate)
print("Excessive irrigation reward:", reward_excessive)
print("Dry soil reward:", reward_dry)

assert reward_no_water > reward_moderate
assert reward_moderate > reward_excessive
assert reward_dry < reward_no_water

print("\nReward ranking tests: PASSED")


# ==========================================
# Test Gymnasium integration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(
    PROJECT_DIR / "data" / "val_weather.csv",
    parse_dates=["time"]
)

env = SmartIrrigationEnv(
    df,
    reward_mode="physics"
)

check_env(env, skip_render_check=True)

state, info = env.reset(seed=42)

total_reward = 0.0
steps = 0

while True:

    action = env.action_space.sample()

    state, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward
    steps += 1

    if terminated or truncated:
        break

assert steps == len(df)

print("\nGYMNASIUM INTEGRATION")
print("=" * 50)
print("Total steps:", steps)
print("Total reward:", round(total_reward, 3))
print("Total irrigation:", info["total_irrigation_mm"])
print("Final soil moisture:", round(info["soil_moisture"], 3))

print("\nAll physics reward tests PASSED!")

env.close()