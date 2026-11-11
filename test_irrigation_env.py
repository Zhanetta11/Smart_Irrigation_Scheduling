"""
Gymnasium Environment Test
"""

import pandas as pd

from pathlib import Path

from gymnasium.utils.env_checker import check_env

from src.irrigation_env import SmartIrrigationEnv


# Load weather data with ET0
DATA_DIR = Path(__file__).resolve().parent / "data"

df = pd.read_csv(
    DATA_DIR / "weather_with_et0.csv",
    parse_dates=["time"]
)

# Select one growing season
df_2020 = df[
    df["time"].dt.year == 2020
].copy()

print("SMART IRRIGATION ENVIRONMENT TEST")
print("=" * 50)

print("Weather records:", len(df_2020))

# Create environment
env = SmartIrrigationEnv(df_2020)

# Validate Gymnasium API
check_env(env, skip_render_check=True)

print("Gymnasium API validation: PASSED")

# Reset environment
observation, info = env.reset(seed=42)

print("\nInitial observation:")
print(observation)

print("\nObservation shape:", observation.shape)
print("Action space:", env.action_space)

assert observation.shape == (5,)

# Simulate one episode
total_reward = 0.0
steps = 0

while True:

    # Random irrigation action
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    total_reward += reward
    steps += 1

    if terminated or truncated:
        break

print("\nSIMULATION RESULTS")
print("=" * 50)

print("Total steps:", steps)
print("Total reward:", round(total_reward, 2))

print(
    "Total irrigation:",
    round(info["total_irrigation_mm"], 2),
    "mm"
)

print(
    "Final soil moisture:",
    round(info["soil_moisture"], 3)
)

assert steps == len(df_2020)

print("\nAll Gymnasium environment tests PASSED!")

env.close()