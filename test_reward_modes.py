import pandas as pd

from src.irrigation_env import SmartIrrigationEnv


df = pd.read_csv(
    "data/val_weather.csv",
    parse_dates=["time"]
)

for mode in ["basic", "physics"]:

    env = SmartIrrigationEnv(
        df,
        reward_mode=mode
    )

    state, info = env.reset(seed=42)

    total_reward = 0.0

    while True:

        # Same fixed action for both environments
        action = 2

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        if terminated or truncated:
            break

    print(
        f"{mode.upper()} | "
        f"Reward: {total_reward:.3f} | "
        f"Water: {info['total_irrigation_mm']:.1f} mm | "
        f"Moisture: {info['soil_moisture']:.3f}"
    )

    env.close()

print("\nReward mode comparison completed!")