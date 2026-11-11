"""
Standard DQN Validation
Compare DQN and threshold-based irrigation on 2024 data.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dqn_agent import DQNAgent
from src.irrigation_env import SmartIrrigationEnv


PROJECT_DIR = Path(__file__).resolve().parent

val_df = pd.read_csv(
    PROJECT_DIR / "data" / "val_weather.csv",
    parse_dates=["time"]
)

agent = DQNAgent(
    state_size=5,
    action_size=4
)

agent.load(
    PROJECT_DIR / "models" / "standard_dqn_best.pt"
)


def evaluate(strategy):

    env = SmartIrrigationEnv(val_df)
    state, _ = env.reset(seed=42)

    records = []

    while True:

        moisture_before = env.soil.get_soil_moisture()

        if strategy == "DQN":
            action = agent.select_action(
                state, training=False
            )

        elif strategy == "Threshold":
            action = (
                3 if moisture_before < 0.25 else 0
            )

        else:
            raise ValueError("Unknown strategy")

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        records.append({
            "soil_moisture": info["soil_moisture"],
            "irrigation_mm": info["irrigation_mm"],
            "drainage_mm": info["drainage_mm"],
            "stress_factor": info["stress_factor"],
            "reward": reward
        })

        if terminated or truncated:
            break

    env.close()

    return pd.DataFrame(records)


def metrics(data):

    return {
        "Irrigation (mm)": data["irrigation_mm"].sum(),
        "Target Days (%)": (
            data["soil_moisture"]
            .between(0.25, 0.30)
            .mean() * 100
        ),
        "Drainage (mm)": data["drainage_mm"].sum(),
        "Average Stress Factor": (
            data["stress_factor"].mean()
        ),
        "Total Reward": data["reward"].sum()
    }


dqn_results = evaluate("DQN")
threshold_results = evaluate("Threshold")

comparison = pd.DataFrame({
    "Standard DQN": metrics(dqn_results),
    "Threshold Baseline": metrics(threshold_results)
}).T

print("\nVALIDATION RESULTS - 2024")
print("=" * 60)

print(comparison.round(3).to_string())

output_path = (
    PROJECT_DIR / "results"
    / "dqn_validation_comparison.csv"
)

output_path.parent.mkdir(exist_ok=True)

comparison.to_csv(output_path)

print("\nResults saved:", output_path)