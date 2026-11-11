
"""
Irrigation Strategy Comparison

Validation year: 2024

Strategies:
1. Standard DQN
2. Physics-Informed DQN
3. Threshold Baseline
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.dqn_agent import DQNAgent
from src.irrigation_env import SmartIrrigationEnv


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)

weather = pd.read_csv(
    PROJECT_DIR / "data" / "val_weather.csv",
    parse_dates=["time"]
)

assert set(weather["time"].dt.year) == {2024}


def load_agent(filename):
    agent = DQNAgent(
        state_size=5,
        action_size=4
    )
    agent.load(MODEL_DIR / filename)
    return agent


standard_agent = load_agent(
    "standard_dqn_best.pt"
)

physics_agent = load_agent(
    "physics_dqn_best.pt"
)


def evaluate(strategy, agent=None):

    # Use the same physical environment for all
    # strategies. Reward is not compared here.
    env = SmartIrrigationEnv(
        weather,
        reward_mode="basic"
    )

    state, _ = env.reset(seed=42)
    records = []

    while True:

        if agent is not None:
            action = agent.select_action(
                state,
                training=False
            )
        else:
            moisture = env.soil.get_soil_moisture()
            action = 3 if moisture < 0.25 else 0

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        records.append({
            "soil_moisture": info["soil_moisture"],
            "irrigation_mm": info["irrigation_mm"],
            "drainage_mm": info["drainage_mm"],
            "stress_factor": info["stress_factor"]
        })

        if terminated or truncated:
            break

    env.close()
    return pd.DataFrame(records)


def calculate_metrics(df):
    return {
        "Irrigation (mm)": df["irrigation_mm"].sum(),
        "Target Days (%)": (
            df["soil_moisture"]
            .between(0.25, 0.30)
            .mean() * 100
        ),
        "Drainage (mm)": df["drainage_mm"].sum(),
        "Stress Factor": df["stress_factor"].mean()
    }


results = {
    "Standard DQN": evaluate(
        "Standard DQN",
        standard_agent
    ),
    "Physics-Informed DQN": evaluate(
        "Physics-Informed DQN",
        physics_agent
    ),
    "Threshold Baseline": evaluate(
        "Threshold Baseline"
    )
}

comparison = pd.DataFrame({
    name: calculate_metrics(data)
    for name, data in results.items()
}).T

print("\nVALIDATION COMPARISON - 2024")
print("=" * 70)
print(comparison.round(3).to_string())

comparison.to_csv(
    RESULTS_DIR / "agents_comparison_2024.csv"
)


# Water savings relative to Standard DQN
standard_water = comparison.loc[
    "Standard DQN", "Irrigation (mm)"
]

physics_water = comparison.loc[
    "Physics-Informed DQN", "Irrigation (mm)"
]

savings = (
    (standard_water - physics_water)
    / standard_water * 100
)

print(
    f"\nPhysics DQN water savings "
    f"vs Standard DQN: {savings:.2f}%"
)


# ==========================================
# Visualization
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

metrics = [
    "Irrigation (mm)",
    "Target Days (%)",
    "Drainage (mm)",
    "Stress Factor"
]

for ax, metric in zip(axes.flat, metrics):
    comparison[metric].plot(
        kind="bar",
        ax=ax
    )
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "agents_comparison_2024.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nComparison completed and saved!")