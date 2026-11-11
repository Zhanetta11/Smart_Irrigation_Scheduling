
"""
Standard DQN Training

Training: 2020-2023
Validation: 2024
Final test: 2025 (reserved for Week 6)

Each episode represents one growing season.
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.dqn_agent import DQNAgent
from src.irrigation_env import SmartIrrigationEnv


# ==========================================
# 1. Configuration
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

EPISODES = 400
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CPU optimization
torch.set_num_threads(2)


# ==========================================
# 2. Load datasets
# ==========================================

train_df = pd.read_csv(
    DATA_DIR / "train_weather.csv",
    parse_dates=["time"]
)

val_df = pd.read_csv(
    DATA_DIR / "val_weather.csv",
    parse_dates=["time"]
)

# Create independent training seasons
training_seasons = {
    int(year): group.sort_values("time")
    .reset_index(drop=True)
    for year, group in train_df.groupby(
        train_df["time"].dt.year
    )
}

print("Training seasons:", list(training_seasons))
print("Validation records:", len(val_df))

assert set(training_seasons) == {
    2020, 2021, 2022, 2023
}

assert len(val_df) == 153


# ==========================================
# 3. Initialize DQN Agent
# ==========================================

agent = DQNAgent(
    state_size=5,
    action_size=4,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.99,
    batch_size=64,
    buffer_size=20000,
    target_update_frequency=250,
    seed=SEED
)


# ==========================================
# 4. Episode simulation
# ==========================================

def run_episode(weather_data, training=True):

    env = SmartIrrigationEnv(weather_data)

    state, info = env.reset()

    total_reward = 0.0
    total_loss = []
    moisture_values = []
    irrigation_values = []
    drainage_values = []

    done = False

    while not done:

        action = agent.select_action(
            state,
            training=training
        )

        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        if training:

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            loss = agent.train_step()

            if loss is not None:
                total_loss.append(loss)

        total_reward += reward

        moisture_values.append(
            info["soil_moisture"]
        )

        irrigation_values.append(
            info["irrigation_mm"]
        )

        drainage_values.append(
            info["drainage_mm"]
        )

        state = next_state

    env.close()

    moisture_array = np.array(moisture_values)

    return {
        "reward": total_reward,
        "irrigation_mm": sum(irrigation_values),
        "drainage_mm": sum(drainage_values),
        "average_moisture": moisture_array.mean(),
        "target_days_pct": (
            np.mean(
                (moisture_array >= 0.25)
                & (moisture_array <= 0.30)
            ) * 100
        ),
        "loss": (
            float(np.mean(total_loss))
            if total_loss else np.nan
        )
    }


# ==========================================
# 5. Training Loop
# ==========================================

print("\nSTARTING DQN TRAINING")
print("=" * 65)

history = []
best_validation_reward = -float("inf")

for episode in range(1, EPISODES + 1):

    # Select one season per episode
    year = random.choice(
        list(training_seasons.keys())
    )

    train_metrics = run_episode(
        training_seasons[year],
        training=True
    )

    agent.decay_epsilon()

    record = {
        "episode": episode,
        "year": year,
        "epsilon": agent.epsilon,
        **train_metrics
    }

    # Validate periodically on unseen 2024 data
    if episode % 10 == 0:

        val_metrics = run_episode(
            val_df,
            training=False
        )

        record["val_reward"] = (
            val_metrics["reward"]
        )

        record["val_irrigation_mm"] = (
            val_metrics["irrigation_mm"]
        )

        record["val_target_days_pct"] = (
            val_metrics["target_days_pct"]
        )

        # Save best model by validation reward
        if (
            val_metrics["reward"]
            > best_validation_reward
        ):

            best_validation_reward = (
                val_metrics["reward"]
            )

            agent.save(
                MODEL_DIR / "standard_dqn_best.pt"
            )

    history.append(record)

    if episode % 20 == 0:

        print(
            f"Episode {episode:3d}/{EPISODES} | "
            f"Year: {year} | "
            f"Reward: {train_metrics['reward']:8.2f} | "
            f"Water: {train_metrics['irrigation_mm']:6.1f} mm | "
            f"Target: {train_metrics['target_days_pct']:5.1f}% | "
            f"Epsilon: {agent.epsilon:.3f}"
        )


# ==========================================
# 6. Save Training Results
# ==========================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    RESULTS_DIR / "dqn_training_history.csv",
    index=False
)

agent.save(
    MODEL_DIR / "standard_dqn_final.pt"
)

print("\nTRAINING COMPLETED!")
print("Best validation reward:", best_validation_reward)


# ==========================================
# 7. Training Visualization
# ==========================================

fig, axes = plt.subplots(
    3, 1,
    figsize=(12, 12)
)

# Reward
axes[0].plot(
    history_df["episode"],
    history_df["reward"],
    alpha=0.4,
    label="Training Reward"
)

axes[0].plot(
    history_df["episode"],
    history_df["reward"].rolling(20).mean(),
    label="20-Episode Moving Average",
    linewidth=2
)

axes[0].set_title("DQN Training Reward")
axes[0].set_ylabel("Reward")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Irrigation
axes[1].plot(
    history_df["episode"],
    history_df["irrigation_mm"]
)

axes[1].set_title("Irrigation Water Consumption")
axes[1].set_ylabel("Irrigation (mm)")
axes[1].grid(alpha=0.3)

# Loss
axes[2].plot(
    history_df["episode"],
    history_df["loss"]
)

axes[2].set_title("DQN Training Loss")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Loss")
axes[2].grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "dqn_training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nSaved model checkpoints and training graphs.")