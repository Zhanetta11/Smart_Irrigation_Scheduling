"""
DQN Agent Test
"""

import numpy as np

from src.dqn_agent import DQNAgent


print("DQN AGENT TEST")
print("=" * 45)

agent = DQNAgent(
    state_size=5,
    action_size=4,
    batch_size=32
)

# Example normalized observation
state = np.array(
    [0.54, 0.52, 0.65, 0.63, 0.28],
    dtype=np.float32
)

# Verify action selection
action = agent.select_action(
    state, training=True
)

print("Selected action:", action)

assert 0 <= action < 4

# Add synthetic experiences
for i in range(100):

    next_state = np.clip(
        state + np.random.normal(0, 0.01, size=5),
        0.0,
        1.0
    ).astype(np.float32)

    agent.remember(
        state=state,
        action=i % 4,
        reward=float(np.random.uniform(-1, 1)),
        next_state=next_state,
        done=(i % 25 == 24)
    )

print("Replay buffer size:", len(agent.memory))

assert len(agent.memory) == 100

# Perform one gradient update
loss = agent.train_step()

print("Training loss:", loss)

assert loss is not None
assert np.isfinite(loss)

# Test greedy action
greedy_action = agent.select_action(
    state, training=False
)

print("Greedy action:", greedy_action)

assert 0 <= greedy_action < 4

print("\nAll DQN agent tests PASSED!")