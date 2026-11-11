# Smart Irrigation: AI-Driven Irrigation Scheduling

### Deep Reinforcement Learning with Physics-Informed Reward Functions
---

## 1. Project Overview

Smart Irrigation is an artificial intelligence project designed to optimize agricultural irrigation scheduling using Deep Reinforcement Learning (DRL) and physical soil-water models.

Traditional irrigation systems often use fixed schedules or simple soil moisture thresholds. These approaches may not fully consider changing weather conditions, evaporation, rainfall, and soil water availability.

This project develops an intelligent irrigation agent that learns when and how much water to apply based on environmental conditions.

The system combines:

- Historical weather data
- FAO-56 Penman-Monteith evapotranspiration calculations
- Soil-water balance simulation
- A custom Gymnasium reinforcement learning environment
- Deep Q-Network (DQN) agents
- Physics-informed reward functions
- Performance evaluation and visualization

The main objective is to reduce unnecessary irrigation water consumption while maintaining suitable soil moisture conditions for plant growth.

The current implementation is a simulation-based research prototype. Physical sensors and automated irrigation hardware are potential future extensions.

---

## 2. Research Problem

Agriculture is one of the major consumers of freshwater resources. Inefficient irrigation can lead to unnecessary water consumption, excessive drainage, and unfavorable soil moisture conditions.

Conventional irrigation strategies may not adapt efficiently to daily changes in environmental conditions.

Reinforcement learning offers an alternative approach by allowing an intelligent agent to learn irrigation decisions through repeated interactions with a simulated environment.

However, a standard reinforcement learning agent does not automatically understand the physical processes that control soil moisture.

This project investigates whether incorporating soil-water balance information into the reward function can improve irrigation efficiency.

### Research Question

**Can integrating FAO-56 Penman-Monteith evapotranspiration calculations and soil-water balance indicators into a reinforcement learning reward function reduce irrigation water consumption compared with a standard DQN agent?**

### Research Hypothesis

A Physics-Informed DQN agent may achieve more water-efficient irrigation decisions than Standard DQN while maintaining suitable soil moisture conditions.

The original research target is to achieve more than 30% water savings. This is an experimental objective, not a result that has already been achieved.

---

## 3. Project Objectives

The project has the following objectives:

1. Collect and analyze historical weather data relevant to irrigation.
2. Calculate reference evapotranspiration using the FAO-56 Penman-Monteith method.
3. Develop a soil-water balance simulation model.
4. Create a custom Gymnasium environment for irrigation scheduling.
5. Implement a threshold-based irrigation baseline.
6. Develop and train a Standard Deep Q-Network agent.
7. Design a Physics-Informed Reward Function.
8. Train a Physics-Informed DQN agent.
9. Compare irrigation strategies using consistent evaluation metrics.
10. Optimize the intelligent irrigation policy and evaluate it on independent test data.

---

## 4. System Architecture

The Smart Irrigation system consists of several connected components.

### Overall Workflow

```text
Historical Weather Dataset
          |
          v
Data Cleaning and Analysis
          |
          v
FAO-56 Penman-Monteith Model
          |
          v
Reference Evapotranspiration (ET0)
          |
          v
Soil-Water Balance Model
          |
          v
Custom Gymnasium Environment
          |
          v
Reinforcement Learning Agent
          |
          v
Irrigation Decision
          |
          v
Soil Moisture Update
          |
          v
Reward Calculation
          |
          v
DQN Learning and Optimization
```

The agent repeatedly interacts with the environment.

At each simulation step:

1. The environment provides information about soil moisture and weather conditions.
2. The agent selects an irrigation action.
3. The soil-water balance model calculates the resulting soil moisture.
4. The environment calculates a reward.
5. The agent uses this experience to improve future irrigation decisions.

Each training episode represents one growing season.

---

## 5. Dataset and Data Processing

The project uses historical weather data covering the years 2020–2025.

The dataset contains environmental variables required for irrigation simulation and evapotranspiration calculations.

### Main Weather Features

| Feature | Description |
|---|---|
| Temperature | Average daily air temperature |
| Relative Humidity | Moisture content of the air |
| Solar Radiation | Solar energy received |
| Precipitation | Daily rainfall |
| Wind Speed | Wind conditions affecting evapotranspiration |
| ET0 | Calculated reference evapotranspiration |

### Dataset Preparation

The preprocessing stage includes:

- Checking missing values
- Detecting duplicate records
- Converting dates into the correct format
- Validating numerical features
- Calculating descriptive statistics
- Visualizing weather patterns
- Preparing training, validation, and testing datasets

### Dataset Split

| Dataset | Years | Purpose |
|---|---|---|
| Training | 2020–2023 | Train reinforcement learning agents |
| Validation | 2024 | Evaluate and select models |
| Testing | 2025 | Final independent evaluation |

The final testing dataset is reserved for Week 6.

This separation helps evaluate how well the trained models perform on previously unseen weather conditions.

---

## 6. FAO-56 Penman-Monteith Model

Reference evapotranspiration (ET0) represents the amount of water transferred from a standardized reference surface to the atmosphere through evaporation and transpiration.

It is an important component of agricultural water management.

This project uses the FAO-56 Penman-Monteith approach to estimate daily reference evapotranspiration.

### FAO-56 Penman-Monteith Equation

\[
ET_0 =
\frac{
0.408\Delta(R_n-G)
+
\gamma\frac{900}{T+273}u_2(e_s-e_a)
}{
\Delta+\gamma(1+0.34u_2)
}
\]

Where:

- ET0 — Reference evapotranspiration (mm/day)
- Δ — Slope of saturation vapor pressure curve
- Rn — Net radiation
- G — Soil heat flux density
- γ — Psychrometric constant
- T — Mean daily air temperature
- u2 — Wind speed at 2 meters
- es — Saturation vapor pressure
- ea — Actual vapor pressure

### Why FAO-56 Is Important

FAO-56 provides a physics-based estimate of atmospheric water demand.

Instead of making irrigation decisions using only soil moisture, the system also considers environmental conditions that influence water loss.

The calculated ET0 is used as an input to the soil-water balance model.

The implementation uses weather-derived inputs and simplifying assumptions that must be considered when interpreting the results.

---

## 7. Soil-Water Balance Model

The soil-water balance model simulates how the amount of water stored in the soil changes over time.

It considers:

- Rainfall
- Irrigation
- Evapotranspiration
- Drainage
- Surface runoff
- Soil water storage

### Water Balance Equation

\[
S_{t+1}=S_t+P_t+I_t-ET_{a,t}-D_t-R_t
\]

Where:

- S — Soil water storage
- P — Precipitation
- I — Irrigation
- ETa — Actual evapotranspiration
- D — Drainage
- R — Runoff

### Soil Parameters

The current simulation uses provisional loam soil parameters.

| Parameter | Value |
|---|---:|
| Root Depth | 0.5 m |
| Field Capacity | 0.30 |
| Wilting Point | 0.15 |
| Saturation | 0.45 |
| Initial Soil Moisture | 0.27 |
| Crop Coefficient | 1.0 |

These parameters describe a simplified soil system and are not calibrated measurements from a real agricultural field.

### Model Behavior

During each simulation day, the model:

1. Receives rainfall and irrigation.
2. Calculates available water.
3. Determines possible surface runoff.
4. Estimates actual evapotranspiration.
5. Calculates drainage above field capacity.
6. Updates soil water storage.
7. Calculates the new soil moisture.

The model was tested under dry weather, irrigation, and heavy rainfall scenarios.

The basic water balance tests passed successfully.

---

## 8. Reinforcement Learning Environment

A custom irrigation environment was implemented using Gymnasium.

Gymnasium provides a standardized interface for reinforcement learning environments.

The environment connects the weather dataset, soil-water balance model, and intelligent irrigation agent.

### State Space

The agent receives five normalized observations:

1. Current soil moisture
2. Air temperature
3. Relative humidity
4. Solar radiation
5. Reference evapotranspiration

These observations describe the current environmental state used by the agent.

### Action Space

The agent selects one of four discrete irrigation actions.

| Action | Irrigation |
|---|---:|
| 0 | 0 mm |
| 1 | 2 mm |
| 2 | 5 mm |
| 3 | 8 mm |

The agent learns to choose an action based on the environmental conditions.

### Episode Structure

One episode represents one growing season.

At the beginning of an episode, the environment resets the soil moisture and simulation counters.

During each step:

- The agent selects an action.
- The environment applies irrigation.
- Soil moisture is updated.
- A reward is calculated.
- The next observation is returned.

The episode ends when all weather records for the selected growing season have been processed.

The Gymnasium environment has passed its basic interface and integration tests.

---

## 9. Reinforcement Learning Models

Three irrigation strategies have been implemented and compared.

### 9.1 Threshold-Based Irrigation

The threshold-based strategy uses predefined soil moisture limits to determine when irrigation is needed.

It serves as a simple reference strategy.

Advantages:

- Easy to understand
- Computationally inexpensive
- Does not require neural network training

Limitations:

- Uses fixed decision rules
- May not adapt optimally to different environmental conditions
- Can allow soil moisture to fall outside the desired range

### 9.2 Standard Deep Q-Network

Standard DQN is a reinforcement learning algorithm that uses a neural network to estimate the expected future reward of different actions.

The agent learns a policy through repeated interaction with the environment.

The implementation uses:

- A neural network
- Experience replay
- Epsilon-greedy exploration
- Discounted future rewards
- A target network
- Gradient-based optimization

The Standard DQN reward encourages suitable soil moisture while penalizing unnecessary irrigation, drainage, and water stress.

### 9.3 Physics-Informed Deep Q-Network

Physics-Informed DQN extends the Standard DQN approach by using a reward function that explicitly incorporates physical indicators of soil-water behavior.

The reward considers:

- Soil moisture conditions
- Irrigation consumption
- Drainage losses
- Water stress

The goal is to encourage the agent to select irrigation actions that maintain suitable moisture while reducing unnecessary water consumption.

The current approach is physics-informed through reward design and the simulation environment. It is not a physics-informed neural network that directly embeds differential equations into the neural network architecture.

---

## 10. DQN Training Configuration

Both DQN models were trained using PyTorch.

### Training Parameters

| Parameter | Value |
|---|---:|
| Training Episodes | 400 |
| State Size | 5 |
| Action Size | 4 |
| Learning Rate | 0.0005 |
| Discount Factor | 0.99 |
| Initial Epsilon | 1.0 |
| Minimum Epsilon | 0.05 |
| Epsilon Decay | 0.99 |
| Batch Size | 64 |
| Replay Buffer Size | 20,000 |
| Target Update Frequency | 250 |
| Random Seed | 42 |

### Training Process

During training:

1. A growing season is selected from 2020–2023.
2. The environment is initialized.
3. The agent observes the environmental state.
4. An irrigation action is selected.
5. The environment calculates the next state and reward.
6. The experience is stored in replay memory.
7. The neural network is updated using sampled experiences.
8. The exploration rate gradually decreases.
9. The agent is periodically evaluated using 2024 data.

The best-performing model checkpoint is saved according to validation reward.

The final model and training history are also saved.

---

## 11. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Neural network implementation and DQN training |
| Gymnasium | Custom reinforcement learning environment |
| NumPy | Numerical calculations and arrays |
| pandas | Dataset processing and analysis |
| Matplotlib | Data visualization and training graphs |
| Jupyter Notebook | Interactive analysis and experiments |
| pathlib | File and directory management |
| dataclasses | Structured soil configuration |
| random | Random season selection and exploration |
| Git | Version control |
| Visual Studio Code | Development environment |

The current implementation runs on a CPU and does not require a dedicated GPU.

---

## 12. Experimental Results

The trained agents were evaluated on the 2024 validation dataset.

### Comparison of Irrigation Strategies

| Metric | Standard DQN | Physics-Informed DQN | Threshold Baseline |
|---|---:|---:|---:|
| Irrigation (mm) | 588.0 | 550.0 | 336.0 |
| Target Moisture Days (%) | 100.0 | 100.0 | 72.55 |
| Drainage (mm) | 111.094 | 86.349 | 32.004 |
| Average Stress Factor | 0.996 | 0.980 | 0.780 |

### Water Savings

Physics-Informed DQN reduced irrigation consumption compared with Standard DQN.

\[
Water\ Savings =
\frac{588-550}{588}\times100
=6.46\%
\]

### Drainage Reduction

Physics-Informed DQN also reduced drainage compared with Standard DQN.

\[
Drainage\ Reduction =
\frac{111.094-86.349}{111.094}\times100
\approx22.3\%
\]

### Interpretation

The Physics-Informed DQN:

- Used 6.46% less irrigation water than Standard DQN.
- Reduced simulated drainage by approximately 22.3%.
- Maintained the target soil moisture range throughout the 2024 validation season.
- Achieved a slightly lower average stress factor than Standard DQN.

The threshold-based strategy used less irrigation water but maintained soil moisture within the target range on only 72.55% of days.

Therefore, irrigation consumption alone is not sufficient to determine the best strategy. Water savings must be considered together with soil moisture control and water stress.

The results are based on simulation rather than real agricultural field experiments.

---

## 13. Training Performance Analysis

### Standard DQN

The Standard DQN training reward increased during the initial episodes and later stabilized around 110–115.

Irrigation consumption varied between growing seasons.

The training loss gradually increased, indicating that additional stability analysis would be useful.

### Physics-Informed DQN

The Physics-Informed DQN training reward increased from low and negative values to approximately 60–65.

The reward became relatively stable after approximately 150–200 episodes.

The percentage of days within the target soil moisture range approached 100% during training.

Irrigation consumption remained variable across episodes.

Training loss fluctuated and showed a gradual upward trend during later training stages.

### Important Note

The reward values of Standard DQN and Physics-Informed DQN should not be compared directly because the models use different reward functions.

The main comparison should focus on irrigation consumption, soil moisture control, drainage, and water stress.

---

## 14. Project Structure

The project is organized into data, source code, models, experiments, tests, and documentation.

```text

Smart_Irrigation/
│
├── .venv/
├── .venv-1/
│
├── data/
│   ├── soil_simulation_results.csv
│   ├── test_weather.csv
│   ├── train_weather.csv
│   ├── val_weather.csv
│   ├── weather_cleaned.csv
│   ├── weather_growing_season.csv
│   ├── weather_raw.csv
│   └── weather_with_et0.csv
│
├── models/
│   ├── physics_dqn_best.pt
│   ├── physics_dqn_final.pt
│   ├── standard_dqn_best.pt
│   └── standard_dqn_final.pt
│
├── notebooks/
│   ├── week1_analysis.ipynb
│   ├── week2_fao56.ipynb
│   ├── week2_soil_simulation.ipynb
│   └── week3_agent_comparison.ipynb
│
├── results/
│   ├── agents_comparison_2024.csv
│   ├── agents_comparison_2024.png
│   ├── dqn_training_curves.png
│   ├── dqn_training_history.csv
│   ├── dqn_validation_comparison.csv
│   ├── et0_growing_seasons.png
│   ├── physics_dqn_training_curves.png
│   ├── physics_dqn_training_history.csv
│   ├── random_agent_20_runs.csv
│   ├── soil_moisture_seasons.png
│   ├── week3_agent_comparison.csv
│   ├── week3_agent_comparison.png
│   └── yearly_irrigation_performance.csv
│
├── src/
│   ├── __pycache__/
│   ├── dqn_agent.py
│   ├── fao56.py
│   ├── irrigation_env.py
│   ├── physics_reward.py
│   └── soil_water_balance.py
│
├── .gitignore
├── compare_agents.py
├── evaluate_dqn.py
├── README.md
├── requirements.txt
├── test_dqn_agent.py
├── test_fao56.py
├── test_irrigation_env.py
├── test_physics_reward.py
├── test_reward_modes.py
├── test_soil_water_balance.py
├── train_physics_dqn.py
└── train.py
```

This structure summarizes the main project components. Exact filenames and additional test scripts may differ in the local implementation.

### Main File Responsibilities

**soil_water_balance.py**

Implements the physical simulation of soil moisture and water storage.

**irrigation_env.py**

Defines the custom Gymnasium environment, state space, action space, reward selection, and episode transitions.

**dqn_agent.py**

Implements the DQN agent, action selection, replay memory, neural network training, and model checkpoint management.

**physics_reward.py**

Defines the physics-informed reward function.

**train.py**

Trains the Standard DQN agent and saves training results.

**train_physics_dqn.py**

Trains the Physics-Informed DQN agent and saves model checkpoints and visualizations.

**compare_agents.py**

Evaluates the trained agents and the threshold-based baseline using the same validation weather data.

**week5_results.md**

Documents the Physics-Informed DQN implementation, training behavior, and validation results.

---

## 15. Installation and Setup

### Requirements

- Python
- pip
- Virtual environment
- Project dependencies from requirements.txt

### Step 1 — Open the Project

Open the Smart_Irrigation folder in Visual Studio Code.

### Step 2 — Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3 — Install Dependencies

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4 — Run Tests

Examples:

```powershell
.\.venv\Scripts\python.exe test_soil_water_balance.py
```

```powershell
.\.venv\Scripts\python.exe test_irrigation_env.py
```

```powershell
.\.venv\Scripts\python.exe test_physics_reward.py
```

These commands assume the test scripts are located in the project root.

### Step 5 — Train Standard DQN

```powershell
.\.venv\Scripts\python.exe train.py
```

### Step 6 — Train Physics-Informed DQN

```powershell
.\.venv\Scripts\python.exe train_physics_dqn.py
```

### Step 7 — Compare Agents

```powershell
.\.venv\Scripts\python.exe compare_agents.py
```

Training is not required every time the project is demonstrated if the trained model checkpoints are already available.

---

## 16. Development Progress

### Week 1 — Weather Data Analysis

**Completed**

- Collected historical weather data.
- Cleaned and validated the dataset.
- Analyzed weather variables.
- Created exploratory visualizations.

### Week 2 — Physical Modeling

**Completed**

- Implemented FAO-56 ET0 calculations.
- Developed the soil-water balance model.
- Tested water storage, irrigation, drainage, and runoff calculations.

### Week 3 — Reinforcement Learning Environment

**Completed**

- Created the custom Gymnasium environment.
- Defined observations and irrigation actions.
- Integrated the soil-water balance model.
- Tested environment interactions.
- Implemented baseline irrigation behavior.

### Week 4 — Standard DQN

**Completed**

- Implemented the DQN agent using PyTorch.
- Trained the model for 400 episodes.
- Saved model checkpoints.
- Generated training graphs.
- Evaluated the agent using validation data.

### Week 5 — Physics-Informed DQN

**Core implementation completed**

- Designed a physics-informed reward function.
- Integrated it into the environment.
- Trained the Physics-Informed DQN for 400 episodes.
- Saved model checkpoints and training history.
- Compared Standard DQN, Physics-Informed DQN, and the threshold baseline.
- Analyzed training graphs and validation results.

### Week 6 — Optimization and Final Evaluation

**Planned**

- Review reward coefficients.
- Conduct reward optimization experiments.
- Evaluate training stability.
- Select the final model using validation data.
- Test the selected models on the reserved 2025 dataset.
- Generate final comparison tables and visualizations.

### Week 7 — Documentation and Presentation

**Planned**

- Finalize technical documentation.
- Improve project reproducibility.
- Review the GitHub repository.
- Prepare the final presentation.
- Demonstrate the system.
- Summarize results, limitations, and future improvements.

---

## 17. Current Limitations

The current system has several limitations.

**Simulation-Based Evaluation**

The project uses a simplified soil-water balance simulation rather than a real agricultural field.

**No Physical Sensor Integration Yet**

ESP32 sensors and real-time soil moisture measurement are not currently integrated into the prototype.

**Simplified Soil Parameters**

The soil parameters are provisional and have not been calibrated using field measurements.

**Limited Irrigation Actions**

The agent can choose only four irrigation amounts: 0, 2, 5, and 8 mm.

**Training Stability**

Training loss and irrigation consumption show variability, requiring additional investigation.

**Generalization**

The final independent evaluation on 2025 weather data has not yet been completed.

**Water Savings Target**

The Physics-Informed DQN currently achieves 6.46% savings compared with Standard DQN on validation data, below the original research target of more than 30%.

---

## 18. Future Improvements

Potential future extensions include:

- Integration with ESP32 soil moisture sensors
- Real-time weather data collection
- Support for multiple soil types
- Crop-specific water requirements
- More realistic crop growth models
- Improved DQN hyperparameters
- Additional reinforcement learning algorithms
- Continuous irrigation control
- A dashboard for monitoring irrigation decisions
- Deployment with physical irrigation equipment

These features are possible extensions and are not part of the completed Week 5 prototype.

---

## 19. Conclusion

The Smart Irrigation project demonstrates how Deep Reinforcement Learning can be combined with physical soil-water models to develop intelligent irrigation scheduling strategies.

The current prototype successfully integrates historical weather data, FAO-56 evapotranspiration calculations, soil-water balance simulation, and a custom Gymnasium environment.

Two DQN agents have been implemented and trained: Standard DQN and Physics-Informed DQN.

Validation results from 2024 show that Physics-Informed DQN reduced irrigation water consumption by 6.46% and drainage by approximately 22.3% compared with Standard DQN, while maintaining soil moisture within the target range throughout the simulated growing season.

Although the original water savings target has not yet been achieved, the results provide a useful foundation for further optimization and evaluation.

The next stages will focus on improving model efficiency, conducting independent testing, and completing the final research documentation.