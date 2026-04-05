# Reinforcement Learning Summative Assignment

## TomatoFarmEnv-v0 — Custom Gymnasium Environment

## Project Summary

This project designs, implements, and compares three reinforcement learning algorithms on a custom **120-day tomato farm management** environment built with the OpenAI Gymnasium framework. An RL agent acts as a farm manager making daily decisions — selecting pesticides, managing irrigation, and calling for expert help — to protect a crop from disease and maximise harvest yield.

Three algorithms are trained and benchmarked:

| Algorithm | Type                           | Library           |
| --------- | ------------------------------ | ----------------- |
| DQN       | Value-Based                    | Stable-Baselines3 |
| REINFORCE | Policy Gradient (Monte-Carlo)  | Custom PyTorch    |
| PPO       | Policy Gradient (Actor-Critic) | Stable-Baselines3 |

Each algorithm runs **10 hyperparameter experiments** (12 for DQN). The best model from each algorithm is saved and can be loaded into `main.py` for a live pygame visualisation or a FastAPI REST server.

---

## Repository Structure

```
.
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Visualization GUI components
├── training/
│   ├── dqn_training.py          # Training script for DQN using SB3
│   ├── pg_training.py           # Training script for PPO/other PG using SB3
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Entry point for running best performing model
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/[your-username]/[repo-name].git
cd [repo-name]
pip install -r requirements.txt
```

### 2. Train the Models

Open each notebook in **Google Colab** (GPU recommended), upload `tomato_farm_env.py` when prompted, then run all cells.

Each notebook trains 10 experiments, saves every model to `saved_models/`, prints a results table, and generates a `2×5` learning curve plot.

```
01_DQN_TomatoFarm.ipynb      →  saved_models/DQN_*.zip
02_REINFORCE_TomatoFarm.ipynb →  saved_models/REINFORCE_*.pt
03_PPO_TomatoFarm.ipynb      →  saved_models/PPO_*.zip
```

### 3. Run the Visual Demo

```bash
# PPO agent (best overall stability)
python main.py --model saved_models/PPO_E01-baseline.zip --algo PPO

# DQN agent
python main.py --model saved_models/DQN_E03-high_lr.zip --algo DQN

# REINFORCE agent
python main.py --model saved_models/REINFORCE_E01-baseline.pt --algo REINFORCE

# Run more episodes
python main.py --model saved_models/PPO_E01-baseline.zip --algo PPO --episodes 5
```

### 4. Run the REST API

```bash
python main.py --model saved_models/PPO_E01-baseline.zip --algo PPO --api
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 5. Headless JSON Output

```bash
python main.py --model saved_models/PPO_E01-baseline.zip --algo PPO --headless
```

Prints one JSON object per step to stdout — pipe into any tool or frontend.

---

## main.py — Full Usage Reference

```
usage: main.py [-h] --model MODEL --algo {PPO,DQN,REINFORCE}
               [--api] [--headless] [--episodes N]
               [--host HOST] [--port PORT]

required arguments:
  --model   Path to saved model (.zip for PPO/DQN, .pt for REINFORCE)
  --algo    Algorithm: PPO | DQN | REINFORCE

optional arguments:
  --api       Start FastAPI server instead of pygame window
  --headless  Print JSON to stdout, no display required
  --episodes  Number of visual demo episodes (default: 3)
  --host      API server host (default: 0.0.0.0)
  --port      API server port (default: 8000)
```

---

## Pygame Visualisation

When running in default (visual) mode, the window shows:

| Panel                             | Description                                                        |
| --------------------------------- | ------------------------------------------------------------------ |
| Farm grid                         | 120 plant cells coloured by health (green) and disease (brown/red) |
| Agent decision box                | The action chosen and the step reward received                     |
| Policy distribution               | Probability bar for each of the 10 actions                         |
| Exploration ↔ Exploitation slider | Red = uncertain, Green = highly confident                          |
| Reward sparkline                  | Episode reward history across all steps                            |
| JSON preview                      | The exact API payload rendered live in the HUD                     |

**Keyboard:** Press `Q` to quit at any time.

---

## REST API Endpoints

| Method | Endpoint      | Description                                     |
| ------ | ------------- | ----------------------------------------------- |
| `GET`  | `/`           | Health check                                    |
| `GET`  | `/reset`      | Start a new 120-day episode                     |
| `GET`  | `/step_auto`  | Agent selects next action → returns state JSON  |
| `POST` | `/step`       | You send `{"action": 0–9}` → returns state JSON |
| `GET`  | `/stream`     | Server-Sent Events stream (one step / 0.5 s)    |
| `GET`  | `/model_info` | Loaded model metadata                           |
| `GET`  | `/docs`       | Auto-generated Swagger UI                       |

### Example JSON Response (`/step_auto`)

```json
{
  "step": 45,
  "done": false,
  "total_reward": 87.3,
  "step_reward": 5.5,
  "action": {
    "id": 2,
    "name": "Fungicide  (Heavy Dose)",
    "probability": 0.843
  },
  "observation": {
    "crop_health": 0.82,
    "disease_type": "FUNGAL",
    "disease_severity": 0.18,
    "growth_stage": "FLOWERING",
    "season_day": 45,
    "budget_remaining": 0.61,
    "temperature_c": 28.4,
    "humidity": 0.72,
    "soil_moisture": 0.54,
    "pest_population": 0.0
  },
  "model": {
    "algo": "PPO",
    "label": "PPO_E01-baseline.zip"
  }
}
```

### Frontend Integration

**JavaScript / React:**

```js
const source = new EventSource("http://localhost:8000/stream");
source.onmessage = (e) => {
  const state = JSON.parse(e.data);
  updateFarmDashboard(state);
};
```

**Flutter / Dart:**

```dart
final res = await http.get(Uri.parse('http://localhost:8000/step_auto'));
final state = jsonDecode(res.body);
final health = state['observation']['crop_health'];
```

---

## Algorithms & Key Results

### DQN — 12 Experiments

Best mean reward: **397.23** (E03, E04, E06, E08)

Key sensitivity:

- **Learning rate** — LR=1e-4 (E02) dropped mean to 331.50 (too slow to converge)
- **Target update interval** — interval=100 (E09) dropped mean to 338.43 (unstable Bellman targets)
- Most other hyperparameters had low impact on this environment

### REINFORCE — 10 Experiments

Best mean reward: **397.23** (E01 baseline)

Key sensitivity:

- **Learning rate** — LR=1e-4 (E02) dropped mean to 366.60
- **Entropy coefficient** — ent=0.10 (E06) dropped mean to 366.20
- 8 out of 10 configurations converged to identical policies — see report for analysis

### PPO — 10 Experiments

Best mean reward: **397.23** (all 10 configurations)

- Most hyperparameter-robust algorithm tested
- All 10 configs converged identically regardless of LR, clip range, or epochs
- Recommended algorithm for this environment

### Algorithm Comparison

| Algorithm | Best Mean  | Std Dev    | Hyp. Sensitivity | Recommended                      |
| --------- | ---------- | ---------- | ---------------- | -------------------------------- |
| DQN       | 397.23     | 139.67     | Moderate         | For off-policy, discrete actions |
| REINFORCE | 397.23     | 139.70     | High             | For learning/comparison only     |
| **PPO**   | **397.23** | **139.67** | **Low**          | **Production deployment**        |

---

## Dependencies

```
gymnasium==1.2.3
stable-baselines3==2.7.1
torch==2.11.0
pygame==2.6.1
numpy==2.4.2
matplotlib==3.10.8
pandas==3.0.1
fastapi==0.135.2
uvicorn==0.42.0
```

Install all: `pip install -r requirements.txt`

---

## Report

The full written report (`RL_Summative_Report.docx`) covers:

- Environment specification (action space, observation space, reward structure)
- Algorithm design and implementation details
- Complete hyperparameter experiment tables for all three algorithms
- Results discussion including the dominant attractor basin finding
- Generalisation analysis across unseen seeds
- Conclusion and recommended algorithm

---

## License

MIT — free to use, modify, and extend for academic and research purposes.
