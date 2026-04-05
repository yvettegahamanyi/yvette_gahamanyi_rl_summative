# TomatoFarmEnv-v0

## Description

TomatoFarmEnv-v0 simulates a **120-day tomato growing season** in which an agent acts as a farm manager responsible for protecting a crop from disease, pest infestation, drought, and budget overruns. The environment models the progression of four distinct disease types — fungal, bacterial, viral, and pest-driven — each of which spreads at a rate influenced by daily weather conditions (temperature, humidity, and rainfall). The agent must select the correct treatment at the right time while managing a finite budget and maintaining adequate soil moisture for healthy plant growth.

The goal is to reach **Day 120** with the highest possible crop health and the lowest possible disease severity in order to maximise the harvest bonus. A poorly managed farm will see crop health degrade to zero before the season ends, resulting in a total crop failure.

This environment is well suited for **value-based methods (DQN)** where the agent learns a Q-value for each discrete treatment action, and for **policy gradient methods (REINFORCE, PPO)** where the agent learns a stochastic policy over the treatment decisions given the continuous weather and crop state.

---

## Action Space

The action space is `Discrete(10)`.

| Value | Action                  | Description                                                                                                                                                                                     |
| ----- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0`   | Do Nothing / Monitor    | Observe the field without intervening. Incurs a passive penalty if disease is spreading unchecked.                                                                                              |
| `1`   | Apply Fungicide (Light) | Low-dose fungicide. Effective against **fungal** disease only. Low cost.                                                                                                                        |
| `2`   | Apply Fungicide (Heavy) | High-dose fungicide. Strongly effective against **fungal** disease. Higher cost.                                                                                                                |
| `3`   | Apply Bactericide       | Targeted treatment for **bacterial** disease only. Penalised if applied to other disease types.                                                                                                 |
| `4`   | Apply Insecticide       | Reduces **pest population** and pest-driven disease severity. Ineffective against fungal/bacterial/viral types.                                                                                 |
| `5`   | Apply Organic Pesticide | Broad-spectrum, mild treatment. Provides a small reduction to any active disease and pest population. Low cost, low penalty risk.                                                               |
| `6`   | Irrigate (Light)        | Adds a moderate amount of soil moisture. Rewarded when soil is dry, penalised when soil is already saturated.                                                                                   |
| `7`   | Irrigate (Heavy)        | Adds a large amount of soil moisture. Highly rewarded in drought conditions, heavily penalised if soil is already wet (root rot risk).                                                          |
| `8`   | Prune Infected Leaves   | Physically removes infected tissue, reducing disease severity at a minor cost to overall crop health. Only rewarded when severity exceeds a meaningful threshold.                               |
| `9`   | Call Agronomist         | Expert consultation. Provides the largest single-step reduction to both disease severity and pest population but carries the highest budget cost. Only rewarded when the situation is critical. |

---

## Observation Space

The observation space is `Box(12,)` with `dtype=float32`.

| Index | Feature            | Min  | Max   | Description                                                       |
| ----- | ------------------ | ---- | ----- | ----------------------------------------------------------------- |
| `0`   | `disease_severity` | 0.0  | 1.0   | Fraction of the crop currently infected                           |
| `1`   | `disease_type`     | 0.0  | 4.0   | Encoded disease: 0=None, 1=Fungal, 2=Bacterial, 3=Viral, 4=Pest   |
| `2`   | `temperature`      | 15.0 | 40.0  | Ambient air temperature in °C                                     |
| `3`   | `humidity`         | 0.0  | 1.0   | Relative humidity (drives disease spread rate)                    |
| `4`   | `rainfall_prob`    | 0.0  | 1.0   | Probability of rainfall today                                     |
| `5`   | `growth_stage`     | 0.0  | 4.0   | 0=Seedling, 1=Vegetative, 2=Flowering, 3=Fruiting, 4=Harvest      |
| `6`   | `season_day`       | 0.0  | 120.0 | Current timestep within the 120-day season                        |
| `7`   | `budget_remaining` | 0.0  | 1.0   | Normalised remaining treatment budget                             |
| `8`   | `soil_moisture`    | 0.0  | 1.0   | Soil water content (< 0.2 = drought stress, > 0.9 = waterlogging) |
| `9`   | `last_action`      | 0.0  | 9.0   | The action taken in the previous timestep                         |
| `10`  | `crop_health`      | 0.0  | 1.0   | Overall fraction of healthy crop biomass remaining                |
| `11`  | `pest_population`  | 0.0  | 1.0   | Normalised aphid / whitefly population                            |

---

## Rewards

The reward is a **shaped scalar** computed at every timestep from three components:

**1. Daily Survival Bonus**
A reward of `+0.5 × crop_health` is given every step to encourage the agent to keep the crop alive throughout the season rather than optimising only for the final harvest.

**2. Action Reward (shaped per action)**
Each action produces a reward or penalty depending on whether it was appropriate for the current state:

- Correct pesticide matched to the active disease type yields a **large positive reward** (up to +14 for an agronomist call).
- Applying the wrong pesticide for the active disease type yields a **small negative reward** (−1.5 to −4.0).
- Irrigating when soil is dry is rewarded; irrigating when soil is already saturated is penalised.
- `Do Nothing` when disease severity exceeds 0.30 yields −2.0 (negligence penalty).
- Actions taken after the budget is exhausted incur a heavy over-budget penalty (`−12 × cost`).

**3. Terminal Reward**
Applied once at episode end:

- **Crop failure** (health reaches 0): `−50`
- **Harvest** (season complete at Day 120): `+100 × crop_health × (1 − disease_severity)`
- **Bankruptcy + catastrophic disease**: `−30`

---

## Starting State

At the beginning of every episode (`reset()`), the environment is initialised as follows:

| Variable           | Starting Value              |
| ------------------ | --------------------------- |
| `crop_health`      | `1.0`                       |
| `disease_severity` | `0.0`                       |
| `disease_type`     | `NONE (0)`                  |
| `pest_population`  | `0.0`                       |
| `season_day`       | `0`                         |
| `growth_stage`     | `SEEDLING (0)`              |
| `budget_remaining` | `1.0`                       |
| `soil_moisture`    | `0.6`                       |
| `temperature`      | Sampled from `U(22, 28)` °C |
| `humidity`         | Sampled from `U(0.4, 0.6)`  |
| `rainfall_prob`    | Sampled from `U(0.1, 0.3)`  |
| `last_action`      | `DO_NOTHING (0)`            |

---

## Episode End

The episode ends under any of the following three conditions:

**Termination** — the episode ends early and the agent has failed:

- `crop_health ≤ 0.0` — the entire crop has died due to disease, drought, or pest damage.
- `budget_remaining ≤ 0.0` AND `disease_severity ≥ 0.9` — the agent has exhausted its budget while the crop is in a near-total disease state with no recovery path.

**Truncation** — the episode ends successfully at the natural season boundary:

- `season_day ≥ 120` — the full 120-day growing season has elapsed and the crop is harvested. The harvest bonus is proportional to the crop health and inverse disease severity at this point.
