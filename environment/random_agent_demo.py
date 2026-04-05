"""
random_agent_demo.py
====================
Demonstrates TomatoFarmEnv-v0 with a purely RANDOM agent.
No model, no training — just exhaustive exploration of the action space
so you can visually inspect all environment components in action.

Run:
    python random_agent_demo.py
"""

import sys
import time
import numpy as np

# ── Try to import pygame early so we get a clean error ─────────────
try:
    import pygame
except ImportError:
    sys.exit("pygame not found. Run: pip install pygame")

from tomato_farm_env import TomatoFarmEnv, ACTION_NAMES, DiseaseType, GrowthStage


# ────────────────────────────────────────────────────────────────────
#  Configuration
# ────────────────────────────────────────────────────────────────────

NUM_EPISODES   = 5        # Number of seasons to simulate
DELAY_SECONDS  = 0.12     # Pause between steps (set 0 for max speed)
SEED           = 42
FORCE_ALL_ACTIONS = True  # Guarantee every action is tried at least once


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────

def print_header():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      TomatoFarmEnv-v0  ·  RANDOM AGENT DEMO                 ║")
    print("║  No model · No training · Pure env exploration              ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Episodes : {NUM_EPISODES:<5}   Step delay : {DELAY_SECONDS:.2f}s                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def print_env_spec(env: TomatoFarmEnv):
    print("─── Environment Specification ───────────────────────────────────")
    print(f"  Action space      : {env.action_space}")
    print(f"  Observation space : {env.observation_space}")
    print(f"  Obs shape         : {env.observation_space.shape}")
    print(f"  Obs low           : {env.observation_space.low}")
    print(f"  Obs high          : {env.observation_space.high}")
    print()
    print("  Actions:")
    for idx, name in ACTION_NAMES.items():
        print(f"    [{idx:2d}]  {name}")
    print()


def print_step(ep, step, action, obs, reward, info):
    bar_h  = "█" * int(obs[10] * 20)   # crop_health
    bar_d  = "█" * int(obs[0]  * 20)   # disease_severity
    sign   = "+" if reward >= 0 else ""
    print(
        f"  Ep{ep} Day{step:3d} │ Act[{action}] {ACTION_NAMES[action]:<26s} │ "
        f"R:{sign}{reward:6.2f} │ "
        f"H:[{bar_h:<20s}] {obs[10]:.2f} │ "
        f"D:[{bar_d:<20s}] {obs[0]:.2f} {info['disease_type']:<10s} │ "
        f"{info['growth_stage']}"
    )


def print_episode_summary(ep, total_r, steps, reason):
    print()
    print(f"  ══ Episode {ep} complete ══════════════════════")
    print(f"     Steps        : {steps}")
    print(f"     Total reward : {total_r:+.2f}")
    print(f"     End reason   : {reason}")
    print()


# ────────────────────────────────────────────────────────────────────
#  Prioritised random sampler
#  Ensures every action is exercised at least once per episode
# ────────────────────────────────────────────────────────────────────

class ExhaustiveSampler:
    """
    Wraps a Gymnasium Discrete space.
    • First N steps cycle through all N actions in random order
      to guarantee full action-space coverage.
    • Afterwards samples uniformly at random.
    """
    def __init__(self, action_space, rng: np.random.Generator):
        self._space = action_space
        self._rng   = rng
        n = action_space.n
        self._queue = list(rng.permutation(n))  # guaranteed coverage list

    def sample(self) -> int:
        if self._queue:
            return self._queue.pop()
        return int(self._space.sample())


# ────────────────────────────────────────────────────────────────────
#  Main demo loop
# ────────────────────────────────────────────────────────────────────

def run_demo():
    print_header()

    env = TomatoFarmEnv(render_mode="human")
    print_env_spec(env)

    rng        = np.random.default_rng(SEED)
    stats: list[dict] = []

    for ep in range(1, NUM_EPISODES + 1):
        obs, info = env.reset(seed=int(rng.integers(0, 99999)))
        sampler   = ExhaustiveSampler(env.action_space, rng) if FORCE_ALL_ACTIONS \
                    else None

        total_reward = 0.0
        step         = 0
        done         = False
        end_reason   = "unknown"

        print(f"─── Episode {ep}/{NUM_EPISODES} ─────────────────────────────────────────────")

        while not done:
            # pygame quit handler
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    print("\n[User closed window — exiting]")
                    _print_stats(stats)
                    return

            # Choose action
            action = sampler.sample() if sampler else int(env.action_space.sample())

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step         += 1

            print_step(ep, step, action, obs, reward, info)

            if DELAY_SECONDS > 0:
                time.sleep(DELAY_SECONDS)

            if terminated or truncated:
                done = True
                if terminated and info["crop_health"] <= 0.01:
                    end_reason = "Crop failure (health=0)"
                elif terminated:
                    end_reason = "Bankruptcy + catastrophic disease"
                else:
                    end_reason = f"Season complete — harvest bonus earned"

        print_episode_summary(ep, total_reward, step, end_reason)
        stats.append({"ep": ep, "steps": step, "reward": total_reward, "reason": end_reason})
        time.sleep(1.5)

    env.close()
    _print_stats(stats)


def _print_stats(stats):
    if not stats:
        return
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              SUMMARY ACROSS ALL EPISODES                    ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    rewards = [s["reward"] for s in stats]
    print(f"  Episodes run  : {len(stats)}")
    print(f"  Avg reward    : {np.mean(rewards):+.2f}")
    print(f"  Best reward   : {max(rewards):+.2f}  (Ep {stats[rewards.index(max(rewards))]['ep']})")
    print(f"  Worst reward  : {min(rewards):+.2f}  (Ep {stats[rewards.index(min(rewards))]['ep']})")
    print()
    for s in stats:
        print(f"  Ep {s['ep']}  steps={s['steps']:3d}  reward={s['reward']:+8.2f}  {s['reason']}")
    print("╚══════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    run_demo()
