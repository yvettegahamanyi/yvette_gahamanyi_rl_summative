"""
TomatoFarmEnv-v0 — Custom Gymnasium Environment
================================================
A reinforcement learning environment simulating a tomato growing season.
An agent must manage crop diseases, pests, irrigation, and budget
across a 120-day season to maximise yield at harvest.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum


# ─────────────────────────────────────────────
#  Enum Definitions
# ─────────────────────────────────────────────

class DiseaseType(IntEnum):
    NONE       = 0
    FUNGAL     = 1   # Early Blight / Late Blight
    BACTERIAL  = 2   # Bacterial Speck / Canker
    VIRAL      = 3   # Tomato Yellow Leaf Curl Virus
    PEST       = 4   # Aphids / Whitefly / Spider Mites

class GrowthStage(IntEnum):
    SEEDLING   = 0   # Days   0 – 20
    VEGETATIVE = 1   # Days  21 – 45
    FLOWERING  = 2   # Days  46 – 70
    FRUITING   = 3   # Days  71 – 100
    HARVEST    = 4   # Days 101 – 120

class Action(IntEnum):
    DO_NOTHING              = 0
    APPLY_FUNGICIDE_LIGHT   = 1
    APPLY_FUNGICIDE_HEAVY   = 2
    APPLY_BACTERICIDE       = 3
    APPLY_INSECTICIDE       = 4
    APPLY_ORGANIC_PESTICIDE = 5
    IRRIGATE_LIGHT          = 6
    IRRIGATE_HEAVY          = 7
    PRUNE_INFECTED_LEAVES   = 8
    CALL_AGRONOMIST         = 9


# ─────────────────────────────────────────────
#  Human-readable labels (used by renderers)
# ─────────────────────────────────────────────

ACTION_NAMES = {
    0: "Do Nothing / Monitor",
    1: "Fungicide  (Light Dose)",
    2: "Fungicide  (Heavy Dose)",
    3: "Bactericide",
    4: "Insecticide",
    5: "Organic Pesticide",
    6: "Irrigate   (Light)",
    7: "Irrigate   (Heavy)",
    8: "Prune Infected Leaves",
    9: "Call Agronomist",
}

DISEASE_COLORS = {
    DiseaseType.NONE:      (50,  180,  50),
    DiseaseType.FUNGAL:    (180, 100,  20),
    DiseaseType.BACTERIAL: (180,  40,  40),
    DiseaseType.VIRAL:     (160,  40, 180),
    DiseaseType.PEST:      (200, 160,  20),
}


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

class TomatoFarmEnv(gym.Env):
    """
    TomatoFarmEnv-v0
    ----------------
    Observation  : Box(12,) — see _build_obs_space()
    Actions      : Discrete(10) — see Action enum
    Reward       : Float — shaped per action + daily health + harvest bonus
    Episode len  : 120 steps (one growing season)

    Terminal Conditions
    ───────────────────
    1. crop_health ≤ 0           → terminated (crop failure)
    2. step_count ≥ MAX_STEPS    → truncated  (season ends, harvest scored)
    3. budget_remaining ≤ 0 AND disease_severity ≥ 0.9
                                 → terminated (bankrupt + catastrophic loss)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}
    MAX_STEPS = 120

    # ── stage thresholds: stage i active while step_count < thresholds[i]
    _STAGE_THRESHOLDS = [20, 45, 70, 100, 120]

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # ── Action Space ───────────────────────────────────────────────
        # 10 discrete actions covering: monitoring, chemical treatments,
        # irrigation management, physical intervention, expert consultation
        self.action_space = spaces.Discrete(10)

        # ── Observation Space ──────────────────────────────────────────
        # 12 continuous features describing crop state, disease, weather
        low = np.array(
            [0.0, 0.0, 15.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 4.0, 40.0, 1.0, 1.0, 4.0, 120.0, 1.0, 1.0, 9.0, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # pygame handles
        self._screen = None
        self._clock  = None
        self._font   = None
        self._font_lg = None

        # internal state (set properly in reset())
        self._rng = np.random.default_rng()

    # ── Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # ── START STATE ────────────────────────────────────────────────
        self.step_count       = 0
        self.crop_health      = 1.0          # [0,1] overall plant health
        self.disease_severity = 0.0          # [0,1] fraction of crop infected
        self.disease_type     = DiseaseType.NONE
        self.temperature      = float(self._rng.uniform(22, 28))   # °C
        self.humidity         = float(self._rng.uniform(0.4, 0.6))
        self.rainfall_prob    = float(self._rng.uniform(0.1, 0.3))
        self.growth_stage     = GrowthStage.SEEDLING
        self.budget_remaining = 1.0          # normalised 0–1
        self.soil_moisture    = 0.6
        self.last_action      = int(Action.DO_NOTHING)
        self.pest_population  = 0.0          # [0,1]

        # telemetry
        self.total_reward     = 0.0
        self.reward_history: list[float] = []
        self.action_history: list[int]   = []
        self.event_log: list[str]        = []

        if self.render_mode == "human":
            self._render_pygame()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self.last_action = action
        self.action_history.append(action)
        reward = 0.0

        # 1. Weather dynamics
        self._update_weather()

        # 2. Disease / pest progression (before treatment)
        self._progress_disease()

        # 3. Growth stage
        self._update_growth_stage()

        # 4. Apply chosen action → shaped reward component
        action_reward = self._apply_action(action)
        reward += action_reward

        # 5. Crop health degradation
        self._update_crop_health()

        # 6. Daily survival bonus — encourages keeping crop alive
        reward += self.crop_health * 0.5

        self.step_count += 1

        # ── TERMINAL CONDITIONS ────────────────────────────────────────
        terminated = False
        truncated  = False
        term_reason = ""

        # T1: Crop completely dead
        if self.crop_health <= 0.0:
            terminated  = True
            reward     -= 50.0
            term_reason = "CROP FAILURE — health reached 0"
            self.event_log.append(f"[Day {self.step_count}] ❌ {term_reason}")

        # T2: Season complete
        elif self.step_count >= self.MAX_STEPS:
            truncated  = True
            bonus      = self.crop_health * 100.0 * (1.0 - self.disease_severity)
            reward    += bonus
            term_reason = f"HARVEST — bonus {bonus:.1f}"
            self.event_log.append(f"[Day {self.step_count}] 🏆 {term_reason}")

        # T3: Bankrupt + catastrophic disease
        elif self.budget_remaining <= 0.0 and self.disease_severity >= 0.9:
            terminated  = True
            reward     -= 30.0
            term_reason = "BANKRUPTCY + CATASTROPHIC DISEASE"
            self.event_log.append(f"[Day {self.step_count}] 💸 {term_reason}")

        self.total_reward += reward
        self.reward_history.append(reward)

        if self.render_mode == "human":
            self._render_pygame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode == "human":
            self._render_pygame()

    def close(self):
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.disease_severity,
            float(self.disease_type),
            self.temperature,
            self.humidity,
            self.rainfall_prob,
            float(self.growth_stage),
            float(self.step_count),
            self.budget_remaining,
            self.soil_moisture,
            float(self.last_action),
            self.crop_health,
            self.pest_population,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            "step":             self.step_count,
            "crop_health":      round(self.crop_health, 3),
            "disease_type":     DiseaseType(int(self.disease_type)).name,
            "disease_severity": round(self.disease_severity, 3),
            "growth_stage":     GrowthStage(int(self.growth_stage)).name,
            "budget_remaining": round(self.budget_remaining, 3),
            "pest_population":  round(self.pest_population, 3),
            "total_reward":     round(self.total_reward, 2),
            "last_action_name": ACTION_NAMES[self.last_action],
        }

    def _update_weather(self):
        self.temperature   += float(self._rng.uniform(-1.5, 1.5))
        self.temperature    = float(np.clip(self.temperature, 15, 40))
        self.humidity      += float(self._rng.uniform(-0.05, 0.05))
        self.humidity       = float(np.clip(self.humidity, 0, 1))
        self.rainfall_prob  = float(self._rng.uniform(0, 1))

        if self.rainfall_prob > 0.7:
            self.soil_moisture = min(1.0, self.soil_moisture + 0.2)
        else:
            self.soil_moisture = max(0.0, self.soil_moisture - 0.05)

    def _progress_disease(self):
        # Random disease onset
        if self.disease_type == DiseaseType.NONE:
            onset_prob = 0.05 + self.humidity * 0.08 + (self.temperature > 30) * 0.05
            if self._rng.random() < onset_prob:
                self.disease_type     = DiseaseType(int(self._rng.integers(1, 5)))
                self.disease_severity = 0.05
                self.event_log.append(
                    f"[Day {self.step_count}] ⚠️  {self.disease_type.name} detected"
                )
        else:
            spread = 0.04 * (1 + self.humidity * 0.8) * (1 - self.crop_health * 0.4)
            self.disease_severity = min(1.0, self.disease_severity + spread)

        # Pest population ticks independently
        if self.pest_population > 0 or self._rng.random() < 0.04:
            self.pest_population = min(
                1.0, self.pest_population + float(self._rng.uniform(0, 0.04))
            )

    def _update_growth_stage(self):
        for i, threshold in enumerate(self._STAGE_THRESHOLDS):
            if self.step_count < threshold:
                self.growth_stage = GrowthStage(i)
                break

    def _update_crop_health(self):
        loss  = self.disease_severity * 0.08
        loss += self.pest_population  * 0.04
        if self.soil_moisture < 0.20:
            loss += 0.06   # drought stress
        if self.soil_moisture > 0.90:
            loss += 0.02   # waterlogging / root rot
        if self.temperature > 36:
            loss += 0.03   # heat stress
        if self.disease_severity > 0.40:
            loss = -(self.disease_severity - 0.40) * 8.0
        self.crop_health = max(0.0, self.crop_health - loss)

    # ── Action logic ────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> float:
        reward = 0.0
        cost   = 0.0

        if action == Action.DO_NOTHING:
            # Passive monitoring — small penalty if untreated disease is spreading
            if self.disease_severity > 0.30:
                reward -= 2.0
            if self.pest_population > 0.40:
                reward -= 1.5

        elif action == Action.APPLY_FUNGICIDE_LIGHT:
            cost = 0.05
            if self.disease_type == DiseaseType.FUNGAL:
                self.disease_severity = max(0.0, self.disease_severity - 0.15)
                reward += 5.0
                self._clear_disease_if_healed()
            else:
                reward -= 1.5   # wrong treatment

        elif action == Action.APPLY_FUNGICIDE_HEAVY:
            cost = 0.12
            if self.disease_type == DiseaseType.FUNGAL:
                self.disease_severity = max(0.0, self.disease_severity - 0.38)
                reward += 11.0
                self._clear_disease_if_healed()
            else:
                reward -= 4.0   # expensive wrong treatment

        elif action == Action.APPLY_BACTERICIDE:
            cost = 0.08
            if self.disease_type == DiseaseType.BACTERIAL:
                self.disease_severity = max(0.0, self.disease_severity - 0.28)
                reward += 9.0
                self._clear_disease_if_healed()
            else:
                reward -= 2.5

        elif action == Action.APPLY_INSECTICIDE:
            cost = 0.08
            if self.disease_type == DiseaseType.PEST or self.pest_population > 0.20:
                self.pest_population  = max(0.0, self.pest_population - 0.45)
                self.disease_severity = max(0.0, self.disease_severity - 0.20)
                reward += 9.0
                self._clear_disease_if_healed()
            else:
                reward -= 2.0

        elif action == Action.APPLY_ORGANIC_PESTICIDE:
            cost = 0.04
            if self.disease_type != DiseaseType.NONE:
                self.disease_severity = max(0.0, self.disease_severity - 0.08)
                reward += 2.0
                if self.disease_severity > 0.30 and self.disease_type != DiseaseType.NONE:
                    reward -= 3.0   # severe disease needs targeted treatment, not organic
            self.pest_population = max(0.0, self.pest_population - 0.10)
            # Remove the unconditional +1.5 — replace with a conditional:
            if self.pest_population > 0.10:
                reward += 1.5
            else:
                reward -= 0.5   # mild penalty for unnecessary application

        elif action == Action.IRRIGATE_LIGHT:
            cost = 0.02
            prev = self.soil_moisture
            self.soil_moisture = min(1.0, self.soil_moisture + 0.15)
            if prev < 0.40:
                reward += 3.5   # crops needed water
            elif prev > 0.70:
                reward -= 1.0   # unnecessary — slight overwater risk

        elif action == Action.IRRIGATE_HEAVY:
            cost = 0.05
            prev = self.soil_moisture
            self.soil_moisture = min(1.0, self.soil_moisture + 0.35)
            if prev < 0.25:
                reward += 7.0   # emergency water saves crop
            elif prev > 0.60:
                reward -= 4.0   # overwatering causes root rot

        elif action == Action.PRUNE_INFECTED_LEAVES:
            cost = 0.01
            if self.disease_severity > 0.15:
                self.disease_severity = max(0.0, self.disease_severity - 0.12)
                self.crop_health      = max(0.0, self.crop_health - 0.04)
                reward += 4.5
            else:
                reward -= 0.5   # unnecessary pruning

        elif action == Action.CALL_AGRONOMIST:
            cost = 0.15
            if self.disease_severity > 0.35 or self.pest_population > 0.40:
                # Expert dramatically reduces both disease and pests
                self.disease_severity = max(0.0, self.disease_severity - 0.25)
                self.pest_population  = max(0.0, self.pest_population  - 0.20)
                reward += 14.0
                self._clear_disease_if_healed()
            else:
                reward -= 2.0   # unnecessary expensive call

        # Budget accounting
        if self.budget_remaining > 0.0:
            self.budget_remaining = max(0.0, self.budget_remaining - cost)
        elif cost > 0.0:
            reward -= cost * 12.0   # over-budget penalty

        return reward

    def _clear_disease_if_healed(self):
        if self.disease_severity <= 0.0:
            self.disease_type = DiseaseType.NONE
            self.event_log.append(f"[Day {self.step_count}] ✅ Disease eradicated")

    # ── Pygame Renderer ──────────────────────────────────────────────────

    def _render_pygame(self):
        import pygame

        W, H = 1100, 720
        if self._screen is None:
            pygame.init()
            pygame.display.set_caption("🍅  TomatoFarmEnv-v0  |  Random Agent Simulation")
            self._screen  = pygame.display.set_mode((W, H))
            self._clock   = pygame.time.Clock()
            self._font    = pygame.font.SysFont("monospace", 13)
            self._font_lg = pygame.font.SysFont("monospace", 18, bold=True)
            self._font_sm = pygame.font.SysFont("monospace", 11)

        # ── palette
        C = {
            "bg":      (22,  33, 22),
            "panel":   (14,  22, 14),
            "border":  (50,  90, 50),
            "white":   (235, 235, 235),
            "gray":    (110, 120, 110),
            "green":   ( 60, 200,  60),
            "red":     (220,  55,  55),
            "yellow":  (240, 210,  55),
            "blue":    ( 70, 130, 220),
            "orange":  (230, 130,  40),
            "purple":  (180,  80, 200),
            "teal":    ( 60, 190, 170),
        }

        self._screen.fill(C["bg"])

        # ── Title bar ──────────────────────────────────────────────
        pygame.draw.rect(self._screen, C["panel"], (0, 0, W, 44))
        title = self._font_lg.render(
            "🍅  TomatoFarmEnv-v0  ·  Random Agent Exploration", True, C["yellow"]
        )
        self._screen.blit(title, (14, 12))
        day_txt = self._font_lg.render(
            f"Day {self.step_count:3d}/120   Total Reward: {self.total_reward:+8.1f}",
            True, C["teal"],
        )
        self._screen.blit(day_txt, (700, 12))

        # ── Farm grid (left) ───────────────────────────────────────
        COLS, ROWS  = 10, 12
        CELL        = 38
        GX, GY      = 16, 56
        PANEL_W     = COLS * CELL + 10

        disease_col = DISEASE_COLORS.get(DiseaseType(int(self.disease_type)), (50, 180, 50))

        for r in range(ROWS):
            for c in range(COLS):
                # Per-plant random jitter so field looks alive
                jitter = self._rng.uniform(-0.08, 0.08)
                ph     = max(0.0, min(1.0, self.crop_health + jitter))
                ds     = max(0.0, min(1.0, self.disease_severity + self._rng.uniform(-0.05, 0.05)))

                healthy_g = int(40 + ph * 160)
                sick_r    = int(ds * 180)
                cell_col  = (sick_r, healthy_g, 30)

                rect = pygame.Rect(GX + c*CELL, GY + r*CELL, CELL-3, CELL-3)
                pygame.draw.rect(self._screen, cell_col, rect, border_radius=6)

                # Tomato berry
                cx = GX + c*CELL + CELL//2 - 1
                cy = GY + r*CELL + CELL//2 - 1
                if ph > 0.55:
                    tom_col = (200, 50, 50)
                elif ph > 0.25:
                    tom_col = (170, 120, 30)
                else:
                    tom_col = (80, 60, 20)
                pygame.draw.circle(self._screen, tom_col, (cx, cy), 7)

                # Disease indicator dot
                if ds > 0.20:
                    pygame.draw.circle(self._screen, disease_col, (cx+8, cy-8), 3)

        # ── Observation panel (right) ──────────────────────────────
        PX = PANEL_W + 30
        pygame.draw.rect(self._screen, C["panel"], (PX-8, 46, W - PX + 8, H - 46))
        pygame.draw.line(self._screen, C["border"], (PX-8, 46), (PX-8, H), 2)

        sec = self._font_lg.render("OBSERVATION SPACE", True, C["yellow"])
        self._screen.blit(sec, (PX, 56))

        obs_rows = [
            ("crop_health",        self.crop_health,       C["green"] if self.crop_health > 0.6 else C["red"]),
            ("disease_severity",   self.disease_severity,  C["red"]   if self.disease_severity > 0.3 else C["green"]),
            ("disease_type",       None,                   C["orange"]),
            ("temperature (°C)",   self.temperature,       C["blue"]),
            ("humidity",           self.humidity,          C["blue"]),
            ("rainfall_prob",      self.rainfall_prob,     C["blue"]),
            ("soil_moisture",      self.soil_moisture,     C["teal"]),
            ("growth_stage",       None,                   C["yellow"]),
            ("season_day",         None,                   C["white"]),
            ("budget_remaining",   self.budget_remaining,  C["green"] if self.budget_remaining > 0.3 else C["red"]),
            ("pest_population",    self.pest_population,   C["red"]   if self.pest_population > 0.3 else C["green"]),
            ("last_action",        None,                   C["purple"]),
        ]

        str_vals = {
            "disease_type":    DiseaseType(int(self.disease_type)).name,
            "growth_stage":    GrowthStage(int(self.growth_stage)).name,
            "season_day":      f"{self.step_count}/120",
            "last_action":     f"[{self.last_action}]",
        }

        for i, (label, val, col) in enumerate(obs_rows):
            y = 86 + i * 40
            lbl = self._font.render(f"  {label}", True, C["gray"])
            self._screen.blit(lbl, (PX, y))

            if label in str_vals:
                display = str_vals[label]
                vtxt = self._font.render(display, True, col)
                self._screen.blit(vtxt, (PX + 220, y))
            else:
                display = f"{val:.3f}"
                vtxt = self._font.render(display, True, col)
                self._screen.blit(vtxt, (PX + 220, y))
                # bar
                bx, bw = PX + 290, 150
                pygame.draw.rect(self._screen, C["gray"],   (bx, y+3, bw, 11), 1, border_radius=4)
                fill = int(np.clip(val, 0, 1) * bw)
                if fill > 0:
                    pygame.draw.rect(self._screen, col, (bx, y+3, fill, 11), border_radius=4)

        # ── Action panel ───────────────────────────────────────────
        AY = 86 + len(obs_rows)*40 + 12
        pygame.draw.line(self._screen, C["border"], (PX-8, AY-8), (W, AY-8), 1)
        asec = self._font_lg.render("ACTION SPACE  (10 actions)", True, C["yellow"])
        self._screen.blit(asec, (PX, AY))

        for idx, name in ACTION_NAMES.items():
            y      = AY + 28 + idx * 22
            active = (idx == self.last_action)
            col    = C["yellow"] if active else C["gray"]
            prefix = "▶ " if active else "  "
            txt    = self._font_sm.render(f"{prefix}[{idx}] {name}", True, col)
            self._screen.blit(txt, (PX, y))
            if active:
                pygame.draw.rect(self._screen, (40, 60, 40),
                                 pygame.Rect(PX-4, y-2, 340, 20), border_radius=4)
                txt = self._font_sm.render(f"{prefix}[{idx}] {name}", True, col)
                self._screen.blit(txt, (PX, y))

        # ── Reward sparkline ───────────────────────────────────────
        if len(self.reward_history) > 1:
            RX, RY, RW, RH = 16, GY + ROWS*CELL + 8, PANEL_W, 60
            pygame.draw.rect(self._screen, C["panel"], (RX, RY, RW, RH))
            pygame.draw.rect(self._screen, C["border"], (RX, RY, RW, RH), 1)
            rlbl = self._font_sm.render("Reward history", True, C["gray"])
            self._screen.blit(rlbl, (RX+4, RY+2))

            hist = self.reward_history[-RW:]
            mn, mx = min(hist), max(hist)
            rng = mx - mn if mx != mn else 1
            pts = []
            for j, rv in enumerate(hist):
                px_ = RX + int(j * RW / max(len(hist)-1, 1))
                py_ = RY + RH - int((rv - mn) / rng * (RH - 14)) - 4
                pts.append((px_, py_))
            if len(pts) > 1:
                col = C["green"] if self.reward_history[-1] >= 0 else C["red"]
                pygame.draw.lines(self._screen, col, False, pts, 2)

        # ── Event log ──────────────────────────────────────────────
        LOG_X, LOG_Y = PX, AY + 28 + 10*22 + 14
        for k, msg in enumerate(self.event_log[-4:]):
            etxt = self._font_sm.render(msg, True, C["teal"])
            self._screen.blit(etxt, (LOG_X, LOG_Y + k*16))

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])
