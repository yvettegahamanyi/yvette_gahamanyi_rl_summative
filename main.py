"""
main.py — TomatoFarmEnv-v0 Trained Agent Demo + REST API
=========================================================
Loads the best saved model (PPO preferred, DQN fallback, REINFORCE last resort)
and runs it in TomatoFarmEnv with a full pygame HUD showing:
  • Farm grid with colour-coded plant health
  • Agent decision overlay (action chosen + reason)
  • Action probability bar chart (policy HUD)
  • Exploration vs Exploitation indicator
  • Live reward sparkline
  • Episode stats panel

JSON Serialisation / FastAPI Integration
-----------------------------------------
Every environment step is serialised to JSON.  Two modes:
  1. python main.py          — live pygame window (demo mode)
  2. python main.py --api    — starts a FastAPI server at http://localhost:8000
                               Frontend can poll /step or stream /stream

FastAPI Endpoints (--api mode):
  GET  /reset          → reset env, return initial state JSON
  POST /step           → {action: int} → next state JSON
  GET  /step_auto      → agent selects action → next state JSON
  GET  /stream         → Server-Sent Events stream of auto-steps
  GET  /model_info     → which model is loaded + hyperparams
  GET  /docs           → Swagger UI (auto-generated)

Usage:
  pip install gymnasium stable-baselines3 torch pygame fastapi uvicorn
  python main.py              # visual demo
  python main.py --api        # REST API server
  python main.py --headless   # JSON output only, no pygame
"""

import sys, os, json, time, argparse
import numpy as np

# ── CLI ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--api",      action="store_true", help="Start FastAPI server")
parser.add_argument("--headless", action="store_true", help="No pygame, print JSON steps")
parser.add_argument("--episodes", type=int, default=3,  help="Demo episodes (visual mode)")
parser.add_argument("--host",     default="0.0.0.0")
parser.add_argument("--port",     type=int, default=8000)
parser.add_argument("--model",     default=None, help="Path to a specific trained model")
parser.add_argument("--algo",      default=None, help="Model algorithm: PPO, DQN, or REINFORCE")
args = parser.parse_args()


# ════════════════════════════════════════════════════════════════
# 1. Load Best Model
# ════════════════════════════════════════════════════════════════

def find_best_model():
    """
    Priority: PPO → DQN → REINFORCE
    Reads the *_info.json files saved by the notebooks.
    """
    if args.model:
        model_path = os.path.expanduser(args.model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        algo_name = args.algo.upper() if args.algo else None
        if algo_name is None:
            upper_path = model_path.upper()
            for candidate in ("PPO", "DQN", "REINFORCE"):
                if candidate in upper_path:
                    algo_name = candidate
                    break
        if algo_name is None:
            raise ValueError(
                "When using --model, also pass --algo (PPO, DQN, or REINFORCE) "
                "if the algorithm cannot be inferred from the filename."
            )

        model_info = {
            "algo": algo_name,
            "label": os.path.basename(model_path),
            "mean_reward": None,
            "model_path": model_path,
        }
        print(f"[main.py] Loading explicit {algo_name} model: {model_path}")
        return algo_name, model_info, model_path

    for algo in ["PPO", "DQN", "REINFORCE"]:
        info_path = f"best_{algo.lower()}_info.json"
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            model_path = info["model_path"]
            print(f"[main.py] Loading best {algo} model: {model_path}")
            print(f"          Mean reward (eval): {format_mean_reward(info.get('mean_reward'))}")
            return algo, info, model_path
    raise FileNotFoundError(
        "No saved model found. Run a notebook first to train and save models."
    )

algo, model_info, model_path = find_best_model()

# Load model based on algorithm
try:
    from environment.tomato_farm_env import (
        TomatoFarmEnv, ACTION_NAMES, DiseaseType, GrowthStage, DISEASE_COLORS
    )
except ModuleNotFoundError:
    from tomato_farm_env import (
        TomatoFarmEnv, ACTION_NAMES, DiseaseType, GrowthStage, DISEASE_COLORS
    )

if algo == "PPO":
    from stable_baselines3 import PPO
    MODEL = PPO.load(model_path)
    def get_action_and_probs(obs):
        action, _ = MODEL.predict(obs, deterministic=True)
        # Get action probabilities from actor network
        import torch
        with torch.no_grad():
            obs_t  = torch.FloatTensor(obs).unsqueeze(0)
            dist   = MODEL.policy.get_distribution(obs_t)
            probs  = dist.distribution.probs.squeeze().numpy()
        return int(action), probs

elif algo == "DQN":
    from stable_baselines3 import DQN
    MODEL = DQN.load(model_path)
    def get_action_and_probs(obs):
        import torch
        action, _ = MODEL.predict(obs, deterministic=True)
        # DQN: derive soft probs from Q-values via softmax
        with torch.no_grad():
            obs_t   = torch.FloatTensor(obs).unsqueeze(0)
            q_vals  = MODEL.q_net(obs_t).squeeze().numpy()
        # Softmax for visualisation (not the true policy, but informative)
        e_q  = np.exp(q_vals - q_vals.max())
        probs = e_q / e_q.sum()
        return int(action), probs

else:  # REINFORCE
    import torch
    try:
        from environment.tomato_farm_env import TomatoFarmEnv as _E
    except ModuleNotFoundError:
        from tomato_farm_env import TomatoFarmEnv as _E

    # Inline PolicyNet (same as notebook)
    import torch.nn as nn
    class PolicyNet(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden),  nn.ReLU(),
                nn.Linear(hidden, act_dim),
            )
        def forward(self, x):
            return torch.softmax(self.net(x), dim=-1)

    ckpt = torch.load(model_path, map_location="cpu")
    env_tmp = _E()
    obs_dim = env_tmp.observation_space.shape[0]
    act_dim = env_tmp.action_space.n
    env_tmp.close()
    _net = PolicyNet(obs_dim, act_dim)
    _net.load_state_dict(ckpt["state_dict"])

    def get_action_and_probs(obs):
        with torch.no_grad():
            probs = _net(torch.FloatTensor(obs)).numpy()
        return int(probs.argmax()), probs


# ════════════════════════════════════════════════════════════════
# 2. JSON Step Serialiser
# ════════════════════════════════════════════════════════════════

def step_to_json(step_num, obs, action, action_probs, reward,
                 terminated, truncated, info, total_reward):
    """
    Serialises one environment step to a JSON-safe dict.
    This is the payload the FastAPI sends to frontend clients.
    """
    return {
        "step":          step_num,
        "done":          bool(terminated or truncated),
        "terminated":    bool(terminated),
        "truncated":     bool(truncated),
        "total_reward":  round(float(total_reward), 2),
        "step_reward":   round(float(reward), 2),
        "action": {
            "id":          int(action),
            "name":        ACTION_NAMES[action],
            "probability": round(float(action_probs[action]), 4),
        },
        "action_probs": {
            ACTION_NAMES[i]: round(float(p), 4)
            for i, p in enumerate(action_probs)
        },
        "observation": {
            "crop_health":       round(float(obs[10]), 3),
            "disease_severity":  round(float(obs[0]),  3),
            "disease_type":      DiseaseType(int(obs[1])).name,
            "temperature_c":     round(float(obs[2]),  1),
            "humidity":          round(float(obs[3]),  3),
            "rainfall_prob":     round(float(obs[4]),  3),
            "growth_stage":      GrowthStage(int(obs[5])).name,
            "season_day":        int(obs[6]),
            "budget_remaining":  round(float(obs[7]),  3),
            "soil_moisture":     round(float(obs[8]),  3),
            "pest_population":   round(float(obs[11]), 3),
        },
        "model": {
            "algo":       model_info["algo"],
            "label":      model_info["label"],
            "mean_reward_eval": model_info["mean_reward"],
        },
    }


def format_mean_reward(value):
    if value is None:
        return "n/a"
    return f"{value:.1f}"


# ════════════════════════════════════════════════════════════════
# 3. FastAPI Server  (--api mode)
# ════════════════════════════════════════════════════════════════

if args.api:
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
    except ImportError:
        sys.exit("pip install fastapi uvicorn")

    app = FastAPI(
        title="TomatoFarmEnv Agent API",
        description=(
            "REST API exposing a trained RL agent navigating TomatoFarmEnv-v0.\n\n"
            "**Integrate into any frontend:** React, Vue, Flutter, Swift.\n"
            "Poll `/step_auto` to get agent decisions, or POST to `/step` with "
            "your own action to compare human vs agent."
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    # Shared server-side env state
    _env   = TomatoFarmEnv()
    _obs, _ = _env.reset(seed=0)
    _total_reward = 0.0
    _step_num     = 0

    @app.get("/reset",
             summary="Reset the farm environment",
             description="Starts a new 120-day growing season. Returns initial observation.")
    def reset_env(seed: int = 42):
        global _env, _obs, _total_reward, _step_num
        _env.close()
        _env = TomatoFarmEnv()
        _obs, _ = _env.reset(seed=seed)
        _total_reward = 0.0
        _step_num     = 0
        action_probs  = np.ones(10) / 10
        return step_to_json(0, _obs, 0, action_probs, 0.0, False, False, {}, 0.0)

    @app.get("/step_auto",
             summary="Agent selects next action",
             description="The trained RL policy picks the optimal action and returns the new state.")
    def step_auto():
        global _obs, _total_reward, _step_num
        action, probs = get_action_and_probs(_obs)
        _obs, reward, terminated, truncated, info = _env.step(action)
        _total_reward += reward
        _step_num     += 1
        payload = step_to_json(_step_num, _obs, action, probs,
                                reward, terminated, truncated, info, _total_reward)
        if terminated or truncated:
            reset_env()
        return payload

    @app.post("/step",
              summary="Human/custom action",
              description="POST {action: 0-9} to override the agent. Useful for human-vs-agent comparison.")
    def step_manual(action: int):
        global _obs, _total_reward, _step_num
        if not 0 <= action <= 9:
            return JSONResponse(status_code=400, content={"error": "action must be 0-9"})
        _, probs = get_action_and_probs(_obs)
        _obs, reward, terminated, truncated, info = _env.step(action)
        _total_reward += reward
        _step_num     += 1
        return step_to_json(_step_num, _obs, action, probs,
                             reward, terminated, truncated, info, _total_reward)

    @app.get("/stream",
             summary="Server-Sent Events stream",
             description="Streams one agent step per second as SSE. Consume with EventSource in JS.")
    def stream():
        def event_gen():
            env  = TomatoFarmEnv()
            obs, _ = env.reset(seed=42)
            total  = 0.0; step = 0
            while True:
                action, probs = get_action_and_probs(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward; step += 1
                payload = step_to_json(step, obs, action, probs,
                                       reward, terminated, truncated, info, total)
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.5)
                if terminated or truncated:
                    obs, _ = env.reset(); total = 0.0; step = 0
                    yield f"data: {json.dumps({'event':'episode_reset'})}\n\n"
            env.close()
        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.get("/model_info",
             summary="Loaded model metadata",
             description="Returns which algorithm and hyperparameters produced the best model.")
    def model_info_endpoint():
        return model_info

    @app.get("/",
             summary="API health check")
    def root():
        return {"status": "ok", "algo": algo, "docs": "/docs"}

    print(f"\n[main.py] Starting FastAPI server at http://{args.host}:{args.port}")
    print(f"[main.py] Swagger UI: http://localhost:{args.port}/docs")
    print(f"[main.py] Example frontend integration:")
    print("""
    // JavaScript (React / Vanilla)
    const source = new EventSource("http://localhost:8000/stream");
    source.onmessage = (e) => {
        const state = JSON.parse(e.data);
        console.log("Day", state.step, "Action:", state.action.name,
                    "Reward:", state.step_reward);
        updateFarmDashboard(state);   // your React component
    };
    """)
    uvicorn.run(app, host=args.host, port=args.port)
    sys.exit(0)


# ════════════════════════════════════════════════════════════════
# 4. Headless mode (--headless)
# ════════════════════════════════════════════════════════════════

if args.headless:
    print(f"[main.py] Headless mode — {algo} agent, 1 episode, JSON output")
    env = TomatoFarmEnv()
    obs, _ = env.reset(seed=42)
    total  = 0.0
    for step_num in range(1, 121):
        action, probs = get_action_and_probs(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        payload = step_to_json(step_num, obs, action, probs,
                                reward, terminated, truncated, info, total)
        print(json.dumps(payload))
        if terminated or truncated:
            break
    env.close()
    sys.exit(0)


# ════════════════════════════════════════════════════════════════
# 5. Pygame Visual Demo (default mode)
# ════════════════════════════════════════════════════════════════

try:
    import pygame
except ImportError:
    sys.exit("pip install pygame")

W, H = 1280, 800

PAL = dict(
    bg      = (14, 22, 14),
    panel   = (10, 18, 10),
    border  = (40, 80, 40),
    white   = (230, 232, 230),
    gray    = (100, 115, 100),
    green   = (50,  200,  50),
    red     = (220,  50,  50),
    yellow  = (240, 210,  50),
    blue    = ( 70, 130, 220),
    orange  = (230, 130,  40),
    purple  = (180,  80, 200),
    teal    = ( 50, 185, 160),
    dark    = ( 20,  35,  20),
    gold    = (255, 200,   0),
)

pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption(f"TomatoFarmEnv — {algo} Agent  |  main.py")
clock  = pygame.time.Clock()

font_sm = pygame.font.SysFont("monospace", 11)
font_md = pygame.font.SysFont("monospace", 13)
font_lg = pygame.font.SysFont("monospace", 16, bold=True)
font_xl = pygame.font.SysFont("monospace", 20, bold=True)


def draw_bar(surface, x, y, w, h, value, col, bg=(30, 50, 30), border=True):
    if border:
        pygame.draw.rect(surface, PAL["border"], (x-1, y-1, w+2, h+2), 1,
                         border_radius=3)
    pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=3)
    fill = int(np.clip(value, 0, 1) * w)
    if fill > 0:
        pygame.draw.rect(surface, col, (x, y, fill, h), border_radius=3)


def draw_action_probs(surface, x, y, probs, chosen_action):
    """Draw horizontal probability bars for all 10 actions."""
    pygame.draw.rect(surface, PAL["panel"], (x, y, 420, 240))
    pygame.draw.rect(surface, PAL["border"], (x, y, 420, 240), 1)
    title = font_md.render("Policy Action Distribution", True, PAL["yellow"])
    surface.blit(title, (x + 8, y + 6))

    for i in range(10):
        row_y  = y + 28 + i * 21
        is_chosen = (i == chosen_action)
        col    = PAL["gold"] if is_chosen else PAL["teal"]
        prefix = font_sm.render(f"[{i}]", True, col)
        surface.blit(prefix, (x + 6, row_y))

        name_surf = font_sm.render(ACTION_NAMES[i][:22], True, col)
        surface.blit(name_surf, (x + 32, row_y))

        bx = x + 220; bw = 150
        draw_bar(surface, bx, row_y + 2, bw, 13, float(probs[i]),
                 col, border=False)
        pct = font_sm.render(f"{probs[i]*100:5.1f}%", True, col)
        surface.blit(pct, (bx + bw + 6, row_y))

        if is_chosen:
            pygame.draw.rect(surface, (50, 70, 30),
                             (x + 4, row_y - 1, 412, 18), border_radius=3)
            surface.blit(prefix, (x + 6, row_y))
            surface.blit(name_surf, (x + 32, row_y))
            draw_bar(surface, bx, row_y + 2, bw, 13, float(probs[i]),
                     col, border=False)
            surface.blit(pct, (bx + bw + 6, row_y))


def draw_farm_grid(surface, env, rng_disp):
    COLS, ROWS = 10, 12
    CELL = 36
    GX, GY = 14, 54
    disease_col = DISEASE_COLORS.get(
        DiseaseType(int(env.disease_type)), (50, 180, 50)
    )
    for r in range(ROWS):
        for c in range(COLS):
            jitter = rng_disp.uniform(-0.07, 0.07)
            ph = float(np.clip(env.crop_health + jitter, 0, 1))
            ds = float(np.clip(env.disease_severity + rng_disp.uniform(-0.04, 0.04), 0, 1))
            cell_col = (int(ds * 170), int(40 + ph * 155), 25)
            rect = pygame.Rect(GX + c * CELL, GY + r * CELL, CELL - 3, CELL - 3)
            pygame.draw.rect(surface, cell_col, rect, border_radius=5)
            cx = GX + c * CELL + CELL // 2
            cy = GY + r * CELL + CELL // 2
            tom_col = (200, 50, 50) if ph > 0.55 else (150, 100, 30) if ph > 0.25 else (80, 55, 20)
            pygame.draw.circle(surface, tom_col, (cx, cy), 6)
            if ds > 0.20:
                pygame.draw.circle(surface, disease_col, (cx + 7, cy - 7), 4)


def draw_reward_sparkline(surface, reward_history, x, y, w, h):
    pygame.draw.rect(surface, PAL["panel"],  (x, y, w, h))
    pygame.draw.rect(surface, PAL["border"], (x, y, w, h), 1)
    lbl = font_sm.render("Reward history", True, PAL["gray"])
    surface.blit(lbl, (x + 4, y + 3))
    if len(reward_history) < 2:
        return
    hist = reward_history[-w:]
    mn, mx = min(hist), max(hist)
    rng = mx - mn if mx != mn else 1
    pts = []
    for j, rv in enumerate(hist):
        px_ = x + int(j * w / max(len(hist) - 1, 1))
        py_ = y + h - int((rv - mn) / rng * (h - 18)) - 4
        pts.append((px_, py_))
    if len(pts) > 1:
        col = PAL["green"] if reward_history[-1] >= 0 else PAL["red"]
        pygame.draw.lines(surface, col, False, pts, 2)


def draw_exploitation_indicator(surface, x, y, probs, step):
    """
    Exploration vs Exploitation:
    High confidence (max_prob > 0.7) = Exploitation (green)
    Low confidence (max_prob < 0.3)  = Exploration  (red)
    """
    max_prob = float(np.max(probs))
    pygame.draw.rect(surface, PAL["panel"], (x, y, 420, 44))
    pygame.draw.rect(surface, PAL["border"], (x, y, 420, 44), 1)

    label = font_md.render("Exploration  <--->  Exploitation", True, PAL["gray"])
    surface.blit(label, (x + 8, y + 5))

    # colour blends from red (explore) to green (exploit)
    blend = max_prob
    col = (int(220 * (1 - blend)), int(200 * blend), 30)
    draw_bar(surface, x + 8, y + 24, 300, 13, blend, col)
    conf_txt = font_sm.render(f"Policy confidence: {max_prob*100:.1f}%", True, PAL["white"])
    surface.blit(conf_txt, (x + 318, y + 24))


# ── Main demo loop ────────────────────────────────────────────────────
rng_disp = np.random.default_rng(7)

for ep in range(args.episodes):
    env = TomatoFarmEnv(render_mode=None)
    obs, _ = env.reset(seed=ep * 17)

    total_reward  = 0.0
    reward_hist   = []
    step_num      = 0
    running       = True
    action        = 0
    action_probs  = np.ones(10) / 10
    step_payload  = {}

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        # ── Agent decides ────────────────────────────────────────
        action, action_probs = get_action_and_probs(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_num     += 1
        reward_hist.append(float(reward))

        # Serialise step to JSON (shown in console + ready for API)
        step_payload = step_to_json(
            step_num, obs, action, action_probs,
            reward, terminated, truncated, info, total_reward
        )

        # ── Draw frame ───────────────────────────────────────────
        screen.fill(PAL["bg"])

        # Title bar
        pygame.draw.rect(screen, PAL["panel"], (0, 0, W, 46))
        pygame.draw.line(screen, PAL["border"], (0, 46), (W, 46), 1)
        ttl = font_xl.render(
            f"TomatoFarmEnv-v0  [{algo} Agent]  "
            f"Day {step_num:3d}/120   Ep {ep+1}/{args.episodes}",
            True, PAL["yellow"]
        )
        screen.blit(ttl, (10, 12))
        rwd_surf = font_xl.render(
            f"Total Reward: {total_reward:+8.1f}", True, PAL["teal"]
        )
        screen.blit(rwd_surf, (870, 12))

        # Farm grid (left column)
        draw_farm_grid(screen, env, rng_disp)

        GRID_W = 10 * 36 + 20
        GRID_H = 12 * 36

        # Reward sparkline below grid
        draw_reward_sparkline(
            screen, reward_hist, 14, 54 + GRID_H + 8, GRID_W - 10, 65
        )

        # ── Right panel ──────────────────────────────────────────
        RX = GRID_W + 18
        pygame.draw.line(screen, PAL["border"], (RX - 6, 46), (RX - 6, H), 1)

        # Action decision box
        pygame.draw.rect(screen, PAL["dark"], (RX, 54, 840, 60), border_radius=6)
        pygame.draw.rect(screen, PAL["border"], (RX, 54, 840, 60), 1, border_radius=6)
        act_lbl = font_lg.render(
            f"Agent Action:  [{action}]  {ACTION_NAMES[action]}", True, PAL["gold"]
        )
        screen.blit(act_lbl, (RX + 10, 64))
        rew_lbl = font_md.render(
            f"Step reward: {reward:+.2f}   "
            f"Crop health: {info.get('crop_health', 0):.2f}   "
            f"Disease: {info.get('disease_type','NONE')} ({info.get('disease_severity',0):.2f})   "
            f"Budget: {info.get('budget_remaining',0):.2f}",
            True, PAL["white"]
        )
        screen.blit(rew_lbl, (RX + 10, 88))

        # Observation panel
        obs_y = 126
        pygame.draw.rect(screen, PAL["panel"], (RX, obs_y, 400, 220))
        pygame.draw.rect(screen, PAL["border"], (RX, obs_y, 400, 220), 1)
        obs_title = font_md.render("Observation Space", True, PAL["yellow"])
        screen.blit(obs_title, (RX + 8, obs_y + 6))

        obs_data = [
            ("crop_health",      obs[10], PAL["green"] if obs[10] > 0.6 else PAL["red"]),
            ("disease_severity", obs[0],  PAL["red"]   if obs[0]  > 0.3 else PAL["green"]),
            ("soil_moisture",    obs[8],  PAL["blue"]),
            ("pest_population",  obs[11], PAL["orange"]),
            ("budget_remaining", obs[7],  PAL["teal"]),
            ("humidity",         obs[3],  PAL["blue"]),
            ("rainfall_prob",    obs[4],  PAL["blue"]),
        ]
        for oi, (name, val, col) in enumerate(obs_data):
            oy = obs_y + 28 + oi * 26
            lbl = font_sm.render(name, True, PAL["gray"])
            screen.blit(lbl, (RX + 8, oy))
            draw_bar(screen, RX + 160, oy + 2, 140, 14, float(val), col)
            vt = font_sm.render(f"{val:.2f}", True, col)
            screen.blit(vt, (RX + 308, oy))

        # Text obs
        text_obs = [
            ("disease_type",  DiseaseType(int(obs[1])).name),
            ("growth_stage",  GrowthStage(int(obs[5])).name),
            ("season_day",    str(int(obs[6]))),
            ("temperature",   f"{obs[2]:.1f}°C"),
        ]
        for ti, (name, val) in enumerate(text_obs):
            ty = obs_y + 28 + (len(obs_data) + ti) * 26  # can exceed panel - handled by surface clip
            if ty < obs_y + 215:
                lbl = font_sm.render(name, True, PAL["gray"])
                screen.blit(lbl, (RX + 8, ty))
                vt = font_sm.render(val, True, PAL["white"])
                screen.blit(vt, (RX + 160, ty))

        # Action probability bars
        draw_action_probs(screen, RX, 358, action_probs, action)

        # Exploration / Exploitation indicator
        draw_exploitation_indicator(screen, RX, 610, action_probs, step_num)

        # JSON Preview (shows serialised step payload)
        jy = 664
        pygame.draw.rect(screen, PAL["panel"], (RX, jy, 840, 90))
        pygame.draw.rect(screen, PAL["border"], (RX, jy, 840, 90), 1)
        jlbl = font_md.render("JSON Step Payload  (API response preview)", True, PAL["yellow"])
        screen.blit(jlbl, (RX + 8, jy + 5))
        # Show first 3 key-value pairs of serialised step
        preview_lines = [
            f'  "step": {step_payload["step"]},  "step_reward": {step_payload["step_reward"]},  "total_reward": {step_payload["total_reward"]},',
            f'  "action": {{"id": {step_payload["action"]["id"]}, "name": "{step_payload["action"]["name"]}", "probability": {step_payload["action"]["probability"]}}},',
            f'  "observation": {{"crop_health": {step_payload["observation"]["crop_health"]}, "disease_type": "{step_payload["observation"]["disease_type"]}", "growth_stage": "{step_payload["observation"]["growth_stage"]}"}},',
        ]
        for li, line in enumerate(preview_lines):
            lt = font_sm.render(line, True, PAL["teal"])
            screen.blit(lt, (RX + 8, jy + 24 + li * 19))

        # Model info footer
        mi_surf = font_sm.render(
            f"Model: {model_info['algo']} [{model_info['label']}]  "
            f"Eval mean reward: {format_mean_reward(model_info['mean_reward'])}   "
            f"Path: {model_info['model_path']}   |  [Q] quit",
            True, PAL["gray"]
        )
        screen.blit(mi_surf, (10, H - 18))

        pygame.display.flip()
        clock.tick(8)   # 8 FPS — slow enough to observe decisions

        if terminated or truncated:
            time.sleep(2.0)
            running = False

    env.close()
    print(f"\n[Episode {ep+1}] Total reward: {total_reward:+.2f}  Steps: {step_num}")
    print(f"  Final JSON payload:")
    print(json.dumps(step_payload, indent=2))

pygame.quit()
print("\n[main.py] Demo complete.")
print("Tip: run  python main.py --api  to start the REST API server.")
