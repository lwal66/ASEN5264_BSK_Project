"""
ppo.py  –  Custom PPO trainer for the BSK-RL spacecraft environment.

Architecture
------------
* Actor-Critic network with shared trunk + separate heads
* Clipped surrogate objective  (Schulman et al., 2017)
* Generalized Advantage Estimation (GAE-λ)
* Entropy bonus for exploration
* Value-function loss clipping (optional, toggled via PPOConfig)

Usage
-----
    python ppo.py                  # train with defaults
    python ppo.py --iters 100      # override number of training iterations
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from Basilisk.architecture import bskLogging

from config import EnvConfig, TrainConfig, PPOConfig
from envs import make_env

ACTION_NAMES = {
    0: "Charge",
    1: "Image",
}


# ---------------------------------------------------------------------------
# Actor-Critic network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic.

    Trunk
    ─────
    Two fully-connected layers (ReLU) shared by both heads.

    Actor head
    ──────────
    Linear → softmax over discrete action space.
    Outputs a Categorical distribution used for action sampling and
    log-probability computation.

    Critic head
    ───────────
    Linear → scalar state-value V(s).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: List[int]):
        super().__init__()

        # Build shared trunk
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Separate heads
        self.actor_head  = nn.Linear(in_dim, act_dim)   # logits
        self.critic_head = nn.Linear(in_dim, 1)          # V(s)

        # Orthogonal initialisation (common PPO practice)
        self._init_weights()

    def _init_weights(self):
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.zeros_(self.actor_head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(self, obs: torch.Tensor):
        """Returns (Categorical distribution, value scalar tensor)."""
        features = self.trunk(obs)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        dist     = Categorical(logits=logits)
        return dist, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.trunk(obs)
        return self.critic_head(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores one iteration's worth of (s, a, r, done, log_pi, V) transitions.
    After collection, compute_returns_and_advantages() fills in GAE targets.
    """

    def __init__(self):
        self.obs:      List[np.ndarray] = []
        self.actions:  List[int]        = []
        self.rewards:  List[float]      = []
        self.dones:    List[bool]        = []
        self.log_probs: List[float]     = []
        self.values:   List[float]      = []

        # Filled in by compute_returns_and_advantages
        self.advantages: np.ndarray | None = None
        self.returns:    np.ndarray | None = None

    def add(self, obs, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float,
        lam: float,
    ):
        """
        Generalized Advantage Estimation (GAE-λ).

        δ_t  = r_t + γ · V(s_{t+1}) · (1 - done_t) − V(s_t)
        A_t  = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + …

        The GAE target is computed in a single reverse pass.
        Returns are A_t + V(s_t)  (used as TD-λ value targets).
        """
        T = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        values_arr = np.array(self.values, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values_arr[t + 1]
            next_non_terminal = 1.0 - float(self.dones[t])

            # TD residual
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - values_arr[t]

            # GAE accumulation
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns    = advantages + values_arr   # V-target = A + V_old

    def as_tensors(self, device: torch.device):
        obs        = torch.as_tensor(np.array(self.obs),       dtype=torch.float32).to(device)
        actions    = torch.as_tensor(np.array(self.actions),   dtype=torch.long).to(device)
        log_probs  = torch.as_tensor(np.array(self.log_probs), dtype=torch.float32).to(device)
        advantages = torch.as_tensor(self.advantages,          dtype=torch.float32).to(device)
        returns    = torch.as_tensor(self.returns,             dtype=torch.float32).to(device)
        values_old = torch.as_tensor(np.array(self.values),    dtype=torch.float32).to(device)
        return obs, actions, log_probs, advantages, returns, values_old

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    env,
    model: ActorCritic,
    hp: PPOConfig,
    device: torch.device,
    env_cfg: EnvConfig,
    seed_offset: int = 0,
) -> Tuple[RolloutBuffer, dict]:
    """
    Run the current policy in the environment until `steps_per_iter` transitions
    have been collected.  Handles episode resets automatically.

    Returns
    -------
    buffer   : filled RolloutBuffer (without GAE yet)
    ep_stats : dict with mean/min/max reward and mean episode length
    """
    buffer = RolloutBuffer()

    ep_rewards:  List[float] = []
    ep_lengths:  List[int]   = []
    ep_reward    = 0.0
    ep_len       = 0

    obs_np, _ = env.reset(seed=env_cfg.seed + seed_offset)

    for _ in range(hp.steps_per_iter):
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            dist, value = model(obs_t)
            action      = dist.sample()
            log_prob    = dist.log_prob(action)

        action_int = action.item()
        try:
            next_obs_np, reward, terminated, truncated, _ = env.step(action_int)
        except RuntimeError:
            # Battery failure corrupts BSK-RL internal state — reset and move on
            next_obs_np, _ = env.reset(seed=env_cfg.seed + seed_offset + len(ep_rewards))
            reward, terminated, truncated = 0.0, True, False
        done = terminated or truncated

        buffer.add(
            obs      = obs_np,
            action   = action_int,
            reward   = float(reward),
            done     = done,
            log_prob = log_prob.item(),
            value    = value.item(),
        )

        ep_reward += float(reward)
        ep_len    += 1
        obs_np     = next_obs_np

        if done or ep_len >= hp.max_ep_steps:
            ep_rewards.append(ep_reward)
            ep_lengths.append(ep_len)
            ep_reward = 0.0
            ep_len    = 0
            obs_np, _ = env.reset(seed=env_cfg.seed + seed_offset + len(ep_rewards))

    # Bootstrap value for the last (possibly incomplete) episode
    with torch.no_grad():
        obs_t      = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
        last_value = model.get_value(obs_t).item()

    buffer.compute_returns_and_advantages(
        last_value = last_value,
        gamma      = hp.gamma,
        lam        = hp.lam,
    )

    stats = {
        "ep_reward_mean": float(np.mean(ep_rewards))  if ep_rewards else float("nan"),
        "ep_reward_min":  float(np.min(ep_rewards))   if ep_rewards else float("nan"),
        "ep_reward_max":  float(np.max(ep_rewards))   if ep_rewards else float("nan"),
        "ep_len_mean":    float(np.mean(ep_lengths))  if ep_lengths else float("nan"),
        "n_episodes":     len(ep_rewards),
    }
    return buffer, stats


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model:     ActorCritic,
    optimizer: optim.Optimizer,
    buffer:    RolloutBuffer,
    hp:        PPOConfig,
    device:    torch.device,
) -> dict:
    """
    Run `n_epochs` passes of minibatch SGD over the collected rollout.

    Loss
    ────
    L = −L_CLIP + vf_coef · L_VF − ent_coef · H

    where
      L_CLIP  = clipped surrogate policy loss
              = E[ min( r_t · A_t,  clip(r_t, 1-ε, 1+ε) · A_t ) ]
              with  r_t = π(a|s) / π_old(a|s)

      L_VF    = value function MSE, optionally clipped
      H       = policy entropy (bonus to encourage exploration)
    """
    obs, actions, old_log_probs, advantages, returns, old_values = buffer.as_tensors(device)

    # Normalise advantages (zero mean, unit variance) for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(buffer)
    metric_accum = {
        "loss_total": 0.0,
        "loss_policy": 0.0,
        "loss_value": 0.0,
        "entropy": 0.0,
        "n_updates": 0,
    }

    indices = np.arange(T)

    for _ in range(hp.n_epochs):
        np.random.shuffle(indices)

        for start in range(0, T, hp.minibatch_size):
            mb_idx = indices[start : start + hp.minibatch_size]
            if len(mb_idx) == 0:
                continue

            mb_obs        = obs[mb_idx]
            mb_actions    = actions[mb_idx]
            mb_old_lp     = old_log_probs[mb_idx]
            mb_adv        = advantages[mb_idx]
            mb_returns    = returns[mb_idx]
            mb_old_values = old_values[mb_idx]

            # ── Forward pass ─────────────────────────────────────────────────
            dist, values = model(mb_obs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy       = dist.entropy().mean()

            # ── Clipped surrogate policy loss ─────────────────────────────────
            log_ratio  = new_log_probs - mb_old_lp
            ratio      = torch.exp(log_ratio)                           # π / π_old

            surr1      = ratio * mb_adv
            surr2      = torch.clamp(ratio, 1.0 - hp.clip_eps, 1.0 + hp.clip_eps) * mb_adv
            loss_pi    = -torch.min(surr1, surr2).mean()

            # ── Value function loss ───────────────────────────────────────────
            if hp.clip_vf_loss:
                # Clip value update to stay close to old value estimate
                values_clipped = mb_old_values + torch.clamp(
                    values - mb_old_values, -hp.clip_eps, hp.clip_eps
                )
                vf_loss1 = (values        - mb_returns) ** 2
                vf_loss2 = (values_clipped - mb_returns) ** 2
                loss_vf  = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
            else:
                loss_vf  = 0.5 * ((values - mb_returns) ** 2).mean()

            # ── Total loss ────────────────────────────────────────────────────
            loss = loss_pi + hp.vf_coef * loss_vf - hp.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
            optimizer.step()

            metric_accum["loss_total"]  += loss.item()
            metric_accum["loss_policy"] += loss_pi.item()
            metric_accum["loss_value"]  += loss_vf.item()
            metric_accum["entropy"]     += entropy.item()
            metric_accum["n_updates"]   += 1

    n = max(metric_accum["n_updates"], 1)
    return {k: v / n for k, v in metric_accum.items() if k != "n_updates"}


# ---------------------------------------------------------------------------
# Post-training evaluation rollout + plotting
# ---------------------------------------------------------------------------

def eval_rollout(model, device, hp, env_cfg, episodes=5, max_steps=100):
    """Run the trained model greedily. Returns per-step and summary DataFrames."""
    env = make_env(env_cfg)
    model.eval()

    all_records     = []
    summary_records = []

    for ep in range(episodes):
        obs_np, _ = env.reset(seed=hp.seed + ep)
        done         = False
        step         = 0
        total_reward = 0.0

        while not done and step < max_steps:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, _ = model(obs_t)
                action  = dist.probs.argmax(dim=-1).item()  # greedy

            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

            record = {
                "episode":      ep,
                "step":         step,
                "action":       action,
                "action_name":  ACTION_NAMES.get(action, str(action)),
                "reward":       float(reward),
                "total_reward": total_reward,
                "terminated":   terminated,
                "truncated":    truncated,
                "battery":      float(next_obs[0]),
                "storage":      float(next_obs[1]),
            }
            # Store remaining obs generically
            for i in range(2, len(next_obs)):
                record[f"obs_{i}"] = float(next_obs[i])
            all_records.append(record)

            obs_np = next_obs
            step  += 1

        summary_records.append({
            "episode":      ep,
            "steps":        step,
            "total_reward": total_reward,
            "terminated":   terminated,
            "truncated":    truncated,
        })

    env.close()
    model.train()
    return pd.DataFrame(all_records), pd.DataFrame(summary_records)


def plot_results(df_steps, title="Post-Training Evaluation"):
    """
    Plot evaluation results across all episodes.
    - Continuous metrics (battery, storage, reward, eclipse): mean +/- std bands
    - Action: step chart per episode (discrete, mean is not meaningful)
    """
    episodes = df_steps["episode"].unique()

    # Truncate all episodes to the same length for band plots
    min_steps = df_steps.groupby("episode")["step"].count().min()
    df_trimmed = df_steps[df_steps["step"] < min_steps]
    steps = sorted(df_trimmed["step"].unique())

    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title, fontsize=13)

    # ── Continuous metrics: mean ± std bands ──────────────────────────────────
    band_metrics = [
        ("battery",  "Battery Fraction", axes[0]),
        ("storage",  "Storage Fraction", axes[1]),
        ("reward",   "Reward",           axes[2]),
    ]
    for col, ylabel, ax in band_metrics:
        data = np.array([
            df_trimmed[df_trimmed["episode"] == ep][col].values
            for ep in episodes
        ])
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        ax.plot(steps, mean, linewidth=1.5, label="Mean")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.25, label="±1 std")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")

    # ── Action: step chart per episode ────────────────────────────────────────
    for ep in episodes:
        ep_df = df_steps[df_steps["episode"] == ep]
        axes[3].step(ep_df["step"], ep_df["action"],
                     where="post", alpha=0.6, label=f"Ep {ep}")
    axes[3].set_ylabel("Action")
    axes[3].set_xlabel("Step")
    axes[3].set_yticks(sorted(df_steps["action"].unique()))
    axes[3].set_yticklabels([ACTION_NAMES.get(a, str(a))
                             for a in sorted(df_steps["action"].unique())])
    axes[3].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(hp: PPOConfig | None = None):
    if hp is None:
        hp = PPOConfig()

    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    torch.manual_seed(hp.seed)
    np.random.seed(hp.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = EnvConfig()  # seed comes from EnvConfig, not PPOConfig
    env     = make_env(env_cfg)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    print(f"Observation dim: {obs_dim}  |  Action dim: {act_dim}")

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model = ActorCritic(obs_dim, act_dim, hp.hidden_sizes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr, eps=1e-5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Output directories + CSV logger ──────────────────────────────────────
    paths = TrainConfig()
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.train_log_dir.mkdir(parents=True, exist_ok=True)
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = paths.train_log_dir / f"{timestamp}_ppo_train.csv"

    csv_header = [
        "iter",
        "ep_reward_mean", "ep_reward_min", "ep_reward_max", "ep_len_mean",
        "n_episodes",
        "loss_total", "loss_policy", "loss_value", "entropy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(csv_header)

    print(f"\nLogging to: {csv_path}")
    print(f"{'Iter':>5}  {'MeanRew':>10}  {'MinRew':>9}  {'MaxRew':>9}  "
          f"{'EpLen':>7}  {'Loss_π':>9}  {'Loss_V':>9}  {'H':>7}")
    print("-" * 80)

    # ── Main training loop ────────────────────────────────────────────────────
    for iteration in range(1, hp.train_iters + 1):

        # 1. Collect rollout with current policy
        buffer, ep_stats = collect_rollout(
            env        = env,
            model      = model,
            hp         = hp,
            device     = device,
            env_cfg    = env_cfg,
            seed_offset = iteration,
        )

        # 2. PPO gradient updates
        loss_stats = ppo_update(model, optimizer, buffer, hp, device)

        # 3. Log to console
        print(
            f"{iteration:5d}  "
            f"{ep_stats['ep_reward_mean']:10.4f}  "
            f"{ep_stats['ep_reward_min']:9.4f}  "
            f"{ep_stats['ep_reward_max']:9.4f}  "
            f"{ep_stats['ep_len_mean']:7.1f}  "
            f"{loss_stats['loss_policy']:9.5f}  "
            f"{loss_stats['loss_value']:9.5f}  "
            f"{loss_stats['entropy']:7.4f}"
        )

        # 4. Log to CSV
        row = [
            iteration,
            ep_stats["ep_reward_mean"], ep_stats["ep_reward_min"],
            ep_stats["ep_reward_max"],  ep_stats["ep_len_mean"],
            ep_stats["n_episodes"],
            loss_stats["loss_total"],   loss_stats["loss_policy"],
            loss_stats["loss_value"],   loss_stats["entropy"],
        ]
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        # 5. Periodic checkpoint
        if iteration % hp.checkpoint_every == 0 or iteration == hp.train_iters:
            ckpt_path = paths.checkpoint_dir / f"ppo_iter_{iteration:04d}.pt"
            torch.save({
                "iteration":    iteration,
                "model_state":  model.state_dict(),
                "optim_state":  optimizer.state_dict(),
                "hyperparams":  hp,
                "obs_dim":      obs_dim,
                "act_dim":      act_dim,
            }, ckpt_path)
            print(f"  └─ Checkpoint saved: {ckpt_path}")

    env.close()
    print("\nTraining complete.")

    # ── Post-training evaluation rollout + plots ──────────────────────────────
    print("\nRunning post-training evaluation...")
    df_steps, df_summary = eval_rollout(model, device, hp, env_cfg)

    df_steps.to_csv(paths.eval_dir / f"{timestamp}_ppo_eval_steps.csv", index=False)
    df_summary.to_csv(paths.eval_dir / f"{timestamp}_ppo_eval_summary.csv", index=False)
    print(f"Eval CSVs saved to: {paths.eval_dir}")

    print("\nEpisode summary:")
    print(df_summary.to_string(index=False))

    plot_results(df_steps, title="Custom PPO — Post-Training Evaluation")

    return model, csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train PPO on BSK-RL spacecraft env")
    parser.add_argument("--iters",          type=int,   default=50,    help="Training iterations")
    parser.add_argument("--steps",          type=int,   default=2048,  help="Steps per iteration")
    parser.add_argument("--lr",             type=float, default=3e-4,  help="Adam learning rate")
    parser.add_argument("--gamma",          type=float, default=0.99,  help="Discount factor")
    parser.add_argument("--lam",            type=float, default=0.95,  help="GAE lambda")
    parser.add_argument("--clip-eps",       type=float, default=0.2,   help="PPO clip epsilon")
    parser.add_argument("--n-epochs",       type=int,   default=10,    help="SGD epochs per iter")
    parser.add_argument("--minibatch",      type=int,   default=64,    help="Minibatch size")
    parser.add_argument("--ent-coef",       type=float, default=0.01,  help="Entropy bonus")
    parser.add_argument("--seed",           type=int,   default=1,     help="Random seed")
    parser.add_argument("--no-clip-vf",     action="store_true",       help="Disable VF loss clipping")
    args = parser.parse_args()

    return PPOConfig(
        train_iters     = args.iters,
        steps_per_iter  = args.steps,
        lr              = args.lr,
        gamma           = args.gamma,
        lam             = args.lam,
        clip_eps        = args.clip_eps,
        n_epochs        = args.n_epochs,
        minibatch_size  = args.minibatch,
        ent_coef        = args.ent_coef,
        seed            = args.seed,
        clip_vf_loss    = not args.no_clip_vf,
    )


if __name__ == "__main__":
    hp    = parse_args()
    train(hp)