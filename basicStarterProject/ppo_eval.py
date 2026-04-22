"""
ppo_eval.py  —  Standalone evaluation script for the custom PPO.

Loads the latest .pt checkpoint from outdir/checkpoints/, runs greedy
rollouts, prints an episode summary, saves CSVs, and produces two figures:

  Figure 1 — Episode metrics (battery, storage, reward, action)
             Mean ± std bands for continuous metrics across all episodes.
             Per-episode step chart for action (discrete).

  Figure 2 — Training curve
             Reads the most recent training CSV from outdir/train_logs/
             and plots mean reward per iteration with min/max shading.

Usage
-----
    python ppo_eval.py                  # evaluate latest checkpoint
    python ppo_eval.py --episodes 10    # run more episodes
    python ppo_eval.py --max-steps 200  # longer episodes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical

from Basilisk.architecture import bskLogging

# Import model architecture and helpers from ppo.py
from ppo import ActorCritic, ACTION_NAMES
from config import EnvConfig, PPOConfig, TrainConfig
from envs import make_env


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Return the most recently saved .pt checkpoint file."""
    candidates = sorted(checkpoint_dir.glob("ppo_iter_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt checkpoints found in {checkpoint_dir}.\n"
            "Run ppo.py first to train the model."
        )
    return candidates[-1]


def load_checkpoint(checkpoint_path: Path):
    """Load model from checkpoint. Returns (model, hyperparams, obs_dim, act_dim)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hp      = ckpt["hyperparams"]

    # hp may be a PPOConfig or old PPOHyperparams — handle both
    hidden_sizes = hp.hidden_sizes if hasattr(hp, "hidden_sizes") else [64, 64]

    model = ActorCritic(obs_dim, act_dim, hidden_sizes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path.name}")
    print(f"  Trained for {ckpt['iteration']} iterations")
    print(f"  Obs dim: {obs_dim}  |  Act dim: {act_dim}")
    print(f"  Hidden sizes: {hidden_sizes}")

    return model, hp, obs_dim, act_dim


# ---------------------------------------------------------------------------
# Greedy rollout
# ---------------------------------------------------------------------------

def run_rollout(model, episodes: int, max_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run greedy (argmax) rollouts. Returns (df_steps, df_summary).
    Handles battery-failure RuntimeErrors gracefully.
    """
    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    env_cfg = EnvConfig()
    env     = make_env(env_cfg)

    all_records     = []
    summary_records = []

    for ep in range(episodes):
        obs_np, _ = env.reset(seed=env_cfg.seed + ep)
        done         = False
        step         = 0
        total_reward = 0.0

        while not done and step < max_steps:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value = model(obs_t)
                action      = dist.probs.argmax(dim=-1).item()  # greedy

            try:
                next_obs, reward, terminated, truncated, _ = env.step(action)
            except RuntimeError:
                # Battery failure — treat as terminal with zero reward
                next_obs, _  = env.reset(seed=env_cfg.seed + ep + episodes)
                reward, terminated, truncated = 0.0, True, False

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

        print(f"  Episode {ep+1:2d}/{episodes}  steps={step:4d}  "
              f"total_reward={total_reward:.4f}  "
              f"{'TERMINATED' if terminated else 'truncated'}")

    env.close()
    return pd.DataFrame(all_records), pd.DataFrame(summary_records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_episode_metrics(df_steps: pd.DataFrame, title: str = "Custom PPO — Evaluation"):
    """
    Figure 1: battery, storage, reward (mean ± std) + action (per episode).
    """
    episodes  = df_steps["episode"].unique()
    min_steps = df_steps.groupby("episode")["step"].count().min()
    df_trim   = df_steps[df_steps["step"] < min_steps]
    steps     = sorted(df_trim["step"].unique())

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=13)

    # ── Mean ± std bands for continuous metrics ───────────────────────────────
    for ax, col, ylabel in [
        (axes[0], "battery", "Battery Fraction"),
        (axes[1], "storage", "Storage Fraction"),
        (axes[2], "reward",  "Step Reward"),
    ]:
        data = np.array([
            df_trim[df_trim["episode"] == ep][col].values
            for ep in episodes
        ])
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        ax.plot(steps, mean, linewidth=1.5, label="Mean")
        ax.fill_between(steps, mean - std, mean + std,
                        alpha=0.25, label="±1 std")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")

    # ── Action: per-episode step chart ────────────────────────────────────────
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
    return fig


def plot_training_curve(train_log_dir: Path, title: str = "Custom PPO — Training Curve"):
    """
    Figure 2: reward mean ± min/max shading across training iterations.
    Reads the most recent training CSV from train_log_dir.
    """
    csvs = sorted(train_log_dir.glob("*_ppo_train.csv"))
    if not csvs:
        print(f"No training CSVs found in {train_log_dir} — skipping training curve.")
        return None

    df = pd.read_csv(csvs[-1])
    print(f"\nTraining log: {csvs[-1].name}  ({len(df)} iterations)")

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=13)

    ax.plot(df["iter"], df["ep_reward_mean"], linewidth=1.5, label="Mean reward")
    ax.fill_between(df["iter"], df["ep_reward_min"], df["ep_reward_max"],
                    alpha=0.2, label="Min / Max")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Episode Reward")
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate custom PPO checkpoint")
    parser.add_argument("--episodes",  type=int, default=5,   help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--no-plot",   action="store_true",   help="Skip plots")
    args = parser.parse_args()

    paths = TrainConfig()
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = find_latest_checkpoint(paths.checkpoint_dir)
    model, hp, obs_dim, act_dim = load_checkpoint(checkpoint_path)

    # ── Run rollouts ──────────────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} greedy episodes (max {args.max_steps} steps each)...")
    df_steps, df_summary = run_rollout(model, args.episodes, args.max_steps)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\nEpisode summary:")
    print(df_summary.to_string(index=False))
    print(f"\nMean total reward: {df_summary['total_reward'].mean():.4f}")
    print(f"Std  total reward: {df_summary['total_reward'].std():.4f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    steps_path   = paths.eval_dir / "ppo_eval_steps.csv"
    summary_path = paths.eval_dir / "ppo_eval_summary.csv"
    df_steps.to_csv(steps_path,   index=False)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {steps_path}")
    print(f"Saved: {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        ckpt_label = checkpoint_path.stem  # e.g. ppo_iter_0020
        fig1 = plot_episode_metrics(df_steps, title=f"Custom PPO — Evaluation ({ckpt_label})")
        fig2 = plot_training_curve(paths.train_log_dir, title="Custom PPO — Training Curve")

        fig1.savefig(paths.eval_dir / "ppo_eval_metrics.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {paths.eval_dir / 'ppo_eval_metrics.png'}")

        if fig2 is not None:
            fig2.savefig(paths.eval_dir / "ppo_training_curve.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_training_curve.png'}")

        plt.show()


if __name__ == "__main__":
    main()