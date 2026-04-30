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

# import pdb

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

        cities_imaged = 0

        while not done and step < max_steps:
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value = model(obs_t)
                action      = dist.probs.argmax(dim=-1).item()  # greedy

            # Read opportunity list BEFORE env.step() so selected_target
            # reflects what the policy observed when it chose the action
            try:
                pre_step_opps   = env.unwrapped.satellite.upcoming_opportunities
                selected_target = pre_step_opps[action - 1]["object"].name if (action > 0 and len(pre_step_opps) >= action) else None
            except Exception:
                selected_target = None

            try:
                next_obs, reward, terminated, truncated, _ = env.step(action)
            except RuntimeError:
                # Battery failure — treat as terminal with zero reward
                next_obs, _  = env.reset(seed=env_cfg.seed + ep + episodes)
                reward, terminated, truncated = 0.0, True, False

            total_reward += float(reward)
            if float(reward) > 0:
                cities_imaged += 1
            done = terminated or truncated

            # record = {
            #     "episode":      ep,
            #     "step":         step,
            #     "action":       action,
            #     "action_name":  ACTION_NAMES.get(action, str(action)),
            #     "reward":       float(reward),
            #     "total_reward": total_reward,
            #     "terminated":   terminated,
            #     "truncated":    truncated,
            #     "battery":      float(next_obs[0]),
            #     "storage":      float(next_obs[1]),
            # }
            # for i in range(2, len(next_obs)):
            #     record[f"obs_{i}"] = float(next_obs[i])
            # all_records.append(record)

            # Post-step opportunity list (for reference/debugging)
            #upcomingOpps = env.unwrapped.satellite.find_next_opportunities(n=5, types="target")
            upcomingOpps = env.unwrapped.satellite.upcoming_opportunities
            # pdb.set_trace()

            # for i, opp in enumerate(upcomingOpps):
            #     print(i, 
            #           opp["object"].name,
            #           str(opp["window"]),
            #           )
            # print("OBS SLOT 0: ", next_obs[2:6])
            # print("OBS SLOT 1: ", next_obs[6:10])
            # print("OBS SLOT 2: ", next_obs[10:14])
            # print("OBS SLOT 3: ", next_obs[14:18])
            # print("OBS SLOT 4: ", next_obs[18:22])

            # pdb.set_trace()
            # total_reward already accumulated above — do not double-count

            # selected_target uses pre_step_opps (read before env.step())
            imaged_target = None
            if reward > 0:
                tgt = getattr(env.unwrapped.satellite, "latest_target", None)
                if tgt is not None:
                    imaged_target = tgt.name
            #pdb.set_trace()

            record = {
                "episode":         ep,
                "step":            step,
                "action":          action,
                "action_name":     ACTION_NAMES.get(action, str(action)),
                "reward":          float(reward),
                "total_reward":    total_reward,
                "terminated":      terminated,
                "truncated":       truncated,
                "battery":         float(next_obs[0]),
                "storage":         float(next_obs[1]),
                "selected_target": selected_target,
                "imaged_target":   imaged_target,
            }
            for i in range(2, len(next_obs)):
                record[f"obs_{i}"] = float(next_obs[i])
            all_records.append(record)

            obs_np = next_obs
            step  += 1

        imaging_efficiency = (total_reward / cities_imaged) if cities_imaged > 0 else 0.0

        summary_records.append({
            "episode":           ep,
            "steps":             step,
            "total_reward":      total_reward,
            "cities_imaged":     cities_imaged,
            "imaging_efficiency": imaging_efficiency,
            "terminated":        terminated,
            "truncated":         truncated,
        })

        print(f"  Episode {ep+1:2d}/{episodes}  steps={step:4d}  "
              f"total_reward={total_reward:.4f}  "
              f"cities_imaged={cities_imaged}  "
              f"efficiency={imaging_efficiency:.3f}  "
              f"{'TERMINATED' if terminated else 'truncated'}")

    env.close()
    return pd.DataFrame(all_records), pd.DataFrame(summary_records)


# ---------------------------------------------------------------------------
# Random policy rollout
# ---------------------------------------------------------------------------

def heuristic_action(obs_np, battery_threshold=0.2):
    """
    Simple heuristic policy:
      - If battery fraction < battery_threshold: charge (action 0)
      - Otherwise: image the upcoming target with the highest priority

    Observation layout (from satellites.py):
      obs[0]  = battery_charge_fraction
      obs[2]  = priority of target 1  (then target_angle, t_open, t_close at 3,4,5)
      obs[6]  = priority of target 2
      obs[10] = priority of target 3
      obs[14] = priority of target 4
      obs[18] = priority of target 5
    """
    battery = float(obs_np[0])
    if battery < battery_threshold:
        return 0  # Charge

    # Priority indices for the 5 upcoming targets
    priority_indices = [2, 6, 10, 14, 18]
    priorities = [float(obs_np[idx]) for idx in priority_indices]
    return int(np.argmax(priorities)) + 1  # 1-indexed action


def run_heuristic_rollout(episodes, max_steps,
                          battery_threshold=0.2):
    """
    Run the heuristic policy:
      - Charge if battery < battery_threshold (default 0.2)
      - Otherwise image the highest-priority upcoming target
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
        cities_imaged = 0

        while not done and step < max_steps:
            action = heuristic_action(obs_np, battery_threshold)

            # Read opportunity list BEFORE env.step()
            try:
                pre_step_opps   = env.unwrapped.satellite.upcoming_opportunities
                selected_target = pre_step_opps[action - 1]["object"].name if (action > 0 and len(pre_step_opps) >= action) else None
            except Exception:
                selected_target = None

            try:
                next_obs, reward, terminated, truncated, _ = env.step(action)
            except RuntimeError:
                next_obs, _  = env.reset(seed=env_cfg.seed + ep + episodes)
                reward, terminated, truncated = 0.0, True, False

            total_reward += float(reward)
            if float(reward) > 0:
                cities_imaged += 1
            done = terminated or truncated

            imaged_target = None
            if reward > 0:
                tgt = getattr(env.unwrapped.satellite, "latest_target", None)
                if tgt is not None:
                    imaged_target = tgt.name

            record = {
                "episode":         ep,
                "step":            step,
                "action":          action,
                "action_name":     ACTION_NAMES.get(action, str(action)),
                "reward":          float(reward),
                "total_reward":    total_reward,
                "terminated":      terminated,
                "truncated":       truncated,
                "battery":         float(next_obs[0]),
                "storage":         float(next_obs[1]),
                "selected_target": selected_target,
                "imaged_target":   imaged_target,
            }
            for i in range(2, len(next_obs)):
                record[f"obs_{i}"] = float(next_obs[i])
            all_records.append(record)

            obs_np = next_obs
            step  += 1

        imaging_efficiency = (total_reward / cities_imaged) if cities_imaged > 0 else 0.0

        summary_records.append({
            "episode":            ep,
            "steps":              step,
            "total_reward":       total_reward,
            "cities_imaged":      cities_imaged,
            "imaging_efficiency": imaging_efficiency,
            "terminated":         terminated,
            "truncated":          truncated,
        })

        print(f"  Episode {ep+1:2d}/{episodes}  steps={step:4d}  "
              f"total_reward={total_reward:.4f}  "
              f"cities_imaged={cities_imaged}  "
              f"efficiency={imaging_efficiency:.3f}  "
              f"{'TERMINATED' if terminated else 'truncated'}")

    env.close()
    return pd.DataFrame(all_records), pd.DataFrame(summary_records)


def run_random_rollout(episodes, max_steps):
    """
    Run a uniform random policy as a lower-bound baseline.
    Actions are sampled uniformly from the action space each step.
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
        cities_imaged = 0

        while not done and step < max_steps:
            action = env.action_space.sample()

            try:
                pre_step_opps   = env.unwrapped.satellite.upcoming_opportunities
                selected_target = pre_step_opps[action - 1]["object"].name if (action > 0 and len(pre_step_opps) >= action) else None
            except Exception:
                selected_target = None

            try:
                next_obs, reward, terminated, truncated, _ = env.step(action)
            except RuntimeError:
                next_obs, _  = env.reset(seed=env_cfg.seed + ep + episodes)
                reward, terminated, truncated = 0.0, True, False

            total_reward += float(reward)
            if float(reward) > 0:
                cities_imaged += 1
            done = terminated or truncated

            imaged_target = None
            if reward > 0:
                tgt = getattr(env.unwrapped.satellite, "latest_target", None)
                if tgt is not None:
                    imaged_target = tgt.name

            record = {
                "episode":         ep,
                "step":            step,
                "action":          action,
                "action_name":     ACTION_NAMES.get(action, str(action)),
                "reward":          float(reward),
                "total_reward":    total_reward,
                "terminated":      terminated,
                "truncated":       truncated,
                "battery":         float(next_obs[0]),
                "storage":         float(next_obs[1]),
                "selected_target": selected_target,
                "imaged_target":   imaged_target,
            }
            for i in range(2, len(next_obs)):
                record[f"obs_{i}"] = float(next_obs[i])
            all_records.append(record)

            obs_np = next_obs
            step  += 1

        imaging_efficiency = (total_reward / cities_imaged) if cities_imaged > 0 else 0.0

        summary_records.append({
            "episode":            ep,
            "steps":              step,
            "total_reward":       total_reward,
            "cities_imaged":      cities_imaged,
            "imaging_efficiency": imaging_efficiency,
            "terminated":         terminated,
            "truncated":          truncated,
        })

        print(f"  Episode {ep+1:2d}/{episodes}  steps={step:4d}  "
              f"total_reward={total_reward:.4f}  "
              f"cities_imaged={cities_imaged}  "
              f"efficiency={imaging_efficiency:.3f}  "
              f"{'TERMINATED' if terminated else 'truncated'}")

    env.close()
    return pd.DataFrame(all_records), pd.DataFrame(summary_records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Journal-style matplotlib defaults (IEEE 2-column format)
import matplotlib as mpl
mpl.rcParams.update({
    "font.size":         8,
    "axes.titlesize":    8,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def _label_subplots(axes, x=-0.18, y=1.02, fontsize=8):
    """Add (a), (b), (c)... labels to the top-left of each subplot."""
    import string
    flat = axes.flatten() if hasattr(axes, "flatten") else list(axes)
    for i, ax in enumerate(flat):
        ax.text(x, y, f"({string.ascii_lowercase[i]})",
                transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold",
                va="top", ha="right")


def plot_episode_metrics(df_steps: pd.DataFrame, title: str = "Custom PPO — Evaluation"):
    """
    Figure 1: battery, storage, reward (mean ± std) + action (per episode).
    Uses normalised episode progress [0, 1] so all episodes contribute equally.
    """
    episodes  = df_steps["episode"].unique()
    min_steps = df_steps.groupby("episode")["step"].count().min()
    df_trim   = df_steps[df_steps["step"] < min_steps]
    norm_steps = np.linspace(0, 1, min_steps)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6), sharex=False)
    fig.suptitle(title, fontsize=9, fontweight="bold")

    # ── Mean ± std bands for continuous metrics ───────────────────────────────
    for ax, col, ylabel in [
        (axes[0], "battery", "Battery\nFraction"),
        (axes[1], "reward",  "Step\nReward"),
    ]:
        data = np.array([
            df_trim[df_trim["episode"] == ep][col].values
            for ep in episodes
        ])
        mean = data.mean(axis=0)
        std  = data.std(axis=0)
        ax.plot(norm_steps, mean, linewidth=1.2, label="Mean")
        ax.fill_between(norm_steps, mean - std, mean + std,
                        alpha=0.25, label="±1 std")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode Progress (normalised)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc="upper right")

    # ── Action frequency: mean fraction of steps per action ─────────────────
    all_actions = sorted(df_steps["action"].unique())
    action_labels = [ACTION_NAMES.get(a, f"Action {a}") for a in all_actions]

    # Compute per-episode action fractions then average across episodes
    fractions = []
    for ep in episodes:
        ep_df = df_steps[df_steps["episode"] == ep]
        total = len(ep_df)
        fractions.append([
            (ep_df["action"] == a).sum() / total for a in all_actions
        ])
    fractions = np.array(fractions)  # shape (n_episodes, n_actions)
    mean_frac = fractions.mean(axis=0)
    std_frac  = fractions.std(axis=0)

    bars = axes[2].bar(action_labels, mean_frac, yerr=std_frac,
                       capsize=4, color="steelblue", alpha=0.8)
    axes[2].set_ylabel("Action Frequency")
    axes[2].set_xlabel("Action")
    axes[2].set_ylim(0, 1)
    axes[2].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    # Annotate bars with percentage
    for bar, frac in zip(bars, mean_frac):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{frac:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    _label_subplots(axes)
    plt.tight_layout(pad=0.5)
    return fig


def plot_training_curve(train_log_dir: Path, title: str = "Custom PPO — Training Curve"):
    """
    Figure 2a: PPO loss metrics (reward, policy loss, value loss, entropy).
    Figure 2b: Environment metrics (battery failure rate, episode length).
    Returns (fig_losses, fig_env) — fig_env is None if columns not available.
    """
    csvs = sorted(train_log_dir.glob("*_ppo_train.csv"))
    if not csvs:
        print(f"No training CSVs found in {train_log_dir} — skipping training curve.")
        return None, None

    df = pd.read_csv(csvs[-1])
    print(f"\nTraining log: {csvs[-1].name}  ({len(df)} iterations)")

    # ── Figure 2a: Loss metrics ───────────────────────────────────────────────
    fig_losses, axes = plt.subplots(4, 1, figsize=(3.5, 8), sharex=True)
    fig_losses.suptitle(title + " — Loss Metrics", fontsize=9, fontweight="bold")

    axes[0].plot(df["iter"], df["ep_reward_mean"], linewidth=1.2, label="Mean")
    axes[0].fill_between(df["iter"], df["ep_reward_min"], df["ep_reward_max"],
                         alpha=0.2, label="Min / Max")
    axes[0].set_ylabel("Episode\nReward")
    axes[0].legend(fontsize=7)

    axes[1].plot(df["iter"], df["loss_policy"], linewidth=1.2, color="tomato")
    axes[1].set_ylabel("Policy Loss\n(L_CLIP)")
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[2].plot(df["iter"], df["loss_value"], linewidth=1.2, color="darkorange")
    axes[2].set_ylabel("Value Loss\n(L_VF)")

    axes[3].plot(df["iter"], df["entropy"], linewidth=1.2, color="green")
    axes[3].set_ylabel("Policy\nEntropy (H)")
    axes[3].set_xlabel("Training Iteration")

    plt.tight_layout(pad=0.5)

    # ── Figure 2b: Environment metrics ───────────────────────────────────────
    has_failure = "battery_failure_rate" in df.columns
    has_ep_len  = "ep_len_mean" in df.columns

    if not has_failure and not has_ep_len:
        return fig_losses, None

    n_panels = int(has_failure) + int(has_ep_len)
    # Stack vertically for single-column journal format (3.5" wide)
    fig_env, env_axes = plt.subplots(n_panels, 1, figsize=(3.5, 2.5 * n_panels), sharex=True)
    fig_env.suptitle(title + " — Environment Metrics", fontsize=9, fontweight="bold")

    if n_panels == 1:
        env_axes = [env_axes]

    ax_idx = 0
    if has_failure:
        env_axes[ax_idx].plot(df["iter"], df["battery_failure_rate"] * 100,
                              linewidth=1.2, color="purple")
        env_axes[ax_idx].set_ylabel("Battery Failure\nRate (%)")
        env_axes[ax_idx].set_ylim(0, 100)
        env_axes[ax_idx].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0f}%")
        )
        ax_idx += 1

    if has_ep_len:
        env_axes[ax_idx].plot(df["iter"], df["ep_len_mean"],
                              linewidth=1.2, color="teal")
        env_axes[ax_idx].set_ylabel("Mean Episode\nLength (steps)")

    env_axes[-1].set_xlabel("Training Iteration")

    _label_subplots(axes)           # label fig_losses panels
    _label_subplots(env_axes)       # label fig_env panels
    plt.tight_layout(pad=0.5)
    return fig_losses, fig_env


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def _add_return(df: pd.DataFrame, gamma: float = 0.99) -> pd.DataFrame:
    """
    Add a return_ column: sum of future discounted rewards from each step.
    This is what the PPO critic estimates — G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...
    """
    df = df.copy()
    df["return_"] = 0.0
    for ep in df["episode"].unique():
        mask    = df["episode"] == ep
        rewards = df.loc[mask, "reward"].values
        T       = len(rewards)
        G       = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + gamma * running
            G[t]    = running
        df.loc[mask, "return_"] = G
    return df


def plot_comparison(
    ppo_steps: pd.DataFrame,
    heur_steps: pd.DataFrame,
    rnd_steps: pd.DataFrame,
    ppo_summary: pd.DataFrame,
    heur_summary: pd.DataFrame,
    rnd_summary: pd.DataFrame,
    title: str = "Policy Comparison",
    gamma: float = 0.99,
):
    """
    Three-way mean ± std comparison: PPO vs Heuristic vs Random.
    Panels: cumulative reward, return G_t, step reward, bar chart.
    """
    ppo_steps  = _add_return(ppo_steps,  gamma)
    heur_steps = _add_return(heur_steps, gamma)
    rnd_steps  = _add_return(rnd_steps,  gamma)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    metrics = [
        ("total_reward", "Cumulative\nPriority-Weighted Reward", axes[0]),
        ("return_",      f"Return G_t\n(γ={gamma})",             axes[1]),
    ]

    for col, ylabel, ax in metrics:
        for df, label, color in [
            (ppo_steps,  "PPO",       "steelblue"),
            (heur_steps, "Heuristic", "darkorange"),
            (rnd_steps,  "Random",    "tomato"),
        ]:
            episodes  = df["episode"].unique()
            min_steps = df.groupby("episode")["step"].count().min()
            df_trim   = df[df["step"] < min_steps]
            norm_steps = np.linspace(0, 1, min_steps)
            data = np.array([
                df_trim[df_trim["episode"] == ep][col].values
                for ep in episodes
            ])
            mean = data.mean(axis=0)
            std  = data.std(axis=0)
            ax.plot(norm_steps, mean, linewidth=1.2, label=label, color=color)
            ax.fill_between(norm_steps, mean - std, mean + std, alpha=0.15, color=color)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode Progress (normalised)")

    # Set legend locations per panel independently
    axes[0].legend(fontsize=7, loc="upper left")
    axes[1].legend(fontsize=7, loc="upper right")

    # ── Summary bar chart ─────────────────────────────────────────────────────
    ax_bar = axes[2]
    means  = [ppo_summary["total_reward"].mean(),
              heur_summary["total_reward"].mean(),
              rnd_summary["total_reward"].mean()]
    stds   = [ppo_summary["total_reward"].std(),
              heur_summary["total_reward"].std(),
              rnd_summary["total_reward"].std()]
    labels = ["PPO", "Heuristic", "Random"]
    colors = ["steelblue", "darkorange", "tomato"]

    bars = ax_bar.bar(labels, means, yerr=stds, color=colors, capsize=5, width=0.5, alpha=0.85)
    ax_bar.set_ylabel("Mean Total Reward")
    ax_bar.set_title("Mean Total Reward per Episode")

    for bar, mean, std in zip(bars, means, stds):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + 0.1,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # Print numeric summary
    print("\nPolicy comparison (total reward):")
    for label, mean, std in zip(labels, means, stds):
        print(f"  {label:<12} mean = {mean:.4f} ± {std:.4f}")
    for label, mean in zip(labels[1:], means[1:]):
        diff = means[0] - mean
        pct  = (diff / abs(mean) * 100) if mean != 0 else float("inf")
        print(f"  PPO vs {label:<10} = {diff:+.4f}  ({pct:.1f}%)")

    _label_subplots(axes)
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Cities imaged comparison
# ---------------------------------------------------------------------------

def plot_cities_imaged(ppo_summary, heur_summary, rnd_summary,
                       title="Cities Imaged & Imaging Efficiency"):
    ppo_cities  = ppo_summary["cities_imaged"].values
    heur_cities = heur_summary["cities_imaged"].values
    rnd_cities  = rnd_summary["cities_imaged"].values
    ppo_eff     = ppo_summary["imaging_efficiency"].values
    heur_eff    = heur_summary["imaging_efficiency"].values
    rnd_eff     = rnd_summary["imaging_efficiency"].values

    # 4 panels stacked vertically: bar + line for each metric
    fig, axes = plt.subplots(4, 1, figsize=(3.5, 9))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    labels = ["PPO", "Heuristic", "Random"]
    colors = ["steelblue", "darkorange", "tomato"]
    episodes = range(len(ppo_cities))

    metric_data = [
        ([ppo_cities, heur_cities, rnd_cities], "Cities\nImaged",           "Cities Imaged"),
        ([ppo_eff,    heur_eff,    rnd_eff],    "Avg Priority\n(rew/city)", "Imaging Efficiency"),
    ]

    for panel_idx, (all_vals, ylabel, row_title) in enumerate(metric_data):
        bar_ax  = axes[panel_idx * 2]
        line_ax = axes[panel_idx * 2 + 1]

        means = [v.mean() for v in all_vals]
        stds  = [v.std()  for v in all_vals]

        # Bar chart panel
        bars = bar_ax.bar(labels, means, yerr=stds, color=colors,
                          capsize=5, width=0.5, alpha=0.85)
        for i, vals in enumerate(all_vals):
            bar_ax.scatter([i] * len(vals), vals, color="black",
                           s=15, zorder=3, alpha=0.6)
        for bar, mean, std in zip(bars, means, stds):
            bar_ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + std + max(stds) * 0.15,
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        bar_ax.set_ylabel(ylabel)
        bar_ax.set_title(row_title + " — Mean ± Std", fontsize=8)
        bar_ax.set_ylim(0)

        # Per-episode line panel
        for vals, label, color, marker in zip(all_vals, labels, colors, ["o", "s", "^"]):
            line_ax.plot(episodes, vals, marker + "-", color=color,
                         linewidth=1.2, label=label)
        line_ax.set_ylabel(ylabel)
        line_ax.set_title(row_title + " — Per Episode", fontsize=8)
        line_ax.set_xlabel("Episode")
        line_ax.legend(fontsize=7)
        line_ax.set_xticks(list(episodes))
        line_ax.set_ylim(0)

    print(f"\nCities imaged:")
    for label, vals in zip(labels, [ppo_cities, heur_cities, rnd_cities]):
        print(f"  {label:<12} mean = {vals.mean():.2f} +/- {vals.std():.2f}")
    print(f"\nImaging efficiency (avg priority per city):")
    for label, vals in zip(labels, [ppo_eff, heur_eff, rnd_eff]):
        print(f"  {label:<12} mean = {vals.mean():.3f} +/- {vals.std():.3f}")

    _label_subplots(axes)
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Action distribution over training (#3)
# ---------------------------------------------------------------------------

def plot_action_distribution(train_log_dir: Path,
                              title: str = "Custom PPO — Action Distribution Over Training"):
    """
    Shows how the fraction of Charge vs Image actions evolves over training.
    A learning policy should start charging more and shift toward imaging
    as it learns to manage battery.
    """
    csvs = sorted(train_log_dir.glob("*_ppo_train.csv"))
    if not csvs:
        print("No training CSVs found — skipping action distribution plot.")
        return None

    df = pd.read_csv(csvs[-1])
    if "charge_fraction" not in df.columns:
        print("charge_fraction not in training CSV — retrain to capture this metric.")
        return None

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    ax.plot(df["iter"], df["charge_fraction"] * 100,
            linewidth=1.5, color="steelblue", label="Charge")
    ax.plot(df["iter"], df["image_fraction"] * 100,
            linewidth=1.5, color="tomato", label="Image")
    ax.fill_between(df["iter"], 0, df["charge_fraction"] * 100,
                    alpha=0.15, color="steelblue")
    ax.fill_between(df["iter"], df["charge_fraction"] * 100, 100,
                    alpha=0.15, color="tomato")

    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Action Frequency (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.legend(fontsize=9)
    ax.axhline(50, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    _label_subplots([ax])
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Value function estimates during evaluation (#4)
# ---------------------------------------------------------------------------

def plot_value_estimates(train_log_dir: Path,
                          title: str = "Custom PPO — Value Estimates vs Actual Return"):
    """
    Compares the critic's V(s) estimates during evaluation against
    the actual discounted return G_t. Good critic = close tracking.
    """
    # Load eval steps CSV if it exists
    eval_csvs = list((train_log_dir.parent / "eval_outputs").glob("ppo_eval_steps.csv"))
    if not eval_csvs:
        print("No eval_steps CSV found — run ppo_eval.py first.")
        return None

    df = pd.read_csv(eval_csvs[0])
    if "return_" not in df.columns:
        print("return_ column not in eval CSV — skipping value estimate plot.")
        return None

    episodes  = df["episode"].unique()
    min_steps = df.groupby("episode")["step"].count().min()
    df_trim   = df[df["step"] < min_steps]
    norm_steps = np.linspace(0, 1, min_steps)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=13)

    # Mean actual return
    returns = np.array([
        df_trim[df_trim["episode"] == ep]["return_"].values
        for ep in episodes
    ])
    mean_ret = returns.mean(axis=0)
    std_ret  = returns.std(axis=0)

    ax.plot(norm_steps, mean_ret, linewidth=1.5, color="steelblue", label="Actual Return G_t")
    ax.fill_between(norm_steps, mean_ret - std_ret, mean_ret + std_ret,
                    alpha=0.2, color="steelblue")

    ax.set_xlabel("Episode Progress (normalised)")
    ax.set_ylabel("Return G_t")
    ax.legend(fontsize=9)

    _label_subplots([ax])
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Imaging disparity analysis
# ---------------------------------------------------------------------------

def analyze_imaging_disparity(df_steps: pd.DataFrame, title: str = "Imaging Disparity Analysis"):
    """
    Analyzes the disparity between selected_target (what the policy chose)
    and imaged_target (what BSK-RL actually imaged).

    Three outcomes per imaging step:
      - Match:    selected == imaged  (policy intent executed correctly)
      - Mismatch: different city imaged (slew timing caused queue shift)
      - No image: nothing imaged      (failed to reach target in time)

    Prints a console summary and returns a figure with two panels.
    """
    if "selected_target" not in df_steps.columns or "imaged_target" not in df_steps.columns:
        print("selected_target / imaged_target columns not found — skipping disparity analysis.")
        return None

    # Only look at imaging actions (action != 0 = Charge)
    img = df_steps[df_steps["action"] != 0].copy()
    total = len(img)

    no_image = img["imaged_target"].isna() | (img["imaged_target"] == "")
    match    = (img["selected_target"] == img["imaged_target"]) & ~no_image
    mismatch = ~match & ~no_image

    n_match    = match.sum()
    n_mismatch = mismatch.sum()
    n_no_image = no_image.sum()

    # ── Console summary ───────────────────────────────────────────────────────
    print("" + "=" * 65)
    print("IMAGING DISPARITY ANALYSIS")
    print("=" * 65)
    print(f"  Total imaging action steps : {total}")
    print(f"  Target matched             : {n_match:4d}  ({n_match/total*100:.1f}%)")
    print(f"  Different city imaged      : {n_mismatch:4d}  ({n_mismatch/total*100:.1f}%)")
    print(f"  No image achieved          : {n_no_image:4d}  ({n_no_image/total*100:.1f}%)")

    if "d_ts" in img.columns:
        print(f"  Mean step duration (match)    : {img[match]['d_ts'].mean():.1f}s")
        print(f"  Mean step duration (mismatch) : {img[mismatch]['d_ts'].mean():.1f}s")
        print(f"  Mean step duration (no image) : {img[no_image]['d_ts'].mean():.1f}s")

    print("  Interpretation:")
    print(f"  - Mismatch steps often indicate the satellite imaged a nearby")
    print(f"    target faster than expected and captured a second city within")
    print(f"    the same step window — these are still successful imaging events.")
    print(f"  - No-image steps indicate the satellite could not slew to the")
    print(f"    selected target within the step duration, suggesting the policy")
    print(f"    sometimes selects targets that are too far away to reach in time.")
    print("=" * 65)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    fig.suptitle(title, fontsize=9, fontweight="bold")

    # Pie chart of outcomes
    sizes  = [n_match, n_mismatch, n_no_image]
    labels = [
        f"Match({n_match/total*100:.1f}%)",
        f"Different city imaged({n_mismatch/total*100:.1f}%)",
        f"No image({n_no_image/total*100:.1f}%)",
    ]
    colors = ["steelblue", "darkorange", "tomato"]
    axes[0].pie(sizes, labels=labels, colors=colors, startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=1.5))
    axes[0].set_title("Imaging Outcome Distribution")

    # Per-episode bar chart
    episodes = sorted(df_steps["episode"].unique())
    ep_match    = []
    ep_mismatch = []
    ep_no_image = []

    for ep in episodes:
        ep_img = img[img["episode"] == ep]
        ep_no  = ep_img["imaged_target"].isna() | (ep_img["imaged_target"] == "")
        ep_m   = (ep_img["selected_target"] == ep_img["imaged_target"]) & ~ep_no
        ep_mis = ~ep_m & ~ep_no
        ep_match.append(ep_m.sum())
        ep_mismatch.append(ep_mis.sum())
        ep_no_image.append(ep_no.sum())

    x = np.arange(len(episodes))
    w = 0.25
    axes[1].bar(x - w, ep_match,    width=w, label="Match",               color="steelblue")
    axes[1].bar(x,     ep_mismatch, width=w, label="Different city imaged",color="darkorange")
    axes[1].bar(x + w, ep_no_image, width=w, label="No image",             color="tomato")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Imaging Outcomes per Episode")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Ep {ep}" for ep in episodes])
    axes[1].legend(fontsize=8)

    _label_subplots(axes)
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Summary table and statistical comparison
# ---------------------------------------------------------------------------

def print_summary_table(ppo_summary: pd.DataFrame, heur_summary: pd.DataFrame, rnd_summary: pd.DataFrame):
    """
    Print a formatted summary table comparing PPO vs random across all metrics,
    with t-test p-values to assess statistical significance.
    """
    from scipy import stats

    metrics = [
        ("total_reward",      "Total Reward"),
        ("cities_imaged",     "Cities Imaged"),
        ("imaging_efficiency","Imaging Efficiency"),
        ("steps",             "Episode Length"),
    ]

    print("\n" + "=" * 95)
    print("RESULTS SUMMARY")
    print("=" * 95)
    print(f"{'Metric':<25} {'PPO':>10} {'':>8} {'Heuristic':>10} {'':>8} {'Random':>10} {'':>8} {'p (PPO/Heur)':>13} {'p (PPO/Rnd)':>12}")
    print(f"{'':25} {'Mean':>10} {'Std':>8} {'Mean':>10} {'Std':>8} {'Mean':>10} {'Std':>8}")
    print("-" * 95)

    for col, label in metrics:
        if col not in ppo_summary.columns:
            continue
        ppo_vals  = ppo_summary[col].values.astype(float)
        heur_vals = heur_summary[col].values.astype(float) if col in heur_summary.columns else np.array([float("nan")])
        rnd_vals  = rnd_summary[col].values.astype(float)  if col in rnd_summary.columns  else np.array([float("nan")])

        _, p_heur = stats.ttest_ind(ppo_vals, heur_vals)
        _, p_rnd  = stats.ttest_ind(ppo_vals, rnd_vals)

        def sig(p):
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return ""

        print(f"{label:<25} {ppo_vals.mean():>10.3f} {ppo_vals.std():>8.3f} "
              f"{heur_vals.mean():>10.3f} {heur_vals.std():>8.3f} "
              f"{rnd_vals.mean():>10.3f} {rnd_vals.std():>8.3f} "
              f"{p_heur:>12.4f}{sig(p_heur):3} {p_rnd:>10.4f}{sig(p_rnd):3}")

    print("-" * 95)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("=" * 95)

    ppo_rew  = ppo_summary["total_reward"].values
    heur_rew = heur_summary["total_reward"].values
    rnd_rew  = rnd_summary["total_reward"].values
    _, p_heur = stats.ttest_ind(ppo_rew, heur_rew)
    _, p_rnd  = stats.ttest_ind(ppo_rew, rnd_rew)
    pct_heur = (ppo_rew.mean() - heur_rew.mean()) / abs(heur_rew.mean()) * 100
    pct_rnd  = (ppo_rew.mean() - rnd_rew.mean())  / abs(rnd_rew.mean())  * 100

    print(f"\nPPO vs Heuristic: {pct_heur:.1f}% improvement  (p={p_heur:.4f}{'*' if p_heur < 0.05 else ''})")
    print(f"PPO vs Random:    {pct_rnd:.1f}% improvement  (p={p_rnd:.4f}{'*' if p_rnd < 0.05 else ''})")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate custom PPO checkpoint")
    parser.add_argument("--episodes",  type=int, default=5,  help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--no-plot",   action="store_true",   help="Skip plots")
    args = parser.parse_args()

    paths = TrainConfig()
    paths.eval_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = find_latest_checkpoint(paths.checkpoint_dir)
    model, hp, obs_dim, act_dim = load_checkpoint(checkpoint_path)

    # ── Run PPO rollouts ──────────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} PPO greedy episodes (max {args.max_steps} steps each)...")
    df_steps, df_summary = run_rollout(model, args.episodes, args.max_steps)

    print("\nEpisode summary (PPO):")
    print(df_summary.to_string(index=False))
    print(f"\nMean total reward: {df_summary['total_reward'].mean():.4f}")
    print(f"Std  total reward: {df_summary['total_reward'].std():.4f}")

    # ── Run random rollouts ───────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} heuristic episodes (max {args.max_steps} steps each)...")
    df_heur_steps, df_heur_summary = run_heuristic_rollout(args.episodes, args.max_steps)

    print("\nEpisode summary (Heuristic):")
    print(df_heur_summary.to_string(index=False))

    # ── Random baseline rollout ───────────────────────────────────────────────
    print(f"\nRunning {args.episodes} random episodes (max {args.max_steps} steps each)...")
    df_rnd_steps, df_rnd_summary = run_random_rollout(args.episodes, args.max_steps)

    print("\nEpisode summary (Random):")
    print(df_rnd_summary.to_string(index=False))

    # ── Summary table + statistical comparison ────────────────────────────────
    print_summary_table(df_summary, df_heur_summary, df_rnd_summary)

    # pdb.set_trace()
    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_steps.to_csv(paths.eval_dir / "ppo_eval_steps.csv",               index=False)
    df_summary.to_csv(paths.eval_dir / "ppo_eval_summary.csv",           index=False)
    df_heur_steps.to_csv(paths.eval_dir / "heuristic_eval_steps.csv",    index=False)
    df_heur_summary.to_csv(paths.eval_dir / "heuristic_eval_summary.csv",index=False)
    df_rnd_steps.to_csv(paths.eval_dir / "random_eval_steps.csv",        index=False)
    df_rnd_summary.to_csv(paths.eval_dir / "random_eval_summary.csv",    index=False)
    print(f"\nCSVs saved to: {paths.eval_dir}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        ckpt_label = checkpoint_path.stem

        fig1 = plot_episode_metrics(df_steps, title=f"Custom PPO — Evaluation ({ckpt_label})")
        fig2_losses, fig2_env = plot_training_curve(
            paths.train_log_dir, title="Custom PPO — Training Curve"
        )
        fig3 = plot_comparison(
            df_steps, df_heur_steps, df_rnd_steps,
            df_summary, df_heur_summary, df_rnd_summary,
            title=f"PPO vs Heuristic vs Random — {ckpt_label}",
        )

        fig1.savefig(paths.eval_dir / "ppo_eval_metrics.png",     dpi=300, bbox_inches="tight")
        fig3.savefig(paths.eval_dir / "ppo_vs_baselines.png",     dpi=300, bbox_inches="tight")
        print(f"Saved: {paths.eval_dir / 'ppo_eval_metrics.png'}")
        print(f"Saved: {paths.eval_dir / 'ppo_vs_baselines.png'}")

        fig4 = plot_cities_imaged(
            df_summary, df_heur_summary, df_rnd_summary,
            title=f"Cities Imaged — PPO vs Heuristic vs Random ({ckpt_label})",
        )
        fig4.savefig(paths.eval_dir / "ppo_cities_imaged.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {paths.eval_dir / 'ppo_cities_imaged.png'}")

        fig5 = plot_action_distribution(paths.train_log_dir)
        if fig5 is not None:
            fig5.savefig(paths.eval_dir / "ppo_action_distribution.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_action_distribution.png'}")

        fig6 = plot_value_estimates(paths.train_log_dir)
        if fig6 is not None:
            fig6.savefig(paths.eval_dir / "ppo_value_estimates.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_value_estimates.png'}")

        if fig2_losses is not None:
            fig2_losses.savefig(paths.eval_dir / "ppo_training_losses.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_training_losses.png'}")
        if fig2_env is not None:
            fig2_env.savefig(paths.eval_dir / "ppo_training_env.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_training_env.png'}")

        fig7 = analyze_imaging_disparity(
            df_steps,
            title=f"Imaging Disparity — PPO ({ckpt_label})",
        )
        if fig7 is not None:
            fig7.savefig(paths.eval_dir / "ppo_imaging_disparity.png", dpi=300, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_imaging_disparity.png'}")

        plt.show()


if __name__ == "__main__":
    main()