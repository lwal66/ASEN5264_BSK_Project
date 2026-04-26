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

import pdb

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

            try:
                next_obs, reward, terminated, truncated, info = env.step(action)
                #print(info.get("time"))
                #print(info.get("simulation_time"))
                #print(info.get("currentTime"))
                #print(info.get("dt"))
                #print(info.get("duration"))
                #pdb.set_trace()
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

            #upcomingOpps = env.unwrapped.satellite.find_next_opportunities(n=5, types="target")
            upcomingOpps = env.unwrapped.satellite.upcoming_opportunities
            pdb.set_trace()

            for i, opp in enumerate(upcomingOpps):
                print(i, 
                      opp["object"].name,
                      str(opp["window"]),
                      )
            print("OBS SLOT 0: ", next_obs[2:6])
            print("OBS SLOT 1: ", next_obs[6:10])
            print("OBS SLOT 2: ", next_obs[10:14])
            print("OBS SLOT 3: ", next_obs[14:18])
            print("OBS SLOT 4: ", next_obs[18:22])
            
            pdb.set_trace()
            total_reward += reward                        


            selected_target = upcomingOpps[action - 1]["object"].name if action > 0 else None
            imaged_target = None
            if reward > 0:
                tgt = getattr(env.unwrapped.satellite, "latest_target", None)
                if tgt is not None:
                    imaged_target = tgt.name
            #pdb.set_trace()

            record = {
                "episode": ep,
                "step": step,
                "d_ts": info.get("d_ts"),
                "required_retasking": info.get("requires_retasking"),
                "action": action,
                "selected_target": selected_target,
                "imaged_target": imaged_target,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "target_0_name": upcomingOpps[0]["object"].name,
                "target_1_name": upcomingOpps[1]["object"].name,
                "target_2_name": upcomingOpps[2]["object"].name,
                "target_3_name": upcomingOpps[3]["object"].name,
                "target_4_name": upcomingOpps[4]["object"].name,
                "battery":      float(next_obs[0]),
                "storage":      float(next_obs[1])
            }
            for i in range(2, len(next_obs)):
                record[f"obs_{i}"] = float(next_obs[i])
            all_records.append(record)

            #pdb.set_trace()

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

def run_random_rollout(episodes: int, max_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a random policy for comparison against the trained PPO.
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
                next_obs, reward, terminated, truncated, info = env.step(action)
            except RuntimeError:
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

            #upcomingOpps = env.unwrapped.satellite.find_next_opportunities(n=5, types="target")
            upcomingOpps = env.unwrapped.satellite.upcoming_opportunities

            for i, opp in enumerate(upcomingOpps):
                print(i, 
                      opp["object"].name,
                      opp["opportunity_open"],
                      opp["opportunity_close"]
                      )
            print("OBS SLOT 0: ", next_obs[2:5])
            print("OBS SLOT 1: ", next_obs[6:9])
            print("OBS SLOT 2: ", next_obs[10:13])
            print("OBS SLOT 3: ", next_obs[14:17])
            print("OBS SLOT 4: ", next_obs[18:21])

            pdb.set_trace()
            total_reward += reward   

            selected_target = upcomingOpps[action - 1]["object"].name if action > 0 else None
            imaged_target = None
            if reward > 0:
                tgt = getattr(env.unwrapped.satellite, "latest_target", None)
                if tgt is not None:
                    imaged_target = tgt.name
            #pdb.set_trace()
            record = {
                "episode": ep,
                "step": step,
                "d_ts": info.get("d_ts"),
                "required_retasking": info.get("requires_retasking"),
                "action": action,
                "selected_target": selected_target,
                "imaged_target": imaged_target,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "target_0_name": upcomingOpps[0]["object"].name,
                "target_1_name": upcomingOpps[1]["object"].name,
                "target_2_name": upcomingOpps[2]["object"].name,
                "target_3_name": upcomingOpps[3]["object"].name,
                "target_4_name": upcomingOpps[4]["object"].name,
                "battery":      float(next_obs[0]),
                "storage":      float(next_obs[1])
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
# Plotting
# ---------------------------------------------------------------------------

def plot_episode_metrics(df_steps: pd.DataFrame, title: str = "Custom PPO — Evaluation"):
    """
    Figure 1: battery, storage, reward (mean ± std) + action (per episode).
    Uses normalised episode progress [0, 1] so all episodes contribute equally.
    """
    episodes  = df_steps["episode"].unique()
    min_steps = df_steps.groupby("episode")["step"].count().min()
    df_trim   = df_steps[df_steps["step"] < min_steps]
    norm_steps = np.linspace(0, 1, min_steps)

    fig, axes = plt.subplots(4, 1, figsize=(10, 9))
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
        ax.plot(norm_steps, mean, linewidth=1.5, label="Mean")
        ax.fill_between(norm_steps, mean - std, mean + std,
                        alpha=0.25, label="±1 std")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode Progress (normalised)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper right")

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

    bars = axes[3].bar(action_labels, mean_frac, yerr=std_frac,
                       capsize=4, color="steelblue", alpha=0.8)
    axes[3].set_ylabel("Action Frequency")
    axes[3].set_xlabel("Action")
    axes[3].set_ylim(0, 1)
    axes[3].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    # Annotate bars with percentage
    for bar, frac in zip(bars, mean_frac):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{frac:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
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
    fig_losses, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig_losses.suptitle(title + " — Loss Metrics", fontsize=13)

    axes[0, 0].plot(df["iter"], df["ep_reward_mean"], linewidth=1.5, label="Mean")
    axes[0, 0].fill_between(df["iter"], df["ep_reward_min"], df["ep_reward_max"],
                             alpha=0.2, label="Min / Max")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].set_xlabel("Training Iteration")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(df["iter"], df["loss_policy"], linewidth=1.5, color="tomato")
    axes[0, 1].set_ylabel("Policy Loss (L_CLIP)")
    axes[0, 1].set_xlabel("Training Iteration")
    axes[0, 1].axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[1, 0].plot(df["iter"], df["loss_value"], linewidth=1.5, color="darkorange")
    axes[1, 0].set_ylabel("Value Loss (L_VF)")
    axes[1, 0].set_xlabel("Training Iteration")

    axes[1, 1].plot(df["iter"], df["entropy"], linewidth=1.5, color="green")
    axes[1, 1].set_ylabel("Policy Entropy (H)")
    axes[1, 1].set_xlabel("Training Iteration")

    plt.tight_layout()

    # ── Figure 2b: Environment metrics ───────────────────────────────────────
    has_failure = "battery_failure_rate" in df.columns
    has_ep_len  = "ep_len_mean" in df.columns

    if not has_failure and not has_ep_len:
        return fig_losses, None

    n_panels = int(has_failure) + int(has_ep_len)
    fig_env, env_axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    fig_env.suptitle(title + " — Environment Metrics", fontsize=13)

    if n_panels == 1:
        env_axes = [env_axes]

    ax_idx = 0
    if has_failure:
        env_axes[ax_idx].plot(df["iter"], df["battery_failure_rate"] * 100,
                              linewidth=1.5, color="purple")
        env_axes[ax_idx].set_ylabel("Battery Failure Rate (%)")
        env_axes[ax_idx].set_xlabel("Training Iteration")
        env_axes[ax_idx].set_ylim(0, 100)
        env_axes[ax_idx].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y:.0f}%")
        )
        ax_idx += 1

    if has_ep_len:
        env_axes[ax_idx].plot(df["iter"], df["ep_len_mean"],
                              linewidth=1.5, color="teal")
        env_axes[ax_idx].set_ylabel("Mean Episode Length (steps)")
        env_axes[ax_idx].set_xlabel("Training Iteration")

    plt.tight_layout()
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
    rnd_steps: pd.DataFrame,
    ppo_summary: pd.DataFrame,
    rnd_summary: pd.DataFrame,
    title: str = "Custom PPO vs Random Policy",
    gamma: float = 0.99,
):
    """
    Side-by-side mean ± std comparison of PPO vs random policy.
    Panels: cumulative reward, mean discounted reward, step reward.
    Plus a summary bar chart of mean total reward.
    """
    # Pre-compute discounted reward for both policies
    ppo_steps = _add_return(ppo_steps, gamma)
    rnd_steps = _add_return(rnd_steps, gamma)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    metrics = [
        ("total_reward",      "Cumulative Reward",        axes[0, 0]),
        ("return_",           f"Return G_t (γ={gamma})",        axes[0, 1]),
        ("reward",            "Step Reward",              axes[1, 0]),
    ]

    for col, ylabel, ax in metrics:
        for df, label, color in [
            (ppo_steps, "PPO",    "steelblue"),
            (rnd_steps, "Random", "tomato"),
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
            ax.plot(norm_steps, mean, linewidth=1.5, label=label, color=color)
            ax.fill_between(norm_steps, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode Progress (normalised)")
        ax.legend(fontsize=8)

    # ── Summary bar chart ─────────────────────────────────────────────────────
    ax_bar = axes[1, 1]
    ppo_mean = ppo_summary["total_reward"].mean()
    rnd_mean = rnd_summary["total_reward"].mean()
    ppo_std  = ppo_summary["total_reward"].std()
    rnd_std  = rnd_summary["total_reward"].std()

    bars = ax_bar.bar(
        ["PPO", "Random"],
        [ppo_mean, rnd_mean],
        yerr=[ppo_std, rnd_std],
        color=["steelblue", "tomato"],
        capsize=6,
        width=0.5,
    )
    ax_bar.set_ylabel("Mean Total Reward")
    ax_bar.set_title("Total Reward Comparison")

    # Annotate bars with values
    for bar, mean, std in zip(bars, [ppo_mean, rnd_mean], [ppo_std, rnd_std]):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.1,
            f"{mean:.2f}",
            ha="center", va="bottom", fontsize=9,
        )

    # Print numeric summary
    diff = ppo_mean - rnd_mean
    pct  = (diff / abs(rnd_mean) * 100) if rnd_mean != 0 else float("inf")
    print("\nPolicy comparison:")
    print(f"  PPO    mean reward = {ppo_mean:.4f} ± {ppo_std:.4f}")
    print(f"  Random mean reward = {rnd_mean:.4f} ± {rnd_std:.4f}")
    print(f"  Difference         = {diff:.4f}  ({pct:.1f}%)")
    if pct < 10:
        print("  Interpretation: weak improvement over random.")
    elif pct < 20:
        print("  Interpretation: modest improvement over random.")
    else:
        print("  Interpretation: meaningful improvement over random.")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Cities imaged comparison
# ---------------------------------------------------------------------------

def plot_cities_imaged(ppo_summary, rnd_summary,
                       title="Cities Imaged & Imaging Efficiency"):
    ppo_cities = ppo_summary["cities_imaged"].values
    rnd_cities = rnd_summary["cities_imaged"].values
    ppo_eff    = ppo_summary["imaging_efficiency"].values
    rnd_eff    = rnd_summary["imaging_efficiency"].values

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(title, fontsize=13)

    labels = ["PPO", "Random"]
    colors = ["steelblue", "tomato"]

    for row, (ppo_vals, rnd_vals, ylabel, row_title) in enumerate([
        (ppo_cities, rnd_cities, "Cities Imaged",        "Cities Imaged per Episode"),
        (ppo_eff,    rnd_eff,    "Avg Priority (reward/city)", "Imaging Efficiency (priority-weighted)"),
    ]):
        means = [ppo_vals.mean(), rnd_vals.mean()]
        stds  = [ppo_vals.std(),  rnd_vals.std()]

        # Bar chart
        bars = axes[row, 0].bar(labels, means, yerr=stds, color=colors,
                                capsize=6, width=0.5, alpha=0.8)
        for i, vals in enumerate([ppo_vals, rnd_vals]):
            axes[row, 0].scatter([i] * len(vals), vals, color="black",
                                 s=20, zorder=3, alpha=0.6)
        for bar, mean in zip(bars, means):
            axes[row, 0].text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + max(stds) * 0.1 + 0.01,
                              f"{mean:.2f}", ha="center", va="bottom", fontsize=9)
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 0].set_title(row_title + " — Mean ± Std")
        axes[row, 0].set_ylim(0)

        # Per-episode line
        episodes = range(len(ppo_vals))
        axes[row, 1].plot(episodes, ppo_vals, "o-", color="steelblue",
                          linewidth=1.5, label="PPO")
        axes[row, 1].plot(episodes, rnd_vals, "s-", color="tomato",
                          linewidth=1.5, label="Random")
        axes[row, 1].set_xlabel("Episode")
        axes[row, 1].set_ylabel(ylabel)
        axes[row, 1].set_title(row_title + " — Per Episode")
        axes[row, 1].legend(fontsize=9)
        axes[row, 1].set_xticks(list(episodes))
        axes[row, 1].set_ylim(0)

    # Print summary
    print(f"\nCities imaged:")
    print(f"  PPO    mean = {ppo_cities.mean():.2f} +/- {ppo_cities.std():.2f}")
    print(f"  Random mean = {rnd_cities.mean():.2f} +/- {rnd_cities.std():.2f}")
    print(f"\nImaging efficiency (avg priority per city):")
    print(f"  PPO    mean = {ppo_eff.mean():.3f} +/- {ppo_eff.std():.3f}")
    print(f"  Random mean = {rnd_eff.mean():.3f} +/- {rnd_eff.std():.3f}")
    if ppo_eff.mean() > rnd_eff.mean():
        print("  PPO is targeting higher-priority cities than random.")
    else:
        print("  PPO is not yet prioritising high-value cities over random.")

    plt.tight_layout()
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

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(title, fontsize=13)

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

    plt.tight_layout()
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

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary table and statistical comparison
# ---------------------------------------------------------------------------

def print_summary_table(ppo_summary: pd.DataFrame, rnd_summary: pd.DataFrame):
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

    col_w = 22
    print("\n" + "=" * 75)
    print("RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Metric':<25} {'PPO Mean':>10} {'PPO Std':>9} {'Rnd Mean':>10} {'Rnd Std':>9} {'p-value':>9}")
    print("-" * 75)

    for col, label in metrics:
        if col not in ppo_summary.columns or col not in rnd_summary.columns:
            continue
        ppo_vals = ppo_summary[col].values.astype(float)
        rnd_vals = rnd_summary[col].values.astype(float)

        t_stat, p_val = stats.ttest_ind(ppo_vals, rnd_vals)

        sig = ""
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        print(f"{label:<25} {ppo_vals.mean():>10.3f} {ppo_vals.std():>9.3f} "
              f"{rnd_vals.mean():>10.3f} {rnd_vals.std():>9.3f} "
              f"{p_val:>8.4f}{sig}")

    print("-" * 75)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")
    print("=" * 75)

    # Overall interpretation
    ppo_rew = ppo_summary["total_reward"].values
    rnd_rew = rnd_summary["total_reward"].values
    _, p_rew = stats.ttest_ind(ppo_rew, rnd_rew)
    pct = (ppo_rew.mean() - rnd_rew.mean()) / abs(rnd_rew.mean()) * 100

    print(f"\nPPO achieves {pct:.1f}% {'improvement' if pct >= 0 else 'degradation'} "
          f"over random in total reward.")
    if p_rew < 0.05:
        print("This difference is statistically significant (p < 0.05).")
    else:
        print("This difference is NOT statistically significant — consider running "
              "more episodes for a stronger result.")
    print()


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

    # ── Run PPO rollouts ──────────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} PPO greedy episodes (max {args.max_steps} steps each)...")
    df_steps, df_summary = run_rollout(model, args.episodes, args.max_steps)
    #pdb.set_trace()
    print("\nEpisode summary (PPO):")
    print(df_summary.to_string(index=False))
    print(f"\nMean total reward: {df_summary['total_reward'].mean():.4f}")
    print(f"Std  total reward: {df_summary['total_reward'].std():.4f}")

    # ── Run random rollouts ───────────────────────────────────────────────────
    print(f"\nRunning {args.episodes} random episodes (max {args.max_steps} steps each)...")
    df_rnd_steps, df_rnd_summary = run_random_rollout(args.episodes, args.max_steps)

    print("\nEpisode summary (Random):")
    print(df_rnd_summary.to_string(index=False))

    # ── Summary table + statistical comparison ────────────────────────────────
    print_summary_table(df_summary, df_rnd_summary)

    pdb.set_trace()
    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_steps.to_csv(paths.eval_dir / "ppo_eval_steps.csv",         index=False)
    df_summary.to_csv(paths.eval_dir / "ppo_eval_summary.csv",     index=False)
    df_rnd_steps.to_csv(paths.eval_dir / "random_eval_steps.csv",  index=False)
    df_rnd_summary.to_csv(paths.eval_dir / "random_eval_summary.csv", index=False)
    print(f"\nCSVs saved to: {paths.eval_dir}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plot:
        ckpt_label = checkpoint_path.stem

        fig1 = plot_episode_metrics(df_steps, title=f"Custom PPO — Evaluation ({ckpt_label})")
        fig2_losses, fig2_env = plot_training_curve(
            paths.train_log_dir, title="Custom PPO — Training Curve"
        )
        fig3 = plot_comparison(
            df_steps, df_rnd_steps,
            df_summary, df_rnd_summary,
            title=f"Custom PPO vs Random — {ckpt_label}",
        )

        fig1.savefig(paths.eval_dir / "ppo_eval_metrics.png",  dpi=150, bbox_inches="tight")
        fig3.savefig(paths.eval_dir / "ppo_vs_random.png",     dpi=150, bbox_inches="tight")
        print(f"Saved: {paths.eval_dir / 'ppo_eval_metrics.png'}")
        print(f"Saved: {paths.eval_dir / 'ppo_vs_random.png'}")

        fig4 = plot_cities_imaged(
            df_summary, df_rnd_summary,
            title=f"Cities Imaged — PPO vs Random ({ckpt_label})",
        )
        fig4.savefig(paths.eval_dir / "ppo_cities_imaged.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {paths.eval_dir / 'ppo_cities_imaged.png'}")

        fig5 = plot_action_distribution(paths.train_log_dir)
        if fig5 is not None:
            fig5.savefig(paths.eval_dir / "ppo_action_distribution.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_action_distribution.png'}")

        fig6 = plot_value_estimates(paths.train_log_dir)
        if fig6 is not None:
            fig6.savefig(paths.eval_dir / "ppo_value_estimates.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_value_estimates.png'}")

        if fig2_losses is not None:
            fig2_losses.savefig(paths.eval_dir / "ppo_training_losses.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_training_losses.png'}")
        if fig2_env is not None:
            fig2_env.savefig(paths.eval_dir / "ppo_training_env.png", dpi=150, bbox_inches="tight")
            print(f"Saved: {paths.eval_dir / 'ppo_training_env.png'}")

        plt.show()


if __name__ == "__main__":
    main()