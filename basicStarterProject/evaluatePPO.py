from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig, TrainConfig
from envs import make_env

import pdb

ACTION_NAMES = {
    0: "Charge",
    1: "Scan",
}


def env_creator(env_config):
    cfg = EnvConfig(**env_config) if env_config else EnvConfig()
    return make_env(cfg)


def build_algo_for_eval():
    register_env("bsk_rl_class_env", env_creator)

    cfg = EnvConfig()

    config = (
        PPOConfig()
        .environment(
            env="bsk_rl_class_env",
            env_config={
                "satellite_name": cfg.satellite_name,
                "episode_time_limit_s": cfg.episode_time_limit_s,
                "seed": cfg.seed,
            },
        )
        .framework("torch")
        .env_runners(num_env_runners=0)
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    return config.build()


def rollout_policy(algo, episodes=5, max_steps=50):
    cfg = EnvConfig()
    env = make_env(cfg)

    all_records = []
    summary_records = []

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.seed + ep)

        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            action = algo.compute_single_action(obs, explore=False)

            # Some RLlib versions may return tuples in some configs
            if isinstance(action, tuple):
                action = action[0]

            action = int(action)

            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            all_records.append({
                "episode": ep,
                "step": step,
                "action": action,
                "action_name": ACTION_NAMES.get(action, str(action)),
                "reward": reward,
                "total_reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
            })

            obs = next_obs
            step += 1
            done = terminated or truncated

        summary_records.append({
            "episode": ep,
            "steps": step,
            "total_reward": total_reward,
            "terminated": terminated,
            "truncated": truncated,
        })

    env.close()

    df_steps = pd.DataFrame(all_records)
    df_summary = pd.DataFrame(summary_records)

    return df_steps, df_summary


def plot_episode(df_steps, episode, fig, axes):
    ep_df = df_steps[df_steps["episode"] == episode].copy()


    axes[0].plot(ep_df["step"], ep_df["reward"], label=f"Episode {episode}")
#   axes[2].plot(ep_df["step"], ep_df["total_reward"], label="Cumulative Reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    #fig.suptitle(f"Episode {episode}", y=0.995)


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)

    # Case 1: checkpoint files are directly in checkpoint_dir (flat structure)
    if (checkpoint_dir / "rllib_checkpoint.json").exists():
        return checkpoint_dir

    # Case 2: RLlib created numbered subfolders e.g. checkpoint_000025
    candidates = sorted(checkpoint_dir.glob("checkpoint_*"))
    if candidates:
        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}.\nRun train.py first."
    )


def main():
    cfg = TrainConfig()
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = find_latest_checkpoint(cfg.checkpoint_dir)
    print(f"Restoring from checkpoint: #{checkpoint_path}")

    algo = build_algo_for_eval()
    algo.restore(str(cfg.checkpoint_dir))

    df_steps, df_summary = rollout_policy(algo, episodes=5, max_steps=100)

    print("\nEpisode summary:")
    print(df_summary)

    df_steps.to_csv(cfg.eval_dir / "eval_steps.csv", index=False)
    df_summary.to_csv(cfg.eval_dir / "eval_summary.csv", index=False)

    fig, axes = plt.subplots(5, 1, figsize=(10, 6), sharex=True)
    for ep in df_summary["episode"]:
        plot_episode(df_steps, ep, fig, axes)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main()