from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

import os
import pandas as pd
import matplotlib.pyplot as plt

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig
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
                "battery": next_obs[0],
                "storage": next_obs[1],
                "eclipse_start": next_obs[2],
                "eclipse_end": next_obs[3],
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


    axes[0].plot(ep_df["step"], ep_df["battery"], label=f"Episode {episode}")
    axes[0].set_ylabel("Battery")

    axes[1].plot(ep_df["step"], ep_df["storage"], label=f"Episode {episode}")
    axes[1].set_ylabel("Storage")

    axes[2].plot(ep_df["step"], ep_df["reward"], label=f"Episode {episode}")
#   axes[2].plot(ep_df["step"], ep_df["total_reward"], label="Cumulative Reward")
    axes[2].set_ylabel("Reward")
    axes[2].legend()

#   axes[3].plot(ep_df["step"], ep_df["eclipse_start"], label="Eclipse Start")
    axes[3].plot(ep_df["step"], ep_df["eclipse_end"], label=f"Episode {episode}")
    axes[3].set_ylabel("Eclipse")
    axes[3].legend()

    axes[4].step(ep_df["step"], ep_df["action"], label=f"Episode {episode}")
    axes[4].set_ylabel("Action")
    axes[4].set_xlabel("Step")
    axes[4].set_yticks(sorted(ep_df["action"].unique()))

    #fig.suptitle(f"Episode {episode}", y=0.995)


def main():
    checkpoint_path = r"C:\Users\mplan\AppData\Local\Temp\tmprhaw2rbc"

    algo = build_algo_for_eval()
    algo.restore(checkpoint_path)

    df_steps, df_summary = rollout_policy(algo, episodes=5, max_steps=100)

    print("\nEpisode summary:")
    print(df_summary)

    os.makedirs("eval_outputs", exist_ok=True)
    df_steps.to_csv("eval_outputs/eval_steps.csv", index=False)
    df_summary.to_csv("eval_outputs/eval_summary.csv", index=False)

    fig, axes = plt.subplots(5, 1, figsize=(10, 6), sharex=True)
    for ep in df_summary["episode"]:
        plot_episode(df_steps, ep, fig, axes)
    plt.show()
    plt.tight_layout()


if __name__ == "__main__":
    main()