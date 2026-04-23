from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig, CitiesConfig
from envs import make_env

PROJECT_ROOT = str(Path(__file__).resolve().parent)


def env_creator(env_config):
    cfg = EnvConfig(**env_config) if env_config else EnvConfig()
    return make_env(cfg)


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "rllib_checkpoint.json").exists():
        return checkpoint_dir
    candidates = sorted(checkpoint_dir.glob("checkpoint_*"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No checkpoint found in {checkpoint_dir}.\nRun trainCities.py first."
    )


def build_algo(cities_cfg: CitiesConfig):
    env_cfg = EnvConfig()

    register_env(cities_cfg.env_name, env_creator)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env=cities_cfg.env_name,
            env_config={
                "satellite_name":       env_cfg.satellite_name,
                "episode_time_limit_s": env_cfg.episode_time_limit_s,
                "seed":                 env_cfg.seed,
            },
            disable_env_checking=False,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length="auto",
        )
        .training(
            lr=cities_cfg.lr,
            gamma=cities_cfg.gamma,
            train_batch_size=cities_cfg.train_batch_size,
            minibatch_size=cities_cfg.minibatch_size,
            num_epochs=cities_cfg.num_sgd_iter,
            model={
                "fcnet_hiddens":    list(cities_cfg.hidden_sizes),
                "fcnet_activation": cities_cfg.activation,
            },
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )

    return config.build()


def action_to_int(action):
    if isinstance(action, tuple):
        action = action[0]
    return int(action)


def rollout_policy(algo, episodes=5, max_steps=1000):
    cfg = EnvConfig()
    env = make_env(cfg)

    step_records = []
    summary_records = []

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.seed + ep)

        done = False
        step = 0
        total_reward = 0.0

        while not done and step < max_steps:
            action = algo.compute_single_action(obs, explore=False)
            action = action_to_int(action)

            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            record = {
                "episode": ep,
                "step": step,
                "action": action,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

            # store all observation elements with generic names
            for i, value in enumerate(next_obs):
                record[f"obs_{i}"] = float(value)

            step_records.append(record)

            obs = next_obs
            step += 1
            done = terminated or truncated

        final_record = {
            "episode": ep,
            "steps": step,
            "total_reward": float(total_reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        # save final observation values too, if available
        for i, value in enumerate(obs):
            final_record[f"final_obs_{i}"] = float(value)

        summary_records.append(final_record)

    env.close()

    df_steps = pd.DataFrame(step_records)
    df_summary = pd.DataFrame(summary_records)
    return df_steps, df_summary


def plot_episode(df_steps, episode, outdir):
    ep_df = df_steps[df_steps["episode"] == episode].copy()
    if ep_df.empty:
        return

    obs_cols = [c for c in ep_df.columns if c.startswith("obs_")]
    n_obs = len(obs_cols)

    # Layout:
    # row 1: reward and cumulative reward
    # row 2: action
    # remaining rows: observations, 1 per subplot
    n_subplots = 2 + n_obs
    fig, axes = plt.subplots(n_subplots, 1, figsize=(11, 2.3 * n_subplots), sharex=True)

    if n_subplots == 1:
        axes = [axes]

    # Reward subplot
    axes[0].plot(ep_df["step"], ep_df["reward"], label="reward")
    axes[0].plot(ep_df["step"], ep_df["total_reward"], label="cumulative_reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    # Action subplot
    axes[1].step(ep_df["step"], ep_df["action"], where="post")
    axes[1].set_ylabel("Action")

    # Observation subplots
    for idx, col in enumerate(obs_cols):
        ax = axes[2 + idx]
        ax.plot(ep_df["step"], ep_df[col])
        ax.set_ylabel(col)

    axes[-1].set_xlabel("Step")
    fig.suptitle(f"Evaluation Episode {episode}", y=0.995)
    plt.tight_layout()

    outpath = outdir / f"episode_{episode:02d}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": PROJECT_ROOT}})

    cities_cfg = CitiesConfig()
    cities_cfg.eval_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = find_latest_checkpoint(cities_cfg.checkpoint_dir)
    print(f"Restoring from checkpoint: {checkpoint_path}")

    algo = build_algo(cities_cfg)
    algo.restore(str(checkpoint_path))

    df_steps, df_summary = rollout_policy(algo=algo, episodes=5, max_steps=1000)

    df_steps.to_csv(cities_cfg.eval_dir / "eval_steps.csv", index=False)
    df_summary.to_csv(cities_cfg.eval_dir / "eval_summary.csv", index=False)

    print("\nEvaluation summary:")
    print(df_summary.to_string(index=False))

    print("\nGenerating episode plots...")
    for ep in sorted(df_summary["episode"].unique()):
        plot_episode(df_steps, ep, cities_cfg.eval_dir)


if __name__ == "__main__":
    main()