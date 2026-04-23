from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

from pathlib import Path

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


def choose_action(policy_type, env, algo, obs):
    if policy_type == "trained":
        action = algo.compute_single_action(obs, explore=False)
        return action_to_int(action)

    if policy_type == "random":
        return int(env.action_space.sample())

    raise ValueError(f"Unknown policy_type: {policy_type}")


def rollout_policy(policy_type, episodes=20, max_steps=1000, algo=None):
    cfg = EnvConfig()
    env = make_env(cfg)

    step_records = []
    episode_records = []

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.seed + ep)

        done = False
        step = 0
        total_reward = 0.0
        positive_reward_steps = 0

        while not done and step < max_steps:
            action = choose_action(policy_type, env, algo, obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            if reward > 0:
                positive_reward_steps += 1

            record = {
                "policy": policy_type,
                "episode": ep,
                "step": step,
                "action": action,
                "reward": float(reward),
                "total_reward": float(total_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

            for i, value in enumerate(next_obs):
                record[f"obs_{i}"] = float(value)

            step_records.append(record)

            obs = next_obs
            step += 1
            done = terminated or truncated

        ep_record = {
            "policy": policy_type,
            "episode": ep,
            "steps": step,
            "total_reward": float(total_reward),
            "positive_reward_steps": positive_reward_steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

        for i, value in enumerate(obs):
            ep_record[f"final_obs_{i}"] = float(value)

        episode_records.append(ep_record)

    env.close()

    df_steps = pd.DataFrame(step_records)
    df_episodes = pd.DataFrame(episode_records)
    return df_steps, df_episodes


def summarize_by_policy(df_episodes):
    summary = (
        df_episodes.groupby("policy")
        .agg(
            episodes=("episode", "count"),
            reward_mean=("total_reward", "mean"),
            reward_std=("total_reward", "std"),
            reward_min=("total_reward", "min"),
            reward_max=("total_reward", "max"),
            steps_mean=("steps", "mean"),
            steps_std=("steps", "std"),
            positive_reward_steps_mean=("positive_reward_steps", "mean"),
            positive_reward_steps_std=("positive_reward_steps", "std"),
        )
        .reset_index()
    )
    return summary


def print_comparison(summary_df):
    print("\nPolicy comparison:")
    print(summary_df.to_string(index=False))

    trained_row = summary_df[summary_df["policy"] == "trained"]
    random_row = summary_df[summary_df["policy"] == "random"]

    if len(trained_row) == 1 and len(random_row) == 1:
        trained_mean = float(trained_row["reward_mean"].iloc[0])
        random_mean = float(random_row["reward_mean"].iloc[0])

        diff = trained_mean - random_mean
        pct = (diff / random_mean * 100.0) if random_mean != 0 else float("inf")

        print("\nReward comparison:")
        print(f"trained mean reward = {trained_mean:.4f}")
        print(f"random  mean reward = {random_mean:.4f}")
        print(f"absolute difference = {diff:.4f}")
        print(f"percent difference  = {pct:.2f}%")

        if pct < 10:
            print("Interpretation: weak evidence of improvement over random.")
        elif pct < 20:
            print("Interpretation: modest improvement over random.")
        else:
            print("Interpretation: meaningful improvement over random.")


def main():
    # Random rollout runs BEFORE ray.init to avoid BSK-RL access-checking issues
    print("Running random policy rollout...")
    cities_cfg = CitiesConfig()
    cities_cfg.compare_dir.mkdir(parents=True, exist_ok=True)

    random_steps, random_episodes = rollout_policy(
        policy_type="random", episodes=20, max_steps=1000, algo=None,
    )

    # Now initialise Ray and restore trained policy
    ray.init(runtime_env={"env_vars": {"PYTHONPATH": PROJECT_ROOT}})

    checkpoint_path = find_latest_checkpoint(cities_cfg.checkpoint_dir)
    print(f"\nRestoring from checkpoint: {checkpoint_path}")

    algo = build_algo(cities_cfg)
    algo.restore(str(checkpoint_path))

    print("\nRunning trained policy rollout...")
    trained_steps, trained_episodes = rollout_policy(
        policy_type="trained", episodes=20, max_steps=1000, algo=algo,
    )

    df_steps    = pd.concat([trained_steps,    random_steps],    ignore_index=True)
    df_episodes = pd.concat([trained_episodes, random_episodes], ignore_index=True)
    df_summary  = summarize_by_policy(df_episodes)

    df_steps.to_csv(cities_cfg.compare_dir / "policy_steps.csv",    index=False)
    df_episodes.to_csv(cities_cfg.compare_dir / "policy_episodes.csv", index=False)
    df_summary.to_csv(cities_cfg.compare_dir / "policy_summary.csv",  index=False)

    print_comparison(df_summary)
    print(f"\nSaved comparison outputs to: {cities_cfg.compare_dir}")


if __name__ == "__main__":
    main()