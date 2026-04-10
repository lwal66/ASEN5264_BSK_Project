from Basilisk.architecture import bskLogging
bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

import os
import pandas as pd
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig
from envs import make_env

import pdb

def env_creator(env_config):
    cfg = EnvConfig(**env_config) if env_config else EnvConfig()
    return make_env(cfg)


def main():
    cfg = EnvConfig()

    register_env("bsk_rl_city_targets_env", env_creator)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="bsk_rl_city_targets_env",
            env_config={
                "satellite_name": cfg.satellite_name,
                "episode_time_limit_s": cfg.episode_time_limit_s,
                "seed": cfg.seed,
            },
            disable_env_checking=False,
        )
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            sample_timeout_s=1200.0,
            rollout_fragment_length="auto",
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            train_batch_size=1000,
            minibatch_size=256,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "tanh",
            },
        )
        .reporting(
            metrics_episode_collection_timeout_s=180,
            metrics_num_episodes_for_smoothing=10,
        )
        .resources(num_gpus=0)
        .debugging(log_level="WARN")
    )

    algo = config.build()

    history = []

    for i in range(40):
        result = algo.train()

        envm = result.get("env_runners", {})
        fault = result.get("fault_tolerance", {})
        timers = result.get("timers", {})
        connectorMetrics = result.get("connector_metrics", {})
        #pdb.set_trace()

        row = {
            "iteration": result.get("training_iteration"),
            "episode_reward_mean": envm.get("episode_return_mean"),
            "episode_reward_min": envm.get("episode_return_min"),
            "episode_reward_max": envm.get("episode_return_max"),
            "episode_len_mean": envm.get("episode_len_mean"),
            "num_enenvv_steps_sampled": result.get("num_env_steps_sampled"),
            "num_episodes": envm.get("num_episodes"),
            "healthy_workers": fault.get("num_healthy_workers"),
            "worker_restarts": fault.get("num_remote_worker_restarts"),
            "iter_time_s": timers.get("training_iteration"),
            "sample_time_s": timers.get("env_runner_sampling_timer"),
            "learn_time_s": timers.get("learner_update_timer"),
        }
        history.append(row)

        # print(
        #     f"iter={row['iteration']} | "
        #     f"reward_mean={row['episode_reward_mean']:.3f} | "
        #     f"reward_min={row['episode_reward_min']:.3f} | "
        #     f"reward_max={row['episode_reward_max']:.3f} | "
        #     f"ep_len_mean={row['episode_len_mean']:.3f} | "
        #     f"steps={row['num_env_steps_sampled']} | "
        #     f"episodes={row['num_episodes']} | "
        #     f"healthy_workers={row['healthy_workers']} | "
        #     f"restarts={row['worker_restarts']} | "
        #     f"iter_time={row['iter_time_s']:.2f}s"
        # )

    os.makedirs("training_outputs_powerDraw", exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv("training_outputs_powerDraw/train_history.csv", index=False)

    checkpoint_dir = algo.save()
    print(f"Saved checkpoint to: {checkpoint_dir}")

    print("\nTraining history:")
    print(df)


if __name__ == "__main__":
    main()