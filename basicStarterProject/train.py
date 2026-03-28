from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig, TrainConfig
from envs import make_env

import pdb

from Basilisk.architecture import bskLogging

def env_creator(env_config):
    cfg = EnvConfig(**env_config) if env_config else EnvConfig()
    return make_env(cfg)

def main():

    bskLogging.setDefaultLogLevel(bskLogging.BSK_WARNING)

    train_cfg = TrainConfig()
    env_cfg = EnvConfig()

    register_env("bsk_rl_basicStartProject_env", env_creator)

    config = (
        PPOConfig()
        .environment(
            env="bsk_rl_basicStartProject_env",
            env_config={
                "satellite_name": env_cfg.satellite_name,
                "episode_time_limit_s": env_cfg.episode_time_limit_s,
                "seed": env_cfg.seed,
            },
        )
        .framework("torch")
        .env_runners(num_env_runners=train_cfg.num_workers)
        .training(
            lr=train_cfg.lr,
            gamma=train_cfg.gamma,
            train_batch_size=train_cfg.train_batch_size,
            minibatch_size=train_cfg.minibatch_size,
            num_sgd_iter=train_cfg.num_sgd_iter
        )
    )

    algo = config.build()

    for i in range(train_cfg.train_iters):
        result = algo.train()
        print(
            f"iter={i+1} "
            f"episode_reward_mean={result.get('episode_reward_mean')}"
            f"episode_len_means={result.get('episode_len_mean')}"
        )
    
    checkpoint_dir = algo.save()
    print(f"Saved checkpoint to: {checkpoint_dir}")

if __name__ == "__main__":
    main()
