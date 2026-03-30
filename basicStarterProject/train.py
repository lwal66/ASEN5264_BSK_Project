from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from config import EnvConfig, TrainConfig
from envs import make_env

from datetime import datetime
import pdb
import csv

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
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    outfilename = f"outdir\{datetime.now().strftime('%Y%m%d_%H%M%S')}_train_basicSim.csv"
    with open(outfilename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "episode_reward_mean", "episode_reward_min", "episode_reward_max", "episode_len_means"])

    for i in range(train_cfg.train_iters):
        result = algo.train()

        rewardMean = result.get('env_runners').get('episode_return_mean')
        rewardMin = result.get('env_runners').get('episode_return_min')
        rewardMax = result.get('env_runners').get('episode_return_max')
        lenMean = result.get('env_runners').get('episode_len_mean')
        row = [str(i+1),str(rewardMean),str(rewardMin),str(rewardMax),str(lenMean)]

        with open(outfilename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    
    checkpoint_dir = algo.save()
    print(f"Saved checkpoint to: {checkpoint_dir}")

if __name__ == "__main__":
    main()
