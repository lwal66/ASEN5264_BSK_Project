from dataclasses import dataclass

@dataclass
class EnvConfig:
    satellite_name: str = "EO1"
    episode_time_limit_s: float = 5400.0 # 90 min
    seed: int = 1

@dataclass
class TrainConfig:
    env_id: str = "StelliteTasking-v1"
    num_workers: int = 1
    train_iters: int = 1
    lr: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 1000
    minibatch_size: int = 256
    num_sgd_iter: int = 2