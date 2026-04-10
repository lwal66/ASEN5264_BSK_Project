from dataclasses import dataclass

@dataclass
class EnvConfig:
    satellite_name: str = "EO1"
    episode_time_limit_s: float = 5*60*60
    seed: int = 1

@dataclass
class TrainConfig:
    env_id: str = "StelliteTasking-v1"
    num_workers: int = 2
    train_iters: int = 25
    lr: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 2000
    minibatch_size: int = 256
    num_sgd_iter: int = 2