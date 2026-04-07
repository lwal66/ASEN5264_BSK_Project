from dataclasses import dataclass
from pathlib import Path
import time

_OUTDIR_ = Path(__file__).resolve().parent / "outdir"

@dataclass
class EnvConfig:
    satellite_name: str = "EO1"
    episode_time_limit_s: float = 3*60*60
    seed: int = 1
    #seed: int = int(time.time())

@dataclass
class TrainConfig:
    env_id: str = "SatelliteTasking-v1"
    num_workers: int = 2
    train_iters: int = 25
    lr: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 2000
    minibatch_size: int = 256
    num_sgd_iter: int = 2

    checkpoint_dir: Path = _OUTDIR_ / "checkpoints"
    train_log_dir:  Path = _OUTDIR_ / "train_logs"
    eval_dir:       Path = _OUTDIR_ / "eval_outputs"