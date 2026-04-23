from dataclasses import dataclass
from pathlib import Path
import time

_OUTDIR_ = Path(__file__).resolve().parent / "outdir"

@dataclass
class EnvConfig:
    satellite_name: str = "EO1"
    episode_time_limit_s: float = 5*60*60
    seed: int = 1
    #seed: int = int(time.time())

@dataclass
class TrainConfig:
    # ── RLlib settings ────────────────────────────────────────────────────────
    env_id:           str   = "SatelliteTasking-v1"
    num_workers:      int   = 2
    train_iters:      int   = 25
    lr:               float = 3e-4
    gamma:            float = 0.99
    train_batch_size: int   = 2000
    minibatch_size:   int   = 256
    num_sgd_iter:     int   = 2

    # ── Output paths ──────────────────────────────────────────────────────────
    checkpoint_dir: Path = _OUTDIR_ / "checkpoints"
    train_log_dir:  Path = _OUTDIR_ / "train_logs"
    eval_dir:       Path = _OUTDIR_ / "eval_outputs"


@dataclass
class PPOConfig:
    # ── Network ───────────────────────────────────────────────────────────────
    hidden_sizes:     list = None   # set in __post_init__

    # ── Rollout ───────────────────────────────────────────────────────────────
    steps_per_iter:   int   = 512  # env steps collected per iteration
    max_ep_steps:     int   = 200   # hard cap per episode inside a rollout

    # ── GAE ───────────────────────────────────────────────────────────────────
    gamma:            float = 0.99  # discount factor
    lam:              float = 0.95  # GAE-λ smoothing parameter

    # ── PPO optimisation ──────────────────────────────────────────────────────
    lr:               float = 3e-4
    n_epochs:         int   = 10    # SGD passes over each batch
    minibatch_size:   int   = 64
    clip_eps:         float = 0.2   # epsilon for clipped surrogate loss
    vf_coef:          float = 0.5   # value loss coefficient
    ent_coef:         float = 0.01  # entropy bonus coefficient
    clip_vf_loss:     bool  = True  # also clip the value function loss
    max_grad_norm:    float = 0.5   # gradient clipping

    # ── Training loop ─────────────────────────────────────────────────────────
    train_iters:      int   = 10
    seed:             int   = 1     # matches EnvConfig.seed
    checkpoint_every: int   = 10    # save a checkpoint every N iterations

    # ── Output paths (shared with TrainConfig) ────────────────────────────────
    checkpoint_dir: Path = _OUTDIR_ / "checkpoints"
    train_log_dir:  Path = _OUTDIR_ / "train_logs"
    eval_dir:       Path = _OUTDIR_ / "eval_outputs"

    def __post_init__(self):
        if self.hidden_sizes is None:
            self.hidden_sizes = [64, 64]