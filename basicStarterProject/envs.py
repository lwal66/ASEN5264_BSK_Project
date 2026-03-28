import gymnasium as gym

from config import EnvConfig
from rewards import make_rewarder
from satellites import make_satellite
from scenarios import make_scenario

def make_env(cfg: EnvConfig | None=None):
    cfg = cfg or EnvConfig()

    satellite = make_satellite(name=cfg.satellite_name)
    scenario = make_scenario()
    rewarder = make_rewarder()

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=satellite,
        scenario=scenario,
        rewarder=rewarder,
        time_limit=cfg.episode_time_limit_s,
    )
    return env
