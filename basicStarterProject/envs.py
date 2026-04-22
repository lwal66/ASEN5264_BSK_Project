import gymnasium as gym

from config import EnvConfig
from rewards import make_rewarder
#from rewards import MyRewarder
from satellites import make_satellite
from scenarios import make_scenario


def make_env(cfg: EnvConfig | None = None):
    cfg = cfg or EnvConfig()

    satellite = make_satellite(name=cfg.satellite_name)
    scenario  = make_scenario()
    rewarder  = make_rewarder()

    env = gym.make(
        "SatelliteTasking-v1",
        satellite=satellite,
        scenario=scenario,
        rewarder=rewarder,
        sim_rate=0.5,
        max_step_duration=300.0,
        time_limit=cfg.episode_time_limit_s,
        failure_penalty=0.0,
        terminate_on_time_limit=True,
        log_level="WARNING",
    )
    return env
