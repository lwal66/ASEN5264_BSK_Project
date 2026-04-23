import gymnasium as gym
import numpy as np

from config import EnvConfig
from rewards import make_rewarder
#from rewards import MyRewarder
from satellites import make_satellite
from scenarios import make_scenario


# Battery fraction below this threshold triggers a failure termination
BATTERY_FAILURE_THRESHOLD = 0.05   # 5%
BATTERY_FAILURE_PENALTY   = -10.0  # reward penalty applied on failure


class BatteryFailureWrapper(gym.Wrapper):
    """
    Terminates the episode and applies a penalty reward when the satellite's
    battery drops below BATTERY_FAILURE_THRESHOLD (obs[0] = battery_charge_fraction).

    This gives the agent an explicit signal that draining the battery is bad,
    making battery management a learnable objective (Level 2).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        battery_fraction = float(obs[0])
        if battery_fraction < BATTERY_FAILURE_THRESHOLD:
            reward     = BATTERY_FAILURE_PENALTY
            terminated = True
            info["battery_failure"] = True

        return obs, reward, terminated, truncated, info


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

    env = BatteryFailureWrapper(env)
    return env