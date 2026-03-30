from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit
from Basilisk.utilities import orbitalMotion
import numpy as np

class StarterImagingSatellite(sats.AccessSatellite):
    """
    Minimual starter Sat

    Keep this compact at first
    - battery fraction
    - storage fraction
    - eclipse state
    - couple simple actions
    """

    observation_spec = [
        obs.SatProperties(
            dict(prop="battery_charge_fraction"),
            dict(prop="storage_level_fraction")
        ),
        obs.Eclipse(),
    ]

    action_spec = [
        act.Charge(duration=600.0),
        act.Scan(duration=120.0)
    ]

    dyn_type = dyn.ContinuousImagingDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

def make_satellite(name: str = "EO1") -> StarterImagingSatellite:
    """
    Factory function so env construction stays clean
    Add sat_args here as project gets more specific
    """
    # oe = orbitalMotion.ClassicElements()
    # oe.a = 7000.0 * 1e3
    # oe.e = 0.1
    # oe.i = np.deg2rad(20.0)
    # oe.omega = 0.0
    # oe.Omega = 0.0
    # oe.f = 0.0

    batteryCapacity = 120.0*3600


    sat_args = {
       
        # Dynamics
        "oe": lambda : random_orbit(
            alt=500,
        ),

        #   "initial_attitude": 


        # Power
        "batteryStorageCapacity": batteryCapacity, # W*s
        #"storedCharge_Init"=lambda: np.random_uniform(0.5, 1.0) * batteryCapacity,
        "basePowerDraw": -20, # W
        "instrumentPowerDraw": -50, # W
    }
    return StarterImagingSatellite(name=name, sat_args=sat_args)