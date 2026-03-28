from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw

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

    dyn_type=dyn.ContinuousImagingDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

def make_satellite(name: str = "EO1") -> StarterImagingSatellite:
    """
    Factory function so env construction stays clean
    Add sat_args here as project gets more specific
    """
    sat_args = {
        # Add deterministic parameters fist
        # Example
        #   "oe": 
        #   "initial_attitude": 
        #   "battery_init"
    }
    return StarterImagingSatellite(name=name, sat_args=sat_args)