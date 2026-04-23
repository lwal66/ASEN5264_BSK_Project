from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_circular_orbit
import numpy as np


class StarterCityImagingSatellite(sats.ImagingSatellite):
    """
    City imaging satellite for Level 2 objectives.

    Actions:  0 = Charge,  1 = Image (up to n_ahead targets ahead)
    Obs:      battery, storage, 5x target opportunity properties, eclipse
    """

    n_ahead = 5

    action_spec = [
        act.Charge(),
        act.Image(n_ahead_image=n_ahead),
    ]

    observation_spec = [
        obs.SatProperties(
            dict(prop="battery_charge_fraction"),
            dict(prop="storage_level_fraction"),
        ),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="target_angle",       norm=np.pi / 2),
            dict(prop="opportunity_open",   norm=300.0),
            dict(prop="opportunity_close",  norm=300.0),
            type="target",
            n_ahead_observe=n_ahead,
        ),
        obs.Eclipse(norm=5700),
    ]

    fsw_type = fsw.SteeringImagerFSWModel


def make_satellite(name: str = "EO1") -> StarterCityImagingSatellite:
    battery_capacity = 120.0 * 3600  # W*s

    sat_args = dict(
        oe=lambda: random_circular_orbit(alt=500),
        batteryStorageCapacity=battery_capacity,
        storedCharge_Init=lambda: battery_capacity * np.random.uniform(0.4, 0.9),
        basePowerDraw=-20.0,
        instrumentPowerDraw=-50.0,
        dataStorageCapacity=200 * 8e6 * 100,
        # Required by CityTargets for access window computation
        imageTargetMinimumElevation=np.arctan(800 / 500),
        # Required by SteeringImagerFSWModel
        imageAttErrorRequirement=0.01,
        imageRateErrorRequirement=0.01,
        # Atmospheric drag (LEO at 500km altitude)
        dragCoeff=2.2,          # drag coefficient (dimensionless)
        # Attitude control
        u_max=0.4,
        K1=0.25,
        K3=3.0,
        omega_max=np.radians(5),
        servo_Ki=5.0,
        servo_P=150 / 5,
    )
    return StarterCityImagingSatellite(name=name, sat_args=sat_args)