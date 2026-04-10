from bsk_rl import act, obs, sats
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_orbit
from Basilisk.utilities import orbitalMotion
import numpy as np

class StarterCityImagingSatellite(sats.ImagingSatellite):

    n_ahead = 5

    action_spec = [
        act.Charge(),
        act.Image(n_ahead_image=n_ahead)
    ]


    observation_spec = [
        obs.SatProperties(
            dict(prop="battery_charge_fraction"),
            dict(prop="storage_level_fraction")
        ),
        obs.OpportunityProperties(
            dict(prop="priority"),
            #dict(prop="r_LB_H", norm=800e3),
            dict(prop="target_angle", norm=np.pi/2),
            dict(prop="opportunity_open", norm=300.0),
            dict(prop="opportunity_close", norm=300.0),
            type="target",
            n_ahead_observe=n_ahead
        ),
        obs.Eclipse(norm=5700)
    ]



    #dyn_type = dyn.ContinuousImagingDynModel
    fsw_type = fsw.SteeringImagerFSWModel



def make_satellite(name: str = "EO1") -> StarterCityImagingSatellite:
    sat_args = dict(
        batteryStorageCapacity=120.0 * 3600,
        storedCharge_Init=lambda: 120.0 * 3600 * np.random.uniform(0.4, 0.9),
        basePowerDraw=-10.0,
        instrumentPowerDraw=-50.0,

        imageAttErrorRequirement=0.01,
        imageRateErrorRequirement=0.01,
        dataStorageCapacity=200 * 8e6 * 100,
        u_max=0.4,
        imageTargetMinimumElevation=np.arctan(800 / 500),
        K1=0.25,
        K3=3.0,
        omega_max=np.radians(5),
        servo_Ki=5.0,
        servo_P=150 / 5,
    )
    return StarterCityImagingSatellite(name=name, sat_args=sat_args)

# def make_satellite(name: str = "EO1") -> StarterImagingSatellite:
#     """
#     Factory function so env construction stays clean
#     Add sat_args here as project gets more specific
#     """
#     # oe = orbitalMotion.ClassicElements()
#     # oe.a = 7000.0 * 1e3
#     # oe.e = 0.1
#     # oe.i = np.deg2rad(20.0)
#     # oe.omega = 0.0
#     # oe.Omega = 0.0
#     # oe.f = 0.0

#     batteryCapacity = 120.0*3600


#     sat_args = {
       
#         # Dynamics
#         "oe": lambda : random_orbit(
#             alt=500,
#         ),

#         #   "initial_attitude": 


#         # Power
#         "batteryStorageCapacity": batteryCapacity, # W*s
#         #"storedCharge_Init"=lambda: np.random_uniform(0.5, 1.0) * batteryCapacity,
#         "basePowerDraw": -20, # W
#         "instrumentPowerDraw": -50, # W
#     }
#     return StarterImagingSatellite(name=name, sat_args=sat_args)