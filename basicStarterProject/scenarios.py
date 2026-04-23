from bsk_rl import scene


def make_scenario():
    """
    City targets scenario for Level 2 objectives.
    250 city targets with priorities for the satellite to image.
    """
    return scene.CityTargets(1000)