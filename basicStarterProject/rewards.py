from bsk_rl import data


def make_rewarder():
    """
    Reward unique city images — each city can only be rewarded once.
    Encourages the satellite to image as many distinct targets as possible.
    """
    return data.UniqueImageReward()
