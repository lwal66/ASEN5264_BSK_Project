from bsk_rl import data

def make_rewarder():
    """
    Minimum reward: reward productive scan time
    """
    return data.ScanningTimeReward()