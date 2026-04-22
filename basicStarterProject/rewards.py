from bsk_rl import data


class PriorityWeightedImageReward(data.UniqueImageReward):
    """
    Extends UniqueImageReward to scale reward by target priority.

    Calls the parent reward() first to handle all unique-image tracking,
    then scales any positive reward by the priority of the imaged target.

    Priority values from CityTargets are in [0, 1].
    A city with priority 1.0 gives full reward; priority 0.5 gives half.
    """

    def reward(self, new_data_dict):
        # Let parent handle unique-image tracking and base reward
        base_rewards = super().reward(new_data_dict)

        priority_rewards = {}
        for sat_id, base_reward in base_rewards.items():
            if base_reward > 0:
                # Find priority of the newly imaged target
                sat_new_data = new_data_dict.get(sat_id)
                priority = 1.0  # safe default
                if sat_new_data is not None:
                    for target in getattr(sat_new_data, "imaged", []):
                        p = getattr(target, "priority", 1.0)
                        priority = float(p)
                        break
                priority_rewards[sat_id] = base_reward * priority
            else:
                priority_rewards[sat_id] = base_reward

        return priority_rewards


def make_rewarder():
    """
    Priority-weighted unique image reward for Level 2 objectives.
    Rewards unique city images scaled by target priority [0, 1].
    """
    return PriorityWeightedImageReward()