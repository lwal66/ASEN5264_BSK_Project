from bsk_rl import data


# class MyRewarder(data.GlobalReward):
#     def reset_pre_sim_init(self):
#         return super().reset_pre_sim_init()
    
#     def reward(self, new_data_dict):
#         print("new_data_dict =")
#         reward = {}

#         for sat_id, new_data in new_data_dict.items():
#             reward[sat_id] = 0.0

#             for item  in getattr(new_data, "imaged", []):
#                 if item not in self.already_rewarded:
#                     reward[sat_id] += 10.0
#                     self.already_rewarded.add(item)

#         return reward

def make_rewarder():
    """
    Minimum reward: reward productive scan time
    """
    return data.UniqueImageReward()