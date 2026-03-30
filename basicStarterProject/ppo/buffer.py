# ppo/buffer.py
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.obs         = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((buffer_size, act_dim), dtype=np.float32)
        self.rewards     = np.zeros(buffer_size, dtype=np.float32)
        self.dones       = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs   = np.zeros(buffer_size, dtype=np.float32)
        self.values      = np.zeros(buffer_size, dtype=np.float32)
        self.advantages  = np.zeros(buffer_size, dtype=np.float32)
        self.returns     = np.zeros(buffer_size, dtype=np.float32)
        self.ptr         = 0

    def store(self, obs, action, reward, done, log_prob, value):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = action
        self.rewards[self.ptr]   = reward
        self.dones[self.ptr]     = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr]    = value
        self.ptr += 1

    def compute_gae(self, last_value):
        gae = 0.0
        for t in reversed(range(self.buffer_size)):
            next_value = last_value if t == self.buffer_size - 1 else self.values[t + 1]
            next_done  = 0.0       if t == self.buffer_size - 1 else self.dones[t + 1]
            delta      = self.rewards[t] + self.gamma * next_value * (1 - next_done) - self.values[t]
            gae        = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batches(self, batch_size):
        indices = np.random.permutation(self.buffer_size)
        for start in range(0, self.buffer_size, batch_size):
            idx = indices[start:start + batch_size]
            yield (
                torch.FloatTensor(self.obs[idx]),
                torch.FloatTensor(self.actions[idx]),
                torch.FloatTensor(self.log_probs[idx]),
                torch.FloatTensor(self.advantages[idx]),
                torch.FloatTensor(self.returns[idx]),
            )

    def reset(self):
        self.ptr = 0