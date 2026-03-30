# ppo/network.py
import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.policy_mean    = nn.Linear(hidden_size, act_dim)
        self.policy_log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head     = nn.Linear(hidden_size, 1)
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, obs):
        features = self.trunk(obs)
        mean     = self.policy_mean(features)
        std      = self.policy_log_std.exp().expand_as(mean)
        value    = self.value_head(features).squeeze(-1)
        return mean, std, value

    def get_action(self, obs):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        mean, std, value = self.forward(obs)
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value