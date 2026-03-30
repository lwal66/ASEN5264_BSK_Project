# envs/mock_spacecraft_env.py
import gymnasium as gym
import numpy as np

class MockSpacecraftEnv(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space      = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.max_steps         = config.get("max_steps", 200)
        self._step             = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        quat   = self.np_random.standard_normal(4).astype(np.float32)
        quat  /= np.linalg.norm(quat)
        rates  = (self.np_random.standard_normal(3) * 0.1).astype(np.float32)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._state = np.concatenate([quat, rates, target])
        return self._state.copy(), {}

    def step(self, action):
        self._step += 1
        quat, rates, target = self._state[:4], self._state[4:7], self._state[7:]
        rates = rates + action * 0.01
        rates *= 0.99
        quat  = quat + 0.5 * rates[0] * quat
        quat /= np.linalg.norm(quat)
        self._state = np.concatenate([quat, rates, target]).astype(np.float32)
        reward      = -(1.0 - abs(quat[0])) - 0.1 * np.linalg.norm(rates) - 0.01 * np.linalg.norm(action)
        terminated  = self._step >= self.max_steps
        return self._state.copy(), reward, terminated, False, {}