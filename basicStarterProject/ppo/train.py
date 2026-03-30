# ppo/train.py
import os
import sys
import torch
import numpy as np
from torch.optim import Adam
sys.path.insert(0, os.path.dirname(__file__))
from network import ActorCritic
from buffer import RolloutBuffer
from loss import ppo_loss
from mock_spacecraft_env import MockSpacecraftEnv

config = {
    "obs_dim": 10, "act_dim": 3, "max_steps": 200, "hidden_size": 64,
    "total_steps": 500_000, "buffer_size": 2048, "batch_size": 64, "n_epochs": 10,
    "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95, "clip_eps": 0.2,
    "value_coeff": 0.5, "entropy_coeff": 0.01, "max_grad_norm": 0.5,
}

def collect_rollout(env, network, buffer):
    obs, _ = env.reset()
    episode_rewards, ep_reward = [], 0.0
    for _ in range(buffer.buffer_size):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = network.get_action(obs_tensor)
        action_np = np.clip(action.squeeze(0).numpy(), env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        ep_reward += reward
        buffer.store(obs, action_np, reward, float(done), log_prob.item(), value.item())
        obs = next_obs
        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs, _ = env.reset()
    with torch.no_grad():
        _, _, last_value = network.get_action(torch.FloatTensor(obs).unsqueeze(0))
    return last_value.item(), episode_rewards

def update(network, optimizer, buffer, config):
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "ratio_mean": []}
    for _ in range(config["n_epochs"]):
        for obs, actions, old_log_probs, advantages, returns in buffer.get_batches(config["batch_size"]):
            loss, info = ppo_loss(network, obs, actions, old_log_probs, advantages, returns,
                                  config["clip_eps"], config["value_coeff"], config["entropy_coeff"])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), config["max_grad_norm"])
            optimizer.step()
            for k, v in info.items():
                metrics[k].append(v)
    return {k: np.mean(v) for k, v in metrics.items()}

def evaluate(env, network, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward, done = 0.0, False
        while not done:
            with torch.no_grad():
                mean, _, _ = network.forward(torch.FloatTensor(obs).unsqueeze(0))
            obs, reward, terminated, truncated, _ = env.step(
                np.clip(mean.squeeze(0).numpy(), env.action_space.low, env.action_space.high))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    return np.mean(rewards)

def train():
    torch.manual_seed(42)
    np.random.seed(42)
    env       = MockSpacecraftEnv(config)
    network   = ActorCritic(config["obs_dim"], config["act_dim"], config["hidden_size"])
    optimizer = Adam(network.parameters(), lr=config["lr"])
    buffer    = RolloutBuffer(config["buffer_size"], config["obs_dim"], config["act_dim"],
                              config["gamma"], config["gae_lambda"])
    total_steps, update_count, best_reward = 0, 0, -np.inf

    print(f"{'Update':>8} {'Steps':>10} {'Eval Reward':>12} {'Policy Loss':>12} {'Value Loss':>12} {'Entropy':>10}")
    print("-" * 70)

    while total_steps < config["total_steps"]:
        last_value, episode_rewards = collect_rollout(env, network, buffer)
        buffer.compute_gae(last_value)
        metrics = update(network, optimizer, buffer, config)
        buffer.reset()
        total_steps  += config["buffer_size"]
        update_count += 1

        if update_count % 10 == 0:
            eval_reward = evaluate(env, network)
            print(f"{update_count:>8} {total_steps:>10} {eval_reward:>12.2f} "
                  f"{metrics['policy_loss']:>12.4f} {metrics['value_loss']:>12.4f} "
                  f"{metrics['entropy']:>10.4f}")
            if eval_reward > best_reward:
                best_reward = eval_reward
                torch.save({"network": network.state_dict(), "optimizer": optimizer.state_dict(),
                            "config": config, "best_reward": best_reward}, "best_model.pt")

    print(f"\nDone. Best eval reward: {best_reward:.2f}")

if __name__ == "__main__":
    train()