# ppo/loss.py
import torch

def ppo_loss(network, obs, actions, old_log_probs, advantages, returns,
             clip_eps=0.2, value_coeff=0.5, entropy_coeff=0.01):
    log_probs, entropy, values = network.evaluate(obs, actions)
    ratio        = torch.exp(log_probs - old_log_probs)
    policy_loss  = -torch.min(ratio * advantages,
                              torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages).mean()
    value_loss   = ((values - returns) ** 2).mean()
    entropy_loss = -entropy.mean()
    total_loss   = policy_loss + value_coeff * value_loss + entropy_coeff * entropy_loss
    return total_loss, {
        "policy_loss": policy_loss.item(),
        "value_loss":  value_loss.item(),
        "entropy":     -entropy_loss.item(),
        "ratio_mean":  ratio.mean().item(),
    }