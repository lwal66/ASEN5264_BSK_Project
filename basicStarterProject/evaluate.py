from config import EnvConfig
from envs import make_env


def run_handcrafted_policy(episodes: int = 3):
    cfg = EnvConfig()
    env = make_env(cfg)

    for ep in range(episodes):
        obs, info = env.reset(seed=cfg.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0

        while not done and step_count < 50:
            # same simple baseline as debug script
            action = int(step_count % 2)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1
            done = terminated or truncated

        print(
            f"episode={ep+1} "
            f"steps={step_count} "
            f"total_reward={total_reward:.4f}"
        )

    env.close()


if __name__ == "__main__":
    run_handcrafted_policy()
